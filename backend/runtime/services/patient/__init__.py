"""
runtime.services.patient
------------------------
Postgres-backed patient service. Sole Postgres credential holder.

Order of operations on every call:
  PDP.decide → SET ROLE app_runtime + RLS vars → query → field trim → audit → return

Security invariants enforced here:
  - Authorization re-derived from PDP on every call; never cached.
  - RLS (app_runtime role + session/patient_scope vars) acts as an independent
    backstop for every clinical query.
  - Denied resources/fields never enter the response.
  - Every call writes an audit row (permit or deny).
  - Patient identifier never reaches the knowledge service.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from runtime.policy.pdp import decide
from runtime.policy.audit import write_audit
from runtime.seed.db import get_conn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource metadata — maps resource name to DB table and allowed columns
# ---------------------------------------------------------------------------

_RESOURCE_META: dict[str, dict] = {
    "demographics": {
        "table":   "patients",
        "columns": ["name", "dob", "sex", "mrn", "address", "insurance_id", "department", "care_team"],
        "single":  True,   # single row per patient
    },
    "conditions": {
        "table":   "conditions",
        "columns": ["code", "label", "onset_date", "status"],
        "single":  False,
    },
    "vitals": {
        "table":   "vitals",
        "columns": ["taken_at", "bp_systolic", "bp_diastolic", "heart_rate", "temp_c", "resp_rate"],
        "single":  False,
    },
    "medications": {
        "table":   "medications",
        "columns": ["drug_name", "dose", "route", "frequency", "prescriber_user_id", "status"],
        "single":  False,
    },
    "lab_results": {
        "table":   "lab_results",
        "columns": ["test_name", "value", "unit", "reference_range", "collected_at", "status", "entered_by_user_id"],
        "single":  False,
    },
    "genetic_markers": {
        "table":   "genetic_markers",
        "columns": ["gene", "variant", "interpretation", "source"],
        "single":  False,
    },
    "clinical_notes": {
        "table":   "clinical_notes",
        "columns": ["author_user_id", "created_at", "body"],
        "single":  False,
    },
}


def _serialize(v: Any) -> Any:
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _set_rls(cur, session_id: str, patient_scope: list[str]) -> None:
    """Switch to app_runtime role and set transaction-local RLS session variables."""
    cur.execute("SET ROLE app_runtime")
    cur.execute("SELECT set_config('app.session_id', %s, true)", (session_id,))
    cur.execute("SELECT set_config('app.patient_scope', %s, true)",
                (",".join(patient_scope),))


# ---------------------------------------------------------------------------
# Tool: get_patient_record
# ---------------------------------------------------------------------------

def get_patient_record(
    patient_id: str,
    resources: list[str],
    claims: dict,
    session_id: str,
) -> dict[str, Any]:
    """
    Fetch one or more resource groups for a pinned patient.

    The patient_id is always injected by ToolClient from the pinned chart,
    never from the model.
    """
    user_id = claims["sub"]
    role_id = claims["role"]

    conn = get_conn()
    try:
        result: dict[str, Any] = {}
        for resource in resources:
            meta = _RESOURCE_META.get(resource)
            if meta is None:
                result[resource] = {"granted": False, "reason": f"unknown resource: {resource}"}
                continue

            decision = decide(conn, claims, "read", resource, patient_id, session_id)

            if decision.effect == "deny":
                result[resource] = {"granted": False, "reason": decision.reason}
                write_audit(
                    conn, session_id, user_id, role_id,
                    "read", resource, patient_id,
                    "deny", decision.reason,
                )
                continue

            # research_analyst path — deidentified binding; individual rows not returned
            if decision.patient_binding == "deidentified":
                result[resource] = {"granted": True, "deidentified": True}
                write_audit(
                    conn, session_id, user_id, role_id,
                    "read", resource, None,
                    "permit", "de-identified access path",
                )
                continue

            # Determine which columns to select
            all_cols = meta["columns"]
            if "*" in decision.allowed_fields:
                cols = all_cols
            else:
                cols = [c for c in decision.allowed_fields if c in all_cols]

            if not cols:
                result[resource] = {"granted": True, "items": []}
                write_audit(
                    conn, session_id, user_id, role_id,
                    "read", resource, patient_id,
                    "permit", "no columns in allow-list",
                )
                continue

            col_list = ", ".join(cols)
            table = meta["table"]

            cur = conn.cursor()
            _set_rls(cur, session_id, decision.patient_scope)

            cur.execute(
                f"SELECT {col_list} FROM {table}"
                f" WHERE session_id = %s AND patient_id = %s",
                (session_id, patient_id),
            )
            rows = cur.fetchall()
            cur.execute("RESET ROLE")

            break_glass = "flag_review" in decision.obligations

            if meta["single"]:
                if rows:
                    fields = {col: _serialize(val) for col, val in zip(cols, rows[0])}
                    result[resource] = {"granted": True, "fields": fields}
                else:
                    result[resource] = {"granted": True, "fields": {}}
            else:
                items = [
                    {col: _serialize(val) for col, val in zip(cols, row)}
                    for row in rows
                ]
                result[resource] = {"granted": True, "items": items}

            write_audit(
                conn, session_id, user_id, role_id,
                "read", resource, patient_id,
                "permit", decision.reason,
                break_glass=break_glass,
                fields_accessed=cols,
            )

        conn.commit()
        return {"ok": True, "patient_id": patient_id, "resources": result}

    except Exception as exc:
        conn.rollback()
        logger.exception("get_patient_record failed")
        return {"ok": False, "code": "internal", "message": str(exc)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool: update_lab_result
# ---------------------------------------------------------------------------

def update_lab_result(
    patient_id: str,
    test_name: str,
    value: Any,
    unit: str,
    claims: dict,
    session_id: str,
) -> dict[str, Any]:
    """Update (or insert if absent) a lab result for the pinned patient."""
    user_id = claims["sub"]
    role_id = claims["role"]

    conn = get_conn()
    try:
        decision = decide(conn, claims, "write", "lab_results", patient_id, session_id)

        if decision.effect == "deny":
            write_audit(
                conn, session_id, user_id, role_id,
                "write", "lab_results", patient_id,
                "deny", decision.reason,
            )
            conn.commit()
            return {"ok": False, "code": "forbidden", "message": decision.reason}

        cur = conn.cursor()
        _set_rls(cur, session_id, decision.patient_scope)

        # Update existing row; if none exists under RLS scope, insert it.
        cur.execute(
            """
            UPDATE lab_results
               SET value = %s, unit = %s, status = 'updated'
             WHERE session_id = %s AND patient_id = %s AND test_name = %s
            """,
            (value, unit, session_id, patient_id, test_name),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO lab_results
                  (session_id, patient_id, test_name, value, unit, status)
                VALUES (%s, %s, %s, %s, %s, 'active')
                """,
                (session_id, patient_id, test_name, value, unit),
            )

        cur.execute("RESET ROLE")

        break_glass = "flag_review" in decision.obligations
        write_audit(
            conn, session_id, user_id, role_id,
            "write", "lab_results", patient_id,
            "permit", decision.reason,
            break_glass=break_glass,
            fields_accessed=["value", "unit"],
        )
        conn.commit()
        return {"ok": True, "patient_id": patient_id, "updated": test_name}

    except Exception as exc:
        conn.rollback()
        logger.exception("update_lab_result failed")
        return {"ok": False, "code": "internal", "message": str(exc)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool: update_medication
# ---------------------------------------------------------------------------

def update_medication(
    patient_id: str,
    drug_name: str,
    dose: str,
    status: str,
    claims: dict,
    session_id: str,
) -> dict[str, Any]:
    """Update (or insert if absent) a medication record for the pinned patient."""
    user_id = claims["sub"]
    role_id = claims["role"]

    conn = get_conn()
    try:
        decision = decide(conn, claims, "write", "medications", patient_id, session_id)

        if decision.effect == "deny":
            write_audit(
                conn, session_id, user_id, role_id,
                "write", "medications", patient_id,
                "deny", decision.reason,
            )
            conn.commit()
            return {"ok": False, "code": "forbidden", "message": decision.reason}

        cur = conn.cursor()
        _set_rls(cur, session_id, decision.patient_scope)

        cur.execute(
            """
            UPDATE medications
               SET dose = %s, status = %s
             WHERE session_id = %s AND patient_id = %s AND drug_name = %s
            """,
            (dose, status, session_id, patient_id, drug_name),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO medications
                  (session_id, patient_id, drug_name, dose, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, patient_id, drug_name, dose, status),
            )

        cur.execute("RESET ROLE")

        break_glass = "flag_review" in decision.obligations
        write_audit(
            conn, session_id, user_id, role_id,
            "write", "medications", patient_id,
            "permit", decision.reason,
            break_glass=break_glass,
            fields_accessed=["dose", "status"],
        )
        conn.commit()
        return {"ok": True, "patient_id": patient_id, "updated": drug_name}

    except Exception as exc:
        conn.rollback()
        logger.exception("update_medication failed")
        return {"ok": False, "code": "internal", "message": str(exc)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool: get_deidentified_aggregate
# ---------------------------------------------------------------------------

def get_deidentified_aggregate(
    claims: dict,
    session_id: str,
) -> dict[str, Any]:
    """
    Return de-identified population statistics for a research analyst.

    No patient_id, name, mrn, dob, or address ever appears in the output.
    Uses the neondb_owner connection (which bypasses RLS) to run aggregate
    queries across all patients in the session, but query construction
    ensures no identity columns are selected.
    """
    user_id = claims["sub"]
    role_id = claims["role"]

    conn = get_conn()
    try:
        decision = decide(conn, claims, "read", "deident_aggregates", None, session_id)

        if decision.effect == "deny":
            write_audit(
                conn, session_id, user_id, role_id,
                "read", "deident_aggregates", None,
                "deny", decision.reason,
            )
            conn.commit()
            return {"ok": False, "code": "forbidden", "message": decision.reason}

        cur = conn.cursor()

        # APOE genotype distribution — no patient identity in output
        cur.execute(
            """
            SELECT variant, COUNT(*) AS count
            FROM   genetic_markers
            WHERE  session_id = %s AND gene = 'APOE'
            GROUP  BY variant
            ORDER  BY count DESC
            """,
            (session_id,),
        )
        apoe_dist = [{"variant": row[0], "count": row[1]} for row in cur.fetchall()]

        # CSF p-tau181 distribution by age band (no dob/name — age band only)
        cur.execute(
            """
            SELECT
                CASE
                    WHEN age_years < 65  THEN '<65'
                    WHEN age_years < 75  THEN '65-74'
                    WHEN age_years < 85  THEN '75-84'
                    ELSE '85+'
                END AS age_band,
                COUNT(*)         AS n,
                ROUND(AVG(lr.value)::numeric, 1) AS avg_ptau181
            FROM lab_results lr
            JOIN (
                SELECT patient_id,
                       EXTRACT(YEAR FROM age(dob))::int AS age_years
                FROM   patients
                WHERE  session_id = %s
            ) p USING (patient_id)
            WHERE lr.session_id = %s AND lr.test_name = 'CSF p-tau181'
            GROUP BY age_band
            ORDER BY age_band
            """,
            (session_id, session_id),
        )
        ptau_dist = [
            {"age_band": r[0], "n": r[1], "avg_ptau181_pg_ml": float(r[2]) if r[2] else None}
            for r in cur.fetchall()
        ]

        # Active medication counts (no names/doses attributed to identifiable patients)
        cur.execute(
            """
            SELECT drug_name, COUNT(DISTINCT patient_id) AS patient_count
            FROM   medications
            WHERE  session_id = %s AND status = 'active'
            GROUP  BY drug_name
            ORDER  BY patient_count DESC
            """,
            (session_id,),
        )
        med_counts = [{"drug_name": r[0], "patient_count": r[1]} for r in cur.fetchall()]

        write_audit(
            conn, session_id, user_id, role_id,
            "read", "deident_aggregates", None,
            "permit", decision.reason,
            fields_accessed=["apoe_distribution", "ptau181_by_age", "active_medications"],
        )
        conn.commit()
        return {
            "ok": True,
            "apoe_distribution": apoe_dist,
            "ptau181_by_age_band": ptau_dist,
            "active_medication_counts": med_counts,
        }

    except Exception as exc:
        conn.rollback()
        logger.exception("get_deidentified_aggregate failed")
        return {"ok": False, "code": "internal", "message": str(exc)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Audit helper: knowledge access logging (called by ToolClient)
# ---------------------------------------------------------------------------

def log_knowledge_access(
    session_id: str,
    user_id: str,
    role_id: str,
    key: str,
    effect: str,
) -> None:
    """
    Write a knowledge-read audit row. Called by ToolClient after a
    query_knowledge call; must not raise — audit failure blocks nothing.
    """
    conn = get_conn()
    try:
        write_audit(
            conn, session_id, user_id, role_id,
            "read", "knowledge", None,
            effect, f"knowledge query: {key}",
        )
        conn.commit()
    except Exception:
        logger.exception("log_knowledge_access audit write failed (non-fatal)")
    finally:
        conn.close()
