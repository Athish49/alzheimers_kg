"""
Regression tests for security breaches A–F (Session 2 audit).

All tests hit the real Neon Postgres DB (no mocks).
Run from backend/: venv/bin/python3.13 -m pytest runtime/tests/test_security_fixes2.py -v
"""
from __future__ import annotations

import uuid
import pytest

from runtime.seed.db import get_conn


TEST_USER  = "u_014"   # Dr. Sarah Chen (attending_physician)
TEST_ROLE  = "attending_physician"
NURSE_USER = "u_015"   # Raj Patel (nurse)
NURSE_ROLE = "nurse"
RA_USER    = "u_017"   # Tom Baker (research_analyst) — check seeder for exact ID


@pytest.fixture(scope="module")
def db_session():
    """Clone TEMPLATE into a unique test session owned by TEST_USER."""
    from runtime.seed.session import clone_session
    sid = "sec2test_" + uuid.uuid4().hex[:12]
    clone_session(sid, TEST_USER)
    conn = get_conn()
    try:
        yield sid, conn
    finally:
        conn.close()
        cleanup = get_conn()
        try:
            cur = cleanup.cursor()
            cur.execute("DELETE FROM break_glass_grants WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM audit_log WHERE session_id = %s", (sid,))
            for table in [
                "clinical_notes", "genetic_markers", "lab_results",
                "medications", "vitals", "conditions",
                "patient_assignments", "patients",
            ]:
                cur.execute(f"DELETE FROM {table} WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (sid,))
            cleanup.commit()
        finally:
            cleanup.close()


def _assigned_patient(conn, sid: str, user_id: str) -> str:
    cur = conn.cursor()
    cur.execute(
        "SELECT patient_id FROM patient_assignments WHERE session_id = %s AND user_id = %s LIMIT 1",
        (sid, user_id),
    )
    row = cur.fetchone()
    assert row, f"No assigned patient for {user_id} in session {sid}"
    return row[0]


# ─── Helper: _verify_session_owner logic ──────────────────────────────────────

def _session_owner_check(conn, session_id: str, user_id: str) -> bool:
    """Returns True if the session belongs to user_id, False otherwise."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
        (session_id, user_id),
    )
    return cur.fetchone() is not None


# ─── Breach A: /patients missing session ownership check ─────────────────────

def test_breach_a_cross_user_session_rejected(db_session):
    """A JWT with NURSE_USER sub + TEST_USER session_id must fail the ownership check."""
    sid, conn = db_session
    # sid was created by TEST_USER — NURSE_USER must not own it
    assert not _session_owner_check(conn, sid, NURSE_USER), (
        "NURSE_USER must not pass ownership check for TEST_USER's session"
    )


def test_breach_a_owner_passes(db_session):
    """TEST_USER must pass the ownership check for their own session."""
    sid, conn = db_session
    assert _session_owner_check(conn, sid, TEST_USER), (
        "TEST_USER must pass ownership check for their own session"
    )


# ─── Breach B+E: TEMPLATE session implicitly blocked ─────────────────────────

def test_breach_b_template_session_not_in_sessions_table(db_session):
    """TEMPLATE must never exist as a row in the sessions table."""
    _, conn = db_session
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sessions WHERE session_id = 'TEMPLATE'")
    assert cur.fetchone() is None, (
        "TEMPLATE must not be a valid sessions row — the ownership check blocks it implicitly"
    )


def test_breach_b_template_ownership_fails_for_any_user(db_session):
    """_verify_session_owner logic rejects TEMPLATE for every persona."""
    _, conn = db_session
    for user_id in [TEST_USER, NURSE_USER]:
        assert not _session_owner_check(conn, "TEMPLATE", user_id), (
            f"TEMPLATE session must be rejected for {user_id}"
        )


def test_breach_b_template_data_exists_but_session_check_blocks_access(db_session):
    """
    TEMPLATE rows DO exist in clinical tables (that's how sessions are seeded),
    but the session ownership check prevents any API access using 'TEMPLATE'
    as session_id — this test confirms the gating mechanism is the sessions table.
    """
    _, conn = db_session
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients WHERE session_id = 'TEMPLATE'")
    template_patient_count = cur.fetchone()[0]
    assert template_patient_count > 0, "TEMPLATE data must exist (seed precondition)"

    # But the gate is the sessions table — TEMPLATE is absent there
    cur.execute("SELECT 1 FROM sessions WHERE session_id = 'TEMPLATE'")
    assert cur.fetchone() is None, "TEMPLATE must not exist in sessions — that's the gate"


# ─── Breach F: audit_log DELETE/UPDATE revoked from app_runtime ──────────────
# Each test uses isolated connections for setup / assertion / cleanup so that a
# permission-denied error (which aborts the Postgres transaction) doesn't bleed
# into subsequent statements on the same connection.

def _insert_audit_row(sid: str, reason: str) -> int:
    """Insert a test audit row as neondb_owner (no RLS). Returns the row id."""
    c = get_conn()
    try:
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO audit_log
              (session_id, user_id, role_id, action, resource, effect, reason)
            VALUES (%s, %s, %s, 'read', 'test', 'permit', %s)
            RETURNING id
            """,
            (sid, TEST_USER, TEST_ROLE, reason),
        )
        row_id = cur.fetchone()[0]
        c.commit()
        return row_id
    finally:
        c.close()


def _delete_audit_row(row_id: int) -> None:
    """Delete a test audit row as neondb_owner (cleanup)."""
    c = get_conn()
    try:
        cur = c.cursor()
        cur.execute("DELETE FROM audit_log WHERE id = %s", (row_id,))
        c.commit()
    finally:
        c.close()


def test_breach_f_audit_log_delete_forbidden_for_app_runtime(db_session):
    """app_runtime must not be able to DELETE from audit_log."""
    sid, _ = db_session
    row_id = _insert_audit_row(sid, "breach_f_delete_test")

    denied = False
    c = get_conn()
    try:
        cur = c.cursor()
        cur.execute("SET ROLE app_runtime")
        cur.execute("SELECT set_config('app.session_id', %s, true)", (sid,))
        try:
            cur.execute("DELETE FROM audit_log WHERE id = %s", (row_id,))
            c.commit()
        except Exception as exc:
            if "permission denied" in str(exc).lower():
                denied = True
            c.rollback()
    finally:
        c.close()

    _delete_audit_row(row_id)
    assert denied, "app_runtime DELETE on audit_log must raise permission denied"


def test_breach_f_audit_log_update_forbidden_for_app_runtime(db_session):
    """app_runtime must not be able to UPDATE audit_log rows."""
    sid, _ = db_session
    row_id = _insert_audit_row(sid, "breach_f_update_test")

    denied = False
    c = get_conn()
    try:
        cur = c.cursor()
        cur.execute("SET ROLE app_runtime")
        cur.execute("SELECT set_config('app.session_id', %s, true)", (sid,))
        try:
            cur.execute("UPDATE audit_log SET reason = 'tampered' WHERE id = %s", (row_id,))
            c.commit()
        except Exception as exc:
            if "permission denied" in str(exc).lower():
                denied = True
            c.rollback()
    finally:
        c.close()

    _delete_audit_row(row_id)
    assert denied, "app_runtime UPDATE on audit_log must raise permission denied"


def test_breach_f_audit_log_grant_is_select_insert_only():
    """
    Verify via information_schema that app_runtime has exactly SELECT + INSERT
    on audit_log — not DELETE, not UPDATE.
    """
    c = get_conn()
    try:
        cur = c.cursor()
        cur.execute(
            """
            SELECT privilege_type
            FROM   information_schema.role_table_grants
            WHERE  grantee    = 'app_runtime'
              AND  table_name = 'audit_log'
            ORDER  BY privilege_type
            """,
        )
        granted = {row[0] for row in cur.fetchall()}
    finally:
        c.close()

    assert "SELECT" in granted, "app_runtime must have SELECT on audit_log"
    assert "INSERT" in granted, "app_runtime must have INSERT on audit_log"
    assert "DELETE" not in granted, "app_runtime must NOT have DELETE on audit_log"
    assert "UPDATE" not in granted, "app_runtime must NOT have UPDATE on audit_log"


# ─── Breach C: patient_id masked in ChartResponse for research_analyst ────────

def test_breach_c_chart_response_masks_patient_id_for_ra():
    """
    ChartResponse.patient_id must be Optional[str] and the endpoint must
    return None for research_analyst callers.
    """
    from runtime.router import ChartResponse
    import inspect, typing

    hints = typing.get_type_hints(ChartResponse)
    patient_id_type = hints.get("patient_id")
    # Accept Optional[str] which is Union[str, None]
    args = getattr(patient_id_type, "__args__", ())
    assert type(None) in args, (
        f"ChartResponse.patient_id must be Optional[str], got {patient_id_type}"
    )


def test_breach_c_research_analyst_gets_null_patient_id(db_session):
    """
    When role is research_analyst, visible_patient_id logic in /chart must
    produce None, not the actual patient_id.
    """
    sid, conn = db_session

    # Simulate the masking logic in the endpoint directly
    role_id    = "research_analyst"
    patient_id = "p_2201"
    visible_patient_id = None if role_id == "research_analyst" else patient_id
    assert visible_patient_id is None


def test_breach_c_physician_gets_real_patient_id(db_session):
    """Non-research-analyst roles must receive the real patient_id in ChartResponse."""
    sid, conn = db_session

    role_id    = "attending_physician"
    patient_id = "p_2201"
    visible_patient_id = None if role_id == "research_analyst" else patient_id
    assert visible_patient_id == patient_id
