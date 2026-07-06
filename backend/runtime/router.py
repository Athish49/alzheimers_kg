"""
runtime.router
--------------
Top-level FastAPI router for the enterprise runtime plane.
Mounted on the existing app in backend/main.py.

Routes added per phase:
  GET  /runtime/health                Phase 0.1
  GET  /runtime/personas              Phase 2.1 — list selectable personas
  POST /runtime/personas/{id}/select  Phase 2.1 — mint JWT + clone session
  POST /runtime/break-glass           Phase 2.3 — write break-glass grant
  GET  /runtime/audit                 Phase 2.4 — read audit log for session
  POST /runtime/orchestrate           Phase 4   — secure clinical Q&A loop
  GET  /runtime/patients              Phase 5   — list patients assigned to caller
  GET  /runtime/chart/{patient_id}    Phase 5   — fetch permitted chart resources (no LLM)
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated, Optional

import jwt as pyjwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

router = APIRouter(prefix="/runtime", tags=["runtime"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify(authorization: Annotated[str, Header()] = "") -> dict:
    """FastAPI dependency: verify Bearer token, return claims."""
    from runtime.auth.jwt import verify_token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return verify_token(token)
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


def _verify_session_owner(session_id: str, user_id: str, conn) -> None:
    """
    Raise 403 if the session does not belong to the authenticated user.
    Also implicitly rejects synthetic session IDs (e.g. 'TEMPLATE') that
    are never inserted into the sessions table.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
        (session_id, user_id),
    )
    if not cur.fetchone():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session does not belong to the authenticated user.",
        )


# ---------------------------------------------------------------------------
# Phase 0.1 — liveness
# ---------------------------------------------------------------------------

@router.get("/health")
def runtime_health() -> dict:
    """Runtime plane liveness check."""
    return {"status": "ok", "plane": "runtime"}


# ---------------------------------------------------------------------------
# Phase 2.1 — Personas
# ---------------------------------------------------------------------------

class PersonaOut(BaseModel):
    user_id:    str
    name:       str
    role_id:    str
    role_label: str
    department: str
    care_team:  str


@router.get("/personas", response_model=list[PersonaOut])
def list_personas() -> list[PersonaOut]:
    """Return all selectable personas (is_persona=true users)."""
    from runtime.seed.db import get_conn
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT u.user_id, u.name, u.role_id, COALESCE(r.label, u.role_id),
                   u.department, u.care_team
            FROM   users u
            LEFT JOIN roles r ON r.role_id = u.role_id
            WHERE  u.is_persona = true
            ORDER  BY u.user_id
            """
        )
        return [
            PersonaOut(
                user_id=r[0], name=r[1], role_id=r[2], role_label=r[3],
                department=r[4], care_team=r[5],
            )
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


class SelectResponse(BaseModel):
    token:      str
    session_id: str
    user_id:    str
    role_id:    str
    name:       str


@router.post("/personas/{user_id}/select", response_model=SelectResponse)
def select_persona(user_id: str) -> SelectResponse:
    """
    Mint a JWT and clone the TEMPLATE into a fresh session sandbox.
    Called when a visitor picks a persona card.
    """
    from runtime.auth.jwt import mint_token
    from runtime.seed.db import get_conn
    from runtime.seed.session import clone_session

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, role_id, department, care_team, is_persona FROM users WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if not row or not row[4]:
            raise HTTPException(status_code=404, detail=f"Persona '{user_id}' not found")
        name, role_id, department, care_team, _ = row
    finally:
        conn.close()

    session_id = "s_" + uuid.uuid4().hex
    clone_session(session_id, user_id)
    token = mint_token(user_id, role_id, department, care_team, session_id)

    return SelectResponse(
        token=token,
        session_id=session_id,
        user_id=user_id,
        role_id=role_id,
        name=name,
    )


# ---------------------------------------------------------------------------
# Phase 2.3 — Break-glass
# ---------------------------------------------------------------------------

class BreakGlassRequest(BaseModel):
    patient_id: str
    reason:     str


class BreakGlassResponse(BaseModel):
    granted:    bool
    expires_in: str  # human-readable


@router.post("/break-glass", response_model=BreakGlassResponse)
def request_break_glass(
    body: BreakGlassRequest,
    claims: dict = Depends(_verify),
) -> BreakGlassResponse:
    """
    Write a time-boxed (15-min) break-glass grant for the pinned patient.
    Requires a non-empty reason. The grant is surfaced as a flagged audit entry
    when the subsequent request is authorized by the PDP.
    """
    from runtime.seed.db import get_conn
    from runtime.policy.audit import write_audit

    if not body.reason.strip():
        raise HTTPException(status_code=400, detail="A reason is required for break-glass access.")

    session_id = claims["session_id"]
    user_id    = claims["sub"]
    role_id    = claims["role"]

    conn = get_conn()
    try:
        cur = conn.cursor()

        # B6: Validate that patient_id exists in this session before granting.
        cur.execute(
            "SELECT 1 FROM patients WHERE patient_id = %s AND session_id = %s",
            (body.patient_id, session_id),
        )
        if not cur.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient '{body.patient_id}' not found in this session.",
            )

        # B5: Upsert semantics — extend existing active grant rather than stacking.
        cur.execute(
            """
            UPDATE break_glass_grants
               SET reason     = %s,
                   expires_at = now() + interval '15 minutes'
             WHERE session_id = %s
               AND user_id    = %s
               AND patient_id = %s
               AND expires_at > now()
            """,
            (body.reason.strip(), session_id, user_id, body.patient_id),
        )
        if cur.rowcount == 0:
            cur.execute(
                """
                INSERT INTO break_glass_grants
                  (session_id, user_id, patient_id, reason, expires_at)
                VALUES (%s, %s, %s, %s, now() + interval '15 minutes')
                """,
                (session_id, user_id, body.patient_id, body.reason.strip()),
            )
        conn.commit()

        # Audit the break-glass request itself
        write_audit(
            conn,
            session_id=session_id,
            user_id=user_id,
            role_id=role_id,
            action="break_glass",
            resource="demographics",
            patient_id=body.patient_id,
            effect="permit",
            reason=body.reason.strip(),
            break_glass=True,
        )
        conn.commit()
    finally:
        conn.close()

    return BreakGlassResponse(granted=True, expires_in="15 minutes")


# ---------------------------------------------------------------------------
# Phase 2.4 — Audit log panel
# ---------------------------------------------------------------------------

class AuditRow(BaseModel):
    id:              int
    ts:              str
    user_id:         str
    role_id:         str
    action:          str
    resource:        str
    patient_id:      Optional[str]
    effect:          str
    reason:          str
    break_glass:     bool
    fields_accessed: list[str]


@router.get("/audit", response_model=list[AuditRow])
def get_audit_log(claims: dict = Depends(_verify)) -> list[AuditRow]:
    """Return audit log rows for the caller's session (most recent first)."""
    from runtime.seed.db import get_conn

    session_id = claims["session_id"]
    user_id    = claims["sub"]
    conn = get_conn()
    try:
        cur = conn.cursor()
        # Verify session ownership before returning audit rows. A forged
        # session_id in the JWT must not allow reading another user's audit log.
        cur.execute(
            "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
            (session_id, user_id),
        )
        if not cur.fetchone():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session does not belong to the authenticated user.",
            )

        cur.execute(
            """
            SELECT id, ts, user_id, role_id, action, resource,
                   patient_id, effect, reason, break_glass, fields_accessed
            FROM   audit_log
            WHERE  session_id = %s
              AND  user_id    = %s
            ORDER  BY id DESC
            LIMIT  200
            """,
            (session_id, user_id),
        )
        return [
            AuditRow(
                id=r[0], ts=r[1].isoformat(), user_id=r[2], role_id=r[3],
                action=r[4], resource=r[5], patient_id=r[6], effect=r[7],
                reason=r[8], break_glass=r[9], fields_accessed=r[10] or [],
            )
            for r in cur.fetchall()
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Phase 4 — Orchestrate: secure clinical Q&A loop
# ---------------------------------------------------------------------------

class OrchestrateRequest(BaseModel):
    question:   str
    patient_id: Optional[str] = None   # the pinned chart patient


class OrchestrateResponse(BaseModel):
    answer:             str
    patient_evidence:   list[str]
    knowledge_evidence: list[dict]
    abstained_on:       list[str]
    session_id:         str


@router.post("/orchestrate", response_model=OrchestrateResponse)
def orchestrate_endpoint(
    body: OrchestrateRequest,
    request: Request,
    claims: dict = Depends(_verify),
) -> OrchestrateResponse:
    """
    Single entry point for the secure clinical Q&A loop.

    Checks LLM usage caps, runs the orchestrator tool-call loop, calls the
    join layer for synthesis, records the cap unit, and returns the answer
    with evidence and abstained_on fields.

    Non-LLM routes (/personas, /audit, /break-glass) never call this endpoint
    and are unaffected by cap state.
    """
    from runtime.gateway import CapExceededError, check_cap, record_call
    from runtime.join import synthesize
    from runtime.orchestrator.agent import orchestrate
    from runtime.seed.db import get_conn

    session_id = claims["session_id"]
    user_id    = claims["sub"]

    conn = get_conn()
    try:
        _verify_session_owner(session_id, user_id, conn)
    finally:
        conn.close()

    # Prefer X-Forwarded-For (set by Render's proxy) for per-IP cap accuracy.
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else None
    )

    try:
        check_cap(session_id, ip)
    except CapExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    try:
        bundle = orchestrate(
            question=body.question,
            claims=claims,
            pinned_patient_id=body.patient_id,
            session_id=session_id,
        )

        result = synthesize(
            question=body.question,
            patient_bundle=bundle["patient_bundle"],
            knowledge_results=bundle["knowledge_results"],
        )
    except Exception:
        import logging
        logging.getLogger(__name__).exception("orchestrate_endpoint unhandled error")
        raise HTTPException(status_code=503, detail="Orchestration temporarily unavailable.")

    record_call(session_id, ip)

    return OrchestrateResponse(
        answer=result["answer"],
        patient_evidence=result["patient_evidence"],
        knowledge_evidence=result["knowledge_evidence"],
        abstained_on=result["abstained_on"],
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Phase 5 — EHR patient list and chart (Synthea data, no PDP)
# ---------------------------------------------------------------------------

class PatientSummary(BaseModel):
    patient_id: str
    name:       str
    dob:        str
    sex:        str
    mrn:        str
    headline:   str


class PatientListResponse(BaseModel):
    total:    int
    page:     int
    limit:    int
    patients: list[PatientSummary]


@router.get("/patients", response_model=PatientListResponse)
def list_patients(
    page: int = 1,
    limit: int = 20,
    search: str = "",
    claims: dict = Depends(_verify),
) -> PatientListResponse:
    """
    Return paginated EHR patients from ehr.patients (Synthea synthetic data).
    If the caller has EHR patient assignments, only those patients are returned.
    Falls back to all patients for original RBAC demo personas (no assignments).
    """
    import runtime.ehr_service as svc

    limit = min(max(limit, 1), 100)
    page  = max(page, 1)
    user_id = claims["sub"]

    assigned_ids = svc.get_ehr_assignments(user_id)

    if assigned_ids:
        rows  = svc.list_patients_filtered(assigned_ids, page=page, limit=limit, search=search)
        total = svc.count_patients_filtered(assigned_ids, search=search)
    else:
        rows  = svc.list_patients(page=page, limit=limit, search=search)
        total = svc.count_patients(search=search)

    return PatientListResponse(
        total=total,
        page=page,
        limit=limit,
        patients=[
            PatientSummary(
                patient_id=r["patient_id"],
                name=r["name"],
                dob=r["dob"] or "",
                sex=r["sex"] or "",
                mrn=r["mrn"] or "",
                headline=r["headline"] or "",
            )
            for r in rows
        ],
    )


class ChartResponse(BaseModel):
    patient_id: str
    resources:  dict


# Maps each chart section key to its permission resource name.
# Sections that require a specific resource to be readable.
_SECTION_RESOURCE: dict[str, str] = {
    "banner":             "demographics",
    "alerts":             "demographics",
    "emergency_contacts": "demographics",
    "allergies":          "demographics",
    "snapshot":           "demographics",
    "encounters":         "encounters",
    "vitals":             "vitals",
    "conditions":         "conditions",
    "medications":        "medications",
    "labs":               "lab_results",
    "procedures":         "procedures",
    "imaging":            "imaging",
    "notes":              "clinical_notes",
    "care_plans":         "care_plans",
    "immunizations":      "immunizations",
    "social_history":     "social_history",
}


def _get_allowed_resources(role_id: str, conn) -> set[str]:
    """Return the set of resource names this role can read."""
    cur = conn.cursor()
    cur.execute(
        "SELECT resource FROM role_permissions WHERE role_id = %s AND action = 'read'",
        (role_id,),
    )
    return {row[0] for row in cur.fetchall()}


# Maps snapshot sub-keys to the permission resource they require.
_SNAPSHOT_RESOURCE: dict[str, str] = {
    "active_problems":    "conditions",
    "active_medications": "medications",
    "latest_vitals":      "vitals",
    "recent_encounters":  "encounters",
    "recent_labs":        "lab_results",
    "allergies":          "medications",
    "alerts":             "demographics",
    "immunization_status":"immunizations",
}

# Maps encounter detail sub-keys (added after header fetch) to their resource.
_ENCOUNTER_RESOURCE: dict[str, str] = {
    "care_team":  "encounters",
    "diagnoses":  "conditions",
    "vitals":     "vitals",
    "procedures": "procedures",
    "medications":"medications",
    "notes":      "clinical_notes",
    "discharge":  "care_plans",
}


def _filter_snapshot(snapshot: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in snapshot.items() if _SNAPSHOT_RESOURCE.get(k, "demographics") in allowed}


def _filter_encounter(detail: dict, allowed: set[str]) -> dict:
    # header keys (encounter_id, visit_type, etc.) are always shown when encounters is allowed
    return {k: v for k, v in detail.items() if _ENCOUNTER_RESOURCE.get(k, "encounters") in allowed}


@router.get("/chart/{patient_id}", response_model=ChartResponse)
def get_chart(
    patient_id: str,
    claims: dict = Depends(_verify),
) -> ChartResponse:
    """
    Fetch a role-filtered EHR chart from ehr.* tables.
    - Specialists (with EHR assignments) can only access their assigned patients.
    - Chart sections are filtered to the caller's role permissions.
    - Original RBAC personas (no EHR assignments) bypass the assignment gate.
    """
    from runtime import ehr_service as svc
    from runtime.seed.db import get_conn

    user_id = claims["sub"]
    role_id = claims["role"]

    # Assignment gate: only enforce for practitioners who have EHR assignments
    assigned_ids = svc.get_ehr_assignments(user_id)
    if assigned_ids and patient_id not in assigned_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not assigned to this patient.",
        )

    banner = svc.get_patient_banner(patient_id)
    if not banner:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    # Determine which resources this role can read
    conn = get_conn()
    try:
        allowed = _get_allowed_resources(role_id, conn)
    finally:
        conn.close()

    # If the role has no permissions on record (e.g. unknown role), allow all
    # to avoid breaking backwards-compat for old demo personas
    if not allowed:
        allowed = set(_SECTION_RESOURCE.values())

    raw_snapshot = svc.get_snapshot(patient_id)
    filtered_snapshot = _filter_snapshot(raw_snapshot, allowed)

    all_resources = {
        "banner":             banner,
        "alerts":             svc.get_patient_alerts(patient_id),
        "emergency_contacts": svc.get_patient_emergency_contacts(patient_id),
        "allergies":          svc.get_allergies(patient_id),
        "snapshot":           filtered_snapshot,
        "encounters":         svc.get_encounters(patient_id),
        "vitals":             svc.get_vitals(patient_id),
        "conditions":         svc.get_conditions(patient_id),
        "medications":        svc.get_medications(patient_id),
        "labs":               svc.get_labs(patient_id),
        "procedures":         svc.get_procedures(patient_id),
        "imaging":            svc.get_imaging(patient_id),
        "notes":              svc.get_notes(patient_id),
        "care_plans":         svc.get_care_plans(patient_id),
        "immunizations":      svc.get_immunizations(patient_id),
        "social_history":     svc.get_social_history(patient_id),
    }

    # Filter sections to what this role is permitted to read
    resources = {
        section: data
        for section, data in all_resources.items()
        if _SECTION_RESOURCE.get(section, "demographics") in allowed
    }

    return ChartResponse(patient_id=patient_id, resources=resources)


@router.get("/chart/{patient_id}/encounter/{encounter_id}")
def get_encounter_detail(
    patient_id: str,
    encounter_id: str,
    claims: dict = Depends(_verify),
) -> dict:
    """
    Fetch role-filtered encounter detail.
    - Assignment gate: same rule as /chart/{patient_id}.
    - Sub-sections (vitals, procedures, meds, notes, discharge) filtered by role permissions.
    """
    from runtime import ehr_service as svc
    from runtime.seed.db import get_conn

    user_id = claims["sub"]
    role_id = claims["role"]

    # Assignment gate
    assigned_ids = svc.get_ehr_assignments(user_id)
    if assigned_ids and patient_id not in assigned_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not assigned to this patient.",
        )

    detail = svc.get_encounter_detail(patient_id, encounter_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Encounter not found.")

    conn = get_conn()
    try:
        allowed = _get_allowed_resources(role_id, conn)
    finally:
        conn.close()

    if not allowed:
        return detail

    return _filter_encounter(detail, allowed)
