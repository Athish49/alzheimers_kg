"""
runtime.router
--------------
Top-level FastAPI router for the enterprise runtime plane.
Mounted on the existing app in backend/main.py.

Routes added per phase:
  GET  /runtime/health         Phase 0.1
  GET  /runtime/personas       Phase 2.1 — list selectable personas
  POST /runtime/personas/{id}/select  Phase 2.1 — mint JWT + clone session
  POST /runtime/break-glass    Phase 2.3 — write break-glass grant
  GET  /runtime/audit          Phase 2.4 — read audit log for session
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated, Optional

import jwt as pyjwt
from fastapi import APIRouter, Depends, Header, HTTPException, status
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
    department: str
    care_team:  str


@router.get("/personas", response_model=list[PersonaOut])
def list_personas() -> list[PersonaOut]:
    """Return the five selectable demo personas."""
    from runtime.seed.db import get_conn
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT u.user_id, u.name, u.role_id, u.department, u.care_team
            FROM   users u
            WHERE  u.is_persona = true
            ORDER  BY u.user_id
            """
        )
        return [
            PersonaOut(user_id=r[0], name=r[1], role_id=r[2], department=r[3], care_team=r[4])
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
    conn = get_conn()
    try:
        cur = conn.cursor()
        # Read as superuser but filter to session (RLS would do this for app_runtime)
        cur.execute(
            """
            SELECT id, ts, user_id, role_id, action, resource,
                   patient_id, effect, reason, break_glass, fields_accessed
            FROM   audit_log
            WHERE  session_id = %s
            ORDER  BY id DESC
            LIMIT  200
            """,
            (session_id,),
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
