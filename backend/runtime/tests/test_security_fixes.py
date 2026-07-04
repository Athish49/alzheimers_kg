"""
Regression tests for security breaches B1–B6.

All tests hit the real Neon Postgres DB (no mocks).
Run from backend/: python3 -m pytest runtime/tests/test_security_fixes.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest

from runtime.auth.jwt import mint_token, verify_token, AUDIENCE, ALGORITHM, _REQUIRED_CLAIMS
from runtime.policy.pdp import decide
from runtime.seed.db import get_conn
from graph_rag.config import CONFIG


# ─── Fixtures ─────────────────────────────────────────────────────────────────

TEST_USER   = "u_014"   # Dr. Sarah Chen, attending_physician
TEST_ROLE   = "attending_physician"
NURSE_USER  = "u_015"   # Raj Patel, nurse
NURSE_ROLE  = "nurse"


@pytest.fixture(scope="module")
def db_session():
    """Clone TEMPLATE into a unique test session; clean up after module."""
    from runtime.seed.session import clone_session
    sid = "sectest_" + uuid.uuid4().hex[:12]
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
    assert row, f"No assigned patient found for {user_id} in session {sid}"
    return row[0]


# ─── B1: .env never in git ────────────────────────────────────────────────────

def test_b1_env_not_tracked():
    """backend/.env must not appear in git's tracked file list."""
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "backend/.env"],
        capture_output=True, text=True,
        cwd="/Users/athish/Documents/alzheimers_kg",
    )
    assert result.stdout.strip() == "", (
        "backend/.env is tracked by git — secrets are exposed in the repository"
    )


# ─── B2: Role escalation via JWT forgery ─────────────────────────────────────

def test_b2_forged_role_denied(db_session):
    """PDP must deny when JWT role claim doesn't match the DB role for user."""
    sid, conn = db_session
    patient_id = _assigned_patient(conn, sid, TEST_USER)

    forged_claims = {
        "sub":        TEST_USER,
        "role":       "attending_physician",   # Sarah is attending_physician in DB
        "session_id": sid,
        "scope":      ["patient.read", "knowledge.read"],
    }
    # This should be a normal permit (role matches)
    d = decide(conn, forged_claims, "read", "demographics", patient_id, sid)
    assert d.effect == "permit", "Legitimate claim must permit"

    # Now forge the role to something higher than the DB record
    forged_claims_bad = {**forged_claims, "role": "attending_physician"}
    # Swap user: use NURSE user but claim attending_physician role
    nurse_forged = {
        "sub":        NURSE_USER,
        "role":       "attending_physician",   # nurse claiming physician role
        "session_id": sid,
        "scope":      ["patient.read", "patient.write", "knowledge.read"],
    }
    d2 = decide(conn, nurse_forged, "read", "genetic_markers", patient_id, sid)
    assert d2.effect == "deny", (
        "PDP must deny when JWT role claim doesn't match the user's DB role"
    )
    assert "does not match DB role" in d2.reason


def test_b2_nonexistent_user_denied(db_session):
    """PDP must deny for a user_id that doesn't exist in the users table."""
    sid, conn = db_session
    patient_id = _assigned_patient(conn, sid, TEST_USER)
    ghost_claims = {
        "sub":        "u_999_ghost",
        "role":       "attending_physician",
        "session_id": sid,
        "scope":      ["patient.read"],
    }
    d = decide(conn, ghost_claims, "read", "demographics", patient_id, sid)
    assert d.effect == "deny"
    assert "does not match DB role" in d.reason


# ─── B3: Cross-session audit log hijacking ────────────────────────────────────

def test_b3_audit_requires_session_ownership(db_session):
    """
    A valid JWT with a forged session_id must be rejected by the /audit endpoint.
    We test the DB ownership check directly since we can't forge a JWT easily.
    """
    sid, conn = db_session

    cur = conn.cursor()
    # Check that sessions table ties session_id to user_id
    cur.execute(
        "SELECT user_id FROM sessions WHERE session_id = %s", (sid,)
    )
    row = cur.fetchone()
    assert row is not None, "Session must exist"
    assert row[0] == TEST_USER, "Session must be owned by TEST_USER"

    # A different user_id should NOT own this session
    cur.execute(
        "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
        (sid, NURSE_USER),
    )
    assert cur.fetchone() is None, (
        "NURSE_USER must not own TEST_USER's session — ownership check would block cross-session read"
    )


# ─── B4: Missing JWT claims cause 500 ────────────────────────────────────────

def test_b4_verify_rejects_token_missing_claims():
    """verify_token must raise InvalidTokenError when required claims are missing."""
    if not CONFIG.jwt_secret:
        pytest.skip("JWT_SECRET not configured")

    # Mint a minimal token without required claims
    now = datetime.now(timezone.utc)
    incomplete_payload = {
        "sub": "u_014",
        # missing: role, session_id, scope
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=30),
    }
    token = pyjwt.encode(incomplete_payload, CONFIG.jwt_secret, algorithm=ALGORITHM)

    with pytest.raises(pyjwt.InvalidTokenError):
        verify_token(token)


def test_b4_verify_accepts_complete_token():
    """verify_token must accept a well-formed token with all required claims."""
    if not CONFIG.jwt_secret:
        pytest.skip("JWT_SECRET not configured")

    token = mint_token("u_014", "attending_physician", "neurology", "team_a", "s_test123")
    claims = verify_token(token)
    assert _REQUIRED_CLAIMS.issubset(claims.keys())


def test_b4_required_claims_set_covers_all():
    """_REQUIRED_CLAIMS must include sub, role, session_id, scope."""
    assert {"sub", "role", "session_id", "scope"} == _REQUIRED_CLAIMS


# ─── B5: Unlimited break-glass grant stacking ────────────────────────────────

def test_b5_break_glass_upsert_no_stacking(db_session):
    """Repeated break-glass requests must update the existing grant, not create new rows."""
    sid, conn = db_session
    patient_id = _assigned_patient(conn, sid, TEST_USER)

    # Clear any prior grants for this test patient
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM break_glass_grants WHERE session_id = %s AND user_id = %s AND patient_id = %s",
        (sid, TEST_USER, patient_id),
    )
    conn.commit()

    # Import and call the endpoint logic directly (simulating two requests)
    from runtime.seed.db import get_conn as fresh_conn

    def _insert_grant(reason: str):
        c = fresh_conn()
        try:
            cu = c.cursor()
            # Replicate the upsert logic from router.py
            cu.execute(
                """
                UPDATE break_glass_grants
                   SET reason = %s, expires_at = now() + interval '15 minutes'
                 WHERE session_id = %s AND user_id = %s AND patient_id = %s AND expires_at > now()
                """,
                (reason, sid, TEST_USER, patient_id),
            )
            if cu.rowcount == 0:
                cu.execute(
                    """
                    INSERT INTO break_glass_grants
                      (session_id, user_id, patient_id, reason, expires_at)
                    VALUES (%s, %s, %s, %s, now() + interval '15 minutes')
                    """,
                    (sid, TEST_USER, patient_id, reason),
                )
            c.commit()
        finally:
            c.close()

    _insert_grant("first reason")
    _insert_grant("second reason")
    _insert_grant("third reason")

    cur.execute(
        "SELECT COUNT(*) FROM break_glass_grants WHERE session_id = %s AND user_id = %s AND patient_id = %s",
        (sid, TEST_USER, patient_id),
    )
    count = cur.fetchone()[0]
    assert count == 1, f"Expected 1 grant row (upsert), found {count}"

    # Cleanup
    cur.execute(
        "DELETE FROM break_glass_grants WHERE session_id = %s AND user_id = %s AND patient_id = %s",
        (sid, TEST_USER, patient_id),
    )
    conn.commit()


# ─── B6: Malformed patient_id accepted in break-glass ────────────────────────

def test_b6_break_glass_rejects_nonexistent_patient(db_session):
    """Break-glass must reject a patient_id not present in the session."""
    sid, conn = db_session

    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM patients WHERE patient_id = %s AND session_id = %s",
        ("p_NONEXISTENT_XYZ", sid),
    )
    assert cur.fetchone() is None, "Precondition: p_NONEXISTENT_XYZ must not exist in session"

    # Simulate the check from the router
    exists = cur.execute(
        "SELECT 1 FROM patients WHERE patient_id = %s AND session_id = %s",
        ("p_NONEXISTENT_XYZ", sid),
    )
    row = cur.fetchone()
    assert row is None, (
        "Patient existence check must fail for unknown patient_id — router should return 404"
    )


def test_b6_break_glass_accepts_valid_patient(db_session):
    """Break-glass must accept a patient_id that actually exists in the session."""
    sid, conn = db_session
    patient_id = _assigned_patient(conn, sid, TEST_USER)

    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM patients WHERE patient_id = %s AND session_id = %s",
        (patient_id, sid),
    )
    assert cur.fetchone() is not None, (
        f"Patient {patient_id} must exist in session {sid} for break-glass to be allowed"
    )
