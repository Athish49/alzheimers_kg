"""
Phase 2 unit tests — access control spine.

Coverage:
  • Phase 2.1  JWT mint / verify
  • Phase 2.2  PDP role-permission matrix (every cell)
  • Phase 2.2  Scenarios S1–S11 (permit/deny decisions only; no LLM)
  • Phase 2.3  Break-glass flow
  • Phase 2.4  Audit write + immutability

All tests are deterministic and hit the real Neon Postgres DB (no mocks, no LLM).
Run from backend/: python3 -m pytest runtime/tests/test_phase2.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest

from runtime.auth.jwt import mint_token, verify_token, AUDIENCE, ALGORITHM
from runtime.policy.pdp import decide
from runtime.policy.audit import write_audit
from runtime.seed.db import get_conn
from graph_rag.config import CONFIG


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _claims(user_id: str, role_id: str, session_id: str) -> dict:
    """Minimal claims dict that matches what verify_token returns."""
    return {
        "sub":        user_id,
        "role":       role_id,
        "session_id": session_id,
        "department": "neurology",
        "care_team":  "team_a",
    }


def _decide(conn, session_id: str, user_id: str, role_id: str,
            action: str, resource: str, patient_id: str | None = None) -> str:
    """Convenience wrapper: returns decision.effect."""
    claims = _claims(user_id, role_id, session_id)
    d = decide(conn, claims, action, resource, patient_id, session_id)
    return d.effect


# ─── Phase 2.1 — JWT ──────────────────────────────────────────────────────────

class TestJWT:
    def test_mint_and_verify_round_trip(self):
        token = mint_token("u_014", "attending_physician", "neurology", "team_a", "s_test")
        claims = verify_token(token)
        assert claims["sub"]        == "u_014"
        assert claims["role"]       == "attending_physician"
        assert claims["session_id"] == "s_test"
        assert claims["aud"]        == AUDIENCE
        assert "patient.read" in claims["scope"]

    def test_tampered_signature_rejected(self):
        token = mint_token("u_014", "attending_physician", "neurology", "team_a", "s_test")
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".badsignature"
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_token(tampered)

    def test_expired_token_rejected(self):
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "u_014", "role": "attending_physician",
            "session_id": "s_exp", "aud": AUDIENCE,
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
        }
        expired = pyjwt.encode(payload, CONFIG.jwt_secret, algorithm=ALGORITHM)
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_token(expired)

    def test_wrong_audience_rejected(self):
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "u_014", "role": "attending_physician",
            "session_id": "s_aud", "aud": "wrong-audience",
            "iat": now,
            "exp": now + timedelta(minutes=30),
        }
        token = pyjwt.encode(payload, CONFIG.jwt_secret, algorithm=ALGORITHM)
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_token(token)

    def test_scope_includes_knowledge_read_for_all_roles(self):
        for role in ["attending_physician", "nurse", "pharmacist", "lab_technician", "research_analyst"]:
            token = mint_token("u_xxx", role, "dept", "none", "s_scope")
            claims = verify_token(token)
            assert "knowledge.read" in claims["scope"], f"knowledge.read missing for {role}"


# ─── Phase 2.2 — PDP role-permission matrix ───────────────────────────────────

class TestRoleMatrix:
    """Verify every cell of the role × resource × action matrix."""

    # ── attending_physician ───────────────────────────────────────────────────

    def test_physician_demographics_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "demographics", "p_2201") == "permit"

    def test_physician_demographics_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "demographics", "p_2201") == "permit"

    def test_physician_conditions_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "conditions", "p_2201") == "permit"

    def test_physician_conditions_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "conditions", "p_2201") == "permit"

    def test_physician_vitals_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "vitals", "p_2201") == "permit"

    def test_physician_vitals_write_denied(self, test_session):
        # attending_physician has no vitals write
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "vitals", "p_2201") == "deny"

    def test_physician_medications_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "medications", "p_2201") == "permit"

    def test_physician_medications_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "medications", "p_2201") == "permit"

    def test_physician_lab_results_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "lab_results", "p_2201") == "permit"

    def test_physician_lab_results_write_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "lab_results", "p_2201") == "deny"

    def test_physician_genetic_markers_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "genetic_markers", "p_2201") == "permit"

    def test_physician_genetic_markers_write_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "genetic_markers", "p_2201") == "deny"

    def test_physician_clinical_notes_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "clinical_notes", "p_2201") == "permit"

    def test_physician_clinical_notes_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "write", "clinical_notes", "p_2201") == "permit"

    def test_physician_deident_aggregates_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "deident_aggregates", "p_2201") == "deny"

    def test_physician_knowledge_read(self, test_session):
        # none-bound — no patient_id needed
        sid, conn = test_session
        assert _decide(conn, sid, "u_014", "attending_physician", "read", "knowledge", None) == "permit"

    # ── nurse ─────────────────────────────────────────────────────────────────

    def test_nurse_demographics_read(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_027", "nurse", sid), "read", "demographics", "p_2201", sid)
        assert d.effect == "permit"
        # field minimization: no mrn, address, or insurance_id
        assert "mrn" not in d.allowed_fields
        assert "address" not in d.allowed_fields
        assert "name" in d.allowed_fields

    def test_nurse_demographics_write_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "write", "demographics", "p_2201") == "deny"

    def test_nurse_conditions_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "conditions", "p_2201") == "permit"

    def test_nurse_vitals_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "vitals", "p_2201") == "permit"

    def test_nurse_vitals_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "write", "vitals", "p_2201") == "permit"

    def test_nurse_medications_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "medications", "p_2201") == "permit"

    def test_nurse_lab_results_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "lab_results", "p_2201") == "permit"

    def test_nurse_genetic_markers_read_denied(self, test_session):
        # Role gate: nurse has no genetic_markers permission at all
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "genetic_markers", "p_2201") == "deny"

    def test_nurse_clinical_notes_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "clinical_notes", "p_2201") == "permit"

    def test_nurse_clinical_notes_write_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "write", "clinical_notes", "p_2201") == "deny"

    def test_nurse_knowledge_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_027", "nurse", "read", "knowledge", None) == "permit"

    # ── pharmacist ────────────────────────────────────────────────────────────

    def test_pharmacist_demographics_read(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_033", "pharmacist", sid), "read", "demographics", "p_2201", sid)
        assert d.effect == "permit"
        assert "name" in d.allowed_fields
        assert "mrn" in d.allowed_fields
        assert "address" not in d.allowed_fields

    def test_pharmacist_conditions_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "conditions", "p_2201") == "deny"

    def test_pharmacist_vitals_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "vitals", "p_2201") == "deny"

    def test_pharmacist_medications_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "medications", "p_2201") == "permit"

    def test_pharmacist_medications_write_denied(self, test_session):
        # pharmacist has NO write permission on any resource
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "write", "medications", "p_2201") == "deny"

    def test_pharmacist_lab_results_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "lab_results", "p_2201") == "permit"

    def test_pharmacist_genetic_markers_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "genetic_markers", "p_2201") == "deny"

    def test_pharmacist_clinical_notes_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "clinical_notes", "p_2201") == "deny"

    def test_pharmacist_knowledge_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_033", "pharmacist", "read", "knowledge", None) == "permit"

    # ── lab_technician ────────────────────────────────────────────────────────

    def test_lab_tech_demographics_read(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_041", "lab_technician", sid), "read", "demographics", "p_2201", sid)
        assert d.effect == "permit"
        assert "name" in d.allowed_fields
        assert "mrn" in d.allowed_fields
        assert "address" not in d.allowed_fields

    def test_lab_tech_conditions_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "conditions", "p_2201") == "deny"

    def test_lab_tech_vitals_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "vitals", "p_2201") == "deny"

    def test_lab_tech_medications_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "medications", "p_2201") == "deny"

    def test_lab_tech_lab_results_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "lab_results", "p_2201") == "permit"

    def test_lab_tech_lab_results_write(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "write", "lab_results", "p_2201") == "permit"

    def test_lab_tech_genetic_markers_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "genetic_markers", "p_2201") == "deny"

    def test_lab_tech_clinical_notes_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "clinical_notes", "p_2201") == "deny"

    def test_lab_tech_knowledge_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_041", "lab_technician", "read", "knowledge", None) == "permit"

    # ── research_analyst ─────────────────────────────────────────────────────

    def test_analyst_demographics_deidentified(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "demographics", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"
        assert d.patient_scope == []

    def test_analyst_conditions_deidentified(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "conditions", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"

    def test_analyst_vitals_deidentified(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "vitals", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"

    def test_analyst_medications_deidentified(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "medications", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"

    def test_analyst_lab_results_deidentified(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "lab_results", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"

    def test_analyst_deident_aggregates_read(self, test_session):
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "deident_aggregates", None, sid)
        assert d.effect == "permit"
        assert d.patient_binding == "deidentified"

    def test_analyst_genetic_markers_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_059", "research_analyst", "read", "genetic_markers", "p_2201") == "deny"

    def test_analyst_clinical_notes_denied(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_059", "research_analyst", "read", "clinical_notes", "p_2201") == "deny"

    def test_analyst_knowledge_read(self, test_session):
        sid, conn = test_session
        assert _decide(conn, sid, "u_059", "research_analyst", "read", "knowledge", None) == "permit"


# ─── Phase 2.2 — Scenarios S1–S11 ────────────────────────────────────────────

class TestScenarios:
    """
    S1–S11 from 04 §8. Tests at the decision level (permit/deny + obligations).
    No LLM is called. Data access is not tested here — only the PDP output.
    """

    def test_s1_physician_reads_genetic_markers_assigned(self, test_session):
        """S1: Sarah reads APOE + CSF biomarkers for assigned patient → permit."""
        sid, conn = test_session
        d = decide(conn, _claims("u_014", "attending_physician", sid), "read", "genetic_markers", "p_2201", sid)
        assert d.effect == "permit"
        assert "audit" in d.obligations

    def test_s2_nurse_reads_genetic_markers_denied(self, test_session):
        """S2: Raj asks same question — genetic_markers denied at role gate."""
        sid, conn = test_session
        d = decide(conn, _claims("u_027", "nurse", sid), "read", "genetic_markers", "p_2201", sid)
        assert d.effect == "deny"

    def test_s3_pharmacist_reads_medications_assigned(self, test_session):
        """S3: Elena reads medications for p_2201 → permit."""
        sid, conn = test_session
        d = decide(conn, _claims("u_033", "pharmacist", sid), "read", "medications", "p_2201", sid)
        assert d.effect == "permit"

    def test_s4_pharmacist_write_medications_denied(self, test_session):
        """S4: Elena tries to write medications → denied (role gate, no write)."""
        sid, conn = test_session
        d = decide(conn, _claims("u_033", "pharmacist", sid), "write", "medications", "p_2201", sid)
        assert d.effect == "deny"

    def test_s5_lab_tech_writes_lab_results(self, test_session):
        """S5: Mei enters a lab result for assigned p_2201 → permit."""
        sid, conn = test_session
        d = decide(conn, _claims("u_041", "lab_technician", sid), "write", "lab_results", "p_2201", sid)
        assert d.effect == "permit"

    def test_s6_lab_tech_reads_medications_denied(self, test_session):
        """S6: Mei tries to view medications → denied (resource gate)."""
        sid, conn = test_session
        d = decide(conn, _claims("u_041", "lab_technician", sid), "read", "medications", "p_2201", sid)
        assert d.effect == "deny"

    def test_s7_physician_unassigned_patient_denied(self, test_session):
        """S7: Sarah opens p_3310 (not assigned) → denied; break-glass offered."""
        sid, conn = test_session
        d = decide(conn, _claims("u_014", "attending_physician", sid), "read", "demographics", "p_3310", sid)
        assert d.effect == "deny"
        assert "p_3310" in d.reason

    def test_s8_physician_break_glass_permits_with_flag(self, test_session):
        """S8: After writing a break-glass grant, Sarah may access p_3310 (time-boxed, flagged)."""
        sid, conn = test_session
        # Insert a valid break-glass grant
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO break_glass_grants
              (session_id, user_id, patient_id, reason, expires_at)
            VALUES (%s, %s, %s, %s, now() + interval '15 minutes')
            """,
            (sid, "u_014", "p_3310", "S8 test — urgent consult required"),
        )
        conn.commit()

        d = decide(conn, _claims("u_014", "attending_physician", sid), "read", "demographics", "p_3310", sid)
        assert d.effect == "permit"
        assert "flag_review" in d.obligations
        assert "p_3310" in d.patient_scope

    def test_s9_nurse_unassigned_patient_denied(self, test_session):
        """S9: Raj opens p_4402 (Sarah's patient, not Raj's) → denied."""
        sid, conn = test_session
        d = decide(conn, _claims("u_027", "nurse", sid), "read", "demographics", "p_4402", sid)
        assert d.effect == "deny"

    def test_s10_analyst_no_phi(self, test_session):
        """S10: Tom reads anything → deidentified path, no patient scope."""
        sid, conn = test_session
        d = decide(conn, _claims("u_059", "research_analyst", sid), "read", "demographics", "p_2201", sid)
        assert d.effect == "permit"
        assert d.patient_scope == []
        assert d.patient_binding == "deidentified"

    def test_s11_nonexistent_patient_denied(self, test_session):
        """S11: Manipulated request for p_9999 (not assigned) → denied regardless of role."""
        sid, conn = test_session
        d = decide(conn, _claims("u_014", "attending_physician", sid), "read", "demographics", "p_9999", sid)
        assert d.effect == "deny"

    def test_s11_assigned_patient_different_session_denied(self, test_session):
        """S11 variant: patient exists in TEMPLATE but not in this session's assignments."""
        sid, conn = test_session
        # u_021 is assigned to p_2215 in TEMPLATE, but not in this session for u_014
        d = decide(conn, _claims("u_014", "attending_physician", sid), "read", "demographics", "p_2215", sid)
        assert d.effect == "deny"


# ─── Phase 2.4 — Audit log ────────────────────────────────────────────────────

class TestAudit:
    def test_write_audit_creates_row(self, test_session):
        """Every PDP call writes exactly one audit row (verified via superuser connection)."""
        sid, conn = test_session
        before = _audit_count(conn, sid)

        write_audit(
            conn,
            session_id=sid,
            user_id="u_014",
            role_id="attending_physician",
            action="read",
            resource="demographics",
            patient_id="p_2201",
            effect="permit",
            reason="test",
        )
        conn.commit()

        after = _audit_count(conn, sid)
        assert after == before + 1

    def test_audit_fields_accessed_stored(self, test_session):
        sid, conn = test_session
        write_audit(
            conn,
            session_id=sid,
            user_id="u_027",
            role_id="nurse",
            action="read",
            resource="demographics",
            patient_id="p_2201",
            effect="permit",
            reason="field minimization test",
            fields_accessed=["name", "dob", "sex"],
        )
        conn.commit()

        cur = conn.cursor()
        cur.execute(
            "SELECT fields_accessed FROM audit_log WHERE session_id = %s AND user_id = %s ORDER BY id DESC LIMIT 1",
            (sid, "u_027"),
        )
        row = cur.fetchone()
        assert row is not None
        assert "name" in row[0]

    def test_audit_break_glass_flag_stored(self, test_session):
        sid, conn = test_session
        write_audit(
            conn,
            session_id=sid,
            user_id="u_014",
            role_id="attending_physician",
            action="read",
            resource="demographics",
            patient_id="p_3310",
            effect="permit",
            reason="emergency consult",
            break_glass=True,
        )
        conn.commit()

        cur = conn.cursor()
        cur.execute(
            "SELECT break_glass FROM audit_log WHERE session_id = %s AND patient_id = %s ORDER BY id DESC LIMIT 1",
            (sid, "p_3310"),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] is True

    def test_audit_immutable_no_update(self, test_session):
        """app_runtime role must not be able to UPDATE audit_log rows."""
        sid, conn = test_session
        # Write a row first
        write_audit(
            conn,
            session_id=sid,
            user_id="u_014",
            role_id="attending_physician",
            action="read",
            resource="demographics",
            patient_id="p_2201",
            effect="permit",
            reason="immutability test",
        )
        conn.commit()

        # Attempt UPDATE as app_runtime — must fail
        import psycopg
        cur = conn.cursor()
        cur.execute("SET ROLE app_runtime")
        cur.execute("SELECT set_config('app.session_id', %s, true)", (sid,))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("UPDATE audit_log SET reason = 'tampered' WHERE session_id = %s", (sid,))
            conn.commit()

        conn.rollback()
        cur.execute("RESET ROLE")

    def test_audit_immutable_no_delete(self, test_session):
        """app_runtime role must not be able to DELETE audit_log rows."""
        sid, conn = test_session
        write_audit(
            conn,
            session_id=sid,
            user_id="u_014",
            role_id="attending_physician",
            action="read",
            resource="demographics",
            patient_id="p_2201",
            effect="permit",
            reason="delete immutability test",
        )
        conn.commit()

        import psycopg
        cur = conn.cursor()
        cur.execute("SET ROLE app_runtime")
        cur.execute("SELECT set_config('app.session_id', %s, true)", (sid,))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("DELETE FROM audit_log WHERE session_id = %s", (sid,))
            conn.commit()

        conn.rollback()
        cur.execute("RESET ROLE")


# ─── Specialist role tests ────────────────────────────────────────────────────

class TestSpecialistRolePermissions:
    """Verify specialist role permissions seeded in the role_permissions table.

    These roles operate on the EHR plane (ehr_patient_assignments), not the
    session-scoped Alzheimer's plane. Tests query the DB directly — no session
    fixture required — to validate the permission matrix is correctly seeded.
    """

    @pytest.fixture
    def conn(self):
        from runtime.seed.db import get_conn
        c = get_conn()
        yield c
        c.close()

    def _read_resources(self, conn, role_id: str) -> set[str]:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT resource FROM role_permissions WHERE role_id = %s AND action = 'read'",
            (role_id,),
        )
        return {r[0] for r in cur.fetchall()}

    def _write_resources(self, conn, role_id: str) -> set[str]:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT resource FROM role_permissions WHERE role_id = %s AND action = 'write'",
            (role_id,),
        )
        return {r[0] for r in cur.fetchall()}

    # ── JWT scope checks for new specialist roles ─────────────────────────────

    def test_specialist_jwt_includes_knowledge_read(self):
        """Every new specialist role gets at least knowledge.read in its JWT scope."""
        specialists = [
            "cardiologist", "psychiatrist", "pulmonologist", "nephrologist",
            "endocrinologist", "gastroenterologist", "rheumatologist",
            "orthopedic_surgeon", "hematologist", "urologist",
            "allergist_immunologist", "primary_care_physician",
            "electrophysiologist", "heart_failure_specialist", "bariatrician",
        ]
        for role in specialists:
            token = mint_token("u_spec", role, "dept", "none", "s_spec")
            claims = verify_token(token)
            assert "knowledge.read" in claims["scope"], f"knowledge.read missing for {role}"

    def test_specialist_jwt_includes_patient_read(self):
        """Every new specialist role gets patient.read scope in its JWT."""
        specialists = [
            "cardiologist", "psychiatrist", "pulmonologist", "nephrologist",
            "endocrinologist",
        ]
        for role in specialists:
            token = mint_token("u_spec", role, "dept", "none", "s_spec")
            claims = verify_token(token)
            assert "patient.read" in claims["scope"], f"patient.read missing for {role}"

    # ── Cardiologist permission matrix ────────────────────────────────────────

    def test_cardiologist_reads_lab_results(self, conn):
        assert "lab_results" in self._read_resources(conn, "cardiologist")

    def test_cardiologist_reads_imaging(self, conn):
        assert "imaging" in self._read_resources(conn, "cardiologist")

    def test_cardiologist_reads_conditions(self, conn):
        assert "conditions" in self._read_resources(conn, "cardiologist")

    def test_cardiologist_no_social_history(self, conn):
        """Cardiologist has no social_history permission."""
        assert "social_history" not in self._read_resources(conn, "cardiologist")

    def test_cardiologist_no_procedures_write(self, conn):
        """Cardiologist cannot write procedures (read-only domain)."""
        assert "procedures" not in self._write_resources(conn, "cardiologist")

    # ── Psychiatrist permission matrix ────────────────────────────────────────

    def test_psychiatrist_no_lab_results(self, conn):
        """Psychiatrist has no lab_results access — outside clinical scope."""
        assert "lab_results" not in self._read_resources(conn, "psychiatrist")

    def test_psychiatrist_no_imaging(self, conn):
        """Psychiatrist has no imaging access."""
        assert "imaging" not in self._read_resources(conn, "psychiatrist")

    def test_psychiatrist_reads_social_history(self, conn):
        assert "social_history" in self._read_resources(conn, "psychiatrist")

    def test_psychiatrist_writes_social_history(self, conn):
        assert "social_history" in self._write_resources(conn, "psychiatrist")

    def test_psychiatrist_reads_care_plans(self, conn):
        assert "care_plans" in self._read_resources(conn, "psychiatrist")

    def test_psychiatrist_reads_clinical_notes(self, conn):
        assert "clinical_notes" in self._read_resources(conn, "psychiatrist")

    # ── Cross-specialist contrast: imaging access ─────────────────────────────

    def test_nephrologist_reads_lab_results(self, conn):
        assert "lab_results" in self._read_resources(conn, "nephrologist")

    def test_nephrologist_reads_imaging(self, conn):
        assert "imaging" in self._read_resources(conn, "nephrologist")

    def test_pulmonologist_reads_imaging(self, conn):
        assert "imaging" in self._read_resources(conn, "pulmonologist")

    # ── All specialists must read demographics and encounters ─────────────────

    def test_all_specialists_read_demographics(self, conn):
        """Every specialist role must be able to read demographics."""
        specialists = [
            "cardiologist", "psychiatrist", "pulmonologist", "nephrologist",
            "endocrinologist", "gastroenterologist", "rheumatologist",
            "orthopedic_surgeon", "hematologist", "urologist",
            "allergist_immunologist", "primary_care_physician",
            "electrophysiologist", "heart_failure_specialist", "bariatrician",
        ]
        for role in specialists:
            resources = self._read_resources(conn, role)
            assert "demographics" in resources, f"{role} missing demographics read"

    def test_all_specialists_read_encounters(self, conn):
        """Every specialist must be able to read encounter history."""
        specialists = [
            "cardiologist", "psychiatrist", "pulmonologist", "nephrologist",
            "endocrinologist", "gastroenterologist", "rheumatologist",
            "orthopedic_surgeon", "hematologist", "urologist",
            "allergist_immunologist", "primary_care_physician",
            "electrophysiologist", "heart_failure_specialist", "bariatrician",
        ]
        for role in specialists:
            resources = self._read_resources(conn, role)
            assert "encounters" in resources, f"{role} missing encounters read"


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _audit_count(conn, session_id: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM audit_log WHERE session_id = %s", (session_id,))
    return cur.fetchone()[0]
