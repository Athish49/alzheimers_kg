"""
Phase 3 tests — Trusted ToolClient shim + Knowledge service + Patient service.

DoD checks:
  3.1  ToolClient strips model-proposed subject keys.
  3.1  ToolClient injects pinned_patient_id for patient-scoped tools.
  3.1  ToolClient returns bad_request when no patient is pinned for a patient tool.
  3.1  ToolClient returns bad_request for unknown tool names.
  3.2  Knowledge service requires knowledge.read scope.
  3.3  get_patient_record: permit → data returned, audit written.
  3.3  get_patient_record: deny (unassigned) → denied marker, audit written.
  3.3  get_patient_record: field minimization enforced (nurse sees no mrn).
  3.3  get_patient_record: deidentified binding → deidentified:true, no PHI.
  3.3  get_patient_record: unknown resource → graceful denied marker.
  3.4  update_lab_result: lab_technician (assigned) → permit, row updated.
  3.4  update_lab_result: attending_physician → deny (wrong role for lab write).
  3.4  update_medication: attending_physician (assigned) → permit, row updated.
  3.4  update_medication: pharmacist → deny.
  3.5  get_deidentified_aggregate: research_analyst → aggregates, no PHI.
  3.5  get_deidentified_aggregate: attending_physician → deny.
  3.5  log_knowledge_access: writes audit row with patient_id=null.
"""

from __future__ import annotations

import pytest

from runtime.orchestrator.tool_client import ToolClient
from runtime.services.patient import (
    get_patient_record,
    update_lab_result,
    update_medication,
    get_deidentified_aggregate,
    log_knowledge_access,
)
from runtime.seed.db import get_conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claims(user_id: str, role_id: str, session_id: str) -> dict:
    from runtime.auth.jwt import _ROLE_SCOPES
    return {
        "sub":        user_id,
        "role":       role_id,
        "session_id": session_id,
        "department": "neurology",
        "care_team":  "team_a",
        "scope":      _ROLE_SCOPES.get(role_id, ["knowledge.read"]),
    }


def _audit_count(conn, session_id: str, action: str, resource: str,
                 patient_id: str | None = None) -> int:
    cur = conn.cursor()
    if patient_id is None:
        cur.execute(
            "SELECT COUNT(*) FROM audit_log WHERE session_id=%s AND action=%s AND resource=%s",
            (session_id, action, resource),
        )
    else:
        cur.execute(
            "SELECT COUNT(*) FROM audit_log WHERE session_id=%s AND action=%s "
            "AND resource=%s AND patient_id=%s",
            (session_id, action, resource, patient_id),
        )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# 3.1 — ToolClient shim
# ---------------------------------------------------------------------------

class TestToolClientShim:
    def test_strips_patient_id_from_args(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        client = ToolClient(claims, "p_2201", sid)

        # Simulate the model proposing args that include a patient key.
        # We pass to a tool that doesn't exist so we only care about strip behavior.
        # Use get_patient_record — patient_id will be re-injected from pinned.
        result = client.call("get_patient_record", {
            "patient_id": "p_EVIL_INJECTION",  # should be stripped
            "resources":  ["demographics"],
        })
        # If strip worked, patient_id used is the pinned one (p_2201), not the injected one.
        assert result.get("ok") is True
        assert result.get("patient_id") == "p_2201"

    def test_strips_all_subject_key_variants(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        client = ToolClient(claims, "p_2201", sid)

        # All subject-key aliases must be stripped.
        result = client.call("get_patient_record", {
            "subject_id": "p_X",
            "patient":    "p_Y",
            "subject":    "p_Z",
            "mrn":        "MRN-00000",
            "chart_id":   "c_1",
            "resources":  ["demographics"],
        })
        assert result.get("ok") is True
        assert result.get("patient_id") == "p_2201"

    def test_injects_pinned_patient(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        client = ToolClient(claims, "p_2201", sid)

        result = client.call("get_patient_record", {"resources": ["conditions"]})
        assert result.get("patient_id") == "p_2201"
        assert "conditions" in result.get("resources", {})

    def test_no_pinned_patient_for_patient_tool(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        client = ToolClient(claims, None, sid)  # no pinned patient

        result = client.call("get_patient_record", {"resources": ["demographics"]})
        assert result["ok"] is False
        assert result["code"] == "bad_request"

    def test_unknown_tool_returns_bad_request(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        client = ToolClient(claims, "p_2201", sid)

        result = client.call("nonexistent_tool", {"foo": "bar"})
        assert result["ok"] is False
        assert result["code"] == "bad_request"

    def test_deidentified_aggregate_not_patient_scoped(self, test_session):
        """get_deidentified_aggregate must not require a pinned patient."""
        sid, _ = test_session
        claims = _claims("u_059", "research_analyst", sid)
        client = ToolClient(claims, None, sid)  # no pinned patient — correct for analyst

        result = client.call("get_deidentified_aggregate", {})
        assert result.get("ok") is True


# ---------------------------------------------------------------------------
# 3.2 — Knowledge service scope gate
# ---------------------------------------------------------------------------

class TestKnowledgeService:
    def test_scope_check_blocks_missing_scope(self, test_session):
        """A claims dict without knowledge.read must be rejected."""
        from runtime.services.knowledge import query_knowledge

        sid, _ = test_session
        bad_claims = {"sub": "u_014", "role": "attending_physician",
                      "session_id": sid, "scope": []}  # scope absent

        result = query_knowledge(bad_claims, sid, key="APOE")
        assert result["ok"] is False
        assert result["code"] == "forbidden"

    def test_scope_present_returns_ok_structure(self, test_session):
        """With knowledge.read scope the service should return the expected shape."""
        from runtime.services.knowledge import query_knowledge

        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)
        result = query_knowledge(claims, sid, key="APOE", context_hint="Alzheimer's risk")

        assert result["ok"] is True
        assert result["key"] == "APOE"
        assert "strategy" in result
        assert "context_text" in result
        assert "evidence" in result


# ---------------------------------------------------------------------------
# 3.3 — get_patient_record
# ---------------------------------------------------------------------------

class TestGetPatientRecord:
    def test_permit_demographics_attending(self, test_session):
        sid, conn = test_session
        claims = _claims("u_014", "attending_physician", sid)
        before = _audit_count(conn, sid, "read", "demographics", "p_2201")

        result = get_patient_record("p_2201", ["demographics"], claims, sid)

        assert result["ok"] is True
        assert result["patient_id"] == "p_2201"
        demo = result["resources"]["demographics"]
        assert demo["granted"] is True
        assert "name" in demo["fields"]
        # Attending gets all fields including mrn
        assert "mrn" in demo["fields"]
        assert _audit_count(conn, sid, "read", "demographics", "p_2201") == before + 1

    def test_deny_unassigned_patient(self, test_session):
        """Attending u_014 is not assigned to p_3310 — should deny."""
        sid, conn = test_session
        claims = _claims("u_014", "attending_physician", sid)

        result = get_patient_record("p_3310", ["demographics"], claims, sid)

        assert result["ok"] is True  # outer call succeeds
        demo = result["resources"]["demographics"]
        assert demo["granted"] is False

    def test_field_minimization_nurse(self, test_session):
        """Nurse should not receive mrn/address/insurance_id for demographics."""
        sid, _ = test_session
        claims = _claims("u_027", "nurse", sid)

        result = get_patient_record("p_2201", ["demographics"], claims, sid)

        assert result["ok"] is True
        demo = result["resources"]["demographics"]
        assert demo["granted"] is True
        fields = demo["fields"]
        # Nurse is allowed: name, dob, sex, department, care_team
        assert "name" in fields
        # These must NOT appear
        assert "mrn" not in fields
        assert "address" not in fields
        assert "insurance_id" not in fields

    def test_field_minimization_pharmacist(self, test_session):
        """Pharmacist gets name + mrn only for demographics."""
        sid, _ = test_session
        claims = _claims("u_033", "pharmacist", sid)

        result = get_patient_record("p_2201", ["demographics"], claims, sid)

        assert result["ok"] is True
        demo = result["resources"]["demographics"]
        assert demo["granted"] is True
        fields = demo["fields"]
        assert "name" in fields
        assert "mrn" in fields
        assert "dob" not in fields
        assert "address" not in fields

    def test_genetic_markers_denied_for_nurse(self, test_session):
        """Nurse has no permission on genetic_markers — must deny."""
        sid, _ = test_session
        claims = _claims("u_027", "nurse", sid)

        result = get_patient_record("p_2201", ["genetic_markers"], claims, sid)

        gm = result["resources"]["genetic_markers"]
        assert gm["granted"] is False

    def test_multiple_resources_mixed_permit_deny(self, test_session):
        """Request demographics+genetic_markers as nurse: one permit, one deny."""
        sid, _ = test_session
        claims = _claims("u_027", "nurse", sid)

        result = get_patient_record("p_2201", ["demographics", "genetic_markers"], claims, sid)

        assert result["ok"] is True
        assert result["resources"]["demographics"]["granted"] is True
        assert result["resources"]["genetic_markers"]["granted"] is False

    def test_deidentified_path_for_research_analyst(self, test_session):
        """Research analyst's read on demographics must return deidentified:true."""
        sid, _ = test_session
        claims = _claims("u_059", "research_analyst", sid)

        result = get_patient_record("p_2201", ["demographics"], claims, sid)

        demo = result["resources"]["demographics"]
        # research_analyst has patient_binding=deidentified — no PHI rows returned
        assert demo["granted"] is True
        assert demo.get("deidentified") is True
        assert "fields" not in demo

    def test_unknown_resource_graceful_deny(self, test_session):
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)

        result = get_patient_record("p_2201", ["ghost_resource"], claims, sid)

        assert result["ok"] is True
        assert result["resources"]["ghost_resource"]["granted"] is False

    def test_conditions_items_returned(self, test_session):
        """Attending should receive a list of condition items, not a single dict."""
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)

        result = get_patient_record("p_2201", ["conditions"], claims, sid)

        assert result["ok"] is True
        cond = result["resources"]["conditions"]
        assert cond["granted"] is True
        assert isinstance(cond["items"], list)
        assert len(cond["items"]) > 0
        assert "label" in cond["items"][0]


# ---------------------------------------------------------------------------
# 3.4 — update_lab_result / update_medication
# ---------------------------------------------------------------------------

class TestWriteTools:
    def test_lab_tech_can_update_lab_result(self, test_session):
        """Lab technician assigned to p_2201 should be permitted."""
        sid, conn = test_session
        claims = _claims("u_041", "lab_technician", sid)
        before = _audit_count(conn, sid, "write", "lab_results", "p_2201")

        result = update_lab_result("p_2201", "CSF p-tau181", 42.5, "pg/mL", claims, sid)

        assert result["ok"] is True
        assert result["updated"] == "CSF p-tau181"
        assert _audit_count(conn, sid, "write", "lab_results", "p_2201") == before + 1

    def test_attending_physician_denied_lab_write(self, test_session):
        """Attending physician has no write permission on lab_results."""
        sid, conn = test_session
        claims = _claims("u_014", "attending_physician", sid)
        before = _audit_count(conn, sid, "write", "lab_results", "p_2201")

        result = update_lab_result("p_2201", "CSF p-tau181", 99.9, "pg/mL", claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"
        # Deny audit row must still be written
        assert _audit_count(conn, sid, "write", "lab_results", "p_2201") == before + 1

    def test_nurse_denied_lab_write(self, test_session):
        sid, _ = test_session
        claims = _claims("u_027", "nurse", sid)

        result = update_lab_result("p_2201", "CSF Aβ42", 800.0, "pg/mL", claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"

    def test_attending_can_update_medication(self, test_session):
        """Attending physician assigned to p_2201 should be permitted to update meds."""
        sid, conn = test_session
        claims = _claims("u_014", "attending_physician", sid)
        before = _audit_count(conn, sid, "write", "medications", "p_2201")

        result = update_medication("p_2201", "Donepezil", "10mg", "active", claims, sid)

        assert result["ok"] is True
        assert result["updated"] == "Donepezil"
        assert _audit_count(conn, sid, "write", "medications", "p_2201") == before + 1

    def test_pharmacist_denied_medication_write(self, test_session):
        """Pharmacist has no write permission on medications (S4 scenario)."""
        sid, conn = test_session
        claims = _claims("u_033", "pharmacist", sid)
        before = _audit_count(conn, sid, "write", "medications", "p_2201")

        result = update_medication("p_2201", "Lecanemab", "10mg/kg", "active", claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"
        assert _audit_count(conn, sid, "write", "medications", "p_2201") == before + 1

    def test_nurse_denied_medication_write(self, test_session):
        """Nurse has no write permission on medications (S6 scenario)."""
        sid, _ = test_session
        claims = _claims("u_027", "nurse", sid)

        result = update_medication("p_2201", "Donepezil", "5mg", "discontinued", claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"

    def test_lab_write_unassigned_patient_denied(self, test_session):
        """Lab tech u_041 is not assigned to p_2208 — should deny."""
        sid, _ = test_session
        claims = _claims("u_041", "lab_technician", sid)

        result = update_lab_result("p_2208", "CSF p-tau181", 55.0, "pg/mL", claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"


# ---------------------------------------------------------------------------
# 3.5 — get_deidentified_aggregate + log_knowledge_access
# ---------------------------------------------------------------------------

class TestDeidentifiedAndAudit:
    def test_research_analyst_gets_aggregates(self, test_session):
        sid, _ = test_session
        claims = _claims("u_059", "research_analyst", sid)

        result = get_deidentified_aggregate(claims, sid)

        assert result["ok"] is True
        assert "apoe_distribution" in result
        assert "ptau181_by_age_band" in result
        assert "active_medication_counts" in result
        # No patient identity in output
        for key in ("patient_id", "name", "mrn", "dob", "address"):
            assert key not in result

    def test_no_phi_in_aggregate_items(self, test_session):
        """Verify none of the aggregate list items contain PHI fields."""
        sid, _ = test_session
        claims = _claims("u_059", "research_analyst", sid)

        result = get_deidentified_aggregate(claims, sid)

        phi_fields = {"patient_id", "name", "mrn", "dob", "address", "insurance_id"}
        for dist_list in [result["apoe_distribution"], result["ptau181_by_age_band"],
                          result["active_medication_counts"]]:
            for item in dist_list:
                assert not phi_fields.intersection(item.keys()), (
                    f"PHI field found in aggregate item: {item}"
                )

    def test_attending_denied_deidentified_aggregate(self, test_session):
        """Attending physician has no permission on deident_aggregates."""
        sid, _ = test_session
        claims = _claims("u_014", "attending_physician", sid)

        result = get_deidentified_aggregate(claims, sid)

        assert result["ok"] is False
        assert result["code"] == "forbidden"

    def test_log_knowledge_access_writes_audit(self, test_session):
        sid, conn = test_session
        before = _audit_count(conn, sid, "read", "knowledge")

        log_knowledge_access(sid, "u_014", "attending_physician", "APOE", "permit")

        assert _audit_count(conn, sid, "read", "knowledge") == before + 1

    def test_log_knowledge_access_null_patient_id(self, test_session):
        """Knowledge audit rows must have patient_id = NULL."""
        sid, conn = test_session

        log_knowledge_access(sid, "u_014", "attending_physician", "lecanemab", "permit")

        cur = conn.cursor()
        cur.execute(
            "SELECT patient_id FROM audit_log "
            "WHERE session_id=%s AND resource='knowledge' "
            "ORDER BY ts DESC LIMIT 1",
            (sid,),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] is None  # patient_id must be NULL
