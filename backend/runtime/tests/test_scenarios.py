"""
Phase 6.1 — S1-S11 end-to-end scenario pass at the orchestrate layer.

These tests call orchestrate() against the real DB (cloned TEMPLATE session)
with mocked LLM calls. They verify that the orchestration layer correctly maps
PDP decisions to the output shape:
  - Denied resources appear in abstained_on / knowledge_keys is empty for them.
  - Granted resources contribute to patient_evidence and (for clinical keys) knowledge_keys.
  - Break-glass override grants access to an unassigned patient with flagging.
  - Research analyst path returns deidentified access with no knowledge keys.

These tests complement test_phase2.py's PDP-level checks (permit/deny).

All tests require DATABASE_URL (integration).
Run: python3 -m pytest runtime/tests/test_scenarios.py -v
"""

from __future__ import annotations

import pytest

from runtime.auth.jwt import _ROLE_SCOPES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _claims(user_id: str, role_id: str, session_id: str, **extra) -> dict:
    return {
        "sub":        user_id,
        "role":       role_id,
        "session_id": session_id,
        "department": extra.get("department", "neurology"),
        "care_team":  extra.get("care_team", "team_a"),
        "scope":      _ROLE_SCOPES.get(role_id, ["knowledge.read"]),
    }


def _run(question: str, claims: dict, patient_id: str | None, monkeypatch) -> dict:
    """Run orchestrate() with mocked LLM and mocked KG; return the bundle."""
    from graph_rag import router as router_module
    from runtime.orchestrator import agent

    class _FakeKGResult:
        strategy_name = "probe"
        context = "[KG-PROBE]"
        raw_data = []

    monkeypatch.setattr(agent, "_propose_resources", lambda q: [
        "demographics", "conditions", "vitals", "medications",
        "lab_results", "genetic_markers", "clinical_notes",
    ])
    monkeypatch.setattr(router_module, "build_context_for_question",
                        lambda q: _FakeKGResult())

    return agent.orchestrate(question, claims, patient_id, claims["session_id"])


# ── S1-S11 ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestScenarios:
    """End-to-end orchestrate() behavior for all eleven demo scenarios."""

    # S1 — Physician reads assigned patient → all resources granted, APOE key derived
    def test_s1_physician_reads_assigned_patient(self, test_session, monkeypatch):
        """S1: Dr. Sarah Chen reads p_2201 (assigned) — full access, APOE in knowledge_keys."""
        sid, _ = test_session
        bundle = _run("What is the genetic risk?", _claims("u_014", "attending_physician", sid),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        gm = resources.get("genetic_markers", {})
        assert gm.get("granted") is True, "Physician should be granted genetic_markers"
        assert "APOE" in bundle["knowledge_keys"], (
            f"APOE must be a knowledge key for physician; keys={bundle['knowledge_keys']}"
        )

    # S2 — Nurse denied genetic_markers
    def test_s2_nurse_denied_genetic_markers(self, test_session, monkeypatch):
        """S2: Raj Patel (nurse) reads p_2201 — genetic_markers denied, no APOE key."""
        sid, _ = test_session
        bundle = _run("What is the genetic risk?", _claims("u_027", "nurse", sid),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        gm = resources.get("genetic_markers", {})
        assert gm.get("granted") is False, "Nurse must be denied genetic_markers"
        assert "APOE" not in bundle["knowledge_keys"], (
            "APOE key must not be derived from denied genetic_markers"
        )

    # S3 — Pharmacist reads medications
    def test_s3_pharmacist_reads_medications(self, test_session, monkeypatch):
        """S3: Elena Rodriguez (pharmacist) reads p_2201 — medications granted."""
        sid, _ = test_session
        bundle = _run("What medications is the patient on?",
                      _claims("u_033", "pharmacist", sid, care_team="none"),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        meds = resources.get("medications", {})
        assert meds.get("granted") is True, "Pharmacist should be granted medications"
        assert "Donepezil" in bundle["knowledge_keys"] or "Lecanemab" in bundle["knowledge_keys"], (
            f"Drug key expected; keys={bundle['knowledge_keys']}"
        )

    # S4 — Pharmacist denied genetic_markers and clinical_notes
    def test_s4_pharmacist_denied_phi_resources(self, test_session, monkeypatch):
        """S4: Pharmacist has no access to genetic_markers or clinical_notes."""
        sid, _ = test_session
        bundle = _run("Tell me everything about the patient.",
                      _claims("u_033", "pharmacist", sid, care_team="none"),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        assert resources.get("genetic_markers", {}).get("granted") is False
        assert resources.get("clinical_notes", {}).get("granted") is False

    # S5 — Lab tech can access lab_results
    def test_s5_lab_tech_reads_lab_results(self, test_session, monkeypatch):
        """S5: Mei Lin (lab_technician) reads p_2201 lab results — granted."""
        sid, _ = test_session
        bundle = _run("What are the CSF biomarker values?",
                      _claims("u_041", "lab_technician", sid, care_team="none"),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        lr = resources.get("lab_results", {})
        assert lr.get("granted") is True, "Lab tech should be granted lab_results"

    # S6 — Lab tech denied medications
    def test_s6_lab_tech_denied_medications(self, test_session, monkeypatch):
        """S6: Mei Lin (lab_technician) — medications denied."""
        sid, _ = test_session
        bundle = _run("What medications is the patient on?",
                      _claims("u_041", "lab_technician", sid, care_team="none"),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        meds = resources.get("medications", {})
        assert meds.get("granted") is False, "Lab tech must be denied medications"

    # S7 — Physician denied unassigned patient
    def test_s7_physician_denied_unassigned_patient(self, test_session, monkeypatch):
        """S7: Sarah tries p_3310 (not her patient) — all resources denied."""
        sid, _ = test_session
        bundle = _run("What is the diagnosis?",
                      _claims("u_014", "attending_physician", sid),
                      "p_3310", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        assert resources, "Resources dict should not be empty (denials still appear)"
        all_denied = all(not v.get("granted") for v in resources.values())
        assert all_denied, (
            f"All resources should be denied for unassigned p_3310; "
            f"granted: {[k for k, v in resources.items() if v.get('granted')]}"
        )
        # No knowledge keys from denied resources
        assert bundle["knowledge_keys"] == []

    # S8 — Break-glass grants access to unassigned patient
    def test_s8_break_glass_grants_access(self, test_session, monkeypatch):
        """S8: After a valid break-glass grant, Sarah can access p_3310."""
        sid, conn = test_session
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO break_glass_grants
              (session_id, user_id, patient_id, reason, expires_at)
            VALUES (%s, 'u_014', 'p_3310', 'S8 urgent consult', now() + interval '15 minutes')
            ON CONFLICT DO NOTHING
            """,
            (sid,),
        )
        conn.commit()

        bundle = _run("What is the diagnosis?",
                      _claims("u_014", "attending_physician", sid),
                      "p_3310", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        demographics = resources.get("demographics", {})
        assert demographics.get("granted") is True, (
            "Break-glass should grant demographics access to p_3310"
        )

    # S9 — Nurse denied unassigned patient
    def test_s9_nurse_denied_unassigned_patient(self, test_session, monkeypatch):
        """S9: Raj tries p_4402 (Sarah's patient, not Raj's) — denied."""
        sid, _ = test_session
        bundle = _run("What are the vitals?",
                      _claims("u_027", "nurse", sid),
                      "p_4402", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        all_denied = all(not v.get("granted") for v in resources.values())
        assert all_denied, (
            f"All resources should be denied for nurse accessing unassigned p_4402"
        )

    # S10 — Research analyst sees deidentified data, no knowledge keys
    def test_s10_analyst_deidentified_no_phi(self, test_session, monkeypatch):
        """S10: Tom Baker (research_analyst) — deidentified access, empty knowledge_keys."""
        sid, _ = test_session
        bundle = _run("What are the lab trends?",
                      _claims("u_059", "research_analyst", sid,
                              department="research", care_team="none"),
                      "p_2201", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        # All granted resources must be deidentified
        for name, entry in resources.items():
            if entry.get("granted"):
                assert entry.get("deidentified"), (
                    f"Resource '{name}' granted to analyst but not marked deidentified"
                )

        # Deidentified resources yield no individual-row knowledge keys
        assert bundle["knowledge_keys"] == [], (
            f"Research analyst should have no knowledge_keys; "
            f"got {bundle['knowledge_keys']}"
        )

    # S11 — Non-existent patient denied
    def test_s11_nonexistent_patient_denied(self, test_session, monkeypatch):
        """S11: Manipulated request for p_9999 — denied for any role."""
        sid, _ = test_session
        bundle = _run("Tell me about the patient.",
                      _claims("u_014", "attending_physician", sid),
                      "p_9999", monkeypatch)

        resources = bundle["patient_bundle"].get("resources", {})
        all_denied = all(not v.get("granted") for v in resources.values())
        assert all_denied, (
            "Non-existent patient p_9999 should produce all-denied resources"
        )
        assert bundle["knowledge_keys"] == []
