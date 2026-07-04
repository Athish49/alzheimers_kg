"""
Phase 6 integration smoke tests.

Six checks that verify cross-layer integration and isolation:
  1. Knowledge service wraps graph_rag.router.build_context_for_question.
  2. S1 end-to-end: physician + p_2201 → synthesis context has PHI AND graph text. [integration]
  3. PHI isolation: patient identifiers never appear in knowledge queries.
  4. Existing /answer endpoint unchanged (mocked pipeline).
  5. Nurse denied genetic_markers → no APOE knowledge key derived. [integration]
  6. Project scoping: every session.run() in retriever.py passes project=. (static)

integration tests (2, 5) require DATABASE_URL and a seeded Neon Postgres DB.
Run all: python3 -m pytest runtime/tests/test_phase6.py -v
Run fast only: python3 -m pytest runtime/tests/test_phase6.py -v -m "not integration"
"""

from __future__ import annotations

import pathlib
import re
import uuid

import pytest

from runtime.auth.jwt import _ROLE_SCOPES


# ── Shared helpers ────────────────────────────────────────────────────────────

def _physician_claims(session_id: str) -> dict:
    return {
        "sub": "u_014", "role": "attending_physician",
        "session_id": session_id, "department": "neurology",
        "care_team": "team_a", "scope": _ROLE_SCOPES["attending_physician"],
    }


def _nurse_claims(session_id: str) -> dict:
    return {
        "sub": "u_027", "role": "nurse",
        "session_id": session_id, "department": "neurology",
        "care_team": "team_a", "scope": _ROLE_SCOPES["nurse"],
    }


class _FakeKGResult:
    strategy_name = "biomarker_probe"
    context = "[KG-PROBE: APOE e4 is a major genetic risk factor for AD]"
    raw_data = [{"gene": "APOE"}]


def _fake_session():
    """Create a fresh integration test session and return its id."""
    from runtime.seed.session import clone_session
    sid = "t6_" + uuid.uuid4().hex[:8]
    clone_session(sid, "u_014")
    return sid


def _drop_session(sid: str) -> None:
    from runtime.seed.db import get_conn
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM audit_log WHERE session_id = %s", (sid,))
        cur.execute("DELETE FROM break_glass_grants WHERE session_id = %s", (sid,))
        for table in [
            "clinical_notes", "genetic_markers", "lab_results",
            "medications", "vitals", "conditions",
            "patient_assignments", "patients",
        ]:
            cur.execute(f"DELETE FROM {table} WHERE session_id = %s", (sid,))
        cur.execute("DELETE FROM sessions WHERE session_id = %s", (sid,))
        conn.commit()
    finally:
        conn.close()


# ── 1. Knowledge service wraps build_context_for_question ────────────────────

class TestKnowledgeServiceReuse:
    """query_knowledge is a thin wrapper — output structure must mirror the upstream call."""

    def test_output_matches_graph_rag_result(self, monkeypatch):
        from graph_rag import router as router_module
        from runtime.services.knowledge import query_knowledge

        call_log: list[str] = []

        def fake_build(question: str):
            call_log.append(question)
            return _FakeKGResult()

        monkeypatch.setattr(router_module, "build_context_for_question", fake_build)

        result = query_knowledge(_physician_claims("s_test"), "s_test", "APOE")

        assert result["ok"] is True
        assert result["key"] == "APOE"
        assert result["strategy"] == _FakeKGResult.strategy_name
        assert result["context_text"] == _FakeKGResult.context
        assert result["evidence"] == _FakeKGResult.raw_data
        assert call_log == ["APOE"], f"Expected question='APOE', got {call_log}"

    def test_context_hint_appended_to_question(self, monkeypatch):
        from graph_rag import router as router_module
        from runtime.services.knowledge import query_knowledge

        call_log: list[str] = []
        monkeypatch.setattr(router_module, "build_context_for_question",
                            lambda q: (call_log.append(q), _FakeKGResult())[1])

        query_knowledge(_physician_claims("s_test"), "s_test", "lecanemab",
                        context_hint="treatment options")

        assert call_log == ["lecanemab treatment options"]

    def test_missing_scope_returns_forbidden(self, monkeypatch):
        from graph_rag import router as router_module
        from runtime.services.knowledge import query_knowledge

        monkeypatch.setattr(router_module, "build_context_for_question",
                            lambda q: _FakeKGResult())

        no_scope = {"sub": "u_014", "role": "attending_physician",
                    "session_id": "s_test", "scope": []}
        result = query_knowledge(no_scope, "s_test", "APOE")
        assert result["ok"] is False
        assert result["code"] == "forbidden"


# ── 2. S1 end-to-end: PHI + graph text in synthesis context ─────────────────

@pytest.mark.integration
class TestS1EndToEndJoin:
    """
    Physician + assigned patient p_2201.
    PHI from Postgres (APOE gene) AND graph context (mocked) must both appear
    in the text passed to the synthesis LLM.
    """

    def test_context_includes_phi_and_graph_text(self, monkeypatch):
        from graph_rag import llm_client as llm_module
        from graph_rag import router as router_module
        from runtime.join import synthesize
        from runtime.orchestrator import agent

        sid = _fake_session()
        try:
            monkeypatch.setattr(agent, "_propose_resources",
                                lambda q: ["genetic_markers", "lab_results"])
            monkeypatch.setattr(router_module, "build_context_for_question",
                                lambda q: _FakeKGResult())

            captured: list[str] = []

            class CaptureLLM:
                def simple_qa(self, question, context, system_prompt=None):
                    captured.append(context)
                    return "Synthesis answer."

            monkeypatch.setattr(llm_module, "_client", CaptureLLM())

            bundle = agent.orchestrate(
                "What is the patient's genetic risk?",
                _physician_claims(sid),
                "p_2201",
                sid,
            )
            synthesize(
                "What is the patient's genetic risk?",
                bundle["patient_bundle"],
                bundle["knowledge_results"],
            )

            assert captured, "LLM synthesis was never called"
            ctx = captured[0]

            # PHI from DB: p_2201's genetic_markers entry has gene="APOE"
            assert "APOE" in ctx, (
                f"APOE (PHI from DB genetic_markers) missing from synthesis context:\n{ctx[:400]}"
            )
            # Graph context from mocked KG
            assert "KG-PROBE" in ctx, (
                f"Knowledge graph text missing from synthesis context:\n{ctx[:400]}"
            )
        finally:
            _drop_session(sid)

    def test_knowledge_keys_contain_apoe(self, monkeypatch):
        """APOE must be derived as a knowledge key from granted genetic_markers."""
        from graph_rag import router as router_module
        from runtime.orchestrator import agent

        sid = _fake_session()
        try:
            monkeypatch.setattr(agent, "_propose_resources",
                                lambda q: ["genetic_markers"])
            monkeypatch.setattr(router_module, "build_context_for_question",
                                lambda q: _FakeKGResult())

            bundle = agent.orchestrate(
                "What is the APOE status?",
                _physician_claims(sid),
                "p_2201",
                sid,
            )
            assert "APOE" in bundle["knowledge_keys"], (
                f"Expected APOE in knowledge_keys, got {bundle['knowledge_keys']}"
            )
        finally:
            _drop_session(sid)


# ── 3. PHI isolation ──────────────────────────────────────────────────────────

class TestPHIIsolation:
    """Patient identifiers must never appear in knowledge queries sent to the graph."""

    def test_patient_id_not_in_knowledge_question(self, monkeypatch):
        from graph_rag import router as router_module
        from runtime.services.knowledge import query_knowledge

        captured: list[str] = []
        monkeypatch.setattr(
            router_module,
            "build_context_for_question",
            lambda q: (captured.append(q), _FakeKGResult())[1],
        )

        query_knowledge(
            _physician_claims("s_test"),
            "s_test",
            "APOE",
            context_hint="patient genetic risk question",
        )

        assert captured, "build_context_for_question was never called"
        phi_tokens = ("p_2201", "p_2208", "Robert Alvarez", "MRN-77012")
        for q in captured:
            for token in phi_tokens:
                assert token not in q, (
                    f"PHI token {token!r} leaked into knowledge query: {q!r}"
                )

    def test_tool_client_strips_patient_id_from_knowledge_args(self, monkeypatch):
        """ToolClient._SUBJECT_KEYS strips patient_id before routing to knowledge service."""
        from graph_rag import router as router_module
        from runtime.orchestrator.tool_client import ToolClient

        captured: list[str] = []
        monkeypatch.setattr(
            router_module,
            "build_context_for_question",
            lambda q: (captured.append(q), _FakeKGResult())[1],
        )

        client = ToolClient(
            claims=_physician_claims("s_test"),
            pinned_patient_id="p_2201",
            session_id="s_test",
        )
        # Model embeds patient_id — ToolClient must strip it before routing
        result = client.call("query_knowledge", {"key": "APOE", "patient_id": "p_2201"})

        assert result.get("ok") is True
        assert captured, "build_context_for_question was never called"
        for q in captured:
            assert "p_2201" not in q, (
                f"patient_id leaked into Neo4j question via ToolClient: {q!r}"
            )

    def test_knowledge_key_is_clinical_concept_not_identifier(self):
        """Keys derived from patient fields must be clinical names, not identifiers."""
        from runtime.orchestrator.agent import _extract_knowledge_keys

        resources = {
            "genetic_markers": {
                "granted": True,
                "items": [{"gene": "APOE", "variant": "ε4/ε4"}],
            },
            "lab_results": {
                "granted": True,
                "items": [{"test_name": "CSF p-tau181", "value": 38}],
            },
            "medications": {
                "granted": True,
                "items": [{"drug_name": "Donepezil", "dose": "10mg"}],
            },
        }
        keys = _extract_knowledge_keys(resources)

        for key in keys:
            assert "p_2201" not in key
            assert "Robert" not in key
            assert "MRN" not in key
        assert "APOE" in keys
        assert "CSF p-tau181" in keys
        assert "Donepezil" in keys


# ── 4. Existing /answer endpoint unchanged ───────────────────────────────────

class TestAnswerEndpointUnchanged:
    def test_post_answer_returns_200_with_expected_fields(self, monkeypatch):
        """Runtime additions must not break the pre-existing /answer endpoint."""
        import runtime.seed as seed_module
        monkeypatch.setattr(seed_module, "startup_reset", lambda: None)

        from graph_rag import pipeline as pipeline_module

        class FakePipeline:
            def answer(self, question, history=None, temperature=0.3,
                       max_tokens=512, return_context=False):
                return {
                    "answer":          "APOE ε4 increases Alzheimer's risk.",
                    "intent_type":     "biomarker",
                    "intent_notes":    "gene/protein intent",
                    "strategy":        "biomarker_strategy",
                    "retrieval_query": "APOE query",
                    "context":         "graph context",
                    "evidence":        {"nodes": [], "edges": []},
                }

        monkeypatch.setattr(pipeline_module, "get_pipeline", lambda: FakePipeline())

        from fastapi.testclient import TestClient
        from main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post("/answer", json={"question": "What does APOE do?"})

        assert resp.status_code == 200
        data = resp.json()
        for field in ("answer", "intent_type", "strategy"):
            assert field in data, f"Field '{field}' missing from /answer response"
        assert data["intent_type"] == "biomarker"


# ── 5. Denied field cannot leak ───────────────────────────────────────────────

@pytest.mark.integration
class TestDeniedFieldNoLeak:
    """
    Nurse is denied genetic_markers → APOE must not appear in knowledge_keys.
    Denied data must not become a knowledge-graph lookup key.
    """

    def test_no_apoe_key_when_genetic_markers_denied(self, monkeypatch):
        from graph_rag import router as router_module
        from runtime.orchestrator import agent

        sid = _fake_session()
        try:
            monkeypatch.setattr(agent, "_propose_resources",
                                lambda q: ["genetic_markers", "lab_results"])
            monkeypatch.setattr(router_module, "build_context_for_question",
                                lambda q: _FakeKGResult())

            bundle = agent.orchestrate(
                "What is the genetic risk?",
                _nurse_claims(sid),
                "p_2201",
                sid,
            )

            resources = bundle["patient_bundle"].get("resources", {})
            gm = resources.get("genetic_markers", {})

            assert gm.get("granted") is False, (
                "Nurse should be denied genetic_markers"
            )
            assert "APOE" not in bundle["knowledge_keys"], (
                f"APOE key must not be derived from denied genetic_markers; "
                f"keys={bundle['knowledge_keys']}"
            )
        finally:
            _drop_session(sid)

    def test_denied_resource_absent_from_synthesis_context(self, monkeypatch):
        """Denied genetic_markers must not appear in the text sent to the LLM."""
        from graph_rag import llm_client as llm_module
        from graph_rag import router as router_module
        from runtime.join import synthesize
        from runtime.orchestrator import agent

        sid = _fake_session()
        try:
            monkeypatch.setattr(agent, "_propose_resources",
                                lambda q: ["genetic_markers"])
            monkeypatch.setattr(router_module, "build_context_for_question",
                                lambda q: _FakeKGResult())

            captured: list[str] = []

            class CaptureLLM:
                def simple_qa(self, question, context, system_prompt=None):
                    captured.append(context)
                    return "No genetic data available."

            monkeypatch.setattr(llm_module, "_client", CaptureLLM())

            bundle = agent.orchestrate(
                "What is the genetic risk?",
                _nurse_claims(sid),
                "p_2201",
                sid,
            )
            result = synthesize(
                "What is the genetic risk?",
                bundle["patient_bundle"],
                bundle["knowledge_results"],
            )

            assert "genetic_markers" in result["abstained_on"], (
                "genetic_markers must appear in abstained_on for nurse"
            )
            # No APOE knowledge block should reach the LLM
            if captured:
                assert "KG-PROBE" not in captured[0], (
                    "Knowledge graph block for denied resource leaked into LLM context"
                )
        finally:
            _drop_session(sid)


# ── 6. Project scoping (static analysis) ─────────────────────────────────────

class TestProjectScoping:
    """
    Every Neo4j session.run() call in retriever.py must pass project= in its args.
    This is a static guard: if a new query is added without project=, the test fails.
    """

    def test_all_session_run_calls_include_project(self):
        retriever_path = (
            pathlib.Path(__file__).parent.parent.parent / "graph_rag" / "retriever.py"
        )
        src = retriever_path.read_text(encoding="utf-8")

        # Extract the argument text of every session.run(...) call, spanning lines.
        bad: list[str] = []
        for match in re.finditer(r"session\.run\(", src):
            start = match.start()
            window = src[start: start + 600]
            depth = 0
            end = 0
            for i, ch in enumerate(window):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            call_text = window[:end]
            if "project" not in call_text:
                bad.append(call_text.replace("\n", " ").strip())

        assert not bad, (
            f"session.run() calls in retriever.py missing project= param:\n"
            + "\n".join(f"  {b}" for b in bad)
        )
