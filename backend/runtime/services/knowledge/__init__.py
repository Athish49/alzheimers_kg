"""
runtime.services.knowledge
--------------------------
Adapter over the existing graph_rag retrieval stack. Neo4j only. No Postgres.

Auth: requires knowledge.read scope in JWT claims.
"""

from __future__ import annotations

from typing import Any

from runtime.auth.jwt import require_scope


def query_knowledge(
    claims: dict,
    session_id: str,
    key: str,
    key_type: str = "general",
    context_hint: str = "",
) -> dict[str, Any]:
    """
    Query the Alzheimer's knowledge graph for a non-PHI key.

    Reuses the existing graph_rag retrieval stack (entity_linker → router →
    retriever → graph_to_text). No patient identifier is accepted or used.
    No Postgres access.

    Parameters
    ----------
    claims       : decoded JWT claims; must contain knowledge.read scope
    session_id   : caller's session (for audit, written by ToolClient)
    key          : graph surface form — e.g. "APOE", "p-tau181", "lecanemab"
    key_type     : informational hint; routing is intent-classified automatically
    context_hint : extra context appended to the query string

    Returns
    -------
    ok=True  → { ok, key, strategy, context_text, evidence }
    ok=False → { ok, code, message }
    """
    try:
        require_scope(claims, "knowledge.read")
    except PermissionError as exc:
        return {"ok": False, "code": "forbidden", "message": str(exc)}

    try:
        from graph_rag.router import build_context_for_question

        question = f"{key} {context_hint}".strip()
        result = build_context_for_question(question)

        return {
            "ok": True,
            "key": key,
            "strategy": result.strategy_name,
            "context_text": result.context,
            "evidence": result.raw_data,
        }
    except Exception as exc:
        return {
            "ok": False,
            "code": "internal",
            "message": f"Knowledge query failed: {exc}",
        }
