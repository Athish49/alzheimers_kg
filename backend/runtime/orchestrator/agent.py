"""
runtime.orchestrator.agent
--------------------------
LLM tool-call loop. NO database handle. NO credential.

Flow (spec 05 §5):
  1. LLM proposes which patient resources to fetch.
  2. ToolClient executes get_patient_record; PDP + RLS trim to granted fields.
  3. From GRANTED fields only, derive non-PHI knowledge keys.
  4. ToolClient executes query_knowledge for each derived key.
  5. Return bundle to join layer for synthesis.

The model can decide WHAT to ask about a patient, never WHICH patient and
never WHICH fields are returned — PDP enforces the field set independently.
A denied field yields no knowledge key, so denied data cannot leak via a
knowledge lookup.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from runtime.orchestrator.tool_client import ToolClient

logger = logging.getLogger(__name__)

_AVAILABLE_RESOURCES: list[str] = [
    "demographics",
    "conditions",
    "vitals",
    "medications",
    "lab_results",
    "genetic_markers",
    "clinical_notes",
]

# Resources that carry graph-resolvable keys; only items from GRANTED entries are used.
_KNOWLEDGE_BEARING_FIELDS: dict[str, str] = {
    "genetic_markers": "gene",
    "lab_results":     "test_name",
    "medications":     "drug_name",
}

_RESOURCE_PROPOSAL_SYSTEM = (
    "You are a resource selector for a secure clinical AI system. "
    "Given a clinical question, output a JSON array of patient record categories needed to answer it. "
    "Choose only from: demographics, conditions, vitals, medications, "
    "lab_results, genetic_markers, clinical_notes. "
    "Respond with ONLY a valid JSON array, no explanation or prose. "
    'Example: ["demographics", "medications", "lab_results"]'
)


def _propose_resources(question: str) -> list[str]:
    """
    Ask the LLM which patient resource categories to request for this question.
    Falls back to all resources on parse error — PDP filters what is actually
    returned, so the fallback is safe.
    """
    from graph_rag.llm_client import get_llm_client
    try:
        llm = get_llm_client()
        raw = llm.chat(
            [{"role": "user", "content": f"Clinical question: {question}\n\nWhich patient record categories are needed?"}],
            system_prompt=_RESOURCE_PROPOSAL_SYSTEM,
            max_tokens=80,
        )
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            proposed = json.loads(match.group())
            valid = [r for r in proposed if r in _AVAILABLE_RESOURCES]
            if valid:
                return valid
    except Exception:
        logger.warning("Resource proposal LLM call failed; requesting all resources.", exc_info=True)
    return list(_AVAILABLE_RESOURCES)


def _extract_knowledge_keys(resources: dict[str, Any]) -> list[str]:
    """
    Return deduplicated knowledge-graph lookup keys derived exclusively from
    GRANTED patient fields.

    Denied resources (granted=False) and de-identified bindings yield no keys,
    ensuring a denied field can never become a knowledge lookup key.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for resource, field in _KNOWLEDGE_BEARING_FIELDS.items():
        entry = resources.get(resource, {})
        if not entry.get("granted"):
            continue
        if entry.get("deidentified"):
            continue
        for item in entry.get("items", []):
            val = item.get(field)
            if val and val not in seen:
                seen.add(val)
                keys.append(val)
    return keys


def orchestrate(
    question: str,
    claims: dict,
    pinned_patient_id: str | None,
    session_id: str,
) -> dict[str, Any]:
    """
    Secure tool-call loop. Returns assembled context bundle for the join layer.

    Returns
    -------
    {
        "patient_bundle":    dict,         # get_patient_record response
        "knowledge_results": list[dict],   # one entry per derived knowledge key
        "knowledge_keys":    list[str],    # keys derived from GRANTED fields
    }

    Does NOT call the synthesis LLM. That is the join layer's responsibility.
    """
    client = ToolClient(
        claims=claims,
        pinned_patient_id=pinned_patient_id,
        session_id=session_id,
    )

    # Step 1 — LLM proposes resources; PDP + RLS enforce what is actually returned.
    if pinned_patient_id:
        proposed_resources = _propose_resources(question)
        patient_result = client.call(
            "get_patient_record",
            {"resources": proposed_resources},
        )
    else:
        patient_result = {"ok": True, "patient_id": None, "resources": {}}

    # Step 2 — derive knowledge keys from GRANTED fields only.
    resources = patient_result.get("resources", {}) if patient_result.get("ok") else {}
    keys = _extract_knowledge_keys(resources)

    # Step 3 — query knowledge graph for each derived key.
    knowledge_results: list[dict] = []
    for key in keys:
        result = client.call(
            "query_knowledge",
            {"key": key, "key_type": "clinical", "context_hint": question},
        )
        knowledge_results.append(result)

    return {
        "patient_bundle":    patient_result,
        "knowledge_results": knowledge_results,
        "knowledge_keys":    keys,
    }
