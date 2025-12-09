"""
graph_rag.router
----------------

Routing logic from:

    user_question
        -> intent (biomarker / drug / phenotype / pathway / gene/protein / general)
        -> graph retrieval strategy
        -> textual context for the LLM.

For v1, retrieval is still centered on Alzheimer's Disease, but we now
use intent-specific context builders from `graph_to_text` to keep
the context ultra-compact and relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .intents import QueryIntent, IntentType, classify_question
from .retriever import GraphRetriever, get_retriever
from . import graph_to_text as gtxt  # <-- NEW: use our context builders


# ---------------------------------------------------------------------
# Routing result dataclass
# ---------------------------------------------------------------------


@dataclass
class RouteResult:
    """
    Result of routing a user question to a graph retrieval strategy.

    Fields
    ------
    intent:
        The classified QueryIntent (type + any detected IDs + notes).

    context:
        The textual context to send to the LLM (typically a multi-section
        summary of the Alzheimer's disease neighborhood).

    strategy_name:
        Short string describing which retrieval strategy was used,
        e.g. "AD_BIOMARKERS_V1", "AD_DRUGS_PATHWAYS_V1", etc.

    debug:
        Optional debug info, such as counts of nodes/edges fetched,
        or internal query choices. v1 uses this minimally.
    """

    intent: QueryIntent
    context: str
    strategy_name: str
    debug: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Core router function
# ---------------------------------------------------------------------


def build_context_for_question(
    question: str,
    retriever: Optional[GraphRetriever] = None,
) -> RouteResult:
    """
    High-level entry point: given a user question, classify its intent
    and build an appropriate graph-derived context string.

    Parameters
    ----------
    question:
        User's natural-language question.
    retriever:
        Optional GraphRetriever instance; if None, uses the singleton
        from graph_rag.retriever.get_retriever().

    Returns
    -------
    RouteResult
        Contains:
            - intent (QueryIntent)
            - context (str)
            - strategy_name (str)
            - debug (dict)
    """
    if retriever is None:
        retriever = get_retriever()

    # 1) Classify the question into an intent
    intent = classify_question(question)

    # 2) Pick an intent-specific context builder
    #
    #    BIOMARKER     -> biomarker-only, grouped by fluid + direction
    #    PHENOTYPE     -> phenotype/symptom-only
    #    DRUG_TRIAL    -> drugs + trials + pathways
    #    PATHWAY       -> same as DRUG_TRIAL for now (drugs + pathways)
    #    GENE_PROTEIN  -> fall back to general AD context (includes genes)
    #    GENERAL       -> general AD context (all sections)
    #
    if intent.type is IntentType.BIOMARKER:
        strategy_name = "AD_BIOMARKERS_V1"
        context = gtxt.build_biomarker_direction_context(retriever)

    elif intent.type is IntentType.PHENOTYPE:
        strategy_name = "AD_PHENOTYPES_V1"
        context = gtxt.build_phenotype_context(retriever)

    elif intent.type in (IntentType.DRUG_TRIAL, IntentType.PATHWAY):
        strategy_name = "AD_DRUGS_PATHWAYS_V1"
        context = gtxt.build_drug_trial_pathway_context(retriever)

    elif intent.type is IntentType.GENE_PROTEIN:
        # For now, reuse the general context (includes gene/protein section).
        strategy_name = "AD_GENES_GENERAL_V1"
        context = gtxt.build_general_ad_context(retriever)

    else:
        # Fallback: general compact Alzheimer’s graph summary.
        strategy_name = "AD_GENERAL_V1"
        context = gtxt.build_general_ad_context(retriever)

    debug: Dict[str, str] = {
        "intent_type": intent.type.name,
        "intent_notes": intent.notes,
        "strategy": strategy_name,
    }

    if intent.focus_entities:
        debug["focus_entities"] = ", ".join(intent.focus_entities)

    return RouteResult(
        intent=intent,
        context=context,
        strategy_name=strategy_name,
        debug=debug,
    )


# ---------------------------------------------------------------------
# Optional tiny helper for quick inspection
# ---------------------------------------------------------------------


def describe_route(result: RouteResult) -> str:
    """
    Build a small human-readable summary of a RouteResult
    (useful for logging or debugging in CLI).
    """
    lines = [
        f"Intent: {result.intent.type.name}",
        f"Notes: {result.intent.notes}",
        f"Strategy: {result.strategy_name}",
    ]
    if result.intent.focus_entities:
        lines.append(f"Focus entities: {', '.join(result.intent.focus_entities)}")
    return "\n".join(lines)