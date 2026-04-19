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
from typing import Any, Dict, Optional

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
    raw_data: Dict[str, Any] = field(default_factory=dict)
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

    # 2) Resolve disease_id once — reused across context + evidence builders
    disease_id = gtxt._resolve_ad_disease_id(retriever)

    # 3) Fetch data once, pass to both context builder and evidence builder
    raw_data: Dict[str, Any] = {}

    if intent.type is IntentType.BIOMARKER:
        strategy_name = "AD_BIOMARKERS_V1"
        if disease_id:
            biomarkers = retriever.get_ad_biomarkers(disease_id)
            context = gtxt.build_biomarker_direction_context(
                retriever, disease_id, biomarkers=biomarkers
            )
            raw_data = gtxt.build_biomarker_evidence(biomarkers)
        else:
            context = gtxt.build_biomarker_direction_context(retriever)

    elif intent.type is IntentType.PHENOTYPE:
        strategy_name = "AD_PHENOTYPES_V1"
        if disease_id:
            phenotypes = retriever.get_ad_phenotypes(disease_id)
            context = gtxt.build_phenotype_context(
                retriever, disease_id, phenotypes=phenotypes
            )
            raw_data = gtxt.build_phenotype_evidence(phenotypes)
        else:
            context = gtxt.build_phenotype_context(retriever)

    elif intent.type is IntentType.DRUG_TRIAL:
        strategy_name = "AD_DRUGS_PATHWAYS_V1"
        if disease_id:
            drugs = retriever.get_ad_drugs(disease_id)
            drug_pws = retriever.get_ad_drug_pathways(disease_id)
            context = gtxt.build_drug_trial_pathway_context(
                retriever, disease_id, drugs=drugs, drug_pathways=drug_pws
            )
            raw_data = gtxt.build_drug_evidence(drugs, drug_pws)
        else:
            context = gtxt.build_drug_trial_pathway_context(retriever)

    elif intent.type is IntentType.PATHWAY:
        strategy_name = "AD_DRUGS_PATHWAYS_V1"
        if disease_id:
            drugs = retriever.get_ad_drugs(disease_id)
            drug_pws = retriever.get_ad_drug_pathways(disease_id)
            context = gtxt.build_drug_trial_pathway_context(
                retriever, disease_id, drugs=drugs, drug_pathways=drug_pws
            )
            raw_data = gtxt.build_pathway_evidence(drug_pws)
        else:
            context = gtxt.build_drug_trial_pathway_context(retriever)

    elif intent.type is IntentType.GENE_PROTEIN:
        strategy_name = "AD_GENES_GENERAL_V1"
        genes_proteins = retriever.get_genes_and_proteins()
        if disease_id:
            biomarkers = retriever.get_ad_biomarkers(disease_id)
            drugs = retriever.get_ad_drugs(disease_id)
            phenotypes = retriever.get_ad_phenotypes(disease_id)
            drug_pws = retriever.get_ad_drug_pathways(disease_id)
            context = gtxt.build_general_ad_context(
                retriever,
                disease_id=disease_id,
                biomarkers=biomarkers,
                drugs=drugs,
                phenotypes=phenotypes,
                drug_pathways=drug_pws,
                genes_proteins=genes_proteins,
            )
        else:
            context = gtxt.build_general_ad_context(retriever, genes_proteins=genes_proteins)
        raw_data = gtxt.build_gene_evidence(genes_proteins)

    else:
        # Fallback: general compact Alzheimer’s graph summary (composite).
        strategy_name = "AD_GENERAL_V1"
        if disease_id:
            biomarkers = retriever.get_ad_biomarkers(disease_id)
            drugs = retriever.get_ad_drugs(disease_id)
            phenotypes = retriever.get_ad_phenotypes(disease_id)
            drug_pws = retriever.get_ad_drug_pathways(disease_id)
            genes_proteins = retriever.get_genes_and_proteins()
            context = gtxt.build_general_ad_context(
                retriever,
                disease_id=disease_id,
                biomarkers=biomarkers,
                drugs=drugs,
                phenotypes=phenotypes,
                drug_pathways=drug_pws,
                genes_proteins=genes_proteins,
            )
            raw_data = gtxt.build_composite_evidence(
                biomarkers=biomarkers,
                drugs=drugs,
                phenotypes=phenotypes,
                drug_pathways=drug_pws,
                genes_proteins=genes_proteins,
            )
        else:
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
        raw_data=raw_data,
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