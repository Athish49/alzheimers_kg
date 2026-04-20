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

    # 3) Fetch only what each intent actually needs.
    raw_data: Dict[str, Any] = {}

    if intent.type is IntentType.BIOMARKER:
        # Fetch: biomarkers only (up to 200 rows — all are relevant)
        strategy_name = "AD_BIOMARKERS_V2"
        if disease_id:
            biomarkers = retriever.get_ad_biomarkers(disease_id, limit=200)
            context = gtxt.build_biomarker_direction_context(
                retriever, disease_id, biomarkers=biomarkers
            )
            raw_data = gtxt.build_biomarker_evidence(biomarkers)
        else:
            context = gtxt.build_biomarker_direction_context(retriever)

    elif intent.type is IntentType.PHENOTYPE:
        # Fetch: phenotypes only (up to 100 rows)
        strategy_name = "AD_PHENOTYPES_V2"
        if disease_id:
            phenotypes = retriever.get_ad_phenotypes(disease_id, limit=100)
            context = gtxt.build_phenotype_context(
                retriever, disease_id, phenotypes=phenotypes
            )
            raw_data = gtxt.build_phenotype_evidence(phenotypes)
        else:
            context = gtxt.build_phenotype_context(retriever)

    elif intent.type is IntentType.DRUG_TRIAL:
        # Fetch: drugs only — trial questions don’t need pathway details.
        # Drug pathways are still included in raw_data for the frontend evidence panel.
        strategy_name = "AD_DRUGS_V2"
        if disease_id:
            drugs = retriever.get_ad_drugs(disease_id, limit=150)
            drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=200)
            context = gtxt.build_drug_only_context(
                retriever, disease_id, drugs=drugs
            )
            raw_data = gtxt.build_drug_evidence(drugs, drug_pws)
        else:
            context = gtxt.build_drug_only_context(retriever)

    elif intent.type is IntentType.PATHWAY:
        # Fetch: drug-pathway edges only — pathway questions don’t need trial phases.
        strategy_name = "AD_PATHWAYS_V2"
        if disease_id:
            drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=250)
            context = gtxt.build_pathway_focused_context(
                retriever, disease_id, drug_pathways=drug_pws
            )
            raw_data = gtxt.build_pathway_evidence(drug_pws)
        else:
            context = gtxt.build_pathway_focused_context(retriever)

    elif intent.type is IntentType.GENE_PROTEIN:
        # Fetch: genes/proteins only — gene questions don’t need biomarker fluids
        # or drug trial phases.
        strategy_name = "AD_GENES_V2"
        genes_proteins = retriever.get_genes_and_proteins(limit=50)
        context = gtxt.build_gene_protein_context(retriever, genes_proteins=genes_proteins)
        raw_data = gtxt.build_gene_evidence(genes_proteins)

    elif intent.type is IntentType.OTHER:
        # Non-AD question: no Neo4j queries, no LLM call (pipeline handles early exit).
        strategy_name = "NOT_AD"
        context = "__OUT_OF_SCOPE__"

    else:
        # GENERAL_AD fallback: all sections, but with tighter limits so the
        # composite context stays within ~6,000 tokens.
        strategy_name = "AD_GENERAL_V2"
        if disease_id:
            biomarkers    = retriever.get_ad_biomarkers(disease_id,    limit=50)
            drugs         = retriever.get_ad_drugs(disease_id,         limit=50)
            phenotypes    = retriever.get_ad_phenotypes(disease_id,    limit=50)
            drug_pws      = retriever.get_ad_drug_pathways(disease_id, limit=75)
            genes_proteins = retriever.get_genes_and_proteins(          limit=15)
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