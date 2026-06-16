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
# Question-level hint detectors
# (called on the original user question, before query rewriting)
# ---------------------------------------------------------------------


def _detect_direction_hint(question: str) -> Optional[str]:
    """Return 'increased', 'decreased', or None based on question wording."""
    q = question.lower()
    if any(w in q for w in ("decreas", "lower", "reduc", "drop", "fall", "diminish", "less")):
        return "decreased"
    if any(w in q for w in ("increas", "elevat", "higher", "rise", "raise", "upregulat")):
        return "increased"
    return None


def _detect_drug_status_hint(question: str) -> Optional[str]:
    """Return a DrugView status bucket name, or None, based on question wording."""
    q = question.lower()
    if "approved" in q or " fda" in q or " ema" in q:
        return "Approved"
    if "phase 3" in q or "phase3" in q or "phase iii" in q:
        return "Phase 3"
    if "discontinue" in q or "terminat" in q or "halt" in q or "fail" in q:
        return "Discontinued"
    return None


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
    *,
    direction_hint: Optional[str] = None,
    status_hint: Optional[str] = None,
) -> RouteResult:
    """
    High-level entry point: given a user question, classify its intent
    and build an appropriate graph-derived context string.

    Parameters
    ----------
    question:
        User’s natural-language question.
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

    # 1) Classify the question into an intent.
    #    entity_matches is already populated by the entity linker inside
    #    classify_question — we reuse it here rather than running the linker again.
    intent = classify_question(question)
    entity_matches = intent.entity_matches  # List[EntityMatch] or []

    # 2) Extract typed entity IDs from the linker results.
    #    AlzPedia nodes are intentionally excluded from gene_ids because the
    #    Gene→Protein Cypher pattern can’t match AlzPedia IDs.
    biomarker_ids: list = [m.node_id for m in entity_matches if m.node_type == "Biomarker"]
    drug_ids:      list = [m.node_id for m in entity_matches if m.node_type == "Drug"]
    gene_ids:      list = [m.node_id for m in entity_matches if m.node_type in ("Gene", "Protein")]

    # 3) Resolve disease_id once — reused across context + evidence builders
    disease_id = gtxt._resolve_ad_disease_id(retriever)

    # 4) Fetch only what each intent actually needs.
    #    For intents where entity IDs are available, try a targeted query first;
    #    fall back to bulk if it returns no rows (e.g., synonyms not yet in graph).
    #    strategy_name is set AFTER we know whether targeted or bulk was used.
    raw_data: Dict[str, Any] = {}
    context: str = ""
    strategy_name: str = ""

    if intent.type is IntentType.BIOMARKER:
        if disease_id:
            if biomarker_ids:
                biomarkers = retriever.get_biomarkers_by_ids(biomarker_ids, disease_id)
                strategy_name = "AD_BIOMARKERS_TARGETED" if biomarkers else "AD_BIOMARKERS_V2"
                if not biomarkers:
                    biomarkers = retriever.get_ad_biomarkers(disease_id, limit=200)
            else:
                biomarkers = retriever.get_ad_biomarkers(disease_id, limit=200)
                strategy_name = "AD_BIOMARKERS_V2"
            context = gtxt.build_biomarker_direction_context(
                retriever, disease_id, biomarkers=biomarkers
            )
            raw_data = gtxt.build_biomarker_evidence(
                biomarkers, direction_filter=direction_hint
            )
        else:
            strategy_name = "AD_BIOMARKERS_V2"
            context = gtxt.build_biomarker_direction_context(retriever)

    elif intent.type is IntentType.PHENOTYPE:
        # Phenotype nodes are not linked by the entity linker in the current graph,
        # so we always use bulk retrieval here.
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
        if disease_id:
            if drug_ids:
                drugs = retriever.get_drugs_by_ids(drug_ids, disease_id)
                drug_pws = retriever.get_drug_pathways_by_drug_ids(drug_ids, disease_id)
                strategy_name = "AD_DRUGS_TARGETED" if drugs else "AD_DRUGS_V2"
                if not drugs:
                    drugs = retriever.get_ad_drugs(disease_id, limit=400)
                    drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=400)
            else:
                drugs = retriever.get_ad_drugs(disease_id, limit=400)
                drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=400)
                strategy_name = "AD_DRUGS_V2"
            context = gtxt.build_drug_only_context(
                retriever, disease_id, drugs=drugs
            )
            raw_data = gtxt.build_drug_evidence(
                drugs, drug_pws, status_hint=status_hint
            )
        else:
            strategy_name = "AD_DRUGS_V2"
            context = gtxt.build_drug_only_context(retriever)

    elif intent.type is IntentType.PATHWAY:
        # Pathway questions are drug-anchored; use drug_ids if available.
        if disease_id:
            if drug_ids:
                drug_pws = retriever.get_drug_pathways_by_drug_ids(drug_ids, disease_id)
                strategy_name = "AD_PATHWAYS_TARGETED" if drug_pws else "AD_PATHWAYS_V2"
                if not drug_pws:
                    drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=250)
            else:
                drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=250)
                strategy_name = "AD_PATHWAYS_V2"
            context = gtxt.build_pathway_focused_context(
                retriever, disease_id, drug_pathways=drug_pws
            )
            raw_data = gtxt.build_pathway_evidence(drug_pws)
        else:
            strategy_name = "AD_PATHWAYS_V2"
            context = gtxt.build_pathway_focused_context(retriever)

    elif intent.type is IntentType.GENE_PROTEIN:
        if gene_ids:
            genes_proteins = retriever.get_proteins_by_ids(gene_ids, limit=100)
            strategy_name = "AD_GENES_TARGETED" if genes_proteins else "AD_GENES_V2"
            if not genes_proteins:
                genes_proteins = retriever.get_genes_and_proteins(limit=50)
        else:
            genes_proteins = retriever.get_genes_and_proteins(limit=50)
            strategy_name = "AD_GENES_V2"
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
            biomarkers     = retriever.get_ad_biomarkers(disease_id,    limit=50)
            drugs          = retriever.get_ad_drugs(disease_id,         limit=100)
            phenotypes     = retriever.get_ad_phenotypes(disease_id,    limit=50)
            drug_pws       = retriever.get_ad_drug_pathways(disease_id, limit=100)
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
        "entity_matches": str(len(entity_matches)),
    }

    if direction_hint:
        debug["direction_hint"] = direction_hint
    if status_hint:
        debug["status_hint"] = status_hint
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