"""
graph_rag.intents
-----------------

Lightweight intent classification for Graph-RAG.

Goal
----
Given a user question, decide what "kind" of graph query we should run:

    - Biomarker-focused
    - Drug / trial-focused
    - Phenotype / symptom-focused
    - Pathway-focused
    - Gene / protein-focused
    - General Alzheimer's Disease overview

This is intentionally rule-based for v1 (no LLM dependency),
so it's cheap, fast, and predictable. Later, you can swap this
for an LLM-based classifier if needed, but the interface should
remain the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class IntentType(Enum):
    """High-level query types we care about."""

    BIOMARKER = auto()
    DRUG_TRIAL = auto()
    PHENOTYPE = auto()
    PATHWAY = auto()
    GENE_PROTEIN = auto()
    GENERAL_AD = auto()
    OTHER = auto()


@dataclass
class QueryIntent:
    """
    Classification result for a user question.

    Fields
    ------
    type:
        Main intent class (biomarker, drug, etc.)

    focus_entities:
        Any explicit entity IDs or labels we detect, e.g.
        "MONDO:0004975", "Aβ42", "donepezil", "CHEBI:53289".

        v1 is very simple; later we can plug in a proper
        entity linker if desired.

    raw_question:
        Original user question (for debugging / logging).

    notes:
        Human-readable notes on why this intent was chosen.
    """

    type: IntentType
    focus_entities: List[str] = field(default_factory=list)
    raw_question: str = ""
    notes: str = ""


# ---------------------------------------------------------------------
# Simple rule-based classifier
# ---------------------------------------------------------------------


# Pre-compiled regex for ID patterns we know exist in the graph
ID_PATTERNS = [
    r"MONDO:\d+",
    r"CHEBI:\d+",
    r"HP:\d+",
    r"GO:\d+",
    r"HGNC:\d+",
    r"PR:\d+",
]


def _extract_potential_ids(text: str) -> List[str]:
    """Extract MONDO/HP/GO/CHEBI/HGNC/PR-style IDs from the question."""
    found: List[str] = []
    for pat in ID_PATTERNS:
        for match in re.findall(pat, text):
            if match not in found:
                found.append(match)
    return found


def classify_question(question: str) -> QueryIntent:
    """
    Classify a question into one of the IntentType categories.

    v1 is purely keyword-based; this is deliberate:
    - deterministic behavior
    - no dependence on any LLM
    - easy to tweak rules as you see real queries

    Later you can replace the internals with a call to an LLM-based
    classifier, but keep the same function signature.
    """
    q_raw = question or ""
    q = q_raw.lower()

    # -----------------------------------------------------------------
    # 1) Identify obvious "domain" keywords
    # -----------------------------------------------------------------
    biomarker_keywords = [
        "biomarker",
        "marker",
        "csf",
        "plasma",
        "serum",
        "fluid",
        "cutoff",
        "sensitivity",
        "specificity",
    ]

    drug_keywords = [
        "drug",
        "treat",
        "treatment",
        "therapy",
        "therapeutic",
        "compound",
        "trial",
        "phase",
        "phase 2",
        "phase 3",
        "approved",
        "approval",
        "status",
        "dosage",
        "dose",
        "company",
    ]

    phenotype_keywords = [
        "symptom",
        "sign",
        "clinical feature",
        "cognitive",
        "memory",
        "language",
        "aphasia",
        "behavior",
        "behaviour",
        "phenotype",
        "presentation",
    ]

    pathway_keywords = [
        "pathway",
        "pathways",
        "go:",
        "signaling",
        "signalling",
        "microglial",
        "synaptic",
        "amyloid cascade",
    ]

    gene_protein_keywords = [
        "gene",
        "genes",
        "protein",
        "proteins",
        "encode",
        "encodes",
        "mutation",
        "variant",
        "variants",
        "hgnc",
        "uniprot",
    ]

    # -----------------------------------------------------------------
    # 2) Score each intent bucket based on keyword hits
    # -----------------------------------------------------------------
    def count_hits(words: List[str]) -> int:
        return sum(1 for w in words if w in q)

    biomarker_hits = count_hits(biomarker_keywords)
    drug_hits = count_hits(drug_keywords)
    phenotype_hits = count_hits(phenotype_keywords)
    pathway_hits = count_hits(pathway_keywords)
    gene_protein_hits = count_hits(gene_protein_keywords)

    # -----------------------------------------------------------------
    # 3) Decide primary intent based on strongest signal
    # -----------------------------------------------------------------
    scores = {
        IntentType.BIOMARKER: biomarker_hits,
        IntentType.DRUG_TRIAL: drug_hits,
        IntentType.PHENOTYPE: phenotype_hits,
        IntentType.PATHWAY: pathway_hits,
        IntentType.GENE_PROTEIN: gene_protein_hits,
    }

    # pick the intent with the highest score
    primary_intent = max(scores, key=scores.get)
    max_score = scores[primary_intent]

    # If no bucket has signal, default to GENERAL_AD if the question
    # looks like it's about Alzheimer's; otherwise OTHER.
    if max_score == 0:
        if "alzheimer" in q or "alzheimers" in q:
            primary_intent = IntentType.GENERAL_AD
            notes = "No strong domain keywords; treated as general Alzheimer's question."
        else:
            primary_intent = IntentType.OTHER
            notes = "No strong domain keywords; classified as OTHER."
    else:
        notes = f"Selected {primary_intent.name} based on keyword hits: {scores}"

    # -----------------------------------------------------------------
    # 4) Extract any explicit IDs (MONDO, HP, GO, etc.)
    # -----------------------------------------------------------------
    ids = _extract_potential_ids(q_raw)

    # -----------------------------------------------------------------
    # 5) Heuristic: if user mentions “biomarker(s)” explicitly,
    #    we should bias to BIOMARKER even if there are other hits.
    # -----------------------------------------------------------------
    if "biomarker" in q or "biomarkers" in q:
        if primary_intent != IntentType.BIOMARKER:
            notes += " Overridden to BIOMARKER due to explicit 'biomarker(s)' mention."
        primary_intent = IntentType.BIOMARKER

    # Similarly, if "trial" or "phase" is present, we strongly bias DRUG_TRIAL.
    if "trial" in q or "phase" in q:
        if primary_intent != IntentType.DRUG_TRIAL and drug_hits > 0:
            notes += " Overridden to DRUG_TRIAL due to trial/phase mention."
            primary_intent = IntentType.DRUG_TRIAL

    return QueryIntent(
        type=primary_intent,
        focus_entities=ids,
        raw_question=q_raw,
        notes=notes,
    )