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

Classification uses a two-pass strategy:
  1. Entity linker (vocabulary match against the live graph) — high precision.
  2. Keyword scoring fallback — used when the linker is unavailable or fires on nothing.

The off-topic gate runs FIRST (before the linker) so clearly irrelevant queries never
reach the graph or LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List


logger = logging.getLogger(__name__)

# Fluid-context words used to disambiguate tau/GFAP/NfL between
# Biomarker (measured in a body fluid) and Gene/Protein (molecular entity).
_FLUID_WORDS = frozenset({
    "csf", "plasma", "serum", "fluid", "blood", "cerebrospinal",
    "liquor", "urine", "saliva",
})

# Unambiguously off-topic domains — classified as OTHER immediately.
# Using a frozenset for O(1) lookup; individual terms are checked with
# word-boundary regex so "sport" doesn't fire on "transport".
_CLEARLY_OFFTOPIC = frozenset({
    "weather", "forecast", "recipe", "cook", "bake", "sport",
    "football", "soccer", "basketball", "baseball", "cricket",
    "movie", "film", "actor", "actress", "music", "song", "album",
    "stock", "invest", "crypto", "bitcoin", "forex",
    "travel", "flight", "hotel", "vacation", "holiday",
    "politics", "election", "president", "government",
    "joke", "poem", "story", "fiction",
})


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
        Explicit ontology IDs found in the question (MONDO, HP, GO, CHEBI …).

    entity_matches:
        Typed EntityMatch objects from the entity linker. Empty when the
        linker is unavailable or the keyword fallback was used. Downstream
        router uses these to run targeted graph queries instead of full
        neighborhood pulls.

    raw_question:
        Original user question (for debugging / logging).

    notes:
        Human-readable notes on why this intent was chosen.
    """

    type: IntentType
    focus_entities: List[str] = field(default_factory=list)
    entity_matches: List[Any] = field(default_factory=list)
    raw_question: str = ""
    notes: str = ""


# ---------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------


def classify_question(question: str) -> QueryIntent:
    """
    Classify a question into one of the IntentType categories.

    Pass 1 — Entity linker (graph vocabulary match):
        High-precision; resolves exact entity names to typed graph nodes.
        Handles ambiguous terms like 'tau' by checking fluid-word context.

    Pass 2 — Keyword scoring fallback:
        Used when the linker is unavailable or matches nothing.
        Deterministic; no LLM dependency.

    Off-topic gate runs before both passes.
    """
    q_raw = question or ""
    q = q_raw.lower()

    # -----------------------------------------------------------------
    # 0) Off-topic gate — FIRST, before any entity linking or scoring.
    #    Word-boundary match so "sport" does not fire on "transport".
    # -----------------------------------------------------------------
    if any(re.search(r"\b" + re.escape(t) + r"\b", q) for t in _CLEARLY_OFFTOPIC):
        return QueryIntent(
            type=IntentType.OTHER,
            focus_entities=[],
            entity_matches=[],
            raw_question=q_raw,
            notes="Matched clearly off-topic domain; classified as OTHER.",
        )

    # Extract any explicit ontology IDs regardless of path taken below.
    ids = _extract_potential_ids(q_raw)

    # -----------------------------------------------------------------
    # 1) Entity linker pass
    # -----------------------------------------------------------------
    entity_matches: List[Any] = []
    try:
        from .entity_linker import get_entity_linker
        linker = get_entity_linker()
        entity_matches = linker.match_entities(q_raw)
    except Exception as exc:
        logger.warning("Entity linker unavailable (%s); falling back to keyword scoring.", exc)

    if entity_matches:
        types_found = {m.node_type for m in entity_matches}

        # Fluid-word context disambiguates tau/GFAP/NfL between
        # "biomarker measured in CSF" vs. "gene/protein entity".
        has_fluid_ctx = any(
            re.search(r"\b" + re.escape(w) + r"\b", q) for w in _FLUID_WORDS
        )

        if "Biomarker" in types_found and ("Gene" in types_found or "Protein" in types_found):
            primary_intent = IntentType.BIOMARKER if has_fluid_ctx else IntentType.GENE_PROTEIN
            notes = (
                f"Entity linker found Biomarker+Gene/Protein; "
                f"fluid-word {'present' if has_fluid_ctx else 'absent'} → {primary_intent.name}."
            )
        elif "Biomarker" in types_found:
            labels = [m.canonical_label for m in entity_matches if m.node_type == "Biomarker"]
            primary_intent = IntentType.BIOMARKER
            notes = f"Entity linker found Biomarker nodes: {labels}."
        elif "Drug" in types_found:
            labels = [m.canonical_label for m in entity_matches if m.node_type == "Drug"]
            primary_intent = IntentType.DRUG_TRIAL
            notes = f"Entity linker found Drug nodes: {labels}."
        elif "Gene" in types_found or "Protein" in types_found:
            primary_intent = IntentType.GENE_PROTEIN
            notes = "Entity linker found Gene/Protein nodes."
        elif "Pathway" in types_found:
            primary_intent = IntentType.PATHWAY
            notes = "Entity linker found Pathway nodes."
        elif "Phenotype" in types_found:
            primary_intent = IntentType.PHENOTYPE
            notes = "Entity linker found Phenotype nodes."
        else:
            # AlzPedia or other node types — treat as general AD context
            primary_intent = IntentType.GENERAL_AD
            notes = f"Entity linker found {types_found}; defaulting to GENERAL_AD."

        # --- Explicit keyword overrides (entity linker path) ---
        # Applied AFTER entity-type assignment so the question's wording always
        # wins over what the entity linker happened to match.
        # Execution order: drug → trial/phase → biomarker → pathway.
        # Each successive check can supersede the previous one (last-wins).

        # "drug(s)" or "medication(s)" explicitly in the question → DRUG_TRIAL.
        # This handles "amyloid-targeting drugs have received FDA approval?" where
        # the entity linker correctly finds amyloid but the *question* is about drugs.
        if re.search(r"\b(drugs?|medications?)\b", q):
            if primary_intent != IntentType.DRUG_TRIAL:
                notes += " Overridden to DRUG_TRIAL due to explicit drug/medication mention."
            primary_intent = IntentType.DRUG_TRIAL

        # "trial" or "phase" + drug context → DRUG_TRIAL
        if "trial" in q or "phase" in q:
            _trial_ctx = any(
                re.search(r"\b" + re.escape(w) + r"\b", q)
                for w in ("drug", "drugs", "treat", "treatment", "therapy", "compound", "trial", "clinical")
            )
            if _trial_ctx and primary_intent != IntentType.DRUG_TRIAL:
                notes += " Overridden to DRUG_TRIAL due to trial/phase mention."
                primary_intent = IntentType.DRUG_TRIAL

        # "biomarker(s)" supersedes any drug override above
        if "biomarker" in q or "biomarkers" in q:
            if primary_intent != IntentType.BIOMARKER:
                notes += " Overridden to BIOMARKER due to explicit 'biomarker(s)' mention."
            primary_intent = IntentType.BIOMARKER

        # "pathway(s)" has the highest explicit-keyword priority
        # (handles "which pathways does lecanemab affect?" where entity linker → Drug)
        if re.search(r"\bpathways?\b", q):
            if primary_intent != IntentType.PATHWAY:
                notes += " Overridden to PATHWAY due to explicit 'pathway(s)' mention."
            primary_intent = IntentType.PATHWAY

        logger.debug("classify_question [linker]: %s → %s", q_raw[:80], primary_intent.name)
        return QueryIntent(
            type=primary_intent,
            focus_entities=ids,
            entity_matches=entity_matches,
            raw_question=q_raw,
            notes=notes,
        )

    # -----------------------------------------------------------------
    # 2) Keyword scoring fallback
    # -----------------------------------------------------------------
    biomarker_keywords = [
        "biomarker", "marker", "csf", "plasma", "serum", "fluid",
        "cutoff", "sensitivity", "specificity",
        "tau", "p-tau", "ptau", "phospho-tau", "tau181", "tau-181",
        "abeta", "amyloid-beta", "amyloid beta",
        "nfl", "neurofilament", "ykl-40", "gfap", "neurogranin", "synaptotagmin",
    ]

    drug_keywords = [
        "drug", "drugs", "treat", "treats", "treatment", "treatments",
        "therapy", "therapies", "therapeutic", "therapeutics",
        "compound", "compounds", "trial", "trials", "phase",
        "phase 2", "phase 3", "approved", "approval",
        "medication", "medications", "dosage", "dose",
        "clinical trial", "clinical trials",
    ]

    phenotype_keywords = [
        "symptom", "symptoms", "sign", "signs",
        "clinical feature", "clinical features",
        "cognitive", "memory", "language", "aphasia",
        "behavior", "behaviour", "phenotype", "phenotypes",
        "presentation", "manifestation", "manifestations",
    ]

    pathway_keywords = [
        "pathway", "pathways", "go:", "signaling", "signalling",
        "microglial", "synaptic", "amyloid cascade",
    ]

    gene_protein_keywords = [
        "gene", "genes", "protein", "proteins", "encode", "encodes",
        "mutation", "variant", "variants", "hgnc", "uniprot",
        "apoe", "apoe4", "presenilin", "psen1", "psen2", "app",
        "bace", "bace1", "trem2", "cd33", "bin1", "sorl1", "clu", "cr1",
    ]

    def count_hits(words: List[str]) -> int:
        return sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", q))

    scores = {
        IntentType.BIOMARKER:    count_hits(biomarker_keywords),
        IntentType.DRUG_TRIAL:   count_hits(drug_keywords),
        IntentType.PHENOTYPE:    count_hits(phenotype_keywords),
        IntentType.PATHWAY:      count_hits(pathway_keywords),
        IntentType.GENE_PROTEIN: count_hits(gene_protein_keywords),
    }

    primary_intent = max(scores, key=scores.get)
    max_score = scores[primary_intent]

    # Tie-break between BIOMARKER and GENE_PROTEIN using fluid-word context,
    # mirroring the entity linker path. "tau protein structure" → no fluid
    # words → GENE_PROTEIN; "tau levels in plasma" → fluid word → BIOMARKER.
    if (
        max_score > 0
        and scores[IntentType.BIOMARKER] == scores[IntentType.GENE_PROTEIN]
        and primary_intent in (IntentType.BIOMARKER, IntentType.GENE_PROTEIN)
    ):
        has_fluid_ctx = any(
            re.search(r"\b" + re.escape(w) + r"\b", q) for w in _FLUID_WORDS
        )
        primary_intent = IntentType.BIOMARKER if has_fluid_ctx else IntentType.GENE_PROTEIN

    if max_score == 0:
        # No keyword signal — could be an AD ontology term the keyword list
        # doesn't cover. Default to GENERAL_AD; the LLM system prompt handles
        # truly irrelevant questions by reporting insufficient context.
        primary_intent = IntentType.GENERAL_AD
        notes = "No domain keywords matched; defaulting to GENERAL_AD for AD-specific assistant."
    else:
        notes = f"Keyword fallback selected {primary_intent.name}: {scores}."

    # --- Explicit keyword overrides (keyword fallback path) ---
    # Same order as entity linker path: drug → trial/phase → biomarker → pathway.

    if re.search(r"\b(drugs?|medications?)\b", q):
        if primary_intent != IntentType.DRUG_TRIAL:
            notes += " Overridden to DRUG_TRIAL due to explicit drug/medication mention."
        primary_intent = IntentType.DRUG_TRIAL

    if "trial" in q or "phase" in q:
        if primary_intent != IntentType.DRUG_TRIAL and scores[IntentType.DRUG_TRIAL] > 0:
            notes += " Overridden to DRUG_TRIAL due to trial/phase mention."
            primary_intent = IntentType.DRUG_TRIAL

    if "biomarker" in q or "biomarkers" in q:
        if primary_intent != IntentType.BIOMARKER:
            notes += " Overridden to BIOMARKER due to explicit 'biomarker(s)' mention."
        primary_intent = IntentType.BIOMARKER

    if re.search(r"\bpathways?\b", q):
        if primary_intent != IntentType.PATHWAY:
            notes += " Overridden to PATHWAY due to explicit 'pathway(s)' mention."
        primary_intent = IntentType.PATHWAY

    logger.debug("classify_question [keyword]: %s → %s", q_raw[:80], primary_intent.name)
    return QueryIntent(
        type=primary_intent,
        focus_entities=ids,
        entity_matches=[],
        raw_question=q_raw,
        notes=notes,
    )
