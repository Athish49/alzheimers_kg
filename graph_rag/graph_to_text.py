"""
graph_rag.graph_to_text
-----------------------

Utilities for turning *structured graph query results* into
LLM-friendly, **ultra-compact** textual context.

This module has two layers:

1) Pure formatters that take Python dict/list structures
   (summarize_biomarkers, summarize_drugs, summarize_phenotypes,
    summarize_drug_pathways, summarize_genes_proteins).

2) Small convenience wrappers that talk to the GraphRetriever and
   build **intent-specific** context strings:

   - build_biomarker_direction_context(...)
   - build_phenotype_context(...)
   - build_drug_trial_pathway_context(...)
   - build_general_ad_context(...)

These wrappers are what the router / pipeline should call.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .config import CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(x: object) -> str:
    """Return a safe string representation (avoid 'None')."""
    if x is None:
        return ""
    return str(x)


def _normalize_direction(raw: Optional[str]) -> str:
    """
    Normalize direction strings from the KG to a small set:
    'increased', 'decreased', 'no_change', 'unknown'.
    """
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    if "increase" in s or "higher" in s or s in ("up", "upregulated"):
        return "increased"
    if "decrease" in s or "lower" in s or s in ("down", "downregulated"):
        return "decreased"
    if "no" in s and "change" in s:
        return "no_change"
    return s or "unknown"


def _fluid_bucket(fluid: Optional[str]) -> str:
    """
    Bucket fluid names to keep things simple:
    - "CSF"
    - "Plasma/Serum"
    - "Other/unspecified"
    """
    if not fluid:
        return "Other/unspecified"
    s = fluid.lower()
    if "csf" in s:
        return "CSF"
    if "plasma" in s or "serum" in s:
        return "Plasma/Serum"
    return "Other/unspecified"


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Simple stable de-duplication."""
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Biomarkers
# ---------------------------------------------------------------------------


def summarize_biomarkers(biomarkers: List[Dict[str, object]]) -> str:
    """
    Ultra-compact summary of AD biomarkers.

    Input: list of rows as returned by GraphRetriever.get_ad_biomarkers:

        {
            "biomarker_id": ...,
            "biomarker_label": ...,
            "analyte": ...,
            "analyte_class": ...,
            "fluid": ...,
            "direction": ...,
            "effect_size": ...,
            "p_value": ...,
            ...
        }

    Output: structured text grouped by fluid + direction.
    """
    if not biomarkers:
        return "No biomarker information was found in the graph.\n"

    # group[(fluid_bucket, direction)] = list of (name, extras_dict)
    groups: Dict[Tuple[str, str], List[Tuple[str, Dict[str, str]]]] = defaultdict(list)

    for row in biomarkers:
        fluid_bucket = _fluid_bucket(_safe_str(row.get("fluid")))
        direction = _normalize_direction(_safe_str(row.get("direction")))

        # Choose a display name for the biomarker
        name = (
            _safe_str(row.get("biomarker_label"))
            or _safe_str(row.get("analyte"))
            or _safe_str(row.get("biomarker_id"))
        )

        if not name:
            continue

        extras = {}
        analyte_class = _safe_str(row.get("analyte_class"))
        if analyte_class:
            extras["class"] = analyte_class

        # effect size and p-value if present
        effect_size = row.get("effect_size")
        if effect_size not in (None, ""):
            extras["effect_size"] = str(effect_size)

        p_value = row.get("p_value")
        if p_value not in (None, ""):
            extras["p"] = str(p_value)

        groups[(fluid_bucket, direction)].append((name, extras))

    # Build text sections
    lines: List[str] = []
    lines.append("### Biomarkers associated with Alzheimer’s disease")
    lines.append(
        "Below is a compact summary of biomarkers grouped by biofluid and "
        "direction of change in Alzheimer’s disease."
    )
    lines.append("")

    # Deterministic order of groups
    fluid_order = ["CSF", "Plasma/Serum", "Other/unspecified"]
    direction_order = ["decreased", "increased", "no_change", "unknown"]

    for fluid in fluid_order:
        section_started = False
        for direction in direction_order:
            key = (fluid, direction)
            if key not in groups:
                continue

            if not section_started:
                lines.append(f"#### {fluid}")
                section_started = True

            human_dir = {
                "decreased": "Decreased",
                "increased": "Increased",
                "no_change": "No clear change",
                "unknown": "Direction unclear",
            }.get(direction, direction.capitalize())

            lines.append(f"- **{human_dir} in AD**:")
            # Aggregate by name to avoid duplicates
            name_to_extras: Dict[str, Dict[str, str]] = {}
            for name, ex in groups[key]:
                if name not in name_to_extras:
                    name_to_extras[name] = dict(ex)
                else:
                    # crude merge: keep first non-empty values
                    for k, v in ex.items():
                        if k not in name_to_extras[name] or not name_to_extras[name][k]:
                            name_to_extras[name][k] = v

            for biomarker_name in sorted(name_to_extras.keys()):
                ex = name_to_extras[biomarker_name]
                ex_bits = []
                if "class" in ex:
                    ex_bits.append(ex["class"])
                if "effect_size" in ex:
                    ex_bits.append(f"effect_size={ex['effect_size']}")
                if "p" in ex:
                    ex_bits.append(f"p={ex['p']}")
                if ex_bits:
                    lines.append(f"  • {biomarker_name} ({', '.join(ex_bits)})")
                else:
                    lines.append(f"  • {biomarker_name}")
            lines.append("")  # blank line between direction groups

        if section_started:
            lines.append("")  # blank line between fluid sections

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Drugs and trials
# ---------------------------------------------------------------------------


def _drug_bucket(row: Dict[str, object]) -> str:
    """
    Bucket drug into a coarse development/approval category based on:
    - status_overall
    - trial_status
    - trial_phase_max
    - has_phase3
    """
    status_overall = _safe_str(row.get("status_overall")).lower()
    trial_status = _safe_str(row.get("trial_status")).lower()
    phase = row.get("trial_phase_max")
    has_phase3 = row.get("has_phase3")

    # "Approved / Phase 3 or beyond"
    if "approved" in status_overall:
        return "Approved or marketed"
    if has_phase3 in (True, "True", "true"):
        return "Phase 3 trials"
    try:
        phase_val = float(phase) if phase not in (None, "") else None
    except Exception:
        phase_val = None
    if phase_val is not None and phase_val >= 3.0:
        return "Phase 3 trials"

    # "Ongoing trials (Phase 1–2, unspecified)"
    if "ongoing" in trial_status or "recruiting" in trial_status:
        if phase_val is not None and phase_val > 0:
            return "Ongoing Phase 1–2 trials"
        return "Ongoing (phase unknown)"

    # "Discontinued / terminated / unknown"
    if any(k in trial_status for k in ("discontinue", "terminated", "halted")):
        return "Discontinued or terminated"

    return "Status unclear / other"


def summarize_drugs(drugs: List[Dict[str, object]]) -> str:
    """
    Ultra-compact summary of AD drugs / therapeutics.

    Input: rows from GraphRetriever.get_ad_drugs:

        {
            "drug_id": ...,
            "drug_label": ...,
            "drug_type": ...,
            "drug_class": ...,
            "status_overall": ...,
            "trial_status": ...,
            "trial_phase_max": ...,
            "has_phase3": ...,
            "trial_count": ...,
            "indication": ...,
        }
    """
    if not drugs:
        return "No AD therapeutics were found in the graph.\n"

    bucket_to_drugs: Dict[str, List[str]] = defaultdict(list)
    drug_meta: Dict[str, Dict[str, str]] = {}

    for row in drugs:
        name = _safe_str(row.get("drug_label")) or _safe_str(row.get("drug_id"))
        if not name:
            continue

        bucket = _drug_bucket(row)
        bucket_to_drugs[bucket].append(name)

        meta = {}
        if row.get("drug_type"):
            meta["type"] = _safe_str(row.get("drug_type"))
        if row.get("drug_class"):
            meta["class"] = _safe_str(row.get("drug_class"))
        if row.get("trial_phase_max") not in (None, ""):
            meta["max_phase"] = _safe_str(row.get("trial_phase_max"))
        if row.get("trial_status"):
            meta["trial_status"] = _safe_str(row.get("trial_status"))

        drug_meta[name] = meta

    lines: List[str] = []
    lines.append("### Therapeutics targeting Alzheimer’s disease")
    lines.append(
        "Drugs are grouped by overall development / trial status. "
        "Only high-level information is included here."
    )
    lines.append("")

    order = [
        "Approved or marketed",
        "Phase 3 trials",
        "Ongoing Phase 1–2 trials",
        "Ongoing (phase unknown)",
        "Discontinued or terminated",
        "Status unclear / other",
    ]

    for bucket in order:
        names = bucket_to_drugs.get(bucket)
        if not names:
            continue

        lines.append(f"#### {bucket}")
        for name in sorted(_dedupe_preserve_order(names)):
            meta = drug_meta.get(name, {})
            bits = []
            if meta.get("type"):
                bits.append(meta["type"])
            if meta.get("class"):
                bits.append(meta["class"])
            if meta.get("max_phase"):
                bits.append(f"max_phase={meta['max_phase']}")
            if meta.get("trial_status"):
                bits.append(f"trial_status={meta['trial_status']}")
            if bits:
                lines.append(f"- {name} ({', '.join(bits)})")
            else:
                lines.append(f"- {name}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Phenotypes / clinical features
# ---------------------------------------------------------------------------


def summarize_phenotypes(phenos: List[Dict[str, object]]) -> str:
    """
    Ultra-compact list of phenotypes / symptoms.

    Input rows:

        {
            "phenotype_id": ...,
            "phenotype_label": ...,
            "onset": ...,
            "frequency": ...,
        }
    """
    if not phenos:
        return "No phenotype / symptom data found for Alzheimer’s disease.\n"

    lines: List[str] = []
    lines.append("### Clinical phenotypes / symptoms of Alzheimer’s disease")
    lines.append(
        "These are symptoms or clinical features linked to Alzheimer’s "
        "disease in the knowledge graph."
    )
    lines.append("")

    for row in sorted(phenos, key=lambda r: _safe_str(r.get("phenotype_label"))):
        label = _safe_str(row.get("phenotype_label")) or _safe_str(
            row.get("phenotype_id")
        )
        onset = _safe_str(row.get("onset"))
        freq = _safe_str(row.get("frequency"))

        bits = []
        if onset:
            bits.append(f"onset={onset}")
        if freq:
            bits.append(f"frequency={freq}")

        if bits:
            lines.append(f"- {label} ({', '.join(bits)})")
        else:
            lines.append(f"- {label}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Drug -> Pathway relationships
# ---------------------------------------------------------------------------


def summarize_drug_pathways(drug_pathways: List[Dict[str, object]]) -> str:
    """
    Ultra-compact summary of pathways affected by AD drugs.

    Input rows:

        {
            "drug_id": ...,
            "drug_label": ...,
            "pathway_id": ...,
            "pathway_label": ...,
            "source": ...,
            "target_notes": ...,
            "action_type": ...,
            "is_primary_target": ...,
        }
    """
    if not drug_pathways:
        return "No pathway-level drug mechanism information was found.\n"

    # Group by drug
    drug_to_paths: Dict[str, List[Tuple[str, Dict[str, str]]]] = defaultdict(list)

    for row in drug_pathways:
        drug_name = _safe_str(row.get("drug_label")) or _safe_str(row.get("drug_id"))
        if not drug_name:
            continue

        pathway = (
            _safe_str(row.get("pathway_label"))
            or _safe_str(row.get("pathway_id"))
            or "Unknown pathway"
        )

        extras = {}
        if row.get("action_type"):
            extras["action_type"] = _safe_str(row.get("action_type"))
        if row.get("is_primary_target") not in (None, ""):
            extras["primary"] = _safe_str(row.get("is_primary_target"))

        drug_to_paths[drug_name].append((pathway, extras))

    lines: List[str] = []
    lines.append("### Pathways affected by Alzheimer’s disease therapeutics")
    lines.append(
        "This section lists drugs and the biological pathways they are "
        "reported to affect in the graph."
    )
    lines.append("")

    for drug_name in sorted(drug_to_paths.keys()):
        lines.append(f"#### {drug_name}")
        seen_paths = set()
        for pathway, extras in drug_to_paths[drug_name]:
            if (drug_name, pathway) in seen_paths:
                continue
            seen_paths.add((drug_name, pathway))

            bits = []
            if "action_type" in extras:
                bits.append(extras["action_type"])
            if "primary" in extras:
                bits.append(f"primary={extras['primary']}")
            if bits:
                lines.append(f"- {pathway} ({', '.join(bits)})")
            else:
                lines.append(f"- {pathway}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Genes and proteins
# ---------------------------------------------------------------------------


def summarize_genes_proteins(
    genes_proteins: List[Dict[str, object]],
    max_items: int = 15,
) -> str:
    """
    Summarize a small subset of Gene -> Protein relationships.

    Input rows:

        {
            "gene_id": ...,
            "gene_symbol": ...,
            "protein_id": ...,
            "protein_label": ...,
        }

    The idea is to give the LLM a flavor of the molecular layer
    without flooding the context.
    """
    if not genes_proteins:
        return "No Gene -> Protein encoding relationships were found.\n"

    lines: List[str] = []
    lines.append("### Example Gene → Protein relationships")
    lines.append(
        "These pairs illustrate genes and the proteins they encode, "
        "drawn from the Alzheimer-related knowledge graph."
    )
    lines.append("")

    # Prefer well-known AD genes if present
    preferred_prefixes = ("APOE", "APP", "MAPT", "PSEN1", "PSEN2")
    preferred: List[str] = []
    others: List[str] = []

    for row in genes_proteins:
        gene_symbol = _safe_str(row.get("gene_symbol")) or _safe_str(row.get("gene_id"))
        protein_label = _safe_str(row.get("protein_label")) or _safe_str(
            row.get("protein_id")
        )
        if not gene_symbol or not protein_label:
            continue

        line = f"- Gene {gene_symbol} encodes protein {protein_label}."
        if any(gene_symbol.upper().startswith(pref) for pref in preferred_prefixes):
            preferred.append(line)
        else:
            others.append(line)

    ordered = _dedupe_preserve_order(preferred) + _dedupe_preserve_order(others)
    for line in ordered[:max_items]:
        lines.append(line)
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Top-level: ultra-compact AD context (from lists)
# ---------------------------------------------------------------------------


def build_ad_ultra_compact_context_from_lists(
    disease_id: str,
    *,
    biomarkers: List[Dict[str, object]],
    drugs: List[Dict[str, object]],
    phenotypes: List[Dict[str, object]],
    drug_pathways: List[Dict[str, object]],
    genes_proteins: List[Dict[str, object]],
) -> str:
    """
    Build a single ultra-compact context string for Alzheimer's disease
    from already-fetched lists.
    """
    lines: List[str] = []

    lines.append("=== Alzheimer’s Disease Graph Context ===")
    lines.append(f"- Disease ID: {disease_id}")
    lines.append("")
    lines.append(
        "This context is a compact summary extracted from an Alzheimer’s "
        "disease knowledge graph. It focuses on biomarkers, therapeutics, "
        "clinical phenotypes, affected pathways, and example gene–protein "
        "relationships."
    )
    lines.append("")

    # Sections
    lines.append(summarize_biomarkers(biomarkers).rstrip())
    lines.append("")
    lines.append(summarize_drugs(drugs).rstrip())
    lines.append("")
    lines.append(summarize_phenotypes(phenotypes).rstrip())
    lines.append("")
    lines.append(summarize_drug_pathways(drug_pathways).rstrip())
    lines.append("")
    lines.append(summarize_genes_proteins(genes_proteins).rstrip())
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Intent-specific wrappers used by the router / pipeline
# ---------------------------------------------------------------------------


def _resolve_ad_disease_id(retriever: GraphRetriever) -> Optional[str]:
    did = retriever.get_alzheimers_disease_id()
    if did:
        return did
    # Fallback to config if retriever couldn't find it
    return getattr(CONFIG, "ad_disease_id", None)


def build_biomarker_direction_context(
    retriever: GraphRetriever,
    disease_id: Optional[str] = None,
    max_rows: int = 200,
) -> str:
    """
    Context for biomarker questions: only Alzheimer's biomarkers,
    grouped by fluid and direction of change.
    """
    if disease_id is None:
        disease_id = _resolve_ad_disease_id(retriever)

    if not disease_id:
        return (
            "The knowledge graph does not contain an Alzheimer's Disease node. "
            "No biomarker information is available."
        )

    biomarkers = retriever.get_ad_biomarkers(disease_id, limit=max_rows)

    header = [
        "=== Alzheimer’s Disease (biomarkers) ===",
        f"- Disease ID: {disease_id}",
        "",
        "The following sections list biomarkers connected to Alzheimer's "
        "disease in the knowledge graph, grouped by biofluid and whether "
        "they tend to increase, decrease, or show no change.",
        "",
    ]

    tail = [
        "",
        "When answering biomarker questions, use ONLY the biomarkers listed "
        "in this context (or clear paraphrases of their names). Do NOT invent "
        "additional biomarkers.",
    ]

    return "\n".join(header) + "\n" + summarize_biomarkers(biomarkers).rstrip() + "\n" + "\n".join(tail)


def build_phenotype_context(
    retriever: GraphRetriever,
    disease_id: Optional[str] = None,
    max_rows: int = 100,
) -> str:
    """
    Context for phenotype / symptom questions: only Phenotype nodes
    linked to Alzheimer's disease.
    """
    if disease_id is None:
        disease_id = _resolve_ad_disease_id(retriever)

    if not disease_id:
        return (
            "The knowledge graph does not contain an Alzheimer's Disease node. "
            "No phenotype information is available."
        )

    phenos = retriever.get_ad_phenotypes(disease_id, limit=max_rows)

    header = [
        "=== Alzheimer’s Disease (clinical phenotypes) ===",
        f"- Disease ID: {disease_id}",
        "",
        "The following clinical phenotypes are explicitly linked to Alzheimer’s "
        "disease in the knowledge graph.",
        "",
    ]

    tail = [
        "",
        "When answering questions about symptoms or phenotypes of Alzheimer's "
        "disease, restrict yourself to these items (or reasonable paraphrases) "
        "instead of inventing new symptoms.",
    ]

    return "\n".join(header) + "\n" + summarize_phenotypes(phenos).rstrip() + "\n" + "\n".join(tail)


def build_drug_trial_pathway_context(
    retriever: GraphRetriever,
    disease_id: Optional[str] = None,
    max_drugs: int = 300,
    max_paths: int = 400,
) -> str:
    """
    Context for drug / trial / pathway questions.

    Includes:
        - therapeutics linked to Alzheimer's disease, summarized by status
        - pathways affected by those drugs
    """
    if disease_id is None:
        disease_id = _resolve_ad_disease_id(retriever)

    if not disease_id:
        return (
            "The knowledge graph does not contain an Alzheimer's Disease node. "
            "No drug / trial / pathway information is available."
        )

    drugs = retriever.get_ad_drugs(disease_id, limit=max_drugs)
    drug_pws = retriever.get_ad_drug_pathways(disease_id, limit=max_paths)

    header = [
        "=== Alzheimer’s Disease: drugs, trials, and pathways ===",
        f"- Disease ID: {disease_id}",
        "",
        "This context lists Alzheimer-related drugs from the knowledge graph, "
        "their development / trial status, and the pathways they are reported "
        "to affect.",
        "",
    ]

    tail = [
        "",
        "When answering questions about Alzheimer's therapeutics and pathways, "
        "restrict yourself to the drugs and pathways that appear in this "
        "context. Do NOT invent new drugs or pathways.",
    ]

    parts = [
        "\n".join(header),
        summarize_drugs(drugs).rstrip(),
        "",
        summarize_drug_pathways(drug_pws).rstrip(),
        "",
        "\n".join(tail),
    ]
    return "\n".join(parts)


def build_general_ad_context(retriever: GraphRetriever) -> str:
    """
    General fallback context: compact Alzheimer’s disease graph summary
    across biomarkers, drugs, phenotypes, pathways, and genes/proteins.

    This is used for GENERAL intent (and as a safe default).
    """
    disease_id = _resolve_ad_disease_id(retriever)
    if not disease_id:
        return (
            "The knowledge graph does not contain an Alzheimer's Disease node. "
            "No general context is available."
        )

    biomarkers = retriever.get_ad_biomarkers(disease_id)
    drugs = retriever.get_ad_drugs(disease_id)
    phenos = retriever.get_ad_phenotypes(disease_id)
    drug_pws = retriever.get_ad_drug_pathways(disease_id)
    genes_proteins = retriever.get_genes_and_proteins()

    return build_ad_ultra_compact_context_from_lists(
        disease_id=disease_id,
        biomarkers=biomarkers,
        drugs=drugs,
        phenotypes=phenos,
        drug_pathways=drug_pws,
        genes_proteins=genes_proteins,
    )


__all__ = [
    "summarize_biomarkers",
    "summarize_drugs",
    "summarize_phenotypes",
    "summarize_drug_pathways",
    "summarize_genes_proteins",
    "build_ad_ultra_compact_context_from_lists",
    "build_biomarker_direction_context",
    "build_phenotype_context",
    "build_drug_trial_pathway_context",
    "build_general_ad_context",
]