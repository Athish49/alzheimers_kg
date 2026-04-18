"""
kg_build.build_edges
--------------------
Build KG edge tables (CSV) from AlzForum processed data + ontology-backed nodes.

Implemented edges (Phase 4, richer graph):
- HAS_BIOMARKER     : Disease -> Biomarker   (from AlzBiomarker)
- TREATS            : Drug    -> Disease     (from Therapeutics trials)
- ENCODES           : Gene    -> Protein     (from PRO.gene_symbol + HGNC)
- TARGETS_PROTEIN   : Drug    -> Protein     (from therapeutics_targets)
- AFFECTS_PATHWAY   : Drug    -> Pathway     (from therapeutics_targets)
- HAS_PHENOTYPE     : Disease -> Phenotype   (seed AD → HPO phenotypes)

Each edge type is written to:
    kg_build/output/edges_<reltype_lower>.csv

This script assumes you've already run:
    python -m kg_build.normalize_entities
so that nodes_*.csv exist under kg_build/output/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from kg_build.paths import (
    ALZFORUM_PROCESSED_DIR,
    KG_OUTPUT_DIR,
    ensure_dirs,
)
from kg_build.schema import get_edge_schema


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _load_nodes(label: str) -> pd.DataFrame:
    """
    Load node table for a given label.

    Example:
        label="Disease" -> kg_build/output/nodes_disease.csv
    """
    filename = f"nodes_{label.lower()}.csv"
    path = KG_OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing node file for label={label}: {path}")
    df = pd.read_csv(path, dtype=str)
    if "id" not in df.columns:
        raise ValueError(f"Node file {path} has no 'id' column.")
    return df


def _find_ad_disease_id(diseases_df: pd.DataFrame) -> Optional[str]:
    """
    Best-effort: find the canonical Alzheimer's disease MONDO ID

    Strategy:
      - Prefer MONDO-sourced row whose label or synonyms contain 'alzheimer'
    """
    df = diseases_df.copy()
    df["label_lower"] = df["label"].astype(str).str.lower()

    if "synonyms" in df.columns:
        df["synonyms_lower"] = df["synonyms"].astype(str).str.lower()
    else:
        df["synonyms_lower"] = ""

    # Filter to rows that look like Alzheimer in label or synonyms
    mask_alz = (
        df["label_lower"].str.contains("alzheimer")
        | df["synonyms_lower"].str.contains("alzheimer")
    )

    candidates = df[mask_alz]

    if candidates.empty:
        return None

    # Prefer MONDO source if present
    if "source" in candidates.columns:
        mondo_rows = candidates[candidates["source"] == "MONDO"]
        if not mondo_rows.empty:
            return mondo_rows.iloc[0]["id"]

    # Fallback: just take the first candidate
    return candidates.iloc[0]["id"]


def _write_edge_csv(
    rel_type: str,
    rows: List[dict],
) -> Path:
    """
    Write list of edge rows to output CSV for a given relationship type.
    """
    edge_schema = get_edge_schema(rel_type)
    if edge_schema is None:
        raise ValueError(f"Unknown edge type: {rel_type}")

    # Ensure directories exist
    ensure_dirs()
    out_dir = KG_OUTPUT_DIR

    # Ensure consistent column order:
    prop_cols = edge_schema.all_props
    columns = ["source_id", "target_id"] + prop_cols

    df = pd.DataFrame(rows, columns=columns)
    out_path = out_dir / f"edges_{rel_type.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"[build_edges] Wrote {len(df):,} rows -> {out_path}")
    return out_path


def _norm(s: str) -> str:
    """Simple string normalizer for matching (lowercase + strip)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().lower()


# ---------------------------------------------------------------------
# HAS_BIOMARKER: Disease -> Biomarker
# From: alzforum/processed/alzbiomarker_effects.csv
# ---------------------------------------------------------------------


def _infer_direction_from_effect(effect_size: float) -> str:
    """
    Infer direction from effect size:
      - > 1  -> 'increased'
      - < 1  -> 'decreased'
      - == 1 or NaN -> 'no_change'
    """
    if pd.isna(effect_size):
        return "no_change"
    try:
        val = float(effect_size)
    except Exception:
        return "no_change"

    if val > 1.0:
        return "increased"
    if val < 1.0:
        return "decreased"
    return "no_change"


def build_has_biomarker_edges() -> Path:
    """
    Build HAS_BIOMARKER edges from AlzBiomarker meta-analysis data.

    Source:
        alzforum/processed/alzbiomarker_effects.csv

    Edge schema (HAS_BIOMARKER):
        - source_label: Disease
        - target_label: Biomarker
        - required props: direction
        - optional: comparison, effect_size_type, effect_size, ci_lower,
                    ci_upper, p_value, n_studies, n_cases, n_controls,
                    study_id, source, source_text, etc.

    For now, we:
      - assume all rows correspond to Alzheimer's disease
      - map to the canonical AD MONDO node
      - map biomarker_key -> Biomarker.id
      - infer direction from effect_size (ratio-like interpretation)
    """
    edge_type = "HAS_BIOMARKER"
    eschema = get_edge_schema(edge_type)
    assert eschema is not None

    # Load node tables (so we only create edges to existing nodes)
    diseases_df = _load_nodes("Disease")
    biomarkers_df = _load_nodes("Biomarker")

    ad_id = _find_ad_disease_id(diseases_df)
    if ad_id is None:
        raise RuntimeError("Could not find Alzheimer disease node in nodes_disease.csv")

    biomarker_ids = set(biomarkers_df["id"].astype(str))

    # Load AlzBiomarker effects
    effects_path = ALZFORUM_PROCESSED_DIR / "alzbiomarker_effects.csv"
    if not effects_path.exists():
        raise FileNotFoundError(f"Missing AlzBiomarker effects file: {effects_path}")

    effects = pd.read_csv(effects_path)

    rows: List[dict] = []

    for _, row in effects.iterrows():
        biomarker_key = str(row.get("biomarker_key", "")).strip()
        if not biomarker_key or biomarker_key not in biomarker_ids:
            # We only attach edges to biomarkers we have as nodes
            continue

        effect_size = row.get("effect_size", np.nan)
        direction = _infer_direction_from_effect(effect_size)

        # Basic metadata we KNOW we have
        comparison = row.get("comparison", "")
        p_value = row.get("p_value", np.nan)
        meta_text = row.get("meta_text", "")

        # Build edge property dict according to schema
        props: Dict[str, object] = {
            "direction": direction,
            "comparison": comparison,
            # We currently don't have effect_size_type in file; leave blank.
            "effect_size_type": "",
            "effect_size": effect_size,
            "ci_lower": "",
            "ci_upper": "",
            "p_value": p_value,
            "n_studies": "",
            "n_cases": "",
            "n_controls": "",
            "study_id": "",
            "source": "AlzBiomarker",
            "source_text": meta_text,
        }

        # Ensure all schema props are present (missing -> blank)
        for prop in eschema.all_props:
            props.setdefault(prop, "")

        rows.append(
            {
                "source_id": ad_id,          # Disease
                "target_id": biomarker_key,  # Biomarker
                **props,
            }
        )

    return _write_edge_csv(edge_type, rows)


# ---------------------------------------------------------------------
# TREATS: Drug -> Disease
# From: alzforum/processed/therapeutics_trials.csv
# ---------------------------------------------------------------------


def _map_indications_to_diseases(
    indication_text: str,
    diseases_df: pd.DataFrame,
) -> List[str]:
    """
    Map a free-text 'indication' string to one or more disease node IDs.

    For now:
      - if the token contains 'alzheimer', map to canonical AD MONDO ID
      - ignore non-AD indications (Parkinson's, PPA, etc.) for this project phase
    """
    if not indication_text or pd.isna(indication_text):
        return []

    ad_id = _find_ad_disease_id(diseases_df)
    if ad_id is None:
        return []

    text = str(indication_text).lower()
    disease_ids: List[str] = []

    # Very simple heuristic: anything mentioning 'alzheimer' or 'mci due to ad'
    if "alzheimer" in text or "mci due to ad" in text or "ad " in text:
        disease_ids.append(ad_id)

    # Deduplicate
    disease_ids = list(dict.fromkeys(disease_ids).keys()) if disease_ids else []
    return disease_ids


def build_treats_edges() -> Path:
    """
    Build TREATS edges from Therapeutics trials summary.

    Source:
        alzforum/processed/therapeutics_trials.csv

    Edge schema (TREATS):
        source_label: Drug
        target_label: Disease
        optional props: status, indication, trial_phase_max, has_phase3,
                        trial_count, approved_regions, source, notes
    """
    edge_type = "TREATS"
    eschema = get_edge_schema(edge_type)
    assert eschema is not None

    # Nodes
    drugs_df = _load_nodes("Drug")
    diseases_df = _load_nodes("Disease")

    drug_ids = set(drugs_df["id"].astype(str))

    # Trials table
    trials_path = ALZFORUM_PROCESSED_DIR / "therapeutics_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing therapeutics_trials file: {trials_path}")

    trials = pd.read_csv(trials_path)

    rows: List[dict] = []

    for _, row in trials.iterrows():
        drug_id = str(row.get("therapeutic_id", "")).strip()
        if not drug_id or drug_id not in drug_ids:
            # Only create edges to drugs that exist as nodes
            continue

        indication_text = row.get("indication", "")
        disease_ids = _map_indications_to_diseases(indication_text, diseases_df)
        if not disease_ids:
            # For now we only model AD-related indications
            continue

        # Edge props
        status = row.get("status", "")
        trial_phase_max = row.get("trial_phase_max", "")
        has_phase3 = row.get("has_phase3", "")
        trial_count = row.get("trial_count", "")
        notes = row.get("notes", "")

        props: Dict[str, object] = {
            "status": status,
            "indication": indication_text,
            "trial_phase_max": trial_phase_max,
            "has_phase3": has_phase3,
            "trial_count": trial_count,
            "approved_regions": "",
            "source": "AlzForum.Therapeutics",
            "notes": notes,
        }

        for prop in eschema.all_props:
            props.setdefault(prop, "")

        for disease_id in disease_ids:
            rows.append(
                {
                    "source_id": drug_id,     # Drug
                    "target_id": disease_id,  # Disease
                    **props,
                }
            )

    return _write_edge_csv(edge_type, rows)


# ---------------------------------------------------------------------
# ENCODES: Gene -> Protein  (HGNC / PRO mapping)
# ---------------------------------------------------------------------


def build_encodes_edges() -> Path:
    """
    Build ENCODES edges using:
      - Gene nodes (HGNC), where label = gene symbol
      - Protein nodes (PRO), where 'gene_symbol' property is filled

    Strategy:
      - Build mapping from normalized gene_symbol -> list of protein IDs
      - For each Gene node, if its symbol appears in that mapping,
        create Gene -[:ENCODES]-> Protein edges.
    """
    edge_type = "ENCODES"
    eschema = get_edge_schema(edge_type)
    if eschema is None:
        raise ValueError("ENCODES edge schema is not defined in schema.py")

    genes_df = _load_nodes("Gene")
    proteins_df = _load_nodes("Protein")

    # Map gene_symbol (normalized) -> list of protein IDs
    gene_sym_to_protein_ids: Dict[str, List[str]] = {}
    if "gene_symbol" in proteins_df.columns:
        for _, prow in proteins_df.iterrows():
            gene_sym = _norm(prow.get("gene_symbol", ""))
            if not gene_sym:
                continue
            pid = prow["id"]
            gene_sym_to_protein_ids.setdefault(gene_sym, []).append(pid)

    rows: List[dict] = []

    for _, grow in genes_df.iterrows():
        gene_id = grow["id"]
        gene_symbol_norm = _norm(grow.get("label", ""))  # HGNC symbol

        if not gene_symbol_norm:
            continue

        protein_ids = gene_sym_to_protein_ids.get(gene_symbol_norm, [])
        if not protein_ids:
            continue

        props: Dict[str, object] = {"source": "HGNC_PRO"}
        for prop in eschema.all_props:
            props.setdefault(prop, "")

        for pid in protein_ids:
            rows.append(
                {
                    "source_id": gene_id,  # Gene
                    "target_id": pid,      # Protein
                    **props,
                }
            )

    return _write_edge_csv(edge_type, rows)


# ---------------------------------------------------------------------
# TARGETS_PROTEIN: Drug -> Protein  (therapeutics_targets)
# AFFECTS_PATHWAY: Drug -> Pathway (therapeutics_targets)
# ---------------------------------------------------------------------


def build_targets_and_pathways_edges() -> (Path, Path):
    """
    Build:
      - TARGETS_PROTEIN edges (Drug -> Protein)
      - AFFECTS_PATHWAY edges (Drug -> Pathway)

    Source:
        alzforum/processed/therapeutics_targets.csv
    """
    # Edge schemas
    eschema_tp = get_edge_schema("TARGETS_PROTEIN")
    if eschema_tp is None:
        raise ValueError("TARGETS_PROTEIN edge schema is not defined in schema.py")

    eschema_ap = get_edge_schema("AFFECTS_PATHWAY")
    if eschema_ap is None:
        raise ValueError("AFFECTS_PATHWAY edge schema is not defined in schema.py")

    # Node tables
    drugs_df = _load_nodes("Drug")
    proteins_df = _load_nodes("Protein")
    genes_df = _load_nodes("Gene")
    pathways_df = _load_nodes("Pathway")

    drug_ids = set(drugs_df["id"].astype(str))

    # --- Protein lookup structures -----------------------------------
    gene_sym_to_protein_ids: Dict[str, List[str]] = {}
    if "gene_symbol" in proteins_df.columns:
        for _, prow in proteins_df.iterrows():
            gene_sym = _norm(prow.get("gene_symbol", ""))
            if not gene_sym:
                continue
            pid = prow["id"]
            gene_sym_to_protein_ids.setdefault(gene_sym, []).append(pid)

    prot_label_to_id: Dict[str, str] = {
        _norm(prow["label"]): prow["id"]
        for _, prow in proteins_df.iterrows()
        if _norm(prow.get("label", ""))
    }

    prot_syn_to_id: Dict[str, str] = {}
    if "synonyms" in proteins_df.columns:
        for _, prow in proteins_df.iterrows():
            pid = prow["id"]
            syns = prow.get("synonyms", "")
            if not isinstance(syns, str) or not syns.strip():
                continue
            for part in syns.split("|"):
                key = _norm(part)
                if key and key not in prot_syn_to_id:
                    prot_syn_to_id[key] = pid

    gene_sym_to_gene_id: Dict[str, str] = {
        _norm(grow.get("label", "")): grow["id"]
        for _, grow in genes_df.iterrows()
        if _norm(grow.get("label", ""))
    }

    # --- Pathway lookup structures -----------------------------------
    # id -> normalized label, for token-overlap matching
    path_id_to_norm_label: Dict[str, str] = {
        prow["id"]: _norm(prow["label"])
        for _, prow in pathways_df.iterrows()
        if _norm(prow.get("label", ""))
    }

    # --- Read therapeutics_targets -----------------------------------
    targets_path = ALZFORUM_PROCESSED_DIR / "therapeutics_targets.csv"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing therapeutics_targets file: {targets_path}")

    targets = pd.read_csv(targets_path)

    tp_rows: List[dict] = []
    ap_rows: List[dict] = []

    for _, row in targets.iterrows():
        drug_id = str(row.get("therapeutic_id", "")).strip()
        if not drug_id or drug_id not in drug_ids:
            continue

        target_name = str(row.get("target_name", "")).strip()
        if not target_name:
            continue

        target_kind = _norm(row.get("target_kind", ""))
        action_type = row.get("action_type", "")
        is_primary_target = row.get("is_primary_target", "")
        target_notes = row.get("target_notes", "")

        name_norm = _norm(target_name)

        # --------------------------
        # TARGETS_PROTEIN candidates
        # --------------------------
        protein_ids_for_target: List[str] = []

        # 1) If target name is a gene symbol we know, use encoded proteins
        if name_norm in gene_sym_to_protein_ids:
            protein_ids_for_target.extend(gene_sym_to_protein_ids[name_norm])

        # 2) Direct match to protein label
        pid = prot_label_to_id.get(name_norm)
        if pid:
            protein_ids_for_target.append(pid)

        # 3) Match to protein synonym
        pid2 = prot_syn_to_id.get(name_norm)
        if pid2:
            protein_ids_for_target.append(pid2)

        protein_ids_for_target = list(dict.fromkeys(protein_ids_for_target))

        if protein_ids_for_target:
            props_tp = {
                "action_type": action_type,
                "is_primary_target": is_primary_target,
                "source": "AlzForum.Therapeutics",
                "target_notes": target_notes,
            }
            for prop in eschema_tp.all_props:
                props_tp.setdefault(prop, "")

            for pid in protein_ids_for_target:
                tp_rows.append(
                    {
                        "source_id": drug_id,  # Drug
                        "target_id": pid,      # Protein
                        **props_tp,
                    }
                )

        # --------------------------
        # AFFECTS_PATHWAY candidates (token-overlap)
        # --------------------------
        is_pathwayish = any(
            kw in target_kind
            for kw in ["pathway", "process", "pathway_or_process"]
        )

        if is_pathwayish:
            name_tokens = set(name_norm.replace("-", " ").split())
            matched_path_ids: List[str] = []

            for path_id, path_label_norm in path_id_to_norm_label.items():
                path_tokens = set(path_label_norm.replace("-", " ").split())
                if name_tokens & path_tokens:
                    matched_path_ids.append(path_id)

            matched_path_ids = list(dict.fromkeys(matched_path_ids))

            if matched_path_ids:
                props_ap = {
                    "action_type": action_type,
                    "is_primary_target": is_primary_target,
                    "source": "AlzForum.Therapeutics",
                    "target_notes": target_notes,
                }
                for prop in eschema_ap.all_props:
                    props_ap.setdefault(prop, "")

                for path_id in matched_path_ids:
                    ap_rows.append(
                        {
                            "source_id": drug_id,  # Drug
                            "target_id": path_id,  # Pathway
                            **props_ap,
                        }
                    )

    tp_path = _write_edge_csv("TARGETS_PROTEIN", tp_rows)
    ap_path = _write_edge_csv("AFFECTS_PATHWAY", ap_rows)
    return tp_path, ap_path


# ---------------------------------------------------------------------
# HAS_PHENOTYPE: Disease -> Phenotype
# ---------------------------------------------------------------------


def build_has_phenotype_edges() -> Path:
    """
    Build HAS_PHENOTYPE edges.

    We currently do NOT have a dedicated disease-phenotype mapping file.
    For this project, we seed clinically reasonable relationships:

        Alzheimer's disease MONDO node
        -> all Phenotype nodes we have (cognitive impairment, memory
           impairment, aphasia, ...)

    This keeps the graph richer and is directionally correct, even if
    not backed by a specific ontology crosswalk in Phase 4.
    """
    edge_type = "HAS_PHENOTYPE"
    eschema = get_edge_schema(edge_type)
    if eschema is None:
        raise ValueError("HAS_PHENOTYPE edge schema is not defined in schema.py")

    diseases_df = _load_nodes("Disease")
    phenotypes_df = _load_nodes("Phenotype")

    ad_id = _find_ad_disease_id(diseases_df)
    if ad_id is None:
        raise RuntimeError("Could not find Alzheimer disease node in nodes_disease.csv")

    rows: List[dict] = []

    for _, prow in phenotypes_df.iterrows():
        pheno_id = prow["id"]
        props: Dict[str, object] = {
            "onset": "",
            "frequency": "",
            "source": "HPO (seeded for AD)",
        }
        for prop in eschema.all_props:
            props.setdefault(prop, "")

        rows.append(
            {
                "source_id": ad_id,   # Disease (AD)
                "target_id": pheno_id,
                **props,
            }
        )

    return _write_edge_csv(edge_type, rows)


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------


def build_all_edges() -> None:
    """
    Build all currently implemented edge CSVs.
    """
    print("[build_edges] Building HAS_BIOMARKER edges...")
    has_biomarker_path = build_has_biomarker_edges()

    print("[build_edges] Building TREATS edges...")
    treats_path = build_treats_edges()

    print("[build_edges] Building ENCODES edges (Gene -> Protein)...")
    encodes_path = build_encodes_edges()

    print("[build_edges] Building TARGETS_PROTEIN and AFFECTS_PATHWAY edges...")
    tp_path, ap_path = build_targets_and_pathways_edges()

    print("[build_edges] Building HAS_PHENOTYPE edges (Disease -> Phenotype)...")
    has_pheno_path = build_has_phenotype_edges()

    print("\n[build_edges] Done.")
    print(f"  - HAS_BIOMARKER    -> {has_biomarker_path}")
    print(f"  - TREATS           -> {treats_path}")
    print(f"  - ENCODES          -> {encodes_path}")
    print(f"  - TARGETS_PROTEIN  -> {tp_path}")
    print(f"  - AFFECTS_PATHWAY  -> {ap_path}")
    print(f"  - HAS_PHENOTYPE    -> {has_pheno_path}")


if __name__ == "__main__":
    build_all_edges()