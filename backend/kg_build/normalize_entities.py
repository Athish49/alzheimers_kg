"""
kg_build.normalize_entities
---------------------------

Phase 4: normalize ontology + AlzForum entity tables into canonical KG node CSVs.

Input (read-only):
    ontology/processed/diseases_mondo.csv
    ontology/processed/drugs_chebi.csv
    ontology/processed/genes_hgnc.csv
    ontology/processed/pathways_go.csv
    ontology/processed/phenotypes_hpo.csv
    ontology/processed/proteins_pro.csv

    alzforum/processed/alzbiomarker_biomarkers.csv
    alzforum/processed/alzpedia_entities.csv
    alzforum/processed/therapeutics_entities_enriched.csv   (if present)
    alzforum/processed/therapeutics_entities.csv            (fallback)
    alzforum/processed/therapeutics_trials.csv              (for Trial nodes)

Output (for later phases):
    kg_build/output/nodes_disease.csv
    kg_build/output/nodes_gene.csv
    kg_build/output/nodes_protein.csv
    kg_build/output/nodes_pathway.csv
    kg_build/output/nodes_phenotype.csv
    kg_build/output/nodes_biomarker.csv
    kg_build/output/nodes_drug.csv
    kg_build/output/nodes_mechanism.csv
    kg_build/output/nodes_company.csv
    kg_build/output/nodes_therapytype.csv
    kg_build/output/nodes_fluid.csv
    kg_build/output/nodes_trial.csv
    kg_build/output/nodes_alzpediaentity.csv

These node CSVs are shaped according to kg_build.schema.NODE_SCHEMAS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from .paths import ONTOLOGY_PROCESSED_DIR, ALZFORUM_PROCESSED_DIR, KG_OUTPUT_DIR
from .schema import get_node_schema


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _ensure_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Ensure the DataFrame has *exactly* the columns defined in NODE_SCHEMAS[label].

    Missing columns are added with NA; extra columns are dropped.
    """
    schema = get_node_schema(label)
    if schema is None:
        raise ValueError(f"Unknown node label in schema: {label}")

    cols = schema.all_props
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA

    # keep only schema-defined columns, in order
    out = out[cols]
    return out


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise raise FileNotFoundError with a nice message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str)


def _slugify(text: str) -> str:
    """Make a filesystem/ID-friendly slug."""
    text = (text or "").strip()
    text = re.sub(r"[^\w]+", "_", text)
    return text.strip("_")


# ---------------------------------------------------------------------
# Builders for each node type
# ---------------------------------------------------------------------


def build_disease_nodes() -> pd.DataFrame:
    """
    Build Disease nodes from MONDO.

    Input: ontology/processed/diseases_mondo.csv
        id, label, iri, synonyms, source
    """
    path = ONTOLOGY_PROCESSED_DIR / "diseases_mondo.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        oid = row["id"]
        rows.append(
            {
                "id": oid,                    # canonical node id
                "label": row["label"],
                "iri": row.get("iri"),
                "mondo_id": oid,
                "umls_cui": pd.NA,
                "mesh_id": pd.NA,
                "icd10": pd.NA,
                "synonyms": row.get("synonyms"),
                "category": pd.NA,
                "source": row.get("source", "MONDO"),
                "raw_source_ids": oid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Disease")


def build_gene_nodes() -> pd.DataFrame:
    """
    Build Gene nodes from HGNC.

    Input: ontology/processed/genes_hgnc.csv
        symbol, name, status, hgnc_id, entrez_id, ensembl_gene_id,
        alias_symbol, prev_symbol
    """
    path = ONTOLOGY_PROCESSED_DIR / "genes_hgnc.csv"
    df = _safe_read_csv(path)

    def join_synonyms(alias: str | None, prev: str | None) -> str | None:
        parts = []
        if isinstance(alias, str) and alias.strip():
            parts.append(alias.strip())
        if isinstance(prev, str) and prev.strip():
            parts.append(prev.strip())
        if not parts:
            return None
        # Keep original pipes inside alias/prev; we just join blocks with "|"
        return "|".join(parts)

    rows = []
    for _, row in df.iterrows():
        hgnc_id = row["hgnc_id"]
        syn = join_synonyms(row.get("alias_symbol"), row.get("prev_symbol"))

        rows.append(
            {
                "id": hgnc_id,
                "label": row["symbol"],
                "iri": pd.NA,   # we don't have an HGNC IRI here
                "hgnc_id": hgnc_id,
                "entrez_id": row.get("entrez_id"),
                "ensembl_id": row.get("ensembl_gene_id"),
                "chromosome": pd.NA,
                "synonyms": syn,
                "source": "HGNC",
                "raw_source_ids": hgnc_id,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Gene")


def build_protein_nodes() -> pd.DataFrame:
    """
    Build Protein nodes from PRO.

    Input: ontology/processed/proteins_pro.csv
        id, label, iri, synonyms, source, gene_symbol
    """
    path = ONTOLOGY_PROCESSED_DIR / "proteins_pro.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        oid = row["id"]
        rows.append(
            {
                "id": oid,
                "label": row["label"],
                "iri": row.get("iri"),
                "uniprot_id": pd.NA,
                "hgnc_id": pd.NA,
                "gene_symbol": row.get("gene_symbol"),
                "synonyms": row.get("synonyms"),
                "source": row.get("source", "PRO"),
                "raw_source_ids": oid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Protein")


def build_pathway_nodes() -> pd.DataFrame:
    """
    Build Pathway nodes from GO.

    Input: ontology/processed/pathways_go.csv
        id, label, iri, source
    """
    path = ONTOLOGY_PROCESSED_DIR / "pathways_go.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        oid = row["id"]
        rows.append(
            {
                "id": oid,
                "label": row["label"],
                "iri": row.get("iri"),
                "go_id": oid,
                "namespace": pd.NA,
                "synonyms": pd.NA,
                "source": row.get("source", "GO"),
                "raw_source_ids": oid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Pathway")


def build_phenotype_nodes() -> pd.DataFrame:
    """
    Build Phenotype nodes from HPO.

    Input: ontology/processed/phenotypes_hpo.csv
        id, label, iri, synonyms, source
    """
    path = ONTOLOGY_PROCESSED_DIR / "phenotypes_hpo.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        oid = row["id"]
        rows.append(
            {
                "id": oid,
                "label": row["label"],
                "iri": row.get("iri"),
                "hpo_id": oid,
                "umls_cui": pd.NA,
                "mesh_id": pd.NA,
                "synonyms": row.get("synonyms"),
                "source": row.get("source", "HPO"),
                "raw_source_ids": oid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Phenotype")


def build_biomarker_nodes() -> pd.DataFrame:
    """
    Build Biomarker nodes from AlzBiomarker.

    Input: alzforum/processed/alzbiomarker_biomarkers.csv
        biomarker_key, analyte_core, analyte_class, fluid, analyte_label_example
    """
    path = ALZFORUM_PROCESSED_DIR / "alzbiomarker_biomarkers.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        bid = row["biomarker_key"]
        rows.append(
            {
                "id": bid,
                "label": row["analyte_label_example"],
                "iri": pd.NA,
                "analyte": row.get("analyte_core"),
                "analyte_class": row.get("analyte_class"),
                "fluid": row.get("fluid"),
                "units": pd.NA,
                "assay_type": pd.NA,
                "source": "AlzBiomarker",
                "raw_source_ids": bid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Biomarker")


def build_drug_nodes() -> pd.DataFrame:
    """
    Build Drug nodes from:
      - ontology/processed/drugs_chebi.csv
      - alzforum/processed/therapeutics_entities_enriched.csv (if available)

    We keep ChEBI IDs (e.g. CHEBI:53289) and AlzForum IDs (e.g. aab-003)
    in the same nodes table, distinguished by their 'source' and 'id' format.
    """
    # --- ChEBI drugs ---------------------------------------------------
    chebi_path = ONTOLOGY_PROCESSED_DIR / "drugs_chebi.csv"
    chebi_df = _safe_read_csv(chebi_path)

    chebi_rows = []
    for _, row in chebi_df.iterrows():
        oid = row["id"]     # CHEBI:XXXX
        chebi_rows.append(
            {
                "id": oid,
                "label": row["label"],
                "iri": row.get("iri"),
                "chebi_id": oid,
                "atc_code": pd.NA,
                "drug_type": pd.NA,
                "drug_class": pd.NA,
                "primary_indication": pd.NA,
                "secondary_indications": pd.NA,
                "mechanism_summary": pd.NA,
                "status_overall": pd.NA,
                "approved_regions": pd.NA,
                "source": row.get("source", "ChEBI"),
                "raw_source_ids": oid,
            }
        )

    # --- AlzForum therapeutics ----------------------------------------
    # Prefer enriched file if present
    enr_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities_enriched.csv"
    base_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities.csv"

    alz_rows = []
    if enr_path.exists():
        df = _safe_read_csv(enr_path)
        for _, row in df.iterrows():
            tid = row["therapeutic_id"]
            # choose best-available fields
            primary_ind = row.get("detail_conditions") or row.get("fda_statuses")
            alz_rows.append(
                {
                    "id": tid,
                    "label": row["name"],
                    "iri": row.get("url"),
                    "chebi_id": pd.NA,         # may be linked later in Phase 5
                    "atc_code": pd.NA,
                    "drug_type": row.get("detail_therapy_type") or row.get("therapy_types"),
                    "drug_class": row.get("detail_target_type") or row.get("target_types"),
                    "primary_indication": primary_ind,
                    "secondary_indications": row.get("detail_approved_for")
                    or row.get("approved_for"),
                    "mechanism_summary": row.get("mechanism_summary"),
                    "status_overall": row.get("status_overall"),
                    "approved_regions": pd.NA,
                    "source": "AlzForum.Therapeutics",
                    "raw_source_ids": tid,
                }
            )
    elif base_path.exists():
        df = _safe_read_csv(base_path)
        for _, row in df.iterrows():
            tid = row["therapeutic_id"]
            alz_rows.append(
                {
                    "id": tid,
                    "label": row["name"],
                    "iri": row.get("url"),
                    "chebi_id": pd.NA,
                    "atc_code": pd.NA,
                    "drug_type": row.get("therapy_types"),
                    "drug_class": row.get("target_types"),
                    "primary_indication": row.get("fda_statuses"),
                    "secondary_indications": row.get("approved_for"),
                    "mechanism_summary": pd.NA,
                    "status_overall": pd.NA,
                    "approved_regions": pd.NA,
                    "source": "AlzForum.Therapeutics",
                    "raw_source_ids": tid,
                }
            )
    else:
        print("[WARN] No therapeutics_entities(enriched).csv found; Drug nodes will only contain ChEBI entries.")

    all_rows = chebi_rows + alz_rows
    out = pd.DataFrame(all_rows)
    return _ensure_columns(out, "Drug")


# ---------------------------------------------------------------------
# NEW node builders: Mechanism, Company, TherapyType, Fluid, Trial, AlzPediaEntity
# ---------------------------------------------------------------------


def build_mechanism_nodes() -> pd.DataFrame:
    """
    Build Mechanism nodes from:
      - alzbiomarker_biomarkers.analyte_class
      - therapeutics_entities[_enriched].target_types / detail_target_type
    """
    mech_map: Dict[str, Dict[str, str]] = {}

    # From biomarker analyte_class
    biom_path = ALZFORUM_PROCESSED_DIR / "alzbiomarker_biomarkers.csv"
    if biom_path.exists():
        bdf = _safe_read_csv(biom_path)
        for _, row in bdf.iterrows():
            cls = row.get("analyte_class")
            if not isinstance(cls, str) or not cls.strip():
                continue
            label = cls.strip()
            mid = "MECH_" + _slugify(label).upper()
            mech_map.setdefault(mid, {
                "id": mid,
                "label": label,
                "category": label.lower(),
                "description": pd.NA,
                "source": "AlzBiomarker",
                "raw_source_ids": label,
            })

    # From therapeutics target types
    enr_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities_enriched.csv"
    base_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities.csv"

    def add_from_target_types(df: pd.DataFrame, col: str, source: str) -> None:
        for _, row in df.iterrows():
            raw = row.get(col)
            if not isinstance(raw, str) or not raw.strip():
                continue
            # split on commas or pipes
            parts = re.split(r"[|,]", raw)
            for p in parts:
                label = p.strip()
                if not label:
                    continue
                mid = "MECH_" + _slugify(label).upper()
                entry = mech_map.get(mid)
                if entry is None:
                    mech_map[mid] = {
                        "id": mid,
                        "label": label,
                        "category": label.lower(),
                        "description": pd.NA,
                        "source": source,
                        "raw_source_ids": label,
                    }
                else:
                    # merge sources
                    existing_src = entry.get("source") or ""
                    if source not in existing_src:
                        entry["source"] = "|".join(
                            [s for s in [existing_src, source] if s]
                        )

    if enr_path.exists():
        df = _safe_read_csv(enr_path)
        add_from_target_types(df, "detail_target_type", "AlzForum.Therapeutics")
        add_from_target_types(df, "target_types", "AlzForum.Therapeutics")
    elif base_path.exists():
        df = _safe_read_csv(base_path)
        add_from_target_types(df, "target_types", "AlzForum.Therapeutics")

    if not mech_map:
        raise FileNotFoundError("No mechanism sources (alzbiomarker_biomarkers / therapeutics_entities) found")

    out = pd.DataFrame(list(mech_map.values()))
    return _ensure_columns(out, "Mechanism")


def build_company_nodes() -> pd.DataFrame:
    """
    Build Company nodes from AlzForum therapeutics entities.
    """
    enr_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities_enriched.csv"
    base_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities.csv"

    company_map: Dict[str, Dict[str, str]] = {}

    def add_companies(df: pd.DataFrame, col: str) -> None:
        for _, row in df.iterrows():
            tid = row.get("therapeutic_id")
            raw = row.get(col)
            if not isinstance(raw, str) or not raw.strip():
                continue
            for part in raw.split(","):
                name = part.strip()
                if not name:
                    continue
                cid = "COMP_" + _slugify(name).upper()
                entry = company_map.get(cid)
                if entry is None:
                    company_map[cid] = {
                        "id": cid,
                        "label": name,
                        "country": pd.NA,
                        "source": "AlzForum.Therapeutics",
                        "raw_source_ids": tid if tid else name,
                    }
                else:
                    # append therapeutic ids
                    existing_raw = entry.get("raw_source_ids") or ""
                    if tid and tid not in existing_raw.split("|"):
                        entry["raw_source_ids"] = "|".join(
                            [s for s in [existing_raw, tid] if s]
                        )

    if enr_path.exists():
        df = _safe_read_csv(enr_path)
        add_companies(df, "detail_company")
        add_companies(df, "companies")
    elif base_path.exists():
        df = _safe_read_csv(base_path)
        add_companies(df, "companies")
    else:
        raise FileNotFoundError("No therapeutics_entities CSVs found for Company nodes")

    out = pd.DataFrame(list(company_map.values()))
    return _ensure_columns(out, "Company")


def build_therapytype_nodes() -> pd.DataFrame:
    """
    Build TherapyType nodes from AlzForum therapeutics entities.
    """
    enr_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities_enriched.csv"
    base_path = ALZFORUM_PROCESSED_DIR / "therapeutics_entities.csv"

    tt_map: Dict[str, Dict[str, str]] = {}

    def add_types(df: pd.DataFrame, col: str) -> None:
        for _, row in df.iterrows():
            raw = row.get(col)
            if not isinstance(raw, str) or not raw.strip():
                continue
            parts = re.split(r"[|,]", raw)
            for p in parts:
                label = p.strip()
                if not label:
                    continue
                tid = "TT_" + _slugify(label).upper()
                if tid not in tt_map:
                    tt_map[tid] = {
                        "id": tid,
                        "label": label,
                        "category": pd.NA,
                        "source": "AlzForum.Therapeutics",
                        "raw_source_ids": label,
                    }

    if enr_path.exists():
        df = _safe_read_csv(enr_path)
        add_types(df, "detail_therapy_type")
        add_types(df, "therapy_types")
    elif base_path.exists():
        df = _safe_read_csv(base_path)
        add_types(df, "therapy_types")
    else:
        raise FileNotFoundError("No therapeutics_entities CSVs found for TherapyType nodes")

    out = pd.DataFrame(list(tt_map.values()))
    return _ensure_columns(out, "TherapyType")


def build_fluid_nodes() -> pd.DataFrame:
    """
    Build Fluid nodes from alzbiomarker_biomarkers.fluid.
    """
    biom_path = ALZFORUM_PROCESSED_DIR / "alzbiomarker_biomarkers.csv"
    df = _safe_read_csv(biom_path)

    fluids = set()
    for _, row in df.iterrows():
        f = row.get("fluid")
        if isinstance(f, str) and f.strip():
            fluids.add(f.strip())

    rows = []
    for f in sorted(fluids):
        fid = "FLUID_" + _slugify(f).upper()
        rows.append(
            {
                "id": fid,
                "label": f,
                "category": pd.NA,
                "source": "AlzBiomarker",
                "raw_source_ids": f,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Fluid")


def build_trial_nodes() -> pd.DataFrame:
    """
    Build Trial nodes from therapeutics_trials.csv.
    """
    trials_path = ALZFORUM_PROCESSED_DIR / "therapeutics_trials.csv"
    df = _safe_read_csv(trials_path)

    rows = []
    for idx, row in df.iterrows():
        tid = row["therapeutic_id"]
        indication = row.get("indication") or ""
        node_id = f"TRIAL_{tid}_{idx}"
        label = f"{tid} trial for {indication}" if indication else f"{tid} trial"

        rows.append(
            {
                "id": node_id,
                "label": label,
                "indication": indication,
                "trial_phase_max": row.get("trial_phase_max"),
                "has_phase3": row.get("has_phase3"),
                "status": row.get("status"),
                "trial_count": row.get("trial_count"),
                "notes": row.get("notes"),
                "source": "AlzForum.Therapeutics",
                "raw_source_ids": tid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "Trial")


def build_alzpediaentity_nodes() -> pd.DataFrame:
    """
    Build AlzPediaEntity nodes from alzpedia_entities.csv.
    """
    path = ALZFORUM_PROCESSED_DIR / "alzpedia_entities.csv"
    df = _safe_read_csv(path)

    rows = []
    for _, row in df.iterrows():
        eid = row["entity_id"]
        rows.append(
            {
                "id": f"ALZPEDIA:{eid}",
                "label": row["name"],
                "url": row.get("url"),
                "synonyms": row.get("synonyms"),
                "short_summary": row.get("short_summary"),
                "category": row.get("category"),
                "has_function_section": row.get("has_function_section"),
                "has_pathology_section": row.get("has_pathology_section"),
                "has_genetics_section": row.get("has_genetics_section"),
                "has_therapeutics_section": row.get("has_therapeutics_section"),
                "source": "AlzPedia",
                "raw_source_ids": eid,
            }
        )

    out = pd.DataFrame(rows)
    return _ensure_columns(out, "AlzPediaEntity")


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def main() -> None:
    KG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    builders: Dict[str, callable] = {
        "Disease": build_disease_nodes,
        "Gene": build_gene_nodes,
        "Protein": build_protein_nodes,
        "Pathway": build_pathway_nodes,
        "Phenotype": build_phenotype_nodes,
        "Biomarker": build_biomarker_nodes,
        "Drug": build_drug_nodes,
        # new nodes
        "Mechanism": build_mechanism_nodes,
        "Company": build_company_nodes,
        "TherapyType": build_therapytype_nodes,
        "Fluid": build_fluid_nodes,
        "Trial": build_trial_nodes,
        "AlzPediaEntity": build_alzpediaentity_nodes,
        # "RiskFactor": build_riskfactor_nodes,   # TODO once AlzRisk is ready
        # "Study": build_study_nodes,             # optional
    }

    for label, fn in builders.items():
        try:
            df = fn()
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {label}: {e}")
            continue

        out_path = KG_OUTPUT_DIR / f"nodes_{label.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Wrote {label} nodes -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()