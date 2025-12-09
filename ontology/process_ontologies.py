# ontology/process_ontologies.py

"""
Process downloaded ontologies into small, Alzheimer-focused subsets.

Input:
    ontology/raw/
        mondo.owl
        hp.owl
        go-basic.owl
        pr.owl
        chebi.owl
        hgnc_complete_set.txt

Output:
    ontology/processed/
        diseases_mondo.csv
        phenotypes_hpo.csv
        pathways_go.csv
        proteins_pro.csv
        drugs_chebi.csv
        genes_hgnc.csv

You can later use these CSVs to populate your Knowledge Graph.
"""

import os
from pathlib import Path
from typing import List, Dict, Iterable, Optional

import pandas as pd
from owlready2 import get_ontology


# ------------------------
# PATHS & CONSTANTS
# ------------------------

ONTOLOGY_ROOT = Path(__file__).resolve().parent
RAW_DIR = ONTOLOGY_ROOT / "raw"
PROCESSED_DIR = ONTOLOGY_ROOT / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Filenames that download_ontologies.py should have created
MONDO_FILE = RAW_DIR / "mondo.owl"
HPO_FILE = RAW_DIR / "hp.owl"
GO_FILE = RAW_DIR / "go-basic.owl"
PRO_FILE = RAW_DIR / "pr.owl"
CHEBI_FILE = RAW_DIR / "chebi.owl"
HGNC_FILE = RAW_DIR / "hgnc_complete_set.txt"

# ------------------------
# CONFIG: TARGET LABELS
# ------------------------

# Disease(s) of interest (MONDO labels)
MONDO_DISEASE_LABELS = {
    # main AD label in MONDO (commonly "Alzheimer disease")
    "alzheimer disease",
    "alzheimer's disease",
    "alzheimer disease, familial",
    "early onset alzheimer disease",
    "late onset alzheimer disease",
}

# Phenotypes (HPO labels)
HPO_PHENOTYPE_LABELS = {
    "memory impairment",
    "cognitive impairment",
    "cognitive decline",
    "aphasia",
    "behavioral abnormality",
    "disorientation",
    "executive function impairment",
}

# GO biological processes / pathways
GO_PATHWAY_LABELS = {
    "amyloid-beta metabolic process",
    "amyloid-beta formation",
    "tau protein phosphorylation",
    "neuroinflammatory response",
    "microglial cell activation",
    "synaptic plasticity",
    "regulation of synaptic plasticity",
    "synaptic signaling",
}

# PRO proteins of interest (labels / name fragments)
PRO_PROTEIN_LABEL_FRAGMENTS = {
    "amyloid beta a4 protein",   # APP
    "amyloid beta",              # generic amyloid-beta
    "microtubule-associated protein tau",  # MAPT
    "presenilin-1",
    "presenilin 1",
    "presenilin-2",
    "presenilin 2",
    "apolipoprotein e",
    "triggering receptor expressed on myeloid cells 2",
}

# HGNC genes of interest (symbols)
HGNC_GENE_SYMBOLS = ["APP", "MAPT", "PSEN1", "PSEN2", "APOE", "TREM2"]

# ChEBI drugs of interest (labels)
CHEBI_DRUG_LABELS = {
    "donepezil",
    "memantine",
    "rivastigmine",
    "galantamine",
}


# ------------------------
# HELPER FUNCTIONS
# ------------------------

def class_label(c) -> Optional[str]:
    """Return a single rdfs:label for an OWL class (lowercased), if available."""
    try:
        if c.label:
            return str(c.label[0]).strip()
    except Exception:
        pass
    return None


def class_synonyms(c) -> List[str]:
    """Attempt to gather synonyms from common annotation properties if present."""
    syns = []
    for attr in ["hasExactSynonym", "has_related_synonym", "hasBroadSynonym", "hasNarrowSynonym", "altLabel"]:
        if hasattr(c, attr):
            try:
                vals = getattr(c, attr)
                syns.extend([str(v).strip() for v in vals])
            except Exception:
                continue
    # remove duplicates
    return sorted(set(syns))


def class_curie(c) -> str:
    """
    Extract a CURIE-like ID from the IRI.
    Example:
        http://purl.obolibrary.org/obo/MONDO_0004975 -> MONDO:0004975
    """
    iri = c.iri
    # IRI often ends with 'PREFIX_########'
    last = iri.split("/")[-1]
    if "_" in last:
        prefix, local = last.split("_", 1)
        return f"{prefix}:{local}"
    return last


def filter_classes_by_label(
    onto,
    allowed_labels_lower: Iterable[str],
    allow_fragment_match: bool = False,
) -> List:
    """
    Filter ontology classes by rdfs:label against a set of allowed labels.

    If allow_fragment_match=True, we treat allowed_labels_lower as substrings to
    search within the class label (useful for proteins or complex labels).
    """
    allowed_labels_lower = {l.lower() for l in allowed_labels_lower}
    matched = []

    for c in onto.classes():
        lbl = class_label(c)
        if not lbl:
            continue
        lbl_lower = lbl.lower()

        if allow_fragment_match:
            if any(fragment in lbl_lower for fragment in allowed_labels_lower):
                matched.append(c)
        else:
            if lbl_lower in allowed_labels_lower:
                matched.append(c)

    return matched


# ------------------------
# MONDO (Disease)
# ------------------------

def process_mondo() -> pd.DataFrame:
    print(f"[MONDO] Loading ontology from {MONDO_FILE}")
    onto = get_ontology(MONDO_FILE.as_uri()).load()

    print("[MONDO] Filtering disease classes of interest…")
    classes = filter_classes_by_label(onto, MONDO_DISEASE_LABELS, allow_fragment_match=False)

    print(f"[MONDO] Found {len(classes)} classes matching target labels.")

    rows = []
    for c in classes:
        lbl = class_label(c) or ""
        curie = class_curie(c)
        syns = class_synonyms(c)
        rows.append(
            {
                "id": curie,
                "label": lbl,
                "iri": c.iri,
                "synonyms": "|".join(syns),
                "source": "MONDO",
            }
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "diseases_mondo.csv"
    df.to_csv(out_path, index=False)
    print(f"[MONDO] Saved {len(df)} rows to {out_path}")
    return df


# ------------------------
# HPO (Phenotypes)
# ------------------------

def process_hpo() -> pd.DataFrame:
    print(f"[HPO] Loading ontology from {HPO_FILE}")
    onto = get_ontology(HPO_FILE.as_uri()).load()

    print("[HPO] Filtering phenotypes of interest…")
    classes = filter_classes_by_label(onto, HPO_PHENOTYPE_LABELS, allow_fragment_match=False)

    print(f"[HPO] Found {len(classes)} classes matching target labels.")

    rows = []
    for c in classes:
        lbl = class_label(c) or ""
        curie = class_curie(c)
        syns = class_synonyms(c)
        rows.append(
            {
                "id": curie,
                "label": lbl,
                "iri": c.iri,
                "synonyms": "|".join(syns),
                "source": "HPO",
            }
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "phenotypes_hpo.csv"
    df.to_csv(out_path, index=False)
    print(f"[HPO] Saved {len(df)} rows to {out_path}")
    return df


# ------------------------
# GO (Pathways / Biological Processes)
# ------------------------

def process_go() -> pd.DataFrame:
    print(f"[GO] Loading ontology from {GO_FILE}")
    onto = get_ontology(GO_FILE.as_uri()).load()

    print("[GO] Filtering biological processes / pathways of interest…")
    classes = filter_classes_by_label(onto, GO_PATHWAY_LABELS, allow_fragment_match=False)

    print(f"[GO] Found {len(classes)} classes matching target labels.")

    rows = []
    for c in classes:
        lbl = class_label(c) or ""
        curie = class_curie(c)
        rows.append(
            {
                "id": curie,
                "label": lbl,
                "iri": c.iri,
                "source": "GO",
            }
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "pathways_go.csv"
    df.to_csv(out_path, index=False)
    print(f"[GO] Saved {len(df)} rows to {out_path}")
    return df


# ------------------------
# PRO (Proteins)
# ------------------------

def process_pro() -> pd.DataFrame:
    print(f"[PRO] Loading ontology from {PRO_FILE}")
    onto = get_ontology(PRO_FILE.as_uri()).load()

    print("[PRO] Filtering protein classes of interest…")
    classes = filter_classes_by_label(
        onto,
        PRO_PROTEIN_LABEL_FRAGMENTS,
        allow_fragment_match=True,  # fragment match because labels can be long
    )

    print(f"[PRO] Found {len(classes)} classes matching target label fragments.")

    rows = []
    for c in classes:
        lbl = class_label(c) or ""
        curie = class_curie(c)
        syns = class_synonyms(c)
        rows.append(
            {
                "id": curie,
                "label": lbl,
                "iri": c.iri,
                "synonyms": "|".join(syns),
                "source": "PRO",
                # gene_symbol will be filled by HGNC mapping later (if possible)
                "gene_symbol": "",
            }
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "proteins_pro.csv"
    df.to_csv(out_path, index=False)
    print(f"[PRO] Saved {len(df)} rows to {out_path}")
    return df


# ------------------------
# ChEBI (Drugs)
# ------------------------

def process_chebi() -> pd.DataFrame:
    print(f"[ChEBI] Loading ontology from {CHEBI_FILE}")
    onto = get_ontology(CHEBI_FILE.as_uri()).load()

    print("[ChEBI] Filtering drug classes of interest…")
    classes = filter_classes_by_label(
        onto,
        CHEBI_DRUG_LABELS,
        allow_fragment_match=False,
    )

    print(f"[ChEBI] Found {len(classes)} classes matching target labels.")

    rows = []
    for c in classes:
        lbl = class_label(c) or ""
        curie = class_curie(c)
        syns = class_synonyms(c)
        rows.append(
            {
                "id": curie,
                "label": lbl,
                "iri": c.iri,
                "synonyms": "|".join(syns),
                "source": "ChEBI",
            }
        )

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "drugs_chebi.csv"
    df.to_csv(out_path, index=False)
    print(f"[ChEBI] Saved {len(df)} rows to {out_path}")
    return df


# ------------------------
# HGNC Genes
# ------------------------

def process_hgnc() -> pd.DataFrame:
    if not HGNC_FILE.exists():
        raise FileNotFoundError(
            f"HGNC gene table not found at {HGNC_FILE}. "
            "Please run download_ontologies.py or download it manually."
        )

    print(f"[HGNC] Loading gene table from {HGNC_FILE}")
    df = pd.read_csv(HGNC_FILE, sep="\t", dtype=str)

    print("[HGNC] Filtering Alzheimer-related genes of interest…")
    df_sub = df[df["symbol"].isin(HGNC_GENE_SYMBOLS)].copy()

    # Keep only some useful columns
    keep_cols = [
        "symbol",
        "name",
        "status",
        "hgnc_id",
        "entrez_id",
        "ensembl_gene_id",
        "alias_symbol",
        "prev_symbol",
    ]
    for col in keep_cols:
        if col not in df_sub.columns:
            df_sub[col] = ""

    df_final = df_sub[keep_cols].reset_index(drop=True)

    out_path = PROCESSED_DIR / "genes_hgnc.csv"
    df_final.to_csv(out_path, index=False)
    print(f"[HGNC] Saved {len(df_final)} rows to {out_path}")
    return df_final


# ------------------------
# COMBINING: PRO Proteins ↔ HGNC Genes (light integration)
# ------------------------

def integrate_proteins_with_genes(
    proteins_df: pd.DataFrame,
    genes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attempt to add gene_symbol info to proteins_pro.csv by matching labels/synonyms
    with HGNC gene symbols.

    For now, we do a simple manual mapping using known biology:
      APP   -> amyloid beta A4 protein
      MAPT  -> microtubule-associated protein tau
      PSEN1 -> presenilin-1
      PSEN2 -> presenilin-2
      APOE  -> apolipoprotein E
      TREM2 -> triggering receptor expressed on myeloid cells 2
    """
    label_to_gene_symbol = {
        "amyloid beta a4 protein": "APP",
        "amyloid beta protein": "APP",
        "microtubule-associated protein tau": "MAPT",
        "presenilin-1": "PSEN1",
        "presenilin 1": "PSEN1",
        "presenilin-2": "PSEN2",
        "presenilin 2": "PSEN2",
        "apolipoprotein e": "APOE",
        "triggering receptor expressed on myeloid cells 2": "TREM2",
    }

    proteins_df = proteins_df.copy()
    proteins_df["gene_symbol"] = ""

    for i, row in proteins_df.iterrows():
        lbl = (row.get("label") or "").lower()
        syns = (row.get("synonyms") or "").lower().split("|") if row.get("synonyms") else []

        gene_symbol = ""
        # Try label direct match
        if lbl in label_to_gene_symbol:
            gene_symbol = label_to_gene_symbol[lbl]
        else:
            # Try synonym-based match
            for syn in syns:
                syn_clean = syn.strip().lower()
                if syn_clean in label_to_gene_symbol:
                    gene_symbol = label_to_gene_symbol[syn_clean]
                    break

        proteins_df.at[i, "gene_symbol"] = gene_symbol

    out_path = PROCESSED_DIR / "proteins_pro.csv"
    proteins_df.to_csv(out_path, index=False)
    print(f"[INTEGRATE] Updated proteins_pro.csv with gene_symbol hints at {out_path}")
    return proteins_df


# ------------------------
# MAIN
# ------------------------

def main():
    print("=== Alzheimer’s KG – Ontology Processing Script ===")
    print(f"[INFO] RAW_DIR       = {RAW_DIR}")
    print(f"[INFO] PROCESSED_DIR = {PROCESSED_DIR}\n")

    # 1) Process each ontology into a small CSV
    mondo_df = process_mondo()
    print()

    hpo_df = process_hpo()
    print()

    go_df = process_go()
    print()

    pro_df = process_pro()
    print()

    chebi_df = process_chebi()
    print()

    hgnc_df = process_hgnc()
    print()

    # 2) Light integration: add gene symbols to PRO table
    pro_df_integrated = integrate_proteins_with_genes(pro_df, hgnc_df)
    print()

    print("=== DONE ===")
    print("Generated processed ontology subsets:")
    for f in sorted(PROCESSED_DIR.glob("*.csv")):
        print(f" - {f.name}")


if __name__ == "__main__":
    main()