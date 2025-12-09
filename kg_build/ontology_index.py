"""
kg_build.ontology_index
-----------------------

Helpers to load and index ontology tables for fast string -> ID lookup.

This module is *read-only* on top of:
    ontology/processed/diseases_mondo.csv
    ontology/processed/drugs_chebi.csv
    ontology/processed/genes_hgnc.csv
    ontology/processed/pathways_go.csv
    ontology/processed/phenotypes_hpo.csv
    ontology/processed/proteins_pro.csv

It builds small in-memory indices that map normalized text terms
(labels, synonyms, symbols) to canonical ontology IDs.

Usage (downstream in Phase 4):

    from kg_build.ontology_index import load_disease_index

    disease_idx = load_disease_index()
    mondo_ids = disease_idx.lookup("Alzheimer's disease")

We intentionally keep this module simple and deterministic:
no fuzzy matching, no external services – just exact-ish lookups
on normalized strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .paths import ONTOLOGY_PROCESSED_DIR


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------


def _norm(s: str) -> str:
    """
    Normalize a free-text term for use as a dictionary key.

    - strip leading/trailing whitespace
    - collapse internal whitespace
    - lowercase

    We can make this more aggressive later (unicode normalization,
    punctuation stripping, Aβ vs Abeta, etc.) without changing callers.
    """
    if s is None:
        return ""
    s = str(s)
    s = " ".join(s.split())
    return s.lower()


def _split_synonyms(val: Optional[str], delim: str = "|") -> List[str]:
    """
    Split a synonyms field like "foo|bar|baz" into ["foo", "bar", "baz"].
    Returns [] if val is None/empty.
    """
    if not val or not isinstance(val, str):
        return []
    parts = [p.strip() for p in val.split(delim)]
    return [p for p in parts if p]


# ---------------------------------------------------------------------
# Core index dataclass
# ---------------------------------------------------------------------


@dataclass
class SimpleOntologyIndex:
    """
    Lightweight index for one ontology table.

    Attributes
    ----------
    name:
        Short name (e.g. "MONDO", "HGNC", "ChEBI").
    df:
        Underlying pandas DataFrame with the raw table.
    id_col:
        Column in df used as canonical ID.
    term_to_ids:
        Mapping from normalized text term -> list of ontology IDs.
        If a term maps to multiple IDs, all are returned;
        the caller decides how to handle ambiguity.
    """

    name: str
    df: pd.DataFrame
    id_col: str
    term_to_ids: Dict[str, List[str]]

    def lookup(self, term: str) -> List[str]:
        """
        Return a list of matching IDs for a given text term.
        Empty list means no match.
        """
        key = _norm(term)
        if not key:
            return []
        return self.term_to_ids.get(key, [])

    def has(self, term: str) -> bool:
        """Convenience: True iff term has at least one matching ID."""
        return bool(self.lookup(term))


def _build_index_from_df(
    *,
    name: str,
    df: pd.DataFrame,
    id_col: str,
    label_cols: Iterable[str],
    synonym_cols: Iterable[str] = (),
    extra_term_cols: Iterable[str] = (),
    synonym_delim: str = "|",
) -> SimpleOntologyIndex:
    """
    Generic helper to build a SimpleOntologyIndex from a DataFrame.

    Parameters
    ----------
    name:
        Short ontology name for debugging.
    df:
        DataFrame with ontology content.
    id_col:
        Column containing canonical IDs (e.g. "id" or "hgnc_id").
    label_cols:
        Columns whose values we treat as canonical labels.
    synonym_cols:
        Columns containing delimited synonym strings.
    extra_term_cols:
        Columns with additional term-like values (e.g. gene_symbol, symbol).
    synonym_delim:
        Delimiter for splitting synonym strings (default: "|").
    """
    if id_col not in df.columns:
        raise ValueError(f"[{name}] id_col '{id_col}' not found in DataFrame columns.")

    # Filter out rows with missing IDs early
    df = df[df[id_col].notna()].copy()

    term_to_ids: Dict[str, List[str]] = {}

    def add_term(term: str, oid: str) -> None:
        key = _norm(term)
        if not key:
            return
        bucket = term_to_ids.setdefault(key, [])
        if oid not in bucket:
            bucket.append(oid)

    for _, row in df.iterrows():
        oid = str(row[id_col])

        # labels
        for col in label_cols:
            if col in df.columns and pd.notna(row[col]):
                add_term(str(row[col]), oid)

        # synonyms (pipe-delimited, usually)
        for col in synonym_cols:
            if col in df.columns and pd.notna(row[col]):
                for syn in _split_synonyms(str(row[col]), delim=synonym_delim):
                    add_term(syn, oid)

        # extra term-like columns (e.g. gene_symbol, symbol)
        for col in extra_term_cols:
            if col in df.columns and pd.notna(row[col]):
                add_term(str(row[col]), oid)

    return SimpleOntologyIndex(
        name=name,
        df=df,
        id_col=id_col,
        term_to_ids=term_to_ids,
    )


# ---------------------------------------------------------------------
# Loaders for each ontology table (with caching)
# ---------------------------------------------------------------------

_CACHE: Dict[str, SimpleOntologyIndex] = {}


def load_disease_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load MONDO disease table and build a text -> MONDO:ID index.

    Uses: ontology/processed/diseases_mondo.csv
    Columns (from your sample):
        id, label, iri, synonyms, source
    """
    cache_key = "disease_mondo"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "diseases_mondo.csv"
    df = pd.read_csv(path)

    idx = _build_index_from_df(
        name="MONDO",
        df=df,
        id_col="id",
        label_cols=["label"],
        synonym_cols=["synonyms"],
        extra_term_cols=[],  # nothing special beyond synonyms
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


def load_drug_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load ChEBI drug table and build text -> CHEBI:ID index.

    Uses: ontology/processed/drugs_chebi.csv
    Columns (from your sample):
        id, label, iri, synonyms, source
    """
    cache_key = "drug_chebi"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "drugs_chebi.csv"
    df = pd.read_csv(path)

    idx = _build_index_from_df(
        name="ChEBI",
        df=df,
        id_col="id",
        label_cols=["label"],
        synonym_cols=["synonyms"],
        extra_term_cols=[],
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


def load_gene_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load HGNC gene table and build text -> HGNC:ID index.

    Uses: ontology/processed/genes_hgnc.csv
    Columns (from your sample):
        symbol, name, status, hgnc_id, entrez_id,
        ensembl_gene_id, alias_symbol, prev_symbol
    """
    cache_key = "gene_hgnc"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "genes_hgnc.csv"
    df = pd.read_csv(path, dtype=str)

    # Use hgnc_id as canonical ID, but include symbol + name + alias/prev as terms
    idx = _build_index_from_df(
        name="HGNC",
        df=df,
        id_col="hgnc_id",
        label_cols=["symbol", "name"],
        synonym_cols=["alias_symbol", "prev_symbol"],
        extra_term_cols=[],  # could add more if needed
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


def load_protein_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load PRO protein table and build text -> PR:ID index.

    Uses: ontology/processed/proteins_pro.csv
    Columns (from your sample):
        id, label, iri, synonyms, source, gene_symbol
    """
    cache_key = "protein_pro"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "proteins_pro.csv"
    df = pd.read_csv(path, dtype=str)

    idx = _build_index_from_df(
        name="PRO",
        df=df,
        id_col="id",
        label_cols=["label"],
        synonym_cols=["synonyms"],
        extra_term_cols=["gene_symbol"],  # handy: APP, MAPT, etc.
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


def load_pathway_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load GO pathways table and build text -> GO:ID index.

    Uses: ontology/processed/pathways_go.csv
    Columns (from your sample):
        id, label, iri, source
    (no synonyms column in your sample)
    """
    cache_key = "pathway_go"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "pathways_go.csv"
    df = pd.read_csv(path, dtype=str)

    idx = _build_index_from_df(
        name="GO",
        df=df,
        id_col="id",
        label_cols=["label"],
        synonym_cols=[],      # none in sample
        extra_term_cols=[],   # could add later if we add synonyms
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


def load_phenotype_index(reload: bool = False) -> SimpleOntologyIndex:
    """
    Load HPO phenotypes table and build text -> HP:ID index.

    Uses: ontology/processed/phenotypes_hpo.csv
    Columns (from your sample):
        id, label, iri, synonyms, source
    """
    cache_key = "phenotype_hpo"
    if not reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    path = ONTOLOGY_PROCESSED_DIR / "phenotypes_hpo.csv"
    df = pd.read_csv(path, dtype=str)

    idx = _build_index_from_df(
        name="HPO",
        df=df,
        id_col="id",
        label_cols=["label"],
        synonym_cols=["synonyms"],
        extra_term_cols=[],
        synonym_delim="|",
    )
    _CACHE[cache_key] = idx
    return idx


# ---------------------------------------------------------------------
# Debug / sanity check
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Quick sanity check printout when run as a script
    loaders = [
        ("Disease (MONDO)", load_disease_index),
        ("Drug (ChEBI)", load_drug_index),
        ("Gene (HGNC)", load_gene_index),
        ("Protein (PRO)", load_protein_index),
        ("Pathway (GO)", load_pathway_index),
        ("Phenotype (HPO)", load_phenotype_index),
    ]

    for name, fn in loaders:
        try:
            idx = fn()
        except FileNotFoundError as e:
            print(f"[WARN] {name}: {e}")
            continue

        print(f"{name}: {len(idx.df)} rows, {len(idx.term_to_ids)} unique terms")
        # Show a couple of example terms if available
        sample_terms = list(idx.term_to_ids.keys())[:5]
        for t in sample_terms:
            print(f"  '{t}' -> {idx.term_to_ids[t]}")
        print()