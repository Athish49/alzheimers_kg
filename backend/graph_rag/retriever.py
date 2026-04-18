"""
graph_rag.retriever
-------------------

Utilities for retrieving subgraphs from Neo4j and converting them into
LLM-friendly textual context for Graph RAG.

Current focus:
- Centered on Alzheimer's disease (MONDO node) as the main disease.
- Pulls a local neighborhood of:
    * Biomarkers (HAS_BIOMARKER)
    * Drugs / therapeutics (TREATS)
    * Phenotypes / symptoms (HAS_PHENOTYPE)
    * Pathways affected by AD drugs (AFFECTS_PATHWAY)
    * Genes and proteins (ENCODES + INVOLVED_IN_PATHWAY, where available)

This module does NOT talk to the LLM directly; it only prepares context
strings. LLM calls are done via `graph_rag.llm_client`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


from neo4j import GraphDatabase, Driver, Session

from .config import CONFIG
from .graph_to_text import build_ad_ultra_compact_context_from_lists

# ---------------------------------------------------------------------
# Core retriever class
# ---------------------------------------------------------------------


@dataclass
class GraphRetriever:
    """
    Small wrapper around the Neo4j Python driver for KG retrieval.

    Parameters
    ----------
    uri:
        Neo4j Bolt URI, e.g. "bolt://localhost:7687".
    user:
        Neo4j username.
    password:
        Neo4j password.
    database:
        Neo4j database name (often "neo4j").
    """

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    # ------------------------------------------------------------------
    # Basic driver plumbing
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    def _session(self) -> Session:
        """Get a read session."""
        return self._driver.session(database=self.database)

    # ------------------------------------------------------------------
    # Helpers: disease lookup
    # ------------------------------------------------------------------

    def get_disease_id_by_name(self, name: str) -> Optional[str]:
        """
        Find a Disease node whose label or synonyms contain the given name.

        Returns
        -------
        disease_id or None
        """
        q = """
        MATCH (d:Disease)
        WHERE toLower(d.label) CONTAINS toLower($name)
           OR toLower(coalesce(d.synonyms, "")) CONTAINS toLower($name)
        RETURN d.id AS id
        LIMIT 1
        """
        with self._session() as session:
            rec = session.run(q, name=name).single()
            return rec["id"] if rec else None

    def get_alzheimers_disease_id(self) -> Optional[str]:
        """
        Convenience helper: find canonical Alzheimer's Disease node.

        Uses "alzheimer" as search term; falls back to MONDO ID if present.
        """
        # 1) Try a semantic lookup
        did = self.get_disease_id_by_name("alzheimer")
        if did:
            return did

        # 2) Fallback to specific MONDO ID we expect from normalize_entities
        q = """
        MATCH (d:Disease {id: 'MONDO:0004975'})
        RETURN d.id AS id
        LIMIT 1
        """
        with self._session() as session:
            rec = session.run(q).single()
            return rec["id"] if rec else None

    # ------------------------------------------------------------------
    # Subgraph retrieval around Alzheimer's Disease
    # ------------------------------------------------------------------

    def get_ad_biomarkers(
        self, disease_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Get biomarker edges for a given disease.
        """
        q = """
        MATCH (d:Disease {id: $did})-[r:HAS_BIOMARKER]->(b:Biomarker)
        RETURN
            b.id            AS biomarker_id,
            b.label         AS biomarker_label,
            b.analyte       AS analyte,
            b.analyte_class AS analyte_class,
            b.fluid         AS fluid,
            r.direction     AS direction,
            r.comparison    AS comparison,
            r.effect_size   AS effect_size,
            r.p_value       AS p_value
        ORDER BY biomarker_label
        LIMIT $limit
        """
        with self._session() as session:
            return [dict(rec) for rec in session.run(q, did=disease_id, limit=limit)]

    def get_ad_drugs(
        self, disease_id: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Get therapeutics (Drug -> Disease via TREATS) for a given disease.
        """
        q = """
        MATCH (dr:Drug)-[r:TREATS]->(d:Disease {id: $did})
        RETURN
            dr.id           AS drug_id,
            dr.label        AS drug_label,
            dr.drug_type    AS drug_type,
            dr.drug_class   AS drug_class,
            dr.status_overall AS status_overall,
            r.status        AS trial_status,
            r.trial_phase_max AS trial_phase_max,
            r.has_phase3    AS has_phase3,
            r.trial_count   AS trial_count,
            r.indication    AS indication
        ORDER BY drug_label
        LIMIT $limit
        """
        with self._session() as session:
            return [dict(rec) for rec in session.run(q, did=disease_id, limit=limit)]

    def get_ad_phenotypes(
        self, disease_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get phenotypes / symptoms for a given disease.
        """
        q = """
        MATCH (d:Disease {id: $did})-[r:HAS_PHENOTYPE]->(p:Phenotype)
        RETURN
            p.id        AS phenotype_id,
            p.label     AS phenotype_label
        ORDER BY phenotype_label
        LIMIT $limit
        """
        with self._session() as session:
            return [dict(rec) for rec in session.run(q, did=disease_id, limit=limit)]

    def get_ad_drug_pathways(
        self, disease_id: str, limit: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Get pathways affected by drugs that treat the given disease.

        Pattern:
            (dr:Drug)-[:TREATS]->(d:Disease)
            (dr)-[r:AFFECTS_PATHWAY]->(pw:Pathway)
        """
        q = """
        MATCH (dr:Drug)-[:TREATS]->(d:Disease {id: $did})
        MATCH (dr)-[r:AFFECTS_PATHWAY]->(pw:Pathway)
        RETURN
            dr.id        AS drug_id,
            dr.label     AS drug_label,
            pw.id        AS pathway_id,
            pw.label     AS pathway_label,
            r.source      AS source,
            r.target_notes AS target_notes,
            r.action_type AS action_type,
            r.is_primary_target AS is_primary_target
        ORDER BY drug_label, pathway_label
        LIMIT $limit
        """
        with self._session() as session:
            return [dict(rec) for rec in session.run(q, did=disease_id, limit=limit)]

    def get_genes_and_proteins(
        self, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Get a sample of Gene -> Protein relationships.

        This is not currently anchored to AD specifically (Phase 4 design),
        but still useful context when the user asks about molecular biology.
        """
        q = """
        MATCH (g:Gene)-[r:ENCODES]->(p:Protein)
        RETURN
            g.id       AS gene_id,
            g.label    AS gene_symbol,
            p.id       AS protein_id,
            p.label    AS protein_label
        ORDER BY gene_symbol, protein_label
        LIMIT $limit
        """
        with self._session() as session:
            return [dict(rec) for rec in session.run(q, limit=limit)]

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def build_ad_context(self) -> str:
        """
        Build an ultra-compact textual context summarizing the Alzheimer's
        disease neighborhood in the KG.

        This now delegates to `graph_rag.graph_to_text.build_ad_ultra_compact_context`
        so all formatting / aggregation is centralized there.

        Returns
        -------
        context : str
        """
        disease_id = self.get_alzheimers_disease_id()
        if not disease_id:
            return (
                "The knowledge graph does not appear to contain an Alzheimer's "
                "Disease node (Disease label containing 'Alzheimer')."
            )

        # Fetch neighborhood pieces (raw, structured dicts)
        biomarkers = self.get_ad_biomarkers(disease_id)
        drugs = self.get_ad_drugs(disease_id)
        phenos = self.get_ad_phenotypes(disease_id)
        drug_pws = self.get_ad_drug_pathways(disease_id)
        genes_proteins = self.get_genes_and_proteins()

        # Delegate formatting to graph_to_text
        context = build_ad_ultra_compact_context_from_lists(
            disease_id=disease_id,
            biomarkers=biomarkers,
            drugs=drugs,
            phenotypes=phenos,
            drug_pathways=drug_pws,
            genes_proteins=genes_proteins,
        )
        return context

    # ------------------------------------------------------------------
    # High-level entry point for RAG
    # ------------------------------------------------------------------

    def get_context_for_question(self, question: str) -> str:
        """
        HIGH-LEVEL: given a user question, return a context string.

        Phase 5, v1:
        ------------
        We ignore the specific question here and always return the
        Alzheimer's Disease-centered ultra-compact context.
        The router + LLM handle intent and question wording.

        Later, we can:
            - parse the question to detect disease / entity mentions
            - switch to different anchors (e.g., Parkinson's)
            - use entity linking for more precise neighborhoods
        """
        return self.build_ad_context()


# ---------------------------------------------------------------------
# Singleton-style accessor
# ---------------------------------------------------------------------


_retriever: Optional[GraphRetriever] = None


def get_retriever() -> GraphRetriever:
    """
    Get (and lazily create) the project-wide GraphRetriever.

    Uses connection details from `graph_rag.config.CONFIG`.
    """
    global _retriever
    if _retriever is None:
        _retriever = GraphRetriever(
            uri=CONFIG.neo4j_uri,
            user=CONFIG.neo4j_user,
            password=CONFIG.neo4j_password,
            database=CONFIG.neo4j_db,
        )
    return _retriever