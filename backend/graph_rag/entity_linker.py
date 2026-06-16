"""
graph_rag.entity_linker
-----------------------

Builds an in-memory vocabulary from the Neo4j graph and resolves entity
mentions in a user query to canonical node IDs.

Design principles
-----------------
- Synonyms live in the DATA (graph node properties), not in code.
  Adding new synonyms = update the graph node; zero code change.
- One AuraDB query at startup; results cached for the process lifetime.
- Returns ALL matches; callers decide how to disambiguate by type.
- Word-boundary matching prevents "gene" firing on "degenerate" etc.

Supported node types and their synonym sources
----------------------------------------------
  Biomarker      — b.label, b.analyte, b.synonyms (added in Step 0)
  Drug           — d.label, d.synonyms (added in Step 0)
  Gene           — g.label, g.synonyms (pipe-separated, e.g. "tau|PPND|MAPT")
  Protein        — p.label, p.synonyms (pipe-separated, 95% coverage)
                   NOTE: only synonyms <= 35 chars are indexed to avoid
                   technical PRO IDs like "UniProtKB:P10636-8, Thr-181, MOD:00047"
                   that will never appear in natural-language queries.
  Phenotype      — ph.label, ph.synonyms (pipe-separated)
  AlzPediaEntity — ae.label, ae.synonyms (comma-separated)
  Pathway        — pw.label only (no synonyms in graph)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum synonym length for Protein nodes.
# Protein synonyms include long technical strings like:
#   "UniProtKB:P10636-8, Thr-181, MOD:00047|hMAPT/iso:Tau-F/Phos:1"
# These will never appear in user queries. We only index short, human-readable ones.
_MAX_PROTEIN_SYNONYM_LEN = 35


@dataclass(frozen=True)
class EntityMatch:
    """A single entity resolved from the query. Immutable once created."""
    node_id: str          # canonical graph node ID, e.g. "tau_p181_csf"
    node_type: str        # "Biomarker" | "Drug" | "Gene" | "Protein" | "Phenotype" | "Pathway" | "AlzPedia"
    canonical_label: str  # human-readable label as stored in the graph
    matched_term: str     # the specific synonym/term that matched in the query


@dataclass
class EntityLinker:
    """
    Vocabulary-based entity linker for the Alzheimer KG.

    Instantiate once and reuse across requests. Call build_vocab() after
    instantiation to populate the vocabulary from AuraDB.

    Parameters
    ----------
    driver : neo4j.Driver
        An open Neo4j driver pointing at AuraDB.
    database : str
        Neo4j database name (usually "neo4j").
    project : str
        Project scope tag used on all nodes.
    """

    driver: object  # neo4j.Driver — typed as object to avoid import at module level
    database: str = "neo4j"
    project: str = "alzheimerskg"

    # Internal: term (lowercase) -> list of EntityMatch candidates
    _vocab: Dict[str, List[EntityMatch]] = field(default_factory=dict, init=False, repr=False)
    # Pre-sorted term list (longest first) so match_entities() doesn't sort on every call.
    _sorted_terms: List[str] = field(default_factory=list, init=False, repr=False)
    _built: bool = field(default=False, init=False, repr=False)

    def build_vocab(self) -> None:
        """
        Query AuraDB for all node labels and synonyms, build the in-memory
        vocabulary. Call once at startup; results persist for process lifetime.

        Raises on any AuraDB failure so the caller (get_entity_linker) can
        decide whether to retry or propagate. The linker is never left in a
        partially-built state — _built stays False until all queries succeed.
        """
        t0 = time.monotonic()
        vocab: Dict[str, List[EntityMatch]] = {}

        def _add(term: str, node_id: str, node_type: str, canonical_label: str) -> None:
            """Normalise term and insert an EntityMatch, skipping blanks and duplicates."""
            key = term.strip().lower()
            if not key or len(key) < 2:
                return
            match = EntityMatch(
                node_id=node_id,
                node_type=node_type,
                canonical_label=canonical_label,
                matched_term=term.strip(),
            )
            bucket = vocab.setdefault(key, [])
            # Avoid indexing the same node twice under the same term
            if not any(m.node_id == node_id for m in bucket):
                bucket.append(match)

        with self.driver.session(database=self.database) as session:

            # --- Biomarker nodes ---
            # Index: label, analyte (the core measurable), synonyms (added in Step 0)
            result = session.run(
                """
                MATCH (b:Biomarker {project: $project})
                RETURN b.id AS id, b.label AS label,
                       b.analyte AS analyte,
                       b.analyte_class AS analyte_class,
                       coalesce(b.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                for term in (label, rec["analyte"], rec["analyte_class"]):
                    if term:
                        _add(term, nid, "Biomarker", label)
                for syn in (rec["synonyms"] or "").split("|"):
                    if syn.strip():
                        _add(syn, nid, "Biomarker", label)

            # --- Drug nodes ---
            # Index: label, synonyms (added in Step 0)
            result = session.run(
                """
                MATCH (d:Drug {project: $project})
                RETURN d.id AS id, d.label AS label,
                       coalesce(d.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                _add(label, nid, "Drug", label)
                for syn in (rec["synonyms"] or "").split("|"):
                    if syn.strip():
                        _add(syn, nid, "Drug", label)

            # --- Gene nodes ---
            # Index: label (gene symbol), pipe-separated synonyms
            result = session.run(
                """
                MATCH (g:Gene {project: $project})
                RETURN g.id AS id, g.label AS label,
                       coalesce(g.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                _add(label, nid, "Gene", label)
                for syn in (rec["synonyms"] or "").split("|"):
                    if syn.strip():
                        _add(syn, nid, "Gene", label)

            # --- Protein nodes ---
            # Index: label, pipe-separated synonyms (short ones only — see _MAX_PROTEIN_SYNONYM_LEN)
            result = session.run(
                """
                MATCH (p:Protein {project: $project})
                RETURN p.id AS id, p.label AS label,
                       coalesce(p.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                if label and len(label) <= _MAX_PROTEIN_SYNONYM_LEN:
                    _add(label, nid, "Protein", label)
                for syn in (rec["synonyms"] or "").split("|"):
                    s = syn.strip()
                    if s and len(s) <= _MAX_PROTEIN_SYNONYM_LEN:
                        _add(s, nid, "Protein", label)

            # --- Phenotype nodes ---
            # Index: label, pipe-separated synonyms
            result = session.run(
                """
                MATCH (ph:Phenotype {project: $project})
                RETURN ph.id AS id, ph.label AS label,
                       coalesce(ph.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                _add(label, nid, "Phenotype", label)
                for syn in (rec["synonyms"] or "").split("|"):
                    if syn.strip():
                        _add(syn, nid, "Phenotype", label)

            # --- AlzPediaEntity nodes ---
            # Index: label, comma-separated synonyms
            result = session.run(
                """
                MATCH (ae:AlzPediaEntity {project: $project})
                RETURN ae.id AS id, ae.label AS label,
                       coalesce(ae.synonyms, '') AS synonyms
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                _add(label, nid, "AlzPedia", label)
                for syn in (rec["synonyms"] or "").split(","):
                    if syn.strip():
                        _add(syn, nid, "AlzPedia", label)

            # --- Pathway nodes ---
            # No synonyms in graph — label only
            result = session.run(
                """
                MATCH (pw:Pathway {project: $project})
                RETURN pw.id AS id, pw.label AS label
                """,
                project=self.project,
            )
            for rec in result:
                nid = rec["id"]
                label = rec["label"] or ""
                _add(label, nid, "Pathway", label)

        # All queries succeeded — atomically publish the new vocab.
        # sorted_terms cached here so match_entities() never re-sorts per request.
        self._vocab = vocab
        self._sorted_terms = sorted(vocab.keys(), key=len, reverse=True)
        self._built = True

        elapsed = time.monotonic() - t0
        node_types = {m.node_type for matches in vocab.values() for m in matches}
        logger.info(
            "EntityLinker vocabulary built in %.2fs: %d unique terms across %d node types (%s).",
            elapsed,
            len(vocab),
            len(node_types),
            ", ".join(sorted(node_types)),
        )

    def match_entities(self, query: str) -> List[EntityMatch]:
        """
        Find all entity mentions in the query string.

        Uses word-boundary regex so "gene" does not fire on "degenerate",
        "tau" does not fire on "tau-related" (it does match "tau" in "tau protein").

        Longer, more specific terms are matched first (e.g., "tau-p181" before
        "tau"), so the returned matched_term always reflects the most specific
        synonym that fired.

        Returns
        -------
        List[EntityMatch]
            Deduplicated by node_id. If a node matches via multiple synonyms,
            only the first (longest) synonym match is kept.
        """
        if not self._built:
            logger.warning("EntityLinker.match_entities() called before build_vocab(). Returning [].")
            return []

        q_lower = query.lower()
        seen_ids: set = set()
        results: List[EntityMatch] = []

        for term in self._sorted_terms:
            if re.search(r"\b" + re.escape(term) + r"\b", q_lower):
                for match in self._vocab[term]:
                    if match.node_id not in seen_ids:
                        seen_ids.add(match.node_id)
                        # Produce a new EntityMatch recording which term fired
                        results.append(EntityMatch(
                            node_id=match.node_id,
                            node_type=match.node_type,
                            canonical_label=match.canonical_label,
                            matched_term=term,
                        ))

        return results

    def entity_types_found(self, query: str) -> List[str]:
        """Convenience: return just the list of node types found in query."""
        return list({m.node_type for m in self.match_entities(query)})


# ---------------------------------------------------------------------------
# Singleton accessor (mirrors pattern used by retriever and llm_client)
# ---------------------------------------------------------------------------

_linker: Optional[EntityLinker] = None


def get_entity_linker() -> EntityLinker:
    """
    Get (and lazily create + initialize) the project-wide EntityLinker.

    Uses the GraphRetriever's underlying Neo4j driver so no second connection
    is opened.

    If build_vocab() fails, the singleton slot is NOT set — the next call
    will retry, and classify_question()'s try/except degrades gracefully to
    keyword scoring in the meantime.
    """
    global _linker
    if _linker is not None:
        return _linker

    from .retriever import get_retriever
    from .config import CONFIG

    retriever = get_retriever()
    candidate = EntityLinker(
        driver=retriever._driver,
        database=CONFIG.neo4j_db,
        project=CONFIG.project_name,
    )
    candidate.build_vocab()   # raises on failure — _linker stays None so next call retries
    _linker = candidate       # only assigned after a successful build
    return _linker
