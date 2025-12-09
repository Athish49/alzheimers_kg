"""
graph_rag.neo4j_client
----------------------

Thin wrapper around the Neo4j Python driver.

Responsibilities
----------------
- Create a singleton Neo4j driver using config from `graph_rag.config`.
- Provide small helpers for read / write queries.
- Provide a couple of convenience helpers that the Graph-RAG layer will use
  (e.g., fetching a Disease node by canonical ID).

Usage
-----
from graph_rag.neo4j_client import get_neo4j_client

client = get_neo4j_client()

# Simple read
rows = client.read(
    "MATCH (d:Disease {id: $id}) RETURN d",
    {"id": "MONDO:0004975"},
)
for r in rows:
    print(r["d"])

# Remember to close the driver on shutdown (CLI / scripts)
client.close()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Driver, Result  # type: ignore

from .config import CONFIG


@dataclass
class Neo4jClient:
    """
    Lightweight Neo4j client.

    Parameters
    ----------
    uri:
        Bolt or Neo4j URI, e.g. "bolt://localhost:7687"
    user:
        Username, e.g. "neo4j"
    password:
        Password for the user.
    database:
        Database name, e.g. "neo4j".
    """

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    # ------------------------------------------------------------------
    # Core low-level methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying driver."""
        self._driver.close()

    def _run(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        readonly: bool = True,
    ) -> Result:
        """
        Run a Cypher query and return the raw neo4j.Result.

        Normally you should use .read() / .write() which wrap this and
        return a list of dicts.
        """
        parameters = parameters or {}
        access_mode = "READ" if readonly else "WRITE"
        # neo4j driver uses access mode on session
        with self._driver.session(database=self.database, default_access_mode=access_mode) as session:  # type: ignore[arg-type]
            return session.run(query, parameters)

    def read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a read-only Cypher query and return list-of-dicts (record.data()).
        """
        result = self._run(query, parameters, readonly=True)
        return [record.data() for record in result]

    def write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a write Cypher query and return list-of-dicts (record.data()).
        """
        result = self._run(query, parameters, readonly=False)
        return [record.data() for record in result]

    # ------------------------------------------------------------------
    # Convenience helpers used by Graph-RAG
    # ------------------------------------------------------------------

    def get_disease_by_id(self, disease_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a Disease node by its canonical id property (e.g. MONDO:0004975).

        Assumes you imported the CSVs in a way that keeps a property `id`.
        """
        rows = self.read(
            """
            MATCH (d:Disease {id: $id})
            RETURN d
            """,
            {"id": disease_id},
        )
        if not rows:
            return None
        return rows[0]["d"]

    def get_node_by_label_and_id(self, label: str, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Generic helper to fetch a node with a given label and `id` property.
        """
        query = f"""
        MATCH (n:{label} {{id: $id}})
        RETURN n
        """
        rows = self.read(query, {"id": node_id})
        if not rows:
            return None
        return rows[0]["n"]

    def neighbors(
        self,
        label: str,
        node_id: str,
        rel_types: Optional[Iterable[str]] = None,
        direction: str = "both",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch 1-hop neighbors of a given node.

        Parameters
        ----------
        label:
            Label of the center node, e.g. "Disease".
        node_id:
            Canonical `id` property of the center node.
        rel_types:
            Optional iterable of relationship type names to restrict to
            (e.g. ["HAS_BIOMARKER", "TREATS"]). If None, use all types.
        direction:
            "out", "in", or "both".
        limit:
            Maximum number of relationships to return.

        Returns
        -------
        List of dicts with keys:
            - src  : center node
            - rel  : relationship
            - nbr  : neighbor node
        """
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be one of 'out', 'in', 'both'")

        if rel_types:
            rel_pattern = "|".join(rel_types)
            rel_pattern = f":{rel_pattern}"
        else:
            rel_pattern = ""

        if direction == "out":
            pattern = f"(n:{label} {{id: $id}})-[r{rel_pattern}]->(m)"
        elif direction == "in":
            pattern = f"(n:{label} {{id: $id}})<-[r{rel_pattern}]-(m)"
        else:
            pattern = f"(n:{label} {{id: $id}})-[r{rel_pattern}]-(m)"

        query = f"""
        MATCH {pattern}
        RETURN n AS src, r AS rel, m AS nbr
        LIMIT $limit
        """

        return self.read(query, {"id": node_id, "limit": limit})


# ----------------------------------------------------------------------
# Singleton-style accessor
# ----------------------------------------------------------------------

_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """
    Get (and lazily create) a project-wide Neo4jClient instance.

    Uses connection settings from graph_rag.config.CONFIG by default.
    """
    global _client
    if _client is None:
        _client = Neo4jClient(
            uri=CONFIG.neo4j_uri,
            user=CONFIG.neo4j_user,
            password=CONFIG.neo4j_password,
            database=CONFIG.neo4j_db,
        )
    return _client