"""
kg_build.migrate_project_tag
-----------------------------

One-time migration: stamp all existing Neo4j nodes and relationships
with `project = 'alzheimerskg'` so the shared AuraDB instance can
distinguish this project's data from other projects.

Also creates per-label indexes on the `project` property so that
filtered lookups remain fast.

Usage
-----
From the backend/ directory (with venv active):

    python -m kg_build.migrate_project_tag

Or point at a different Neo4j instance via env vars before running:

    NEO4J_URI=bolt://... NEO4J_PASSWORD=... python -m kg_build.migrate_project_tag

The script is idempotent: nodes/relationships that already have
`project` set are left unchanged.
"""

from __future__ import annotations

import sys

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Connection — reads from graph_rag config so .env values are honoured
# ---------------------------------------------------------------------------

from graph_rag.config import CONFIG

URI      = CONFIG.neo4j_uri
USER     = CONFIG.neo4j_user
PASSWORD = CONFIG.neo4j_password
DB       = CONFIG.neo4j_db
PROJECT  = CONFIG.project_name

# Node labels present in this KG
NODE_LABELS = [
    "Disease",
    "Biomarker",
    "Drug",
    "Phenotype",
    "Pathway",
    "Gene",
    "Protein",
]


def run_migration() -> None:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session(database=DB) as session:
            # ------------------------------------------------------------------
            # 1. Tag all untagged nodes
            # ------------------------------------------------------------------
            print(f"[migrate] Stamping nodes with project='{PROJECT}' ...")
            result = session.run(
                "MATCH (n) WHERE n.project IS NULL "
                "SET n.project = $project "
                "RETURN count(n) AS updated",
                project=PROJECT,
            )
            rec = result.single()
            node_count = rec["updated"] if rec else 0
            print(f"[migrate]   → {node_count} node(s) updated.")

            # ------------------------------------------------------------------
            # 2. Tag all untagged relationships
            # ------------------------------------------------------------------
            print(f"[migrate] Stamping relationships with project='{PROJECT}' ...")
            result = session.run(
                "MATCH ()-[r]-() WHERE r.project IS NULL "
                "SET r.project = $project "
                "RETURN count(r) AS updated",
                project=PROJECT,
            )
            rec = result.single()
            rel_count = rec["updated"] if rec else 0
            print(f"[migrate]   → {rel_count} relationship(s) updated.")

            # ------------------------------------------------------------------
            # 3. Create indexes on project for each node label
            # ------------------------------------------------------------------
            print("[migrate] Creating indexes on project property ...")
            for label in NODE_LABELS:
                index_name = f"idx_{label.lower()}_project"
                try:
                    session.run(
                        f"CREATE INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) ON (n.project)"
                    )
                    print(f"[migrate]   → index '{index_name}' ensured.")
                except Exception as exc:
                    # Non-fatal: index may already exist under a different name
                    print(f"[migrate]   ! Could not create index for {label}: {exc}")

    finally:
        driver.close()

    print(
        f"\n[migrate] Done. "
        f"Nodes updated: {node_count}, Relationships updated: {rel_count}\n"
        f"All future queries will filter by project='{PROJECT}'."
    )


if __name__ == "__main__":
    run_migration()
