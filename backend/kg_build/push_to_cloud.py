"""
kg_build.push_to_cloud
----------------------

Sync the local Neo4j (Desktop) data to Neo4j AuraDB (cloud).

What it does
------------
1. Reads every node/relationship for project='alzheimerskg' from LOCAL Neo4j.
2. Deletes all project data from the CLOUD instance (clean slate).
3. Writes nodes (by label) into the cloud — MERGE on (id, project).
4. Writes relationships (by type) into the cloud — MATCH endpoints,
   MERGE relationship, SET all properties.

All reads/writes are batched (BATCH_SIZE rows at a time) to stay within
AuraDB memory limits.

Usage
-----
From backend/ (with venv active):

    # Dry run — shows counts, writes nothing to cloud
    python -m kg_build.push_to_cloud

    # Actually push
    python -m kg_build.push_to_cloud --execute

Env vars used (from backend/.env via graph_rag.config):
    LOCAL  — NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DB
    CLOUD  — CLOUD_NEO4J_URI / CLOUD_NEO4J_USER / CLOUD_NEO4J_PASSWORD / CLOUD_NEO4J_DB
    PROJECT_NAME — defaults to "alzheimerskg"
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List

from neo4j import GraphDatabase, Driver

# Load .env before importing CONFIG
from graph_rag.config import CONFIG
from kg_build.schema import NODE_SCHEMAS, EDGE_SCHEMAS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 500  # rows per UNWIND batch
PROJECT = CONFIG.project_name

# ---------------------------------------------------------------------------
# Driver factories
# ---------------------------------------------------------------------------


def _local_driver() -> Driver:
    return GraphDatabase.driver(
        CONFIG.neo4j_uri,
        auth=(CONFIG.neo4j_user, CONFIG.neo4j_password),
    )


def _cloud_driver() -> tuple[Driver, str]:
    import os
    # dotenv was already loaded by graph_rag.config — values are in os.environ
    uri  = os.environ.get("CLOUD_NEO4J_URI",      "").strip()
    user = os.environ.get("CLOUD_NEO4J_USER",     "neo4j").strip()
    pwd  = os.environ.get("CLOUD_NEO4J_PASSWORD", "").strip()
    db   = os.environ.get("CLOUD_NEO4J_DB",       "neo4j").strip()

    if not uri or not pwd:
        raise RuntimeError(
            "CLOUD_NEO4J_URI and CLOUD_NEO4J_PASSWORD must be set in backend/.env\n"
            "  CLOUD_NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io\n"
            "  CLOUD_NEO4J_PASSWORD=..."
        )
    print(f"  Cloud    : {uri} / db={db}")
    return GraphDatabase.driver(uri, auth=(user, pwd)), db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batches(items: List[Any], size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _read_nodes(driver: Driver, db: str, label: str) -> List[Dict[str, Any]]:
    """
    Return all nodes of a given label belonging to this project from LOCAL.
    Each row is a plain dict of all properties.
    """
    q = f"""
    MATCH (n:{label} {{project: $project}})
    RETURN properties(n) AS props
    """
    with driver.session(database=db) as session:
        return [rec["props"] for rec in session.run(q, project=PROJECT)]


def _read_relationships(
    driver: Driver, db: str, rel_type: str, src_label: str, tgt_label: str
) -> List[Dict[str, Any]]:
    """
    Return all relationships of a given type (anchored on project nodes) from LOCAL.
    Each row: {source_id, target_id, props}
    """
    q = f"""
    MATCH (a:{src_label} {{project: $project}})-[r:{rel_type}]->(b:{tgt_label})
    RETURN a.id AS source_id, b.id AS target_id, properties(r) AS props
    """
    with driver.session(database=db) as session:
        return [
            {"source_id": rec["source_id"], "target_id": rec["target_id"], "props": dict(rec["props"])}
            for rec in session.run(q, project=PROJECT)
        ]


def _merge_nodes_batch(
    driver: Driver, db: str, label: str, batch: List[Dict[str, Any]]
) -> int:
    """MERGE a batch of nodes into the cloud, SET all properties."""
    q = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id, project: row.project}})
    SET n += row
    RETURN count(n) AS cnt
    """
    with driver.session(database=db) as session:
        rec = session.run(q, rows=batch).single()
        return rec["cnt"] if rec else 0


def _merge_relationships_batch(
    driver: Driver,
    db: str,
    rel_type: str,
    src_label: str,
    tgt_label: str,
    batch: List[Dict[str, Any]],
) -> int:
    """MERGE a batch of relationships into the cloud, SET all properties."""
    q = f"""
    UNWIND $rows AS row
    MATCH (a:{src_label} {{id: row.source_id, project: $project}})
    MATCH (b:{tgt_label} {{id: row.target_id, project: $project}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += row.props
    SET r.project = $project
    RETURN count(r) AS cnt
    """
    with driver.session(database=db) as session:
        rec = session.run(q, rows=batch, project=PROJECT).single()
        return rec["cnt"] if rec else 0


def _clear_project_from_cloud(driver: Driver, db: str, dry_run: bool) -> None:
    """Delete all nodes (and their relationships) for this project from cloud."""
    count_q = "MATCH (n {project: $project}) RETURN count(n) AS cnt"
    with driver.session(database=db) as session:
        rec = session.run(count_q, project=PROJECT).single()
        existing = rec["cnt"] if rec else 0

    print(f"  Cloud currently has {existing} node(s) with project='{PROJECT}'.")
    if dry_run:
        print("  [dry-run] Skipping cloud clear.")
        return

    if existing == 0:
        print("  Nothing to clear.")
        return

    # DETACH DELETE in batches to avoid out-of-memory on large graphs
    deleted = 0
    while True:
        with driver.session(database=db) as session:
            rec = session.run(
                "MATCH (n {project: $project}) "
                "WITH n LIMIT $limit "
                "DETACH DELETE n "
                "RETURN count(n) AS cnt",
                project=PROJECT,
                limit=BATCH_SIZE,
            ).single()
            batch_deleted = rec["cnt"] if rec else 0
        deleted += batch_deleted
        if batch_deleted == 0:
            break

    print(f"  Cleared {deleted} node(s) (+ their relationships) from cloud.")


# ---------------------------------------------------------------------------
# Main push logic
# ---------------------------------------------------------------------------


def push(dry_run: bool = True) -> None:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Alzheimer KG — push local → cloud")
    print(f"  Project  : {PROJECT}")
    print(f"  Local    : {CONFIG.neo4j_uri} / db={CONFIG.neo4j_db}")

    local_drv = _local_driver()
    local_db  = CONFIG.neo4j_db

    try:
        cloud_drv, cloud_db = _cloud_driver()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # Step 1 — read everything from local
        # ------------------------------------------------------------------
        print("\n[1/3] Reading from local Neo4j ...")

        node_data: Dict[str, List[Dict[str, Any]]] = {}
        for label in NODE_SCHEMAS:
            rows = _read_nodes(local_drv, local_db, label)
            node_data[label] = rows
            if rows:
                print(f"  {label:20s} : {len(rows):>6,} nodes")

        rel_data: Dict[str, List[Dict[str, Any]]] = {}
        for rel_type, schema in EDGE_SCHEMAS.items():
            rows = _read_relationships(
                local_drv, local_db, rel_type, schema.source_label, schema.target_label
            )
            rel_data[rel_type] = rows
            if rows:
                print(f"  {rel_type:30s} : {len(rows):>6,} relationships")

        total_nodes = sum(len(v) for v in node_data.values())
        total_rels  = sum(len(v) for v in rel_data.values())
        print(f"\n  Total: {total_nodes:,} nodes, {total_rels:,} relationships to push.")

        if total_nodes == 0:
            print(
                f"\nWARNING: No nodes found in local Neo4j for project='{PROJECT}'.\n"
                "  Have you run the migration script yet?\n"
                "  python -m kg_build.migrate_project_tag"
            )
            return

        # ------------------------------------------------------------------
        # Step 2 — clear cloud
        # ------------------------------------------------------------------
        print("\n[2/3] Clearing existing project data from cloud ...")
        _clear_project_from_cloud(cloud_drv, cloud_db, dry_run)

        if dry_run:
            print("\n[3/3] [dry-run] Skipping writes to cloud.")
            print("\nRe-run with --execute to actually push the data.")
            return

        # ------------------------------------------------------------------
        # Step 3 — write nodes
        # ------------------------------------------------------------------
        print("\n[3/3] Writing to cloud Neo4j ...")
        print("  — Nodes —")
        t0 = time.time()
        written_nodes = 0
        for label, rows in node_data.items():
            if not rows:
                continue
            label_count = 0
            for batch in _batches(rows, BATCH_SIZE):
                label_count += _merge_nodes_batch(cloud_drv, cloud_db, label, batch)
            written_nodes += label_count
            print(f"    {label:20s} : {label_count:>6,} merged")

        # — Relationships —
        print("  — Relationships —")
        written_rels = 0
        for rel_type, rows in rel_data.items():
            if not rows:
                continue
            schema = EDGE_SCHEMAS[rel_type]
            rel_count = 0
            for batch in _batches(rows, BATCH_SIZE):
                rel_count += _merge_relationships_batch(
                    cloud_drv, cloud_db,
                    rel_type, schema.source_label, schema.target_label,
                    batch,
                )
            written_rels += rel_count
            print(f"    {rel_type:30s} : {rel_count:>6,} merged")

        elapsed = time.time() - t0
        print(
            f"\nDone in {elapsed:.1f}s — "
            f"{written_nodes:,} nodes, {written_rels:,} relationships pushed to cloud."
        )

    finally:
        local_drv.close()
        cloud_drv.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Push local Neo4j data (project='alzheimerskg') to Neo4j AuraDB.\n"
            "Runs in dry-run mode by default."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually clear the cloud and write data. Without this flag the script is a dry run.",
    )
    args = parser.parse_args()
    push(dry_run=not args.execute)


if __name__ == "__main__":
    main()
