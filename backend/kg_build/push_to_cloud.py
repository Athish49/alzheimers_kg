"""
kg_build.push_to_cloud
----------------------

Load neo4j_import/ CSV files into a Neo4j database.

The CSVs in neo4j_import/ are the authoritative source. This script
pushes them to whichever database you specify via --target.

What it does
------------
1. Reads every node/relationship CSV from backend/neo4j_import/.
2. Deletes ALL project data from the target database (clean slate —
   DETACH DELETE in batches, so no orphan nodes or duplicate edges
   can survive between runs).
3. Writes nodes by label — MERGE on (id, project).
4. Writes relationships by type — MATCH endpoints, MERGE relationship,
   SET all properties.

All writes are batched (BATCH_SIZE rows at a time) to stay within
AuraDB memory limits.

Usage
-----
From backend/ (with venv active):

    # Dry run — shows counts, writes nothing
    python -m kg_build.push_to_cloud --target cloud
    python -m kg_build.push_to_cloud --target local

    # Actually push
    python -m kg_build.push_to_cloud --target cloud --execute
    python -m kg_build.push_to_cloud --target local --execute

Env vars used (from backend/.env):
    LOCAL  — NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DB
    CLOUD  — CLOUD_NEO4J_URI / CLOUD_NEO4J_USER /
             CLOUD_NEO4J_PASSWORD / CLOUD_NEO4J_DB
    PROJECT_NAME — defaults to "alzheimerskg"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from neo4j import GraphDatabase, Driver

from graph_rag.config import CONFIG
from kg_build.schema import EDGE_SCHEMAS, NODE_SCHEMAS
from kg_build.paths import NEO4J_IMPORT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 500
PROJECT = CONFIG.project_name

# ---------------------------------------------------------------------------
# Driver factory
# ---------------------------------------------------------------------------


def _get_driver(target: str) -> Tuple[Driver, str]:
    """Return (driver, database_name) for the requested target."""
    if target == "local":
        uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687").strip()
        user = os.environ.get("NEO4J_USER",     "neo4j").strip()
        pwd  = os.environ.get("NEO4J_PASSWORD", "").strip()
        db   = os.environ.get("NEO4J_DB",       "neo4j").strip()
        if not pwd:
            raise RuntimeError(
                "NEO4J_PASSWORD must be set in backend/.env for --target local"
            )
        print(f"  Target   : local  ({uri} / db={db})")
    else:
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
        print(f"  Target   : cloud  ({uri} / db={db})")

    return GraphDatabase.driver(uri, auth=(user, pwd)), db


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _plain_col(col: str) -> str:
    """
    Strip Neo4j import type annotations from a CSV column header.

      'id:ID(Biomarker)'          → 'id'
      'source_id:START_ID(Drug)'  → 'source_id'
      'target_id:END_ID(Disease)' → 'target_id'
      'label'                     → 'label'
    """
    return re.split(r"[:(]", col)[0]


def _load_node_csv(label: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load backend/neo4j_import/neo4j_nodes_{label}.csv.
    Returns None if the file does not exist.
    Empty-string values are dropped so they don't overwrite real data with "".
    """
    path = NEO4J_IMPORT / f"neo4j_nodes_{label.lower()}.csv"
    if not path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {_plain_col(k): v for k, v in row.items()}
            cleaned = {k: v for k, v in cleaned.items() if v != ""}
            cleaned["project"] = PROJECT  # always set; CSVs may not include this column
            rows.append(cleaned)
    return rows


def _load_edge_csv(rel_type: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load backend/neo4j_import/neo4j_edges_{rel_type}.csv.
    Returns None if the file does not exist.
    Each row is returned as {source_id, target_id, props}.
    """
    path = NEO4J_IMPORT / f"neo4j_edges_{rel_type.lower()}.csv"
    if not path.exists():
        return None

    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {_plain_col(k): v for k, v in row.items()}
            source_id = cleaned.pop("source_id", None)
            target_id = cleaned.pop("target_id", None)
            # Drop project from props — it gets set separately on the relationship
            props = {k: v for k, v in cleaned.items() if v != "" and k != "project"}
            rows.append({"source_id": source_id, "target_id": target_id, "props": props})
    return rows


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _batches(items: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _merge_nodes_batch(
    driver: Driver, db: str, label: str, batch: List[Dict[str, Any]]
) -> int:
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


def _clear_project(driver: Driver, db: str, target: str, dry_run: bool) -> None:
    """DETACH DELETE all project nodes from the target. Batched to avoid OOM."""
    with driver.session(database=db) as session:
        rec = session.run(
            "MATCH (n {project: $project}) RETURN count(n) AS cnt", project=PROJECT
        ).single()
        existing = rec["cnt"] if rec else 0

    print(f"  {target} currently has {existing} node(s) with project='{PROJECT}'.")
    if dry_run:
        print("  [dry-run] Skipping clear.")
        return
    if existing == 0:
        print("  Nothing to clear.")
        return

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
            n = rec["cnt"] if rec else 0
        deleted += n
        if n == 0:
            break

    print(f"  Cleared {deleted} node(s) (+ their relationships).")


# ---------------------------------------------------------------------------
# Main push logic
# ---------------------------------------------------------------------------


def push(target: str = "cloud", dry_run: bool = True) -> None:
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Alzheimer KG — CSV → {target}")
    print(f"  Project  : {PROJECT}")
    print(f"  CSV dir  : {NEO4J_IMPORT}")

    try:
        driver, db = _get_driver(target)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # Step 1 — read CSVs
        # ------------------------------------------------------------------
        print("\n[1/3] Reading CSV files ...")

        node_data: Dict[str, List[Dict[str, Any]]] = {}
        for label in NODE_SCHEMAS:
            rows = _load_node_csv(label)
            if rows is not None:
                node_data[label] = rows
                print(f"  {label:20s} : {len(rows):>6,} nodes")

        edge_data: Dict[str, List[Dict[str, Any]]] = {}
        for rel_type in EDGE_SCHEMAS:
            rows = _load_edge_csv(rel_type)
            if rows is not None:
                edge_data[rel_type] = rows
                print(f"  {rel_type:30s} : {len(rows):>6,} relationships")

        total_nodes = sum(len(v) for v in node_data.values())
        total_rels  = sum(len(v) for v in edge_data.values())
        print(f"\n  Total: {total_nodes:,} nodes, {total_rels:,} relationships to push.")

        if total_nodes == 0:
            print(f"\nWARNING: No node CSV files found in {NEO4J_IMPORT}")
            return

        # ------------------------------------------------------------------
        # Step 2 — clean slate
        # ------------------------------------------------------------------
        print(f"\n[2/3] Clearing existing project data from {target} ...")
        _clear_project(driver, db, target, dry_run)

        if dry_run:
            print("\n[3/3] [dry-run] Skipping writes.")
            print("Re-run with --execute to actually push the data.")
            return

        # ------------------------------------------------------------------
        # Step 3 — write nodes then relationships
        # ------------------------------------------------------------------
        print(f"\n[3/3] Writing to {target} Neo4j ...")
        t0 = time.time()

        print("  — Nodes —")
        written_nodes = 0
        for label, rows in node_data.items():
            label_count = 0
            for batch in _batches(rows, BATCH_SIZE):
                label_count += _merge_nodes_batch(driver, db, label, batch)
            written_nodes += label_count
            print(f"    {label:20s} : {label_count:>6,} merged")

        print("  — Relationships —")
        written_rels = 0
        for rel_type, rows in edge_data.items():
            schema = EDGE_SCHEMAS[rel_type]
            rel_count = 0
            for batch in _batches(rows, BATCH_SIZE):
                rel_count += _merge_relationships_batch(
                    driver, db,
                    rel_type, schema.source_label, schema.target_label,
                    batch,
                )
            written_rels += rel_count
            print(f"    {rel_type:30s} : {rel_count:>6,} merged")

        elapsed = time.time() - t0
        print(
            f"\nDone in {elapsed:.1f}s — "
            f"{written_nodes:,} nodes, {written_rels:,} relationships pushed to {target}."
        )

    finally:
        driver.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Push neo4j_import/ CSVs to a Neo4j database.\n"
            "Always does a full clean-slate replace: all project nodes are\n"
            "DETACH-DELETEd before writing, so no duplicates can accumulate.\n"
            "Runs in dry-run mode by default; add --execute to write."
        )
    )
    parser.add_argument(
        "--target",
        choices=["local", "cloud"],
        default="cloud",
        help=(
            "Target database. "
            "'local' uses NEO4J_URI/USER/PASSWORD/DB from .env. "
            "'cloud' uses CLOUD_NEO4J_URI/USER/PASSWORD/DB from .env. "
            "Default: cloud."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually clear and write. Without this flag the script is a dry run.",
    )
    args = parser.parse_args()
    push(target=args.target, dry_run=not args.execute)


if __name__ == "__main__":
    main()
