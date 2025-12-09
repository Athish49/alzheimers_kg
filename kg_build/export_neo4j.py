"""
kg_build.export_neo4j
---------------------
Prepare Neo4j-import-ready CSVs from Phase 4 canonical node/edge tables.

Input (Phase 4 outputs)
-----------------------
- kg_build/output/nodes_*.csv
    e.g. nodes_biomarker.csv, nodes_disease.csv, ...

- kg_build/output/edges_*.csv
    e.g. edges_has_biomarker.csv, edges_treats.csv, ...

Output (for Neo4j)
------------------
- neo4j_import/neo4j_nodes_<slug>.csv
    - id:ID(Label), <other props...>

- neo4j_import/neo4j_edges_<slug>.csv
    - source_id:START_ID(<SourceLabel>),
      target_id:END_ID(<TargetLabel>),
      <other props...>

Where <slug> is the lowercase form of the label / edge type
(consistent with nodes_*.csv and edges_*.csv naming).

Usage
-----
From project root (where pyproject.toml / .venv live):

    python -m kg_build.export_neo4j

After this, you can point Neo4j (or neo4j-admin) to the 'neo4j_import'
directory. We are not generating Cypher here yet; this script focuses
on producing clean CSVs with proper :ID and :START_ID/:END_ID typing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from kg_build.paths import (
    KG_OUTPUT_DIR,
    NEO4J_IMPORT,
    ensure_dirs,
)
from kg_build.schema import NODE_SCHEMAS, EDGE_SCHEMAS


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _slug_from_node_label(label: str) -> str:
    """
    Convert a NodeSchema label (e.g. 'Biomarker', 'Disease')
    into the slug used in filenames (e.g. 'biomarker', 'disease').
    """
    return label.lower()


def _slug_from_edge_type(edge_type: str) -> str:
    """
    Convert an edge type (e.g. 'HAS_BIOMARKER', 'TREATS')
    into the slug used in filenames (e.g. 'has_biomarker', 'treats').
    """
    return edge_type.lower()


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load a CSV into (fieldnames, list-of-rows) using DictReader."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    """Write a list-of-dicts CSV with given fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------------------------------------------------------------------
# Node export: nodes_*.csv -> neo4j_nodes_*.csv
# -------------------------------------------------------------------


def export_nodes_to_neo4j() -> Dict[str, Path]:
    """
    Export all known node tables into Neo4j-style CSVs.

    Returns
    -------
    mapping : dict
        slug -> output Path for each node CSV written.
    """
    out: Dict[str, Path] = {}

    for label, schema in NODE_SCHEMAS.items():
        slug = _slug_from_node_label(label)
        src_path = KG_OUTPUT_DIR / f"nodes_{slug}.csv"
        if not src_path.exists():
            # Not all node types may be present yet; skip silently.
            continue

        fieldnames, rows = _read_csv(src_path)
        if not fieldnames:
            continue

        # Expect first column to be 'id'
        original_id_col = fieldnames[0]
        if original_id_col != "id":
            raise ValueError(
                f"Expected first column of {src_path} to be 'id', "
                f"got '{original_id_col}'"
            )

        neo_id_col = f"{original_id_col}:ID({label})"
        new_fieldnames = [neo_id_col] + fieldnames[1:]

        # Transform rows: move 'id' -> 'id:ID(Label)'
        neo_rows: List[Dict[str, str]] = []
        for row in rows:
            new_row = dict(row)  # shallow copy
            # pop the old key and assign to the new Neo4j-typed column
            new_row[neo_id_col] = new_row.pop("id")
            neo_rows.append(new_row)

        neo_path = NEO4J_IMPORT / f"neo4j_nodes_{slug}.csv"
        _write_csv(neo_path, new_fieldnames, neo_rows)
        out[slug] = neo_path

    return out


# -------------------------------------------------------------------
# Edge export: edges_*.csv -> neo4j_edges_*.csv
# -------------------------------------------------------------------


def export_edges_to_neo4j() -> Dict[str, Path]:
    """
    Export all known edge tables into Neo4j-style CSVs.

    Returns
    -------
    mapping : dict
        slug -> output Path for each edge CSV written.
    """
    out: Dict[str, Path] = {}

    for edge_type, schema in EDGE_SCHEMAS.items():
        slug = _slug_from_edge_type(edge_type)
        src_path = KG_OUTPUT_DIR / f"edges_{slug}.csv"
        if not src_path.exists():
            # Not all edge types may be present yet; skip silently.
            continue

        fieldnames, rows = _read_csv(src_path)
        if not fieldnames:
            continue

        # Expect at least: source_id, target_id, ...
        if len(fieldnames) < 2:
            raise ValueError(
                f"Expected at least two columns (source_id, target_id) in {src_path}"
            )

        src_col = fieldnames[0]
        tgt_col = fieldnames[1]
        if src_col != "source_id" or tgt_col != "target_id":
            raise ValueError(
                f"Expected first two columns of {src_path} to be "
                f"'source_id', 'target_id', got '{src_col}', '{tgt_col}'"
            )

        start_col = f"{src_col}:START_ID({schema.source_label})"
        end_col = f"{tgt_col}:END_ID({schema.target_label})"
        new_fieldnames = [start_col, end_col] + fieldnames[2:]

        # Transform rows: move 'source_id'/'target_id' to Neo4j-typed columns
        neo_rows: List[Dict[str, str]] = []
        for row in rows:
            new_row = dict(row)
            new_row[start_col] = new_row.pop("source_id")
            new_row[end_col] = new_row.pop("target_id")
            neo_rows.append(new_row)

        neo_path = NEO4J_IMPORT / f"neo4j_edges_{slug}.csv"
        _write_csv(neo_path, new_fieldnames, neo_rows)
        out[slug] = neo_path

    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    ensure_dirs()

    print("[export_neo4j] Exporting nodes...")
    node_files = export_nodes_to_neo4j()
    print(f"[export_neo4j] Exported {len(node_files)} node CSV(s).")

    print("[export_neo4j] Exporting edges...")
    edge_files = export_edges_to_neo4j()
    print(f"[export_neo4j] Exported {len(edge_files)} edge CSV(s).")

    print("\n[export_neo4j] Summary (for Neo4j import):")
    if node_files:
        print("  Node CSVs:")
        for slug, path in sorted(node_files.items()):
            print(f"    - {slug:12s} -> {path}")
    else:
        print("  Node CSVs: none exported")

    if edge_files:
        print("  Edge CSVs:")
        for slug, path in sorted(edge_files.items()):
            print(f"    - {slug:12s} -> {path}")
    else:
        print("  Edge CSVs: none exported")

    print(
        "\n[export_neo4j] You can now point Neo4j to the 'neo4j_import' folder.\n"
        "For example (Cypher LOAD CSV, assuming files are in Neo4j's import dir):\n"
        "  LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes_disease.csv' AS row\n"
        "  MERGE (d:Disease {id: row.id})\n"
        "  SET d.label = row.label;\n"
        "\n"
        "Or use neo4j-admin 'database import' with these files.\n"
    )


if __name__ == "__main__":
    main()