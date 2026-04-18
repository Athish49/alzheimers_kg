"""
kg_build.schema
----------------
Central definition of the Knowledge Graph schema for the Alzheimer's project.

This module defines:

- NodeSchema  : canonical node labels and their properties
- EdgeSchema  : canonical relationship types and edge properties

Phase 4 scripts (normalize_entities.py, build_edges.py, export_neo4j.py)
should rely on these schemas instead of hard-coding labels/columns.

The actual *data* for these comes from:
- AlzForum processed CSVs (alzforum/processed/)
- Ontology lookups (ontology/processed/)

We aim for a schema that is:
- Expressive enough for current Phase 3 data
- Stable for future extensions (more ontologies, more edges)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# Dataclasses for schema objects
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSchema:
    """
    Schema for a node label in the KG.

    Attributes
    ----------
    label:
        Neo4j-style node label (e.g. "Disease", "Protein").
    description:
        Human-readable description of this node type.
    required_props:
        Properties that *must* be present for every node of this label.
        At minimum, we enforce "id" and "label".
    optional_props:
        Properties that may or may not be present.
    """

    label: str
    description: str
    required_props: List[str] = field(default_factory=list)
    optional_props: List[str] = field(default_factory=list)

    @property
    def all_props(self) -> List[str]:
        """Return the union of required and optional properties."""
        # Avoid duplicates while preserving order
        seen = set()
        out: List[str] = []
        for p in self.required_props + self.optional_props:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out


@dataclass(frozen=True)
class EdgeSchema:
    """
    Schema for a relationship type (edge) in the KG.

    Attributes
    ----------
    type:
        Relationship type (e.g. "HAS_BIOMARKER", "TREATS").
    description:
        Human-readable description.
    source_label:
        Label of the source node (tail).
    target_label:
        Label of the target node (head).
    required_props:
        Edge properties that must exist.
    optional_props:
        Edge properties that are nice-to-have but optional.
    """

    type: str
    description: str
    source_label: str
    target_label: str
    required_props: List[str] = field(default_factory=list)
    optional_props: List[str] = field(default_factory=list)

    @property
    def all_props(self) -> List[str]:
        """Return the union of required and optional properties."""
        seen = set()
        out: List[str] = []
        for p in self.required_props + self.optional_props:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out


# ---------------------------------------------------------------------
# Node schemas (core node labels)
# ---------------------------------------------------------------------

# NOTE:
# - Every node has at least: id, label
# - *_id fields hold ontology IDs where possible (MONDO, HGNC, etc.)
# - "source" can be comma-separated AlzForum / ontology sources


NODE_SCHEMAS: Dict[str, NodeSchema] = {
    # Disease
    "Disease": NodeSchema(
        label="Disease",
        description="Disease / disorder entities (e.g. Alzheimer disease, MCI).",
        required_props=["id", "label"],
        optional_props=[
            "iri",           # <--- NEW
            "mondo_id",
            "umls_cui",
            "mesh_id",
            "icd10",
            "synonyms",
            "category",
            "source",
            "raw_source_ids",
        ],
    ),

    # Protein
    "Protein": NodeSchema(
        label="Protein",
        description="Proteins / gene products (e.g. APP, tau, BACE1).",
        required_props=["id", "label"],
        optional_props=[
            "iri",          # <--- NEW
            "uniprot_id",
            "hgnc_id",
            "gene_symbol",
            "synonyms",
            "source",
            "raw_source_ids",
        ],
    ),

    # Gene
    "Gene": NodeSchema(
        label="Gene",
        description="Genes (e.g. APP, PSEN1, APOE).",
        required_props=["id", "label"],
        optional_props=[
            "iri",          # <--- NEW (if we ever attach an IRI from HGNC)
            "hgnc_id",
            "entrez_id",
            "ensembl_id",
            "chromosome",
            "synonyms",
            "source",
            "raw_source_ids",
        ],
    ),

    # Pathway
    "Pathway": NodeSchema(
        label="Pathway",
        description="Biological processes / pathways (mostly GO terms).",
        required_props=["id", "label"],
        optional_props=[
            "iri",          # <--- NEW
            "go_id",
            "namespace",
            "synonyms",
            "source",
            "raw_source_ids",
        ],
    ),

    # Biomarker – optional but harmless to add iri for future mapping
    "Biomarker": NodeSchema(
        label="Biomarker",
        description="Assayable biomarkers (often fluid-based analytes).",
        required_props=["id", "label"],
        optional_props=[
            "iri",           # <--- optional, future-proof
            "analyte",
            "analyte_class",
            "fluid",
            "units",
            "assay_type",
            "source",
            "raw_source_ids",
        ],
    ),

    # Phenotype
    "Phenotype": NodeSchema(
        label="Phenotype",
        description="Clinical signs, symptoms, and phenotypes (HPO-driven).",
        required_props=["id", "label"],
        optional_props=[
            "iri",          # <--- NEW
            "hpo_id",
            "umls_cui",
            "mesh_id",
            "synonyms",
            "source",
            "raw_source_ids",
        ],
    ),

    # Drug (for ChEBI iri)
    "Drug": NodeSchema(
        label="Drug",
        description="Therapeutics / interventions (AlzForum Therapeutics, CHEBI).",
        required_props=["id", "label"],
        optional_props=[
            "iri",              # <--- NEW
            "chebi_id",
            "atc_code",
            "drug_type",
            "drug_class",
            "primary_indication",
            "secondary_indications",
            "mechanism_summary",
            "status_overall",
            "approved_regions",
            "source",
            "raw_source_ids",
        ],
    ),

    "RiskFactor": NodeSchema(
        label="RiskFactor",
        description="Non-genetic or genetic risk / protective factors (AlzRisk).",
        required_props=["id", "label"],
        optional_props=[
            "category",      # cardiovascular, metabolic, lifestyle, hormonal, genetic
            "direction",     # increased_risk, protective, null (overall)
            "short_summary",
            "source",
            "raw_source_ids",
        ],
    ),
    
    "Study": NodeSchema(
        label="Study",
        description="Meta-analyses or clinical trial groups (optional node type).",
        required_props=["id", "label"],
        optional_props=[
            "citation",
            "year",
            "pubmed_id",
            "doi",
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: Mechanism / pathology nodes (amyloid, tau, etc.)
    "Mechanism": NodeSchema(
        label="Mechanism",
        description="Pathophysiologic mechanism or pathology (e.g. Amyloid, Tau, Other neurotransmitters).",
        required_props=["id", "label"],
        optional_props=[
            "category",      # amyloid, tau, other_neurotransmitters, other
            "description",
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: Company nodes (drug sponsors)
    "Company": NodeSchema(
        label="Company",
        description="Organizations / companies developing therapeutics.",
        required_props=["id", "label"],
        optional_props=[
            "country",
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: TherapyType nodes (immunotherapy, small molecule, DNA/RNA-based)
    "TherapyType": NodeSchema(
        label="TherapyType",
        description="Therapeutic modality (e.g. Immunotherapy (passive), Small Molecule, DNA/RNA-based).",
        required_props=["id", "label"],
        optional_props=[
            "category",   # high-level grouping if we want (biologic, small_molecule, gene_therapy)
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: Fluid nodes (CSF, plasma, serum, etc.)
    "Fluid": NodeSchema(
        label="Fluid",
        description="Biofluid or sample type in which biomarkers are measured (e.g. CSF, Plasma, Plasma/Serum).",
        required_props=["id", "label"],
        optional_props=[
            "category",   # e.g. central, peripheral
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: Trial nodes (aggregated clinical trials per drug/indication)
    "Trial": NodeSchema(
        label="Trial",
        description="Aggregated clinical trial record for a drug-indication pair.",
        required_props=["id", "label"],
        optional_props=[
            "indication",
            "trial_phase_max",
            "has_phase3",
            "status",          # ongoing, discontinued, completed
            "trial_count",
            "notes",
            "source",
            "raw_source_ids",
        ],
    ),

    # NEW: AlzPedia entity nodes (textual gene/protein pages)
    "AlzPediaEntity": NodeSchema(
        label="AlzPediaEntity",
        description="AlzPedia entry representing a gene, protein, or concept.",
        required_props=["id", "label"],
        optional_props=[
            "url",
            "synonyms",
            "short_summary",
            "category",         # protein_or_gene, pathway, etc.
            "has_function_section",
            "has_pathology_section",
            "has_genetics_section",
            "has_therapeutics_section",
            "source",
            "raw_source_ids",
        ],
    ),
}


# ---------------------------------------------------------------------
# Edge schemas (relationship types)
# ---------------------------------------------------------------------

EDGE_SCHEMAS: Dict[str, EdgeSchema] = {
    # --- Disease <-> Biomarker (from AlzBiomarker) -------------------
    "HAS_BIOMARKER": EdgeSchema(
        type="HAS_BIOMARKER",
        description=(
            "Biomarker evidence for a disease (e.g. CSF Aβ42 decreased in AD "
            "vs controls)."
        ),
        source_label="Disease",
        target_label="Biomarker",
        required_props=[
            "direction",          # increased, decreased, no_change
        ],
        optional_props=[
            "comparison",         # e.g. AD vs Control
            "disease_group",
            "control_group",
            "effect_size_type",   # SMD, OR, log(OR), etc.
            "effect_size",
            "ci_lower",
            "ci_upper",
            "p_value",
            "n_studies",
            "n_cases",
            "n_controls",
            "study_id",           # optional FK -> Study
            "source",             # e.g. AlzBiomarker
            "source_text",
        ],
    ),

    # --- RiskFactor -> Disease (from AlzRisk) ------------------------
    "INCREASES_RISK_OF": EdgeSchema(
        type="INCREASES_RISK_OF",
        description="Risk factor increases risk of a disease (AlzRisk).",
        source_label="RiskFactor",
        target_label="Disease",
        required_props=[
            "direction",          # increased_risk, protective, null
        ],
        optional_props=[
            "outcome",            # AD, all-cause dementia, etc.
            "population",         # midlife, late-life, etc.
            "effect_size_type",   # RR, HR, OR
            "effect_size",
            "ci_lower",
            "ci_upper",
            "p_value",
            "n_studies",
            "quality_flags",
            "study_id",
            "source",
            "source_text",
        ],
    ),

    # --- Drug -> Disease (from therapeutics_trials) ------------------
    "TREATS": EdgeSchema(
        type="TREATS",
        description=(
            "Therapeutic trials / approval for a disease "
            "(e.g. lecanemab treats early AD, status=approved)."
        ),
        source_label="Drug",
        target_label="Disease",
        required_props=[],
        optional_props=[
            "status",            # approved, ongoing, discontinued, failed, unknown
            "indication",        # mild AD, MCI due to AD, early AD, etc.
            "trial_phase_max",   # max phase reached (1/2/3/4)
            "has_phase3",        # bool-ish
            "trial_count",       # rough count from AlzForum
            "approved_regions",
            "source",            # AlzForum.Therapeutics
            "notes",             # free-text summary from FDA status
        ],
    ),

    # --- Drug -> Protein (from therapeutics_targets) -----------------
    "TARGETS_PROTEIN": EdgeSchema(
        type="TARGETS_PROTEIN",
        description=(
            "Therapeutic targets a specific protein (e.g. BACE1, tau, APP)."
        ),
        source_label="Drug",
        target_label="Protein",
        required_props=[],
        optional_props=[
            "action_type",       # inhibitor, antibody, agonist, modulator, etc.
            "is_primary_target", # bool-ish
            "source",            # AlzForum.Therapeutics, literature, etc.
            "target_notes",      # snippet from mechanism text
        ],
    ),

    # --- Drug -> Pathway (when target is more process-like) ----------
    "AFFECTS_PATHWAY": EdgeSchema(
        type="AFFECTS_PATHWAY",
        description=(
            "Therapeutic affects a biological pathway or process "
            "(e.g. neuroinflammation, amyloid processing)."
        ),
        source_label="Drug",
        target_label="Pathway",
        required_props=[],
        optional_props=[
            "action_type",
            "is_primary_target",
            "source",
            "target_notes",
        ],
    ),

    # --- Gene -> Protein ---------------------------------------------
    "ENCODES": EdgeSchema(
        type="ENCODES",
        description="Gene encodes a protein (HGNC / UniProt mapping).",
        source_label="Gene",
        target_label="Protein",
        required_props=[],
        optional_props=[
            "source",       # HGNC, UniProt
        ],
    ),

    # --- Protein -> Pathway ------------------------------------------
    "INVOLVED_IN_PATHWAY": EdgeSchema(
        type="INVOLVED_IN_PATHWAY",
        description="Protein participates in a biological process/pathway (GO).",
        source_label="Protein",
        target_label="Pathway",
        required_props=[],
        optional_props=[
            "evidence_code",   # GO evidence (EXP, IEA, TAS, etc.)
            "source",
        ],
    ),

    # --- Disease -> Phenotype (symptoms) -----------------------------
    "HAS_PHENOTYPE": EdgeSchema(
        type="HAS_PHENOTYPE",
        description="Disease presents with a given phenotype/symptom (HPO).",
        source_label="Disease",
        target_label="Phenotype",
        required_props=[],
        optional_props=[
            "onset",           # early, late, variable
            "frequency",       # common, rare, etc.
            "source",
        ],
    ),

    # === NEW: Mechanism / pathology bridge ===========================

    # Disease -> Mechanism (e.g. AD involves amyloid & tau pathology)
    "INVOLVES_PATHOLOGY": EdgeSchema(
        type="INVOLVES_PATHOLOGY",
        description="Disease involves a given pathophysiologic mechanism (e.g. amyloid, tau).",
        source_label="Disease",
        target_label="Mechanism",
        required_props=[],
        optional_props=[
            "role",         # primary, secondary, speculative
            "source",
        ],
    ),

    # Drug -> Mechanism (drug targets a mechanism / pathology class)
    "TARGETS_PATHOLOGY": EdgeSchema(
        type="TARGETS_PATHOLOGY",
        description="Therapeutic targets a pathophysiologic mechanism (e.g. amyloid-related, tau).",
        source_label="Drug",
        target_label="Mechanism",
        required_props=[],
        optional_props=[
            "action_type",      # antibody, small_molecule, gene_therapy...
            "is_primary_target",
            "source",
            "target_notes",
        ],
    ),

    # Biomarker -> Mechanism (biomarker reflects a mechanism)
    "REFLECTS_PATHOLOGY": EdgeSchema(
        type="REFLECTS_PATHOLOGY",
        description="Biomarker reflects a given pathophysiologic mechanism (e.g. amyloid biomarker).",
        source_label="Biomarker",
        target_label="Mechanism",
        required_props=[],
        optional_props=[
            "analyte_core",
            "analyte_class",
            "fluid",
            "source",
        ],
    ),

    # === NEW: AlzPedia-based gene associations =======================

    # AlzPediaEntity -> Gene (page represents a gene / protein)
    "REPRESENTS_GENE": EdgeSchema(
        type="REPRESENTS_GENE",
        description="AlzPedia entity corresponds to a specific gene.",
        source_label="AlzPediaEntity",
        target_label="Gene",
        required_props=[],
        optional_props=[
            "match_strategy",   # exact_symbol, synonym_match, fuzzy
            "source",
        ],
    ),

    # Gene -> Disease (associated via AlzPedia / genetics evidence)
    "ASSOCIATED_WITH_DISEASE": EdgeSchema(
        type="ASSOCIATED_WITH_DISEASE",
        description="Gene associated with a disease (e.g. AD risk gene).",
        source_label="Gene",
        target_label="Disease",
        required_props=[],
        optional_props=[
            "evidence_type",    # gwas, linkage, candidate_gene, etc.
            "source",
        ],
    ),

    # === NEW: Company / therapy type metadata ========================

    # Drug -> Company
    "DEVELOPED_BY": EdgeSchema(
        type="DEVELOPED_BY",
        description="Drug is/was developed or sponsored by a company.",
        source_label="Drug",
        target_label="Company",
        required_props=[],
        optional_props=[
            "role",        # sponsor, originator, partner
            "source",
        ],
    ),

    # Drug -> TherapyType
    "HAS_THERAPY_TYPE": EdgeSchema(
        type="HAS_THERAPY_TYPE",
        description="Drug has a given therapeutic modality (immunotherapy, small molecule, etc.).",
        source_label="Drug",
        target_label="TherapyType",
        required_props=[],
        optional_props=[
            "source",
        ],
    ),

    # === NEW: Biomarker measurement context ==========================

    # Biomarker -> Fluid
    "MEASURED_IN": EdgeSchema(
        type="MEASURED_IN",
        description="Biomarker is measured in a given biofluid (CSF, plasma, etc.).",
        source_label="Biomarker",
        target_label="Fluid",
        required_props=[],
        optional_props=[
            "source",
        ],
    ),

    # === NEW: Trial graph structure ==================================

    # Drug -> Trial
    "HAS_TRIAL": EdgeSchema(
        type="HAS_TRIAL",
        description="Drug has a clinical trial record for a given indication.",
        source_label="Drug",
        target_label="Trial",
        required_props=[],
        optional_props=[
            "source",
        ],
    ),

    # Trial -> Disease
    "FOR_DISEASE": EdgeSchema(
        type="FOR_DISEASE",
        description="Trial is for a specific disease / indication.",
        source_label="Trial",
        target_label="Disease",
        required_props=[],
        optional_props=[
            "indication_label",   # raw text from AlzForum
            "source",
        ],
    ),
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def get_node_schema(label: str) -> Optional[NodeSchema]:
    """Return the NodeSchema for a given label, or None if unknown."""
    return NODE_SCHEMAS.get(label)


def get_edge_schema(rel_type: str) -> Optional[EdgeSchema]:
    """Return the EdgeSchema for a given relationship type, or None if unknown."""
    return EDGE_SCHEMAS.get(rel_type)


def list_node_labels() -> List[str]:
    """Return all defined node labels."""
    return sorted(NODE_SCHEMAS.keys())


def list_edge_types() -> List[str]:
    """Return all defined relationship types."""
    return sorted(EDGE_SCHEMAS.keys())


if __name__ == "__main__":
    # Small self-check / debug printout
    print("Node labels:")
    for lbl in list_node_labels():
        ns = NODE_SCHEMAS[lbl]
        print(f"  - {lbl}: required={ns.required_props}, optional={len(ns.optional_props)} props")

    print("\nEdge types:")
    for et in list_edge_types():
        es = EDGE_SCHEMAS[et]
        print(
            f"  - {et}: {es.source_label} -> {es.target_label}, "
            f"required={es.required_props}, optional={len(es.optional_props)} props"
        )