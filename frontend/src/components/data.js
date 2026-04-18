export const categories = [
  {
    id: "BIOMARKER", label: "Biomarkers",
    blurb: "Fluids, effect sizes, p-values",
    prompts: [
      "What CSF biomarkers are elevated in Alzheimer's disease?",
      "Which plasma biomarkers are decreased in AD?",
    ],
  },
  {
    id: "DRUG_TRIAL", label: "Drug & Trials",
    blurb: "Status, phase, CHEBI IDs",
    prompts: [
      "Which amyloid-targeting drugs have received FDA approval?",
      "What drugs are currently in Phase 3 trials for AD?",
    ],
  },
  {
    id: "GENE_PROTEIN", label: "Genes & Proteins",
    blurb: "HGNC, UniProt, GWAS evidence",
    prompts: [
      "Which genes are most strongly associated with Alzheimer's disease by GWAS?",
      "What protein does APOE encode?",
    ],
  },
  {
    id: "PHENOTYPE", label: "Phenotypes",
    blurb: "HPO IDs, onset, frequency",
    prompts: [
      "What are the most frequent symptoms of Alzheimer's disease?",
      "At what stage does agitation typically appear?",
    ],
  },
  {
    id: "PATHWAY", label: "Pathways",
    blurb: "GO IDs, action types",
    prompts: [
      "Which biological pathways does lecanemab affect?",
      "What is the role of amyloid precursor processing in AD?",
    ],
  },
  {
    id: "GENERAL_AD", label: "General AD",
    blurb: "Composite multi-facet views",
    prompts: [
      "Give me an overview of the Alzheimer's disease landscape.",
      "What are the main disease mechanisms in AD?",
    ],
  },
];
