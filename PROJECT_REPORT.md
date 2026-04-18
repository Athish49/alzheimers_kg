# Alzheimer's Knowledge Graph RAG — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Backend API Reference](#3-backend-api-reference)
4. [Intent Classification](#4-intent-classification)
5. [Retrieval Strategies](#5-retrieval-strategies)
6. [Knowledge Graph Schema](#6-knowledge-graph-schema)
7. [LLM & Configuration](#7-llm--configuration)
8. [Frontend](#8-frontend)
9. [Data Flow](#9-data-flow)
10. [Setup & Configuration](#10-setup--configuration)

---

## 1. Project Overview

**Alzheimer's Knowledge Graph RAG** is a Retrieval-Augmented Generation (RAG) system that answers clinical and research questions about Alzheimer's disease (AD). Instead of relying on a generic LLM, it queries a structured **Neo4j knowledge graph** containing curated AD entities (biomarkers, drugs, phenotypes, genes, pathways), retrieves the most relevant subgraph, and feeds that context to a **local Ollama LLM** to generate grounded, citation-anchored answers.

### Who it helps

| User | How it helps |
|------|-------------|
| Clinicians | Quickly look up biomarker profiles, phenotype onset/frequency, and approved drug options |
| Researchers | Explore trial pipelines, pathway mechanisms, and gene–protein associations |
| Caregivers / patients | Plain-language answers about symptoms, risk factors, and treatment options |
| Data scientists | Inspect structured graph context alongside model-generated answers for downstream analysis |

### Key design choices

- **Deterministic routing** — question intent is classified with keyword rules (no LLM call needed), so context selection is fast and reproducible.
- **Grounded answers** — the LLM is strictly instructed to extract from the retrieved context only, minimising hallucination.
- **Transparent metadata** — every response includes the detected intent, retrieval strategy, and (optionally) the raw context used.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────┐
│  React + Vite Frontend  (localhost:5173)                    │
│  SearchBar → App.jsx → MessageList                         │
└────────────────────────────────┬───────────────────────────┘
                                 │  POST /answer
                                 ▼
┌────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                          │
│                                                            │
│  pipeline.py ──► router.py ──► intents.py                  │
│       │                                                    │
│       ▼                                                    │
│  retriever.py  ◄──► Neo4j  (bolt://localhost:7687)         │
│       │                                                    │
│       ▼                                                    │
│  graph_to_text.py  (structured text for prompt)            │
│       │                                                    │
│       ▼                                                    │
│  llm_client.py ──► Ollama  (localhost:11434)               │
│                    model: llama3.2:3b                      │
└────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI + Uvicorn |
| Database | Neo4j (bolt driver) |
| LLM runtime | Ollama (`llama3.2:3b`) |
| NLP / entity detection | spaCy, scispacy, regex |
| Frontend | React 19 + Vite 7 |

---

## 3. Backend API Reference

Base URL: `http://localhost:8000`

CORS is enabled for `http://localhost:5173` (Vite dev server).

---

### POST `/answer`

The single entrypoint for all question-answering. Accepts a natural-language question, routes it to the appropriate graph retrieval strategy, and returns an LLM-generated answer with full metadata.

#### Request

**Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | `string` | yes | — | Natural-language question about Alzheimer's disease |
| `temperature` | `float` | no | `null` (uses config default `0.2`) | LLM sampling temperature (0.0 = deterministic) |
| `max_tokens` | `integer` | no | `400` | Maximum tokens in the generated answer |
| `return_context` | `boolean` | no | `false` | When `true`, includes the raw retrieved graph context in the response |

**Example request body:**

```json
{
  "question": "What CSF biomarkers are elevated in Alzheimer's disease?",
  "temperature": 0.1,
  "max_tokens": 500,
  "return_context": true
}
```

#### Response

**Content-Type:** `application/json`

| Field | Type | Always present | Description |
|-------|------|----------------|-------------|
| `answer` | `string` | yes | LLM-generated answer grounded in retrieved context |
| `intent_type` | `string` | yes | Detected question intent (see [Intent Classification](#4-intent-classification)) |
| `intent_notes` | `string \| null` | yes | Human-readable note from the classifier (e.g. detected entities) |
| `strategy` | `string` | yes | Retrieval strategy used (see [Retrieval Strategies](#5-retrieval-strategies)) |
| `context` | `string \| null` | only if `return_context: true` | Raw text context passed to the LLM |

**Intent type values:**

| Value | Meaning |
|-------|---------|
| `BIOMARKER` | Question about biomarkers, fluids, assays |
| `DRUG_TRIAL` | Question about drugs, clinical trials, therapies |
| `PHENOTYPE` | Question about symptoms or clinical phenotypes |
| `PATHWAY` | Question about biological pathways or mechanisms |
| `GENE_PROTEIN` | Question about genes or proteins |
| `GENERAL_AD` | General Alzheimer's disease question |
| `OTHER` | Does not match any specific AD category |

**Strategy values:**

| Value | When used |
|-------|----------|
| `AD_BIOMARKERS_V1` | Intent = `BIOMARKER` |
| `AD_PHENOTYPES_V1` | Intent = `PHENOTYPE` |
| `AD_DRUGS_PATHWAYS_V1` | Intent = `DRUG_TRIAL` or `PATHWAY` |
| `AD_GENES_GENERAL_V1` | Intent = `GENE_PROTEIN` or `GENERAL_AD` |
| `AD_GENERAL_V1` | Intent = `OTHER` |

**Example response (with `return_context: true`):**

```json
{
  "answer": "In cerebrospinal fluid (CSF), the most consistently elevated biomarkers in Alzheimer's disease are total tau (t-tau) and phosphorylated tau (p-tau), along with decreased amyloid-beta 42 (Aβ42). The Aβ42/Aβ40 ratio is also reduced and serves as a sensitive indicator of amyloid pathology.",
  "intent_type": "BIOMARKER",
  "intent_notes": "Detected biomarker focus with fluid keyword 'CSF'",
  "strategy": "AD_BIOMARKERS_V1",
  "context": "## AD Biomarkers by Fluid\n\n### CSF\n**Increased:** t-tau (effect_size: 2.1, p<0.001), p-tau181 ...\n**Decreased:** Aβ42 (effect_size: -1.8, p<0.001) ..."
}
```

**Example response (with `return_context: false` — default):**

```json
{
  "answer": "Lecanemab (brand name Leqembi) and donanemab are among the recently approved amyloid-targeting therapies for early Alzheimer's disease. Aducanumab received accelerated FDA approval but has since been withdrawn from the market.",
  "intent_type": "DRUG_TRIAL",
  "intent_notes": null,
  "strategy": "AD_DRUGS_PATHWAYS_V1",
  "context": null
}
```

#### Error responses

| HTTP status | Cause |
|-------------|-------|
| `422 Unprocessable Entity` | Missing required field (`question`) or invalid field types |
| `500 Internal Server Error` | Neo4j connection failure, Ollama timeout, or unhandled exception |

---

## 4. Intent Classification

Classification is **rule-based and deterministic** — no LLM is needed. The classifier in [`graph_rag/intents.py`](graph_rag/intents.py) scores the question against keyword sets for each intent category and returns the highest-scoring match.

### Detection logic

1. **Entity extraction** — regex patterns scan for ontology IDs embedded in the question:
   - `MONDO:xxxxxxx` → Disease
   - `CHEBI:xxxxxxx` → Chemical / Drug
   - `HP:xxxxxxx` → Phenotype (HPO term)
   - `GO:xxxxxxx` → Pathway (Gene Ontology)
   - `HGNC:xxxxxxx` or `PR:xxxxxxx` → Gene / Protein

2. **Keyword scoring** — each intent has a curated keyword list. The question is lowercased and scored by keyword frequency.

3. **Explicit bias** — direct mentions of biomarkers (e.g. "amyloid", "tau", "p-tau") or trial terms ("phase 3", "FDA approved") boost the relevant intent regardless of score.

### Keyword categories (examples)

| Intent | Example keywords |
|--------|-----------------|
| `BIOMARKER` | amyloid, tau, p-tau, csf, plasma, biomarker, blood test, aβ42 |
| `DRUG_TRIAL` | drug, treatment, therapy, clinical trial, phase 3, approved, lecanemab |
| `PHENOTYPE` | symptom, sign, memory loss, cognitive, dementia, agitation |
| `PATHWAY` | pathway, mechanism, signaling, cascade, mtor, apoptosis |
| `GENE_PROTEIN` | gene, protein, apoe, app, psen1, mutation, variant |
| `GENERAL_AD` | alzheimer, alzheimer's, ad, dementia, neurodegeneration |

---

## 5. Retrieval Strategies

Each strategy fetches a different subgraph from Neo4j and formats it as structured text for the LLM prompt.

### `AD_BIOMARKERS_V1`

**Used for:** Questions about biomarkers, fluid-based tests, diagnostic markers.

**Retrieves:**
- All `HAS_BIOMARKER` edges from the canonical AD node (`MONDO:0004975`)
- Properties: analyte name, analyte class, fluid, direction (increased/decreased/no_change), effect size, p-value, comparison group

**Formatted as:** Grouped by fluid (CSF → Plasma/Serum → Other), then by direction (Increased → Decreased → No change → Unknown).

**Example context excerpt:**
```
## AD Biomarkers by Fluid and Direction

### CSF
**Increased:**
- t-tau (Protein) | effect_size: 2.1 | p_value: <0.001
- p-tau181 (Phosphoprotein) | effect_size: 1.9 | p_value: <0.001

**Decreased:**
- Aβ42 (Peptide) | effect_size: -1.8 | p_value: <0.001
```

---

### `AD_PHENOTYPES_V1`

**Used for:** Questions about symptoms, clinical features, or disease presentation.

**Retrieves:**
- All `HAS_PHENOTYPE` edges from the AD node
- Properties: phenotype label, HPO ID, onset, frequency

**Formatted as:** Flat list with onset and frequency annotations.

**Example context excerpt:**
```
## Alzheimer's Disease Phenotypes

- Memory impairment (HP:0002354) | onset: Adult | frequency: Very frequent
- Disorientation (HP:0100543) | onset: Adult | frequency: Frequent
- Agitation (HP:0000718) | onset: Adult | frequency: Occasional
```

---

### `AD_DRUGS_PATHWAYS_V1`

**Used for:** Questions about treatments, clinical trials, or biological mechanisms/pathways.

**Retrieves:**
- All `TREATS` edges (Drug → AD): drug name, CHEBI ID, drug type, approval status, trial phase, trial count, indication
- All `AFFECTS_PATHWAY` edges (Drug → Pathway): action type, pathway name, GO ID, primary target flag

**Formatted as:**
- Drugs grouped by development status (Approved → Phase 3 → Phase 1–2 → Discontinued → Status unclear)
- Pathways grouped by drug, listing all pathway targets and action types

**Example context excerpt:**
```
## AD Drug Landscape

### Approved
- Lecanemab (CHEBI:170976) | Monoclonal antibody | Amyloid-targeting
- Donepezil (CHEBI:53289) | Small molecule | Cholinesterase inhibitor

### Phase 3
- Donanemab | Monoclonal antibody | ...

## Drug–Pathway Relationships

### Lecanemab
- Amyloid precursor processing pathway (GO:0042987) | action: inhibition | primary_target: true
```

---

### `AD_GENES_GENERAL_V1`

**Used for:** Gene/protein questions, or general AD questions not matched to a more specific strategy.

**Retrieves:**
- All `ENCODES` edges (Gene → Protein): gene symbol, HGNC ID, protein name, UniProt ID
- All `INVOLVED_IN_PATHWAY` edges (Protein → Pathway)
- All `ASSOCIATED_WITH_DISEASE` edges (Gene → AD): evidence type (GWAS/linkage/candidate)
- Full AD biomarker, drug, and phenotype context (same as above strategies)

**Formatted as:** Gene–Protein pairings (up to 15, prioritising AD risk genes like APOE, APP, PSEN1) plus all other sections.

---

### `AD_GENERAL_V1`

**Used for:** Questions classified as `OTHER` — catch-all for questions not matching any focused intent.

**Retrieves:** Full general AD context (all sections: biomarkers, drugs, phenotypes, gene–protein links).

---

## 6. Knowledge Graph Schema

The Neo4j graph is built from curated Alzheimer's data. Schema definitions live in [`kg_build/schema.py`](kg_build/schema.py).

### Node types

| Label | Required properties | Key optional properties |
|-------|--------------------|-----------------------|
| `Disease` | `id`, `label` | `mondo_id`, `umls_cui`, `icd10`, `synonyms`, `source` |
| `Biomarker` | `id`, `label` | `analyte`, `analyte_class`, `fluid`, `units`, `assay_type` |
| `Drug` | `id`, `label` | `chebi_id`, `drug_type`, `drug_class`, `status_overall`, `approved_regions` |
| `Phenotype` | `id`, `label` | `hpo_id`, `umls_cui`, `synonyms` |
| `Gene` | `id`, `label` | `hgnc_id`, `entrez_id`, `ensembl_id`, `chromosome` |
| `Protein` | `id`, `label` | `uniprot_id`, `hgnc_id`, `gene_symbol` |
| `Pathway` | `id`, `label` | `go_id`, `namespace` |
| `RiskFactor` | `id`, `label` | `category`, `direction` (`increased_risk` \| `protective`) |
| `Trial` | `id`, `label` | `indication`, `trial_phase_max`, `has_phase3`, `status`, `trial_count` |
| `Mechanism` | `id`, `label` | `category` (`amyloid` \| `tau` \| `other_neurotransmitters`), `description` |
| `Company` | `id`, `label` | `country` |
| `AlzPediaEntity` | `id`, `label` | `url`, `category`, `has_function_section`, `has_pathology_section` |

### Canonical AD node

The system anchors all disease-centric queries to:

```
id: "MONDO:0004975"
label: "Alzheimer disease"
```

### Edge types

| Relationship | Source → Target | Key properties |
|-------------|----------------|---------------|
| `HAS_BIOMARKER` | Disease → Biomarker | `direction` (required), `effect_size`, `p_value`, `comparison` |
| `TREATS` | Drug → Disease | `status`, `trial_phase_max`, `has_phase3`, `trial_count`, `indication` |
| `HAS_PHENOTYPE` | Disease → Phenotype | `onset`, `frequency` |
| `ENCODES` | Gene → Protein | `source` |
| `INVOLVED_IN_PATHWAY` | Protein → Pathway | `evidence_code` |
| `AFFECTS_PATHWAY` | Drug → Pathway | `action_type`, `is_primary_target` |
| `TARGETS_PROTEIN` | Drug → Protein | `action_type`, `is_primary_target` |
| `INCREASES_RISK_OF` | RiskFactor → Disease | `direction`, `effect_size` |
| `INVOLVES_PATHOLOGY` | Disease → Mechanism | `role` (`primary` \| `secondary` \| `speculative`) |
| `TARGETS_PATHOLOGY` | Drug → Mechanism | `action_type`, `is_primary_target` |
| `REFLECTS_PATHOLOGY` | Biomarker → Mechanism | `analyte_core`, `analyte_class` |
| `DEVELOPED_BY` | Drug → Company | `role` (`sponsor` \| `originator` \| `partner`) |
| `HAS_THERAPY_TYPE` | Drug → TherapyType | — |
| `MEASURED_IN` | Biomarker → Fluid | — |
| `HAS_TRIAL` | Drug → Trial | — |
| `FOR_DISEASE` | Trial → Disease | `indication_label` |
| `ASSOCIATED_WITH_DISEASE` | Gene → Disease | `evidence_type` (`gwas` \| `linkage` \| `candidate_gene`) |
| `REPRESENTS_GENE` | AlzPediaEntity → Gene | `match_strategy` |

---

## 7. LLM & Configuration

### LLM client (`graph_rag/llm_client.py`)

Wraps the Ollama HTTP API. Two methods are exposed:

**`chat(messages, system_prompt, temperature, top_p, num_ctx, max_tokens)`**
- Sends a raw multi-turn conversation to `${OLLAMA_BASE_URL}/chat`
- `messages` follows the OpenAI-style format: `[{"role": "user"|"assistant"|"system", "content": "..."}]`

**`simple_qa(question, context, system_prompt, temperature, max_tokens)`**
- Convenience wrapper used by the pipeline
- Constructs a single user message: `Context:\n{context}\n\nQuestion: {question}`
- Injects a strict system prompt that forbids hallucination: *"Answer only from the provided context. Do not add information not present in the context."*

### System prompt (used in pipeline)

The pipeline uses a domain-specific system prompt that instructs the model to:
- Be a knowledgeable Alzheimer's research assistant
- Extract and synthesise information only from the provided context
- Present findings clearly for clinical or research audiences
- Acknowledge explicitly when the context does not contain enough information to answer

### Configuration (`graph_rag/config.py`)

All settings are read from environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `12345678` | Neo4j password |
| `NEO4J_DB` | `neo4j` | Neo4j database name |
| `LLM_PROVIDER` | `ollama` | LLM backend (only `ollama` supported) |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/api` | Ollama API base URL |
| `LLM_TEMPERATURE` | `0.2` | Default sampling temperature |
| `TOP_P` | `0.9` | Top-p nucleus sampling |
| `NUM_CTX` | `4096` | LLM context window size (tokens) |
| `LLM_TIMEOUT` | `60` | HTTP timeout for Ollama requests (seconds) |
| `AD_DISEASE_ID` | `MONDO:0004975` | Canonical AD node ID in the graph |
| `GRAPH_RAG_MAX_HOPS` | `2` | Maximum graph traversal depth |
| `MAX_EDGES` | `300` | Maximum edges returned per query |
| `TOPK_BIOMARKERS` | `50` | Max biomarkers retrieved |
| `TOPK_DRUGS` | `50` | Max drugs retrieved |
| `TOPK_PATHWAYS` | `50` | Max pathways retrieved |
| `TOPK_GENES` | `50` | Max genes retrieved |

---

## 8. Frontend

**Location:** [`ui/`](ui/)  
**Stack:** React 19, Vite 7, Lucide icons  
**Dev server:** `http://localhost:5173`

### Components

| Component | File | Responsibility |
|-----------|------|---------------|
| `App` | [`ui/src/App.jsx`](ui/src/App.jsx) | Root state management, API call, session restart |
| `SearchBar` | [`ui/src/components/SearchBar.jsx`](ui/src/components/SearchBar.jsx) | Text input, submit on Enter or button click, loading state |
| `MessageList` | [`ui/src/components/MessageList.jsx`](ui/src/components/MessageList.jsx) | Chat bubble rendering, expandable context/intent/strategy details, auto-scroll |

### App state

```js
{
  hasStarted: boolean,     // false = landing screen shown
  messages: [              // chat history
    {
      role: "user" | "assistant",
      content: string,
      intent_type?: string,
      strategy?: string,
      context?: string
    }
  ],
  isLoading: boolean       // spinner while API call is in flight
}
```

### API call (in `App.jsx`)

```js
const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/answer`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, return_context: true })
});
```

`VITE_API_BASE_URL` defaults to `http://localhost:8000` if not set in `.env`.

---

## 9. Data Flow

```
1. User types question in SearchBar
          │
          ▼
2. App.jsx POSTs { question } to POST /answer
          │
          ▼
3. FastAPI receives QuestionRequest
          │
          ▼
4. router.py → intents.py
   classify_question(question)
   → IntentType + detected entities + notes
          │
          ▼
5. router.py selects context builder:
   BIOMARKER        → build_biomarker_direction_context()
   PHENOTYPE        → build_phenotype_context()
   DRUG_TRIAL/PATH  → build_drug_trial_pathway_context()
   GENE_PROTEIN/    → build_general_ad_context()
   GENERAL_AD
   OTHER            → build_general_ad_context()
          │
          ▼
6. retriever.py runs Cypher queries against Neo4j
   get_ad_biomarkers()    → HAS_BIOMARKER edges
   get_ad_drugs()         → TREATS edges
   get_ad_phenotypes()    → HAS_PHENOTYPE edges
   get_ad_drug_pathways() → AFFECTS_PATHWAY edges
   get_genes_and_proteins()→ ENCODES + INVOLVED_IN_PATHWAY edges
          │
          ▼
7. graph_to_text.py formats raw records → structured text
   summarize_biomarkers()     → fluid/direction groups
   summarize_drugs()          → status groups
   summarize_phenotypes()     → annotated list
   summarize_drug_pathways()  → per-drug pathway list
   summarize_genes_proteins() → gene→protein pairs
          │
          ▼
8. pipeline.py builds LLM prompt:
   system_prompt + context_text + question
          │
          ▼
9. llm_client.py POSTs to Ollama /api/chat
   model: llama3.2:3b
   temperature: 0.2 (default)
          │
          ▼
10. LLM generates answer grounded in context
          │
          ▼
11. FastAPI returns AnswerResponse JSON
          │
          ▼
12. MessageList.jsx renders answer + expandable metadata
```

---

## 10. Setup & Configuration

### Prerequisites

| Service | Version | Notes |
|---------|---------|-------|
| Python | ≥ 3.10 | |
| Neo4j | ≥ 5.x | Must have AD knowledge graph loaded |
| Ollama | latest | Must have `llama3.2:3b` pulled |
| Node.js | ≥ 18 | For frontend only |

### Backend setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (or create a .env file)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export OLLAMA_BASE_URL=http://localhost:11434/api

# Start the API server
uvicorn graph_rag.pipeline:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend setup

```bash
cd ui
npm install

# Optional: set API base URL
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

npm run dev        # dev server at localhost:5173
npm run build      # production build → ui/dist/
```

### Ollama model

```bash
ollama pull llama3.2:3b
ollama serve       # starts on port 11434 by default
```

### Quick health check

```bash
# Backend
curl http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Alzheimer disease?"}'

# Expected: JSON with answer, intent_type, strategy fields
```

---

*Generated from source code analysis. For the latest schema changes, refer to [`kg_build/schema.py`](kg_build/schema.py) and [`graph_rag/pipeline.py`](graph_rag/pipeline.py).*
