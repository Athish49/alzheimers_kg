# Alzheimer's Knowledge Graph RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about Alzheimer's disease by combining a structured Knowledge Graph (Neo4j) with a Large Language Model (Ollama).

## Overview

This project builds a knowledge graph from various Alzheimer's disease datasets (AlzForum, etc.) and uses it to ground LLM responses. It features:
- **Knowledge Graph Construction**: Scripts to normalize entities and build edges (Biomarkers, Drugs, Genes, etc.).
- **Graph RAG Pipeline**: A Python backend that routes questions, retrieves graph context, and generates answers.
- **Interactive UI**: A React + Vite frontend for chatting with the system.

## Repository Structure

- `kg_build/`: Scripts to build the Knowledge Graph (normalization, edge creation, export).
- `graph_rag/`: The RAG pipeline (FastAPI backend, Retriever, Router, LLM Client).
- `ui/`: React + Vite frontend application.
- `neo4j_import/`: Output directory for CSVs ready for Neo4j import.
- `alzforum/`: Source data directories (processed data).

## Prerequisites

- **Python 3.8+**
- **Node.js & npm**
- **Neo4j Database** (Local instance recommended)
- **Ollama** (Local LLM server)
    - Model: `llama3.2:3b` (default)

## Installation & Setup

### 1. Backend Setup (Python)

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend Setup (React)

Navigate to the `ui` directory and install dependencies:

```bash
cd ui
npm install
cd ..
```

### 3. Database & LLM Setup

1.  **Neo4j**: Ensure your Neo4j database is running.
    - Default URI: `bolt://localhost:7687`
    - Default Auth: `neo4j` / `12345678` (Change in `graph_rag/config.py` or via env vars if needed).
2.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the model:
    ```bash
    ollama pull llama3.2:3b
    ollama serve
    ```

### 4. Data Acquisition & Processing

Before building the graph, you must download and process the source data. Run these scripts in the following order:

**Ontologies**:
```bash
python -m ontology.download_ontologies
python -m ontology.process_ontologies
```

**AlzForum Data**:
```bash
python -m alzforum.download_alzforum
python -m alzforum.process_alzbiomarker
python -m alzforum.process_alzpedia
python -m alzforum.process_therapeutics
python -m alzforum.process_therapeutic_details
```

### 5. Knowledge Graph Construction

Build the graph artifacts and import them into Neo4j. Run these in order:

1.  **Normalize Entities**:
    ```bash
    python -m kg_build.normalize_entities
    ```
2.  **Build Edges**:
    ```bash
    python -m kg_build.build_edges
    ```
3.  **Export for Neo4j**:
    ```bash
    python -m kg_build.export_neo4j
    ```
4.  **Import to Neo4j**:
    - Copy the CSV files from `neo4j_import/` to your Neo4j `import` directory.
    - Use `LOAD CSV` commands or `neo4j-admin database import` to load the data.
    - *Note: See `kg_build/export_neo4j.py` output for specific import instructions.*

## Usage

### Start the Backend

From the project root (this spins up the FastAPI server):

```bash
python -m graph_rag.pipeline
```
The API will be available at `http://localhost:8000`.

### Start the Frontend

In a separate terminal, from the `ui` directory:

```bash
cd ui
npm run dev
```
Open your browser to the URL shown (usually `http://localhost:5173`).

## Configuration

Key configurations can be found in `graph_rag/config.py`. You can override them using environment variables:
- `NEO4J_URI`, `NEO4J_PASSWORD`
- `OLLAMA_BASE_URL`, `LLM_MODEL`
