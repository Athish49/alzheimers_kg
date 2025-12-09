"""
graph_rag.config
----------------

Central configuration for the Graph-RAG layer.

This version is configured to use a *local* Ollama model by default:

    - provider: "ollama"
    - model:    "llama3.2:3b"
    - base_url: "http://localhost:11434/api"  (default Ollama endpoint)

You can override anything via env vars without changing code:

    export OLLAMA_BASE_URL="http://localhost:11434/api"
    export LLM_MODEL="llama3.2:3b"

Neo4j connection is also fully env-driven:

    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="your-password"
    export NEO4J_DB="neo4j"
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Neo4j configuration
# ---------------------------------------------------------------------


NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DB: str = os.getenv("NEO4J_DB", "neo4j")


# ---------------------------------------------------------------------
# LLM configuration – tuned for local Ollama
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    """
    Generic LLM configuration.

    For this project we default to a local Ollama server running llama3.2:3b.
    Other parts of the code will use this config to call the Ollama HTTP API.
    """

    provider: str = "ollama"      # "ollama" by default
    model: str = "llama3.2:3b"    # your local model tag
    # IMPORTANT: include /api so llm_client can call `${base_url}/chat`
    base_url: str = "http://localhost:11434/api"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 4096
    max_tokens: int = 800
    timeout_s: int = 60


LLM_CONFIG = LLMConfig(
    provider=os.getenv("LLM_PROVIDER", "ollama"),
    model=os.getenv("LLM_MODEL", "llama3.2:3b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    top_p=float(os.getenv("LLM_TOP_P", "0.9")),
    num_ctx=int(os.getenv("LLM_NUM_CTX", "4096")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
    timeout_s=int(os.getenv("LLM_TIMEOUT_S", "60")),
)


# ---------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------


# Canonical MONDO ID for Alzheimer disease (matches your nodes_disease.csv)
DEFAULT_AD_DISEASE_ID: str = os.getenv("AD_DISEASE_ID", "MONDO:0004975")

# How many neighbors / paths to pull by default for local subgraphs
DEFAULT_MAX_HOPS: int = int(os.getenv("GRAPH_RAG_MAX_HOPS", "2"))
DEFAULT_MAX_EDGES: int = int(os.getenv("GRAPH_RAG_MAX_EDGES", "300"))

# Default top-k limits for different retrievers
TOPK_BIOMARKERS: int = int(os.getenv("GRAPH_RAG_TOPK_BIOMARKERS", "50"))
TOPK_DRUGS: int = int(os.getenv("GRAPH_RAG_TOPK_DRUGS", "50"))
TOPK_PATHWAYS: int = int(os.getenv("GRAPH_RAG_TOPK_PATHWAYS", "50"))
TOPK_GENES: int = int(os.getenv("GRAPH_RAG_TOPK_GENES", "50"))


# ---------------------------------------------------------------------
# Convenience aggregate (used by the rest of the code)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    # Neo4j
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD
    neo4j_db: str = NEO4J_DB

    # Keep the nested LLM config for flexibility
    llm: LLMConfig = LLM_CONFIG

    # Also expose flat attributes used by llm_client.get_llm_client()
    ollama_base_url: str = LLM_CONFIG.base_url
    llm_model: str = LLM_CONFIG.model
    llm_temperature: float = LLM_CONFIG.temperature
    llm_top_p: float = LLM_CONFIG.top_p
    llm_num_ctx: int = LLM_CONFIG.num_ctx
    llm_timeout: int = LLM_CONFIG.timeout_s

    # Graph defaults
    ad_disease_id: str = DEFAULT_AD_DISEASE_ID
    max_hops: int = DEFAULT_MAX_HOPS
    max_edges: int = DEFAULT_MAX_EDGES


CONFIG = AppConfig()