"""
graph_rag.config
----------------

Central configuration for the Graph-RAG layer.

Reads from backend/.env (loaded relative to this file, so CWD doesn't matter).
All values can also be overridden with real environment variables, which take
precedence over .env.

Auto-detection
--------------
ENVIRONMENT
    Explicitly set "local" or "cloud" in .env.
    If unset, detected from cloud-platform env vars:
        RENDER, RAILWAY_ENVIRONMENT, HEROKU_APP_NAME, GAE_ENV,
        AWS_LAMBDA_FUNCTION_NAME, WEBSITE_SITE_NAME (Azure),
        K_SERVICE / GOOGLE_CLOUD_PROJECT (GCP Cloud Run),
        FLY_APP_NAME
    Falls back to "local" if none found.

LLM_PROVIDER
    Explicitly set in .env, or auto-selected:
        local  → ollama
        cloud  → first provider whose API key is present:
                 anthropic → openai → groq → gemini
                 If no key at all → groq (free tier; llm_client raises a
                 clear error at startup asking for GROQ_API_KEY or GEMINI_API_KEY).

NEO4J
    local  → NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DB
    cloud  → CLOUD_NEO4J_URI / CLOUD_NEO4J_USER /
             CLOUD_NEO4J_PASSWORD / CLOUD_NEO4J_DB

URLs
    local  → http://localhost:5173 / http://localhost:8000
    cloud  → FRONTEND_URL / BACKEND_URL (must be set explicitly)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Load .env relative to this file (backend/.env), regardless of CWD
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)  # real env vars always win

logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    """Return env var value, falling back to default when empty or unset."""
    return os.getenv(key, "").strip() or default


# ---------------------------------------------------------------------------
# Known cloud-platform sentinel env vars
# ---------------------------------------------------------------------------

_CLOUD_SENTINELS = [
    "RENDER",
    "RAILWAY_ENVIRONMENT",
    "HEROKU_APP_NAME",
    "GAE_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_EXECUTION_ENV",
    "WEBSITE_SITE_NAME",    # Azure App Service
    "K_SERVICE",            # GCP Cloud Run
    "GOOGLE_CLOUD_PROJECT",
    "FLY_APP_NAME",
]

# Default models per provider
_DEFAULT_MODELS: dict[str, str] = {
    "ollama":    "llama3.2:3b",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "groq":      "llama-3.1-8b-instant",
    "gemini":    "gemini-2.5-flash",
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_environment() -> str:
    """Return 'local' or 'cloud'."""
    explicit = _env("ENVIRONMENT").lower()
    if explicit in ("local", "cloud"):
        return explicit
    if any(os.getenv(var) for var in _CLOUD_SENTINELS):
        logger.info("Cloud environment detected from platform env vars.")
        return "cloud"
    return "local"


def _detect_llm_provider(environment: str) -> str:
    """Return the LLM provider to use."""
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        return explicit

    if environment == "local":
        return "ollama"

    # Cloud: pick the first provider with a key present
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("GEMINI_API_KEY"):
        return "gemini"

    # No keys at all — default to groq (free tier).
    # get_llm_client() will raise a descriptive error so the user knows
    # what to set in .env.
    logger.warning(
        "No LLM API key found in .env. Defaulting provider to 'groq'. "
        "Set GROQ_API_KEY or GEMINI_API_KEY in backend/.env for cloud use."
    )
    return "groq"


def _resolve_api_key(provider: str) -> str:
    """Return the API key for the given provider."""
    key_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq":      "GROQ_API_KEY",
        "gemini":    "GEMINI_API_KEY",
    }
    # Provider-specific key takes priority, then the generic LLM_API_KEY
    return _env(key_map.get(provider, "")) or _env("LLM_API_KEY")


def _detect_fallback_provider(primary: str) -> tuple[str, str, str]:
    """
    Return (provider, model, api_key) for a fallback LLM.

    Picks the first cloud provider (other than `primary`) that has a key set.
    Returns ('', '', '') if no fallback is available.
    """
    _key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "groq":      "GROQ_API_KEY",
        "gemini":    "GEMINI_API_KEY",
    }
    for provider in ("anthropic", "openai", "groq", "gemini"):
        if provider == primary:
            continue
        key = _env(_key_map[provider])
        if key:
            return provider, _DEFAULT_MODELS[provider], key
    return "", "", ""


# ---------------------------------------------------------------------------
# Resolve all values
# ---------------------------------------------------------------------------

ENVIRONMENT:  str = _detect_environment()
LLM_PROVIDER: str = _detect_llm_provider(ENVIRONMENT)

# LLM
LLM_MODEL:       str   = _env("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "llama3.2:3b")
LLM_API_KEY:     str   = _resolve_api_key(LLM_PROVIDER)

# Fallback LLM (used when primary provider is rate-limited)
_fb = _detect_fallback_provider(LLM_PROVIDER)
LLM_FALLBACK_PROVIDER: str = _fb[0]
LLM_FALLBACK_MODEL:    str = _fb[1]
LLM_FALLBACK_API_KEY:  str = _fb[2]
OLLAMA_BASE_URL: str   = _env("OLLAMA_BASE_URL", "http://localhost:11434/api")
LLM_TEMPERATURE: float = float(_env("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P:       float = float(_env("LLM_TOP_P",       "0.9"))
LLM_NUM_CTX:     int   = int(_env("LLM_NUM_CTX",       "4096"))
LLM_MAX_TOKENS:  int   = int(_env("LLM_MAX_TOKENS",    "800"))
LLM_TIMEOUT_S:   int   = int(_env("LLM_TIMEOUT_S",     "60"))

# Neo4j — local vs cloud
if ENVIRONMENT == "cloud":
    NEO4J_URI:      str = _env("CLOUD_NEO4J_URI",      "")
    NEO4J_USER:     str = _env("CLOUD_NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD: str = _env("CLOUD_NEO4J_PASSWORD", "")
    NEO4J_DB:       str = _env("CLOUD_NEO4J_DB",       "neo4j")
else:
    NEO4J_URI:      str = _env("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER:     str = _env("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD: str = _env("NEO4J_PASSWORD", "12345678")
    NEO4J_DB:       str = _env("NEO4J_DB",       "neo4j")

# URLs — local vs cloud
if ENVIRONMENT == "cloud":
    FRONTEND_URL: str = _env("FRONTEND_URL", "")
    BACKEND_URL:  str = _env("BACKEND_URL",  "")
else:
    FRONTEND_URL: str = _env("FRONTEND_URL", "http://localhost:5173")
    BACKEND_URL:  str = _env("BACKEND_URL",  "http://localhost:8000")

# Graph RAG constants
PROJECT_NAME:          str = _env("PROJECT_NAME",              "alzheimerskg")
DEFAULT_AD_DISEASE_ID: str = _env("AD_DISEASE_ID",             "MONDO:0004975")
DEFAULT_MAX_HOPS:      int  = int(_env("GRAPH_RAG_MAX_HOPS",   "2"))
DEFAULT_MAX_EDGES:     int  = int(_env("GRAPH_RAG_MAX_EDGES",  "300"))
TOPK_BIOMARKERS:       int  = int(_env("GRAPH_RAG_TOPK_BIOMARKERS", "50"))
TOPK_DRUGS:            int  = int(_env("GRAPH_RAG_TOPK_DRUGS",      "50"))
TOPK_PATHWAYS:         int  = int(_env("GRAPH_RAG_TOPK_PATHWAYS",   "50"))
TOPK_GENES:            int  = int(_env("GRAPH_RAG_TOPK_GENES",      "50"))


# ---------------------------------------------------------------------------
# Aggregate config dataclass (same public interface as before)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    # Environment
    environment:  str = ENVIRONMENT
    llm_provider: str = LLM_PROVIDER

    # LLM
    llm_model:       str   = LLM_MODEL
    llm_api_key:     str   = LLM_API_KEY
    ollama_base_url: str   = OLLAMA_BASE_URL
    llm_temperature: float = LLM_TEMPERATURE
    llm_top_p:       float = LLM_TOP_P
    llm_num_ctx:     int   = LLM_NUM_CTX
    llm_timeout:     int   = LLM_TIMEOUT_S

    # Fallback LLM
    llm_fallback_provider: str = LLM_FALLBACK_PROVIDER
    llm_fallback_model:    str = LLM_FALLBACK_MODEL
    llm_fallback_api_key:  str = LLM_FALLBACK_API_KEY

    # Neo4j
    neo4j_uri:      str = NEO4J_URI
    neo4j_user:     str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD
    neo4j_db:       str = NEO4J_DB

    # URLs
    frontend_url: str = FRONTEND_URL
    backend_url:  str = BACKEND_URL

    # Graph RAG
    project_name:  str = PROJECT_NAME
    ad_disease_id: str = DEFAULT_AD_DISEASE_ID
    max_hops:      int = DEFAULT_MAX_HOPS
    max_edges:     int = DEFAULT_MAX_EDGES


CONFIG = AppConfig()

logger.info(
    "Config loaded | environment=%s | llm_provider=%s | model=%s | neo4j=%s",
    CONFIG.environment,
    CONFIG.llm_provider,
    CONFIG.llm_model,
    CONFIG.neo4j_uri,
)
