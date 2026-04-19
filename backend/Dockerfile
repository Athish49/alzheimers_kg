# =============================================================================
# Alzheimer KG — Backend API
# =============================================================================
# Serves: graph_rag.pipeline:app  (FastAPI on port 8000)
#
# Build:
#   docker build -t alzkg-backend .
#
# Run (local, Neo4j also local via host.docker.internal):
#   docker run -p 8000:8000 --env-file .env alzkg-backend
#
# Run (cloud — pass cloud Neo4j + LLM key):
#   docker run -p 8000:8000 \
#     -e ENVIRONMENT=cloud \
#     -e CLOUD_NEO4J_URI=neo4j+s://... \
#     -e CLOUD_NEO4J_USER=... \
#     -e CLOUD_NEO4J_PASSWORD=... \
#     -e CLOUD_NEO4J_DB=... \
#     -e LLM_PROVIDER=groq \
#     -e GROQ_API_KEY=... \
#     alzkg-backend
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies — installed before copying app code so this layer
# is cached across code-only changes
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application code
# (.dockerignore excludes venv/, .env, kg_build/output/, neo4j_import/, etc.)
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

# Health check: hit the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Render injects $PORT; fall back to 8000 for local Docker runs
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
