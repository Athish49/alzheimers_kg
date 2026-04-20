"""
main.py
-------
FastAPI entry point for the Alzheimer Graph RAG API.

Start locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Render / Docker:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from graph_rag.pipeline import get_pipeline, QuestionRequest, AnswerResponse
from graph_rag.llm_client import LLMUnavailableError

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

VERSION = "1.0.0"
START_TIME: datetime = datetime.now(timezone.utc)

app = FastAPI(
    title="Alzheimer Graph RAG API",
    description="Answer Alzheimer-related questions using the knowledge graph + LLM.",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    start_time: str
    uptime_seconds: float


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health_check() -> HealthResponse:
    """Returns deployment status, version, and uptime."""
    now = datetime.now(timezone.utc)
    uptime = (now - START_TIME).total_seconds()
    return HealthResponse(
        status="ok",
        version=VERSION,
        start_time=START_TIME.isoformat(),
        uptime_seconds=round(uptime, 2),
    )


# ---------------------------------------------------------------------------
# Q&A endpoint
# ---------------------------------------------------------------------------


@app.post("/answer", tags=["rag"])
def answer_question(payload: QuestionRequest):
    pipe = get_pipeline()
    try:
        res = pipe.answer(
            question=payload.question,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            return_context=payload.return_context,
        )
    except LLMUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "The AI service is temporarily unavailable due to high demand. "
                    "Please try again in a moment."
                )
            },
        )
    return AnswerResponse(
        answer=res["answer"],
        intent_type=res["intent_type"],
        intent_notes=res["intent_notes"],
        strategy=res["strategy"],
        context=res.get("context"),
        evidence=res.get("evidence"),
    )


# ---------------------------------------------------------------------------
# Local dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
