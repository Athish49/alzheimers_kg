"""
runtime.router
--------------

Top-level FastAPI router for the enterprise runtime plane.
Mounted on the existing app in backend/main.py.

New routes added here as phases complete:
  GET  /runtime/health       — runtime plane liveness (this item)
  POST /personas             — Phase 2.1 (JWT mint)
  POST /orchestrate          — Phase 4.1 (tool-call loop)
  POST /break-glass          — Phase 2.3
  GET  /audit                — Phase 2.4
"""

from fastapi import APIRouter

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/health")
def runtime_health() -> dict:
    """Runtime plane liveness check."""
    return {"status": "ok", "plane": "runtime"}
