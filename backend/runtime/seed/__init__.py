"""
runtime.seed
------------
Public API for the seed / startup module.

startup_reset()  — called once on boot from main.py:
  1. apply_schema()    (idempotent DDL)
  2. reset()           (wipe non-TEMPLATE sessions, re-verify template)
  3. check_kg_health() (warn if KG nodes missing)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def startup_reset() -> None:
    """Idempotent boot sequence for the runtime plane."""
    from .schema import apply_schema
    from .session import reset
    from .kg_health import check_kg_health

    logger.info("Runtime startup: applying schema...")
    apply_schema()

    logger.info("Runtime startup: resetting sessions...")
    reset()

    logger.info("Runtime startup: checking knowledge graph...")
    check_kg_health()

    logger.info("Runtime startup complete.")
