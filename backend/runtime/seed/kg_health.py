"""
runtime.seed.kg_health
----------------------
Boot-time check that the existing alzheimerskg graph is reachable and
that the five key nodes the Knowledge service depends on resolve via the
existing entity linker.

No Neo4j writes. No PHI crosses into this check.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ["APOE", "abeta42", "p-tau181", "donepezil", "lecanemab"]


def check_kg_health() -> bool:
    """
    Return True if the alzheimerskg graph is reachable and all required key
    nodes resolve. Logs warnings (does not raise) so a KG outage doesn't
    prevent the rest of the system from starting.
    """
    try:
        from graph_rag.entity_linker import get_entity_linker
        linker = get_entity_linker()
    except Exception as exc:
        logger.warning("KG health: entity linker unavailable — %s", exc)
        return False

    missing = []
    for key in _REQUIRED_KEYS:
        matches = linker.match_entities(key)
        if not matches:
            missing.append(key)
        else:
            logger.info(
                "KG health: '%s' → %d node(s) (%s)",
                key,
                len(matches),
                ", ".join(m.node_id for m in matches),
            )

    if missing:
        logger.warning("KG health: key nodes NOT found in graph: %s", missing)
        return False

    logger.info("KG health check passed — all %d key nodes resolved.", len(_REQUIRED_KEYS))
    return True
