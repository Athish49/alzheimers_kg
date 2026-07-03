"""
runtime.gateway
---------------
In-process rate limiting and LLM usage caps. No access decisions made here.

Caps enforced:
  - Per-session LLM call count (resets with the session sandbox).
  - Per-IP LLM call count per hour.
  - Global daily LLM call count (resets at UTC midnight).

All counters live in process memory. On a Render free-tier single-worker
deployment this is correct; on cold-start the counters reset, which is
fine — the "self-heals on spin-down" property means daily usage resets too.

Non-LLM routes (/personas, /audit, /break-glass) never call check_cap,
so the access-control and audit paths remain available even when capped.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from graph_rag.config import CONFIG


class CapExceededError(RuntimeError):
    """Raised when any LLM usage cap has been hit."""

    COOLDOWN_MESSAGE = (
        "The demo is cooling down — the LLM call limit has been reached for now. "
        "Audit, chart viewing, and break-glass still work. "
        "Try again shortly or pick a different persona to start a fresh session."
    )


# ---------------------------------------------------------------------------
# In-memory counter state
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# session_id → count
_session_counts: dict[str, int] = defaultdict(int)

# ip → (count, hour_start_utc)
_ip_counts: dict[str, tuple[int, int]] = {}

# (count, date_str)
_global_state: list = [0, _today := datetime.now(timezone.utc).date().isoformat()]


def _utc_hour() -> int:
    return datetime.now(timezone.utc).hour


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_cap(session_id: str, ip_address: Optional[str]) -> None:
    """
    Raise CapExceededError if the next LLM call would exceed any cap.
    Call this BEFORE each LLM synthesis call; call record_call AFTER success.
    """
    ip = ip_address or "unknown"
    with _lock:
        # Per-session cap
        if _session_counts[session_id] >= CONFIG.cap_llm_per_session:
            raise CapExceededError(CapExceededError.COOLDOWN_MESSAGE)

        # Per-IP per-hour cap
        if ip in _ip_counts:
            count, hour = _ip_counts[ip]
            if hour == _utc_hour() and count >= CONFIG.cap_llm_per_ip_per_hour:
                raise CapExceededError(CapExceededError.COOLDOWN_MESSAGE)

        # Global daily cap
        today = _utc_date_str()
        if _global_state[1] != today:
            _global_state[0] = 0
            _global_state[1] = today
        if _global_state[0] >= CONFIG.cap_llm_global_per_day:
            raise CapExceededError(CapExceededError.COOLDOWN_MESSAGE)


def record_call(session_id: str, ip_address: Optional[str]) -> None:
    """Increment all usage counters. Call this AFTER a successful LLM synthesis call."""
    ip = ip_address or "unknown"
    with _lock:
        _session_counts[session_id] += 1

        hour = _utc_hour()
        if ip in _ip_counts:
            count, stored_hour = _ip_counts[ip]
            if stored_hour == hour:
                _ip_counts[ip] = (count + 1, hour)
            else:
                _ip_counts[ip] = (1, hour)
        else:
            _ip_counts[ip] = (1, hour)

        today = _utc_date_str()
        if _global_state[1] != today:
            _global_state[0] = 0
            _global_state[1] = today
        _global_state[0] += 1


def reset_session(session_id: str) -> None:
    """Clear the per-session counter. Called when a new sandbox is cloned."""
    with _lock:
        _session_counts.pop(session_id, None)
