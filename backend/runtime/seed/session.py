"""
runtime.seed.session
--------------------
Session lifecycle: clone and reset.

clone_session(session_id, user_id)
    Copies all TEMPLATE clinical + assignment rows into the new session.
    Creates the session row. Called on persona selection.

reset()
    Wipes all non-TEMPLATE session data. Re-seeds template if absent.
    Called on every boot so the public demo always starts fresh.
"""

from __future__ import annotations

import logging

from .db import transaction
from .seeder import seed, verify_template

logger = logging.getLogger(__name__)

TEMPLATE = "TEMPLATE"

# Tables that are cloned per-session (order respects FK deps)
_SESSION_TABLES = [
    "patients",
    "patient_assignments",
    "conditions",
    "vitals",
    "medications",
    "lab_results",
    "genetic_markers",
    "clinical_notes",
]

# Columns to copy for each session-scoped table (excludes identity/auto columns)
_TABLE_COLUMNS: dict[str, list[str]] = {
    "patients": [
        "patient_id", "name", "dob", "sex", "mrn",
        "address", "insurance_id", "department", "care_team", "headline",
    ],
    "patient_assignments": [
        "user_id", "patient_id", "relationship", "care_team",
    ],
    "conditions": [
        "patient_id", "code", "label", "onset_date", "status",
    ],
    "vitals": [
        "patient_id", "taken_at", "bp_systolic", "bp_diastolic",
        "heart_rate", "temp_c", "resp_rate",
    ],
    "medications": [
        "patient_id", "drug_name", "dose", "route",
        "frequency", "prescriber_user_id", "status",
    ],
    "lab_results": [
        "patient_id", "test_name", "value", "unit",
        "reference_range", "collected_at", "status", "entered_by_user_id",
    ],
    "genetic_markers": [
        "patient_id", "gene", "variant", "interpretation", "source",
    ],
    "clinical_notes": [
        "patient_id", "author_user_id", "created_at", "body",
    ],
}


def clone_session(session_id: str, user_id: str) -> None:
    """
    Copy all TEMPLATE rows into `session_id` and create the sessions row.
    The caller (persona selection endpoint) provides the new session_id and user_id.
    """
    with transaction() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions(session_id, user_id) VALUES(%s, %s)"
            " ON CONFLICT DO NOTHING",
            (session_id, user_id),
        )
        for table in _SESSION_TABLES:
            cols = _TABLE_COLUMNS[table]
            col_list = ", ".join(cols)
            cur.execute(
                f"INSERT INTO {table}(session_id, {col_list})"
                f" SELECT %s, {col_list} FROM {table} WHERE session_id = %s",
                (session_id, TEMPLATE),
            )
    logger.info("Session cloned: %s (user=%s)", session_id, user_id)


def reset() -> None:
    """
    On-boot cleanup:
      1. Delete all non-TEMPLATE data from session-scoped tables.
      2. Clear sessions, audit_log, break_glass_grants entirely.
      3. Re-seed template if rows are missing (idempotent).
    """
    with transaction() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM break_glass_grants")
        cur.execute("DELETE FROM audit_log")
        cur.execute("DELETE FROM sessions")
        for table in reversed(_SESSION_TABLES):
            cur.execute(f"DELETE FROM {table} WHERE session_id <> %s", (TEMPLATE,))
    logger.info("Reset complete: non-TEMPLATE sessions cleared.")

    if not verify_template():
        logger.info("Template rows missing — running seed.")
        seed()
    else:
        logger.info("Template verified OK.")
