"""
Pytest configuration for runtime tests.

Provides a module-scoped `session_id` fixture that:
  1. Clones the TEMPLATE into a fresh test sandbox session
  2. Yields (session_id, open_conn) for test use
  3. Deletes all test session data on teardown
"""
from __future__ import annotations

import uuid
import pytest

from runtime.seed.session import clone_session
from runtime.seed.db import get_conn


TEST_USER = "u_014"  # Sarah Chen (attending_physician) — used for the session owner


@pytest.fixture(scope="module")
def test_session():
    """Clone TEMPLATE into a unique test session; clean up after all tests."""
    sid = "test_" + uuid.uuid4().hex[:12]
    clone_session(sid, TEST_USER)

    conn = get_conn()
    try:
        yield sid, conn
    finally:
        conn.close()
        # Tear down: wipe all rows belonging to this test session
        cleanup = get_conn()
        try:
            cur = cleanup.cursor()
            cur.execute("DELETE FROM break_glass_grants WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM audit_log WHERE session_id = %s", (sid,))
            for table in [
                "clinical_notes", "genetic_markers", "lab_results",
                "medications", "vitals", "conditions",
                "patient_assignments", "patients",
            ]:
                cur.execute(f"DELETE FROM {table} WHERE session_id = %s", (sid,))
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (sid,))
            cleanup.commit()
        finally:
            cleanup.close()
