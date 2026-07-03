"""
runtime.seed.db
---------------
Database connection factory for the runtime plane.

Superuser connections (neondb_owner) are used for migrations, seeding, and
session cloning — the table owner bypasses RLS so the TEMPLATE rows are
accessible without any session variable.

App-role connections (SET ROLE app_runtime) are used for every per-request
operation; they are subject to RLS enforced by session variables.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg import Connection

from graph_rag.config import CONFIG

logger = logging.getLogger(__name__)


def get_conn(autocommit: bool = False) -> Connection:
    """
    Open a superuser (neondb_owner) connection.
    Caller is responsible for closing it or using as a context manager.
    """
    if not CONFIG.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to backend/.env and to Render env vars."
        )
    return psycopg.connect(CONFIG.database_url, autocommit=autocommit)


@contextmanager
def transaction() -> Generator[Connection, None, None]:
    """Superuser connection in a transaction block (auto-commit on success, rollback on error)."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
