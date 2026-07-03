"""
runtime.policy.audit
--------------------
Audit writer — every PDP decision and every data access writes one row.

Uses the app_runtime DB role for INSERT so the append-only grant is
exercised on every write (UPDATE/DELETE will be rejected at the DB level).
"""

from __future__ import annotations

from typing import Optional


def write_audit(
    conn,
    session_id: str,
    user_id: str,
    role_id: str,
    action: str,
    resource: str,
    patient_id: Optional[str],
    effect: str,
    reason: str,
    break_glass: bool = False,
    fields_accessed: Optional[list[str]] = None,
) -> None:
    """
    Append one row to audit_log via the app_runtime role.

    The caller passes an open connection (superuser). We switch to
    app_runtime + set the session variable, INSERT, then reset the role
    so the connection is left in a clean state for the caller.
    """
    cur = conn.cursor()
    cur.execute("SET ROLE app_runtime")
    # SET LOCAL does not accept parameterized placeholders; use set_config() instead.
    # is_local=true makes the setting transaction-local, same effect as SET LOCAL.
    cur.execute("SELECT set_config('app.session_id', %s, true)", (session_id,))
    cur.execute(
        """
        INSERT INTO audit_log
          (session_id, user_id, role_id, action, resource,
           patient_id, effect, reason, break_glass, fields_accessed)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            session_id, user_id, role_id, action, resource,
            patient_id, effect, reason, break_glass,
            fields_accessed or [],
        ),
    )
    cur.execute("RESET ROLE")
