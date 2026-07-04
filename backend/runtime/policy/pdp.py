"""
runtime.policy.pdp
------------------
Policy Decision Point (PDP).

decide(claims, action, resource, pinned_patient_id, session_id) -> Decision

Algorithm (04 §5):
  1. Role gate   — role_permissions[role, resource, action] present?
  2. Binding gate — none / deidentified / assigned (+ break-glass check)
  3. Field gate  — slice allowed_fields from role_permissions
  4. Always attach "audit" obligation

The pinned_patient_id is an INPUT to authorization, never a grant.
Authorization is re-derived from the DB on every call; never cached.

This function accepts a psycopg connection managed by the caller
(services/patient/ or tests). It never creates its own connection so
the Postgres credential stays in the caller's scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Decision:
    effect:         str         # "permit" | "deny"
    patient_scope:  list[str]   # ids to set in app.patient_scope
    allowed_fields: list[str]   # role's field allow-list for this resource
    obligations:    list[str]   # always ["audit"]; break-glass adds "flag_review"
    reason:         str
    patient_binding: str = ""   # "none" | "deidentified" | "assigned" (for caller use)


def decide(
    conn,
    claims: dict,
    action: str,
    resource: str,
    pinned_patient_id: Optional[str],
    session_id: str,
) -> Decision:
    """
    Evaluate a request against the policy store.

    Parameters
    ----------
    conn              : open psycopg connection (caller-managed, superuser)
    claims            : decoded JWT claims (sub, role, session_id, ...)
    action            : "read" | "write"
    resource          : resource enum (demographics, lab_results, ...)
    pinned_patient_id : the chart open on screen; None for knowledge calls
    session_id        : the visitor's sandbox session id
    """
    user_id = claims["sub"]
    role_id = claims["role"]

    cur = conn.cursor()

    # ── 0. Role integrity check ───────────────────────────────────────────
    # JWT role is a hint, not a grant. Verify against the users table on
    # every call so a forged role claim cannot escalate privileges.
    cur.execute("SELECT role_id FROM users WHERE user_id = %s", (user_id,))
    user_row = cur.fetchone()
    if not user_row or user_row[0] != role_id:
        return Decision(
            effect="deny",
            patient_scope=[],
            allowed_fields=[],
            obligations=["audit"],
            reason=f"JWT role claim '{role_id}' does not match DB role for user '{user_id}'",
        )

    # ── 1. Role gate ──────────────────────────────────────────────────────
    cur.execute(
        """
        SELECT patient_binding, allowed_fields
        FROM   role_permissions
        WHERE  role_id  = %s
          AND  resource = %s
          AND  action   = %s
        """,
        (role_id, resource, action),
    )
    row = cur.fetchone()
    if not row:
        return Decision(
            effect="deny",
            patient_scope=[],
            allowed_fields=[],
            obligations=["audit"],
            reason=f"role '{role_id}' has no {action} permission on '{resource}'",
        )

    patient_binding, allowed_fields = row

    # ── 2. Binding gate ───────────────────────────────────────────────────

    # knowledge and other non-patient-bound resources
    if patient_binding == "none":
        return Decision(
            effect="permit",
            patient_scope=[],
            allowed_fields=allowed_fields,
            obligations=["audit"],
            reason="resource is not patient-bound",
            patient_binding="none",
        )

    # analyst de-identified path — no PHI, no identity
    if patient_binding == "deidentified":
        return Decision(
            effect="permit",
            patient_scope=[],
            allowed_fields=allowed_fields,
            obligations=["audit"],
            reason="de-identified aggregate access",
            patient_binding="deidentified",
        )

    # assigned — check the patient_assignments table
    if not pinned_patient_id:
        return Decision(
            effect="deny",
            patient_scope=[],
            allowed_fields=[],
            obligations=["audit"],
            reason="patient-bound resource requires a pinned patient",
        )

    cur.execute(
        """
        SELECT patient_id
        FROM   patient_assignments
        WHERE  session_id = %s
          AND  user_id    = %s
        """,
        (session_id, user_id),
    )
    assigned_ids = [r[0] for r in cur.fetchall()]

    if pinned_patient_id in assigned_ids:
        return Decision(
            effect="permit",
            patient_scope=assigned_ids,
            allowed_fields=allowed_fields,
            obligations=["audit"],
            reason=f"user is assigned to patient {pinned_patient_id}",
            patient_binding="assigned",
        )

    # ── 2b. Break-glass check ─────────────────────────────────────────────
    cur.execute(
        """
        SELECT id
        FROM   break_glass_grants
        WHERE  session_id  = %s
          AND  user_id     = %s
          AND  patient_id  = %s
          AND  expires_at  > now()
        """,
        (session_id, user_id, pinned_patient_id),
    )
    grant = cur.fetchone()
    if grant:
        return Decision(
            effect="permit",
            patient_scope=assigned_ids + [pinned_patient_id],
            allowed_fields=allowed_fields,
            obligations=["audit", "flag_review"],
            reason="break-glass grant active — access flagged for review",
            patient_binding="assigned",
        )

    return Decision(
        effect="deny",
        patient_scope=[],
        allowed_fields=[],
        obligations=["audit"],
        reason=f"user '{user_id}' is not assigned to patient '{pinned_patient_id}'",
    )
