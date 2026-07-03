"""
runtime.orchestrator.tool_client
---------------------------------
Trusted ToolClient shim — the single place where identity binds to requests.

The model proposes tool calls (name + args). This shim:
  1. Strips any patient/subject identifier the model may have embedded.
  2. Substitutes the authoritative pinned_patient_id for patient-scoped tools.
  3. Routes to the appropriate service function.
  4. Logs knowledge-service accesses via the patient service's audit writer.

Security invariant:
  The model can decide WHAT to ask about a patient, never WHICH patient.
  patient_id is always injected here from the pinned chart, not from the model.

This module holds NO database handle and imports NO psycopg client.
"""

from __future__ import annotations

from typing import Any

# All keys the model might use to propose a subject — any of these are stripped.
_SUBJECT_KEYS: frozenset[str] = frozenset({
    "patient_id", "subject_id", "patient", "subject", "mrn", "chart_id",
})

# Tools that operate on a specific patient (subject injected from pinned chart).
_PATIENT_TOOLS: frozenset[str] = frozenset({
    "get_patient_record",
    "update_lab_result",
    "update_medication",
})


class ToolClient:
    """
    Stateless per-request shim.

    Instantiate once per request with the authenticated context,
    then call .call() for each tool the model proposes.
    """

    def __init__(
        self,
        claims: dict,
        pinned_patient_id: str | None,
        session_id: str,
    ) -> None:
        self.claims = claims
        self.pinned_patient_id = pinned_patient_id
        self.session_id = session_id

    def call(self, tool_name: str, model_args: dict[str, Any]) -> dict:
        """
        Execute one tool call with identity bound from ctx, not from the model.

        Parameters
        ----------
        tool_name    : one of the four authorized tools
        model_args   : raw args the model proposed (may contain subject fields)

        Returns
        -------
        dict with at least {"ok": bool}; shape matches each service's response.
        """
        # 1. Strip any subject identifier the model may have included.
        args = {k: v for k, v in model_args.items() if k not in _SUBJECT_KEYS}

        # 2. For patient-scoped tools, inject the pinned patient.
        if tool_name in _PATIENT_TOOLS:
            if not self.pinned_patient_id:
                return {
                    "ok": False,
                    "code": "bad_request",
                    "message": "No patient pinned for patient-scoped tool call.",
                }
            args["patient_id"] = self.pinned_patient_id

        # 3. Route to the service.
        return self._route(tool_name, args)

    def _route(self, tool_name: str, args: dict) -> dict:
        if tool_name == "query_knowledge":
            from runtime.services.knowledge import query_knowledge
            result = query_knowledge(
                claims=self.claims,
                session_id=self.session_id,
                **args,
            )
            # Knowledge audit is written by the patient service (sole DB holder).
            try:
                from runtime.services.patient import log_knowledge_access
                log_knowledge_access(
                    session_id=self.session_id,
                    user_id=self.claims["sub"],
                    role_id=self.claims["role"],
                    key=args.get("key", ""),
                    effect="permit" if result.get("ok") else "deny",
                )
            except Exception:
                pass  # audit failure must not block the answer
            return result

        elif tool_name == "get_patient_record":
            from runtime.services.patient import get_patient_record
            return get_patient_record(
                claims=self.claims,
                session_id=self.session_id,
                **args,
            )

        elif tool_name == "update_lab_result":
            from runtime.services.patient import update_lab_result
            return update_lab_result(
                claims=self.claims,
                session_id=self.session_id,
                **args,
            )

        elif tool_name == "update_medication":
            from runtime.services.patient import update_medication
            return update_medication(
                claims=self.claims,
                session_id=self.session_id,
                **args,
            )

        elif tool_name == "get_deidentified_aggregate":
            from runtime.services.patient import get_deidentified_aggregate
            return get_deidentified_aggregate(
                claims=self.claims,
                session_id=self.session_id,
            )

        else:
            return {
                "ok": False,
                "code": "bad_request",
                "message": f"Unknown tool: '{tool_name}'",
            }
