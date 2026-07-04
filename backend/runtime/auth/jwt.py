"""
runtime.auth.jwt
----------------
JWT mint and verify for the runtime plane.

Tokens are minted on persona selection and verified by both services.
The raw token is NEVER given to the LLM.

Claim shape (03 §8):
  sub, role, department, care_team, scope, session_id, aud, iat, exp
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from graph_rag.config import CONFIG

AUDIENCE = "sentinel-runtime"
ALGORITHM = "HS256"

# Coarse service-level scope per role.
# The PDP enforces the fine-grained resource/action/field restrictions.
_ROLE_SCOPES: dict[str, list[str]] = {
    "attending_physician": ["patient.read", "patient.write", "knowledge.read"],
    "nurse":               ["patient.read", "knowledge.read"],
    "pharmacist":          ["patient.read", "knowledge.read"],
    "lab_technician":      ["patient.read", "patient.write", "knowledge.read"],
    "research_analyst":    ["patient.read", "knowledge.read"],
}


def mint_token(
    user_id: str,
    role_id: str,
    department: str,
    care_team: str,
    session_id: str,
) -> str:
    """Mint a signed HS256 JWT for the given persona."""
    if not CONFIG.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")
    now = datetime.now(timezone.utc)
    payload = {
        "sub":        user_id,
        "role":       role_id,
        "department": department,
        "care_team":  care_team,
        "scope":      _ROLE_SCOPES.get(role_id, ["knowledge.read"]),
        "session_id": session_id,
        "aud":        AUDIENCE,
        "iat":        now,
        "exp":        now + timedelta(minutes=CONFIG.jwt_expiry_minutes),
    }
    return jwt.encode(payload, CONFIG.jwt_secret, algorithm=ALGORITHM)


_REQUIRED_CLAIMS = {"sub", "role", "session_id", "scope"}


def verify_token(token: str) -> dict:
    """
    Decode and verify the token. Raises jwt.InvalidTokenError on any failure
    (expired, tampered signature, wrong audience, missing required claims, etc.).
    Returns the decoded claims dict.
    """
    if not CONFIG.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")
    claims = jwt.decode(
        token,
        CONFIG.jwt_secret,
        algorithms=[ALGORITHM],
        audience=AUDIENCE,
    )
    missing = _REQUIRED_CLAIMS - claims.keys()
    if missing:
        raise jwt.exceptions.MissingRequiredClaimError(next(iter(missing)))
    return claims


def require_scope(claims: dict, required: str) -> None:
    """Raise PermissionError if the required scope is absent from the token."""
    if required not in claims.get("scope", []):
        raise PermissionError(f"Token lacks required scope: {required}")
