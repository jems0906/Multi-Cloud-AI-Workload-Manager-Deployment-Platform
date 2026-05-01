from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt import decode as jwt_decode
from jwt.exceptions import InvalidTokenError


@dataclass(frozen=True)
class AuthContext:
    token: str
    role: str
    subject: str | None = None


_security = HTTPBearer(auto_error=False)
_ROLE_LEVELS = {
    "viewer": 1,
    "operator": 2,
    "admin": 3,
}
_AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "auth_audit.log"


def _token_roles() -> dict[str, str]:
    """
    Resolve token->role mapping from environment.

    FLEXAI_TOKEN_ROLES can be set to JSON, for example:
    {"token-a": "viewer", "token-b": "admin"}
    """
    raw = os.getenv("FLEXAI_TOKEN_ROLES", "").strip()
    if not raw:
        return {
            "local-dev-token": "admin",
            "local-operator-token": "operator",
            "local-viewer-token": "viewer",
        }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid FLEXAI_TOKEN_ROLES JSON") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("FLEXAI_TOKEN_ROLES must be a JSON object")

    normalized: dict[str, str] = {}
    for token, role in parsed.items():
        if not isinstance(token, str) or not isinstance(role, str):
            raise RuntimeError("FLEXAI_TOKEN_ROLES must map string token->string role")
        normalized[token] = role.strip().lower()
    return normalized


def _jwt_signature_verification_enabled() -> bool:
    raw = os.getenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true").strip().lower()
    return raw not in {"0", "false", "no"}


@lru_cache(maxsize=8)
def _jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(url)


def _decode_jwt_payload(token: str) -> dict[str, object]:
    decode_options = {
        "verify_exp": False,
        "verify_nbf": False,
        "verify_iat": False,
        "verify_aud": False,
    }

    try:
        if not _jwt_signature_verification_enabled():
            payload = jwt_decode(
                token,
                options={"verify_signature": False, **decode_options},
                algorithms=["HS256", "RS256"],
            )
        else:
            hs_secret = os.getenv("FLEXAI_JWT_HS256_SECRET", "").strip()
            jwks_url = os.getenv("FLEXAI_JWT_JWKS_URL", "").strip()

            if hs_secret:
                payload = jwt_decode(
                    token,
                    key=hs_secret,
                    algorithms=["HS256"],
                    options=decode_options,
                )
            elif jwks_url:
                signing_key = _jwks_client(jwks_url).get_signing_key_from_jwt(token)
                payload = jwt_decode(
                    token,
                    key=signing_key.key,
                    algorithms=["RS256"],
                    options=decode_options,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="JWT verification key not configured",
                )
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return payload


def _validate_jwt_claims(payload: dict[str, object]) -> None:
    now_epoch = int(datetime.now(timezone.utc).timestamp())

    exp = payload.get("exp")
    if exp is not None:
        try:
            if now_epoch >= int(exp):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            if now_epoch < int(nbf):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token not yet valid")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    expected_issuer = os.getenv("FLEXAI_JWT_ISSUER", "").strip()
    if expected_issuer and payload.get("iss") != expected_issuer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer")

    expected_audience = os.getenv("FLEXAI_JWT_AUDIENCE", "").strip()
    if expected_audience:
        audience_claim = payload.get("aud")
        if isinstance(audience_claim, str):
            valid_audience = audience_claim == expected_audience
        elif isinstance(audience_claim, list):
            valid_audience = expected_audience in audience_claim
        else:
            valid_audience = False
        if not valid_audience:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience")


def _write_auth_audit(
    request: Request,
    token: str | None,
    decision: str,
    detail: str,
    required_role: str,
    resolved_role: str | None,
    status_code: int,
) -> None:
    fingerprint = None
    if token:
        fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "decision": decision,
        "detail": detail,
        "required_role": required_role,
        "resolved_role": resolved_role,
        "status_code": status_code,
        "token_fingerprint": fingerprint,
        "client": request.client.host if request.client else None,
    }

    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, separators=(",", ":")) + "\n")


def read_auth_audit_events(
    limit: int = 100,
    decision: str | None = None,
    path_contains: str | None = None,
) -> list[dict[str, Any]]:
    """Read auth audit events in reverse chronological order with optional filters."""
    safe_limit = max(1, min(limit, 1000))
    target_decision = decision.strip().lower() if decision else None
    target_path = path_contains.strip().lower() if path_contains else None

    if not _AUDIT_LOG_PATH.exists():
        return []

    lines = _AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    results: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue
        if target_decision and str(event.get("decision", "")).lower() != target_decision:
            continue
        if target_path and target_path not in str(event.get("path", "")).lower():
            continue

        results.append(event)
        if len(results) >= safe_limit:
            break

    return results


def _authenticate(credentials: HTTPAuthorizationCredentials | None) -> AuthContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials.strip()
    role_mapping = _token_roles()
    role = role_mapping.get(token)
    subject: str | None = None
    if role is None:
        payload = _decode_jwt_payload(token)
        _validate_jwt_claims(payload)
        role_claim = os.getenv("FLEXAI_JWT_ROLE_CLAIM", "role").strip() or "role"
        role_value = payload.get(role_claim)
        if not isinstance(role_value, str):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token role missing")
        role = role_value.strip().lower()
        subject_claim = payload.get("sub")
        if isinstance(subject_claim, str) and subject_claim.strip():
            subject = subject_claim.strip()

    if role not in _ROLE_LEVELS:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid role config")

    return AuthContext(token=token, role=role, subject=subject)


def require_role(min_role: str) -> Callable[[HTTPAuthorizationCredentials | None], AuthContext]:
    expected = min_role.strip().lower()
    if expected not in _ROLE_LEVELS:
        raise ValueError(f"Unknown role {min_role!r}")

    def _dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    ) -> AuthContext:
        token = credentials.credentials.strip() if credentials and credentials.credentials else None

        try:
            auth = _authenticate(credentials)
        except HTTPException as exc:
            _write_auth_audit(
                request=request,
                token=token,
                decision="deny",
                detail=str(exc.detail),
                required_role=expected,
                resolved_role=None,
                status_code=exc.status_code,
            )
            raise

        if _ROLE_LEVELS[auth.role] < _ROLE_LEVELS[expected]:
            _write_auth_audit(
                request=request,
                token=auth.token,
                decision="deny",
                detail="Insufficient role",
                required_role=expected,
                resolved_role=auth.role,
                status_code=status.HTTP_403_FORBIDDEN,
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

        _write_auth_audit(
            request=request,
            token=auth.token,
            decision="allow",
            detail="Authorized",
            required_role=expected,
            resolved_role=auth.role,
            status_code=status.HTTP_200_OK,
        )
        return auth

    return _dependency


def _verify_token(token: str, required_role: str) -> AuthContext:
    """
    Authenticate a raw bearer token string and check the minimum role.

    Used by WebSocket endpoints that cannot rely on FastAPI's Depends()
    because they need to authenticate before accepting the socket.

    Raises :class:`fastapi.HTTPException` (401 or 403) on failure.
    """
    from fastapi.security import HTTPAuthorizationCredentials  # local import avoids circular deps

    creds = HTTPAuthorizationCredentials(scheme="bearer", credentials=token)
    auth = _authenticate(creds)
    expected = required_role.strip().lower()
    if _ROLE_LEVELS.get(auth.role, 0) < _ROLE_LEVELS.get(expected, 0):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    return auth
