"""
API-key authentication for the ControlPlane HTTP layer.

Opt-in and backwards compatible: authentication is enabled **only** when the
``CONTROLPLANE_API_KEY`` environment variable is set. When it is unset (local
dev, the existing test suite, CI) ``verify_api_key`` is a no-op and every
request is allowed — so wiring it as a global dependency changes nothing until
an operator explicitly configures a key.

Enable in production via ``docker-compose.yml`` / the environment:

    CONTROLPLANE_API_KEY=<random-secret>

Clients then send it as the ``X-API-Key`` request header.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Request, status

_ENV_VAR = "CONTROLPLANE_API_KEY"
_HEADER = "X-API-Key"

# Paths that stay open even when a key is configured (probes / API docs).
_PUBLIC_PATHS = frozenset(
    {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
)


def _configured_key() -> str:
    return os.environ.get(_ENV_VAR, "").strip()


def api_key_enabled() -> bool:
    return bool(_configured_key())


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=_HEADER),
) -> None:
    """FastAPI dependency. Raises 401 on a missing/incorrect key when enabled."""
    expected = _configured_key()
    if not expected:
        return  # authentication disabled (default)

    if request.url.path in _PUBLIC_PATHS:
        return

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
