"""
FastAPI dependencies for request-scoped resources.

``get_db`` is the DB-session seam. By default it is a **no-op stub** that
yields ``None`` — persistence is opt-in and SQLAlchemy may not be installed.
It becomes a real session provider in two ways, neither of which affects the
default path:

* ``api.app`` startup, when ``CONTROLPLANE_PERSISTENCE=1``, sets
  ``app.dependency_overrides[get_db] = database.session.get_db``.
* ``conftest.py`` overrides it with an in-memory SQLite session for tests
  that exercise the DB-backed stores.

Endpoints that want persistence declare ``db = Depends(get_db)`` and pass
``db`` through to the service / ``DecisionEngine``; when the stub is active
they simply receive ``None`` and fall back to in-memory behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def get_db() -> Iterator[Any]:
    """Yield a request-scoped DB session, or ``None`` when persistence is off."""
    yield None
