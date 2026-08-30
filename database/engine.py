"""
Engine construction.

``CONTROLPLANE_DATABASE_URL`` selects the backend:

    (unset)                        -> sqlite:///./controlplane.db   (zero-config local dev)
    sqlite:///./controlplane.db    -> file-backed SQLite
    sqlite+pysqlite:///:memory:    -> in-memory SQLite (tests)
    postgresql+psycopg://u:p@host/db -> PostgreSQL (docker-compose / production)

Nothing in this module has side effects at import time.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL = "sqlite:///./controlplane.db"
_ENV_VAR = "CONTROLPLANE_DATABASE_URL"


def database_url() -> str:
    """The configured database URL (env override, else zero-config SQLite)."""
    return os.environ.get(_ENV_VAR) or DEFAULT_DATABASE_URL


def _is_memory_sqlite(url: str) -> bool:
    return url.startswith("sqlite") and (":memory:" in url or url.endswith("sqlite://"))


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """
    Build a SQLAlchemy :class:`Engine` with backend-appropriate options.

    * SQLite gets ``check_same_thread=False`` (FastAPI / TestClient use
      multiple threads) and, for ``:memory:``, a ``StaticPool`` so every
      connection sees the same in-memory database.
    * PostgreSQL gets a small pre-pinged connection pool.
    """
    url = url or database_url()
    kwargs: dict = {"echo": echo, "future": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if _is_memory_sqlite(url):
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = int(os.environ.get("CONTROLPLANE_DB_POOL_SIZE", "5"))
        kwargs["max_overflow"] = int(os.environ.get("CONTROLPLANE_DB_MAX_OVERFLOW", "10"))

    return create_engine(url, **kwargs)
