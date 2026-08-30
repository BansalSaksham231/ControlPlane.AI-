"""
Persistent storage layer for ControlPlane.ai (SQLAlchemy 2.0).

This package is **completely isolated** from the runtime import graph. No
module under ``api/``, ``session/``, ``investigation/``, ``decision/`` or the
test suite imports ``database`` at module load. It is pulled in only when a
caller *explicitly* opts into persistence:

    * ``ControlPlaneService(use_database=True)`` (or ``CONTROLPLANE_PERSISTENCE=1``)
    * ``api.dependencies.get_db`` when wired as a FastAPI dependency
    * ``conftest.py`` (guarded by ``pytest.importorskip("sqlalchemy")``)

Default deployment target is zero-config SQLite (``sqlite:///./controlplane.db``);
setting ``CONTROLPLANE_DATABASE_URL`` to a PostgreSQL DSN switches backends with
no code change.

Nothing here changes the behaviour of the in-memory managers. The DB-backed
stores implement the *same method signatures* as ``session.manager`` and
``investigation.service`` so they can be dropped in via constructor injection.
"""

from __future__ import annotations

__all__ = [
    "Base",
    "DbInteraction",
    "DbDecisionTrace",
    "DbSessionState",
    "DbGovernanceAction",
    "database_url",
    "make_engine",
    "init_engine",
    "get_sessionmaker",
    "get_db",
    "session_scope",
    "DbSessionStore",
    "DbGovernanceStore",
    "DbTraceStore",
]

from database.engine import database_url, make_engine
from database.models import (
    Base,
    DbDecisionTrace,
    DbGovernanceAction,
    DbInteraction,
    DbSessionState,
)
from database.repositories import DbGovernanceStore, DbSessionStore, DbTraceStore
from database.session import get_db, get_sessionmaker, init_engine, session_scope
