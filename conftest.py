"""
Pytest configuration.

1. Ensures the repository root is importable as the top-level package
   namespace (``data``, ``detectors``, ``fusion``, ...) regardless of where
   pytest is invoked from.

2. Reconciles the persistent-storage layer with the existing suite. The
   default managers are in-memory, so the ~786 legacy tests need no database.
   When SQLAlchemy *is* installed, the FastAPI ``get_db`` dependency is
   overridden with a shared in-memory SQLite database for the whole test
   session, and a ``db_session`` fixture is offered to tests that exercise the
   DB-backed stores directly. When SQLAlchemy is absent, everything here is a
   no-op and the suite runs exactly as before.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_HAVE_SQLALCHEMY = importlib.util.find_spec("sqlalchemy") is not None
_HAVE_STREAMLIT = importlib.util.find_spec("streamlit") is not None


# ---------------------------------------------------------------------------
# Streamlit UI test isolation
# ---------------------------------------------------------------------------

if _HAVE_STREAMLIT:

    @pytest.fixture(autouse=True, scope="module")
    def _reset_streamlit_resource_cache():
        """
        Every Streamlit AppTest UI test module builds its own
        ``ControlPlaneService`` via ``st.cache_resource`` (see
        ``streamlit_app.py::get_service``). That cache is process-global, not
        scoped to an individual ``AppTest`` run, so without this reset the
        cached service — and the in-memory session/audit state it owns —
        leaks across UI test modules that reuse the same scenario session ids
        (e.g. ``SESSION-SCEN-B``), making multi-turn assertions depend on
        pytest's file collection order. Clearing it once per module gives
        each UI test module a fresh service, matching a real first page load.
        """
        import streamlit as st

        st.cache_resource.clear()
        yield


# ---------------------------------------------------------------------------
# persistence-layer test wiring (only active when SQLAlchemy is installed)
# ---------------------------------------------------------------------------

if _HAVE_SQLALCHEMY:

    @pytest.fixture(scope="session")
    def _test_engine():
        from database.session import init_engine, reset_engine

        # Shared in-memory database for the whole test session.
        engine = init_engine("sqlite+pysqlite:///:memory:", create_all=True)
        yield engine
        reset_engine()

    @pytest.fixture()
    def db_session(_test_engine):
        """A transactional SQLAlchemy session, rolled back after each test."""
        from sqlalchemy.orm import Session

        connection = _test_engine.connect()
        txn = connection.begin()
        session = Session(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            session.close()
            txn.rollback()
            connection.close()

    @pytest.fixture(scope="session", autouse=True)
    def _override_get_db(_test_engine):
        """
        Point the FastAPI ``get_db`` dependency at the in-memory test DB for
        the whole session. Harmless for tests that never touch it (they get a
        session yielded and ignored).
        """
        try:
            from api.app import app
            from api.dependencies import get_db
            from database.session import get_db as real_get_db
        except Exception:  # pragma: no cover - app import is optional
            yield
            return

        app.dependency_overrides[get_db] = real_get_db
        try:
            yield
        finally:
            app.dependency_overrides.pop(get_db, None)

else:

    @pytest.fixture()
    def db_session():
        pytest.skip("SQLAlchemy is not installed; persistence-layer tests skipped.")
