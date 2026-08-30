"""
Session / unit-of-work management.

``init_engine()`` is the single entry point that binds a process (or a test
session) to a database. It is lazy and idempotent — the first call that needs
a session triggers it with the env-configured URL, and callers that want an
explicit URL (tests, ``docker-compose``) call it up front.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.engine import make_engine
from database.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(
    url: str | None = None, *, echo: bool = False, create_all: bool = True
) -> Engine:
    """
    Bind this process to a database and (optionally) create the schema.

    Idempotent for a given URL: calling it again with the same URL returns
    the existing engine; calling it with a new URL rebinds (used by tests).
    """
    global _engine, _SessionLocal

    if _engine is not None and (url is None or str(_engine.url) == url):
        return _engine

    _engine = make_engine(url, echo=echo)
    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, expire_on_commit=False, class_=Session
    )
    if create_all:
        Base.metadata.create_all(_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def reset_engine() -> None:
    """Drop the process binding (tests only)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Iterator[Session]:
    """
    FastAPI dependency: yield a request-scoped session, commit on success,
    roll back on error, always close.
    """
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for non-request callers (scripts, service wiring)."""
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
