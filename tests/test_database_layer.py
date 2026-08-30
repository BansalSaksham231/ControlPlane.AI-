"""
Persistent storage layer (``database/``).

Entirely skipped unless SQLAlchemy is installed (``pip install -r
requirements-db.txt``) — the default suite runs on the in-memory managers and
never imports this package. When SQLAlchemy is present these tests prove the
DB-backed stores are drop-in replacements for their in-memory counterparts.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from database.models import Base  # noqa: E402
from database.repositories import DbGovernanceStore, DbSessionStore  # noqa: E402
from database.session import init_engine, reset_engine, session_scope  # noqa: E402
from investigation.schemas import InvestigationStatus  # noqa: E402
from session.manager import SessionManager  # noqa: E402
from settings import load_settings  # noqa: E402


@pytest.fixture()
def _engine():
    eng = init_engine("sqlite+pysqlite:///:memory:", create_all=True)
    yield eng
    reset_engine()


def test_schema_creates_all_tables(_engine):
    names = set(Base.metadata.tables)
    assert names == {
        "cp_interactions",
        "cp_session_state",
        "cp_decision_traces",
        "cp_governance_actions",
    }


def test_session_manager_with_db_store_matches_in_memory(_engine):
    cfg = load_settings()
    mem = SessionManager(cfg)
    db = SessionManager(cfg, store=DbSessionStore())

    for mgr in (mem, db):
        mgr.record("S1", overall_risk=0.9, decision="BLOCK", interaction_id="i1",
                   critical=True, critical_trigger="CRITICAL_PII")
        mgr.record("S1", overall_risk=0.1, decision="ALLOW", interaction_id="i2")

    c_mem = mem.contribution("S1", 0.1)
    c_db = db.contribution("S1", 0.1)
    # critical floor is persisted and re-applied identically
    assert c_db["critical_floor"] == c_mem["critical_floor"] > 0.0
    assert c_db["adjusted_overall_risk"] == c_mem["adjusted_overall_risk"]
    assert db.active_sessions() == ["S1"]
    db.reset("S1")
    assert db.active_sessions() == []


def test_db_governance_store_is_append_only_and_computes_effective_decision(_engine):
    store = DbGovernanceStore()
    store.record_action(
        interaction_id="X", action="MODIFY_DECISION", actor="alice", comment="softer",
        previous_status=InvestigationStatus.OPEN, new_status=InvestigationStatus.REVIEWED,
        original_decision="BLOCK", reviewer_decision="HUMAN_REVIEW",
    )
    assert store.current_status("X") is InvestigationStatus.REVIEWED
    assert store.effective_decision("X", "BLOCK") == "HUMAN_REVIEW"
    assert len(store.get_history("X")) == 1

    # append-only: rows cannot be updated
    from database.models import DbGovernanceAction

    with pytest.raises(Exception, match="append-only"):
        with session_scope() as db:
            row = db.query(DbGovernanceAction).first()
            row.comment = "tampered"
            db.flush()
