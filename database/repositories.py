"""
DB-backed stores that are drop-in replacements for the in-memory managers.

Each class mirrors the public method surface of its in-memory counterpart so
it can be injected without touching callers:

    InMemorySessionStore  (session.manager)      <-> DbSessionStore
    GovernanceStore       (investigation.service) <-> DbGovernanceStore
    ControlPlaneService._audit (dict)             <-> DbTraceStore

Every operation runs in its own short transaction (``session_scope``), so a
single long-lived store instance is safe to share across a threaded API
process. Pass an explicit ``session_factory`` (e.g. a test's ``get_db``
override) to bind the store to a specific session/transaction instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import (
    DbDecisionTrace,
    DbGovernanceAction,
    DbInteraction,
    DbSessionState,
)
from database.session import session_scope

SessionFactory = Callable[[], "contextlib.AbstractContextManager[Session]"]


def _default_factory() -> "contextlib.AbstractContextManager[Session]":
    return session_scope()


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------


class DbSessionStore:
    """
    Persistent replacement for ``session.manager.InMemorySessionStore``.

    Contract (identical to the in-memory store):
        get(session_id) -> SessionState | None
        put(state: SessionState) -> None
        pop(session_id) -> None
        ids() -> list[str]
    """

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        self._factory = session_factory or _default_factory

    def get(self, session_id: str):
        from session.schemas import SessionState

        with self._factory() as db:
            row = db.get(DbSessionState, session_id)
            if row is None:
                return None
            return SessionState.model_validate(row.state)

    def put(self, state) -> None:
        snap = getattr(state, "snapshot", None)
        with self._factory() as db:
            row = db.get(DbSessionState, state.session_id)
            dump = state.model_dump(mode="json")
            if row is None:
                db.add(
                    DbSessionState(
                        session_id=state.session_id,
                        interaction_count=state.interaction_count,
                        cumulative_risk=float(state.cumulative_risk),
                        critical_floor=float(getattr(snap, "critical_floor", 0.0) or 0.0),
                        escalated=bool(state.escalated),
                        state=dump,
                    )
                )
            else:
                row.interaction_count = state.interaction_count
                row.cumulative_risk = float(state.cumulative_risk)
                row.critical_floor = float(getattr(snap, "critical_floor", 0.0) or 0.0)
                row.escalated = bool(state.escalated)
                row.state = dump

    def pop(self, session_id: str) -> None:
        with self._factory() as db:
            row = db.get(DbSessionState, session_id)
            if row is not None:
                db.delete(row)

    def ids(self) -> list[str]:
        with self._factory() as db:
            return list(db.scalars(select(DbSessionState.session_id)))


# ---------------------------------------------------------------------------
# governance actions (append-only)
# ---------------------------------------------------------------------------


class DbGovernanceStore:
    """
    Persistent replacement for ``investigation.service.GovernanceStore``.
    Same method surface: ``record_action`` / ``get_actions`` /
    ``get_all_actions`` / ``current_status`` / ``get_history`` /
    ``effective_decision``.
    """

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        self._factory = session_factory or _default_factory

    # -- writes ---------------------------------------------------------- #

    def record_action(
        self,
        *,
        interaction_id: str,
        action,
        actor: str,
        comment: str,
        previous_status,
        new_status,
        original_decision: str,
        reviewer_decision: str | None = None,
        timestamp: datetime | None = None,
    ):
        import uuid

        from investigation.schemas import GovernanceAction

        ts = timestamp or datetime.now(timezone.utc)
        # ``action_id`` must be unique but rows here are append-only (a
        # ``before_update`` guard rejects any later write to a row, so it
        # cannot be filled in after an initial insert via flush()). A
        # separate "max(id) + 1" read-then-insert would let two concurrent
        # record_action calls read the same max and collide on the column's
        # unique constraint, so it is generated up front instead — random,
        # not sequential, but collision-free without a DB round-trip and
        # without ever touching the row again after its one insert.
        # Chronological order is unaffected: reads always sort by the
        # table's own autoincrement id (see _rows_for / get_all_actions),
        # never by parsing this string.
        action_id = f"GOV-{uuid.uuid4().hex[:12].upper()}"
        record = GovernanceAction(
            action_id=action_id,
            interaction_id=interaction_id,
            timestamp=ts,
            actor=actor or "reviewer",
            action=action,
            comment=comment,
            previous_status=previous_status,
            new_status=new_status,
            original_decision=original_decision,
            reviewer_decision=reviewer_decision,
        )
        with self._factory() as db:
            db.add(
                DbGovernanceAction(
                    action_id=action_id,
                    interaction_id=interaction_id,
                    actor=record.actor,
                    action=_val(action),
                    comment=comment or "",
                    previous_status=_val(previous_status),
                    new_status=_val(new_status),
                    original_decision=original_decision,
                    reviewer_decision=reviewer_decision,
                    payload=record.model_dump(mode="json"),
                    created_at=ts,
                )
            )
        return record

    # -- reads --------------------------------------------------------- #

    def _rows_for(self, db: Session, interaction_id: str) -> list[DbGovernanceAction]:
        return list(
            db.scalars(
                select(DbGovernanceAction)
                .where(DbGovernanceAction.interaction_id == interaction_id)
                .order_by(DbGovernanceAction.id.asc())
            )
        )

    def get_actions(self, interaction_id: str) -> list:
        from investigation.schemas import GovernanceAction

        with self._factory() as db:
            return [
                GovernanceAction.model_validate(r.payload)
                for r in self._rows_for(db, interaction_id)
            ]

    def get_all_actions(self) -> list:
        from investigation.schemas import GovernanceAction

        with self._factory() as db:
            rows = db.scalars(
                select(DbGovernanceAction).order_by(DbGovernanceAction.id.asc())
            )
            return [GovernanceAction.model_validate(r.payload) for r in rows]

    def current_status(self, interaction_id: str):
        from investigation.schemas import InvestigationStatus

        actions = self.get_actions(interaction_id)
        return actions[-1].new_status if actions else InvestigationStatus.OPEN

    def get_history(self, interaction_id: str) -> list:
        return self.get_actions(interaction_id)

    def effective_decision(self, interaction_id: str, original_decision: str) -> str:
        from investigation.schemas import GovernanceActionType

        for action in reversed(self.get_actions(interaction_id)):
            if (
                action.action is GovernanceActionType.MODIFY_DECISION
                and action.reviewer_decision
            ):
                return action.reviewer_decision
        return original_decision


# ---------------------------------------------------------------------------
# decision traces + interactions (audit log)
# ---------------------------------------------------------------------------


class DbTraceStore:
    """
    Persistent replacement for ``ControlPlaneService._audit`` /
    ``_interactions`` / ``_audit_order``. Traces are append-only.
    """

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        self._factory = session_factory or _default_factory

    def put(self, interaction, trace) -> None:
        redacted = trace.redacted_dump()
        fd = trace.final_decision
        with self._factory() as db:
            if db.get(DbDecisionTrace, interaction.interaction_id) is not None:
                return  # immutable: first write wins
            db.add(
                DbInteraction(
                    interaction_id=interaction.interaction_id,
                    session_id=getattr(interaction, "session_id", None),
                    application=_val(getattr(interaction, "application", "")),
                    prompt=getattr(interaction, "prompt", "") or "",
                    response=getattr(interaction, "response", "") or "",
                    context=getattr(interaction, "context", "") or "",
                    payload=_safe_dump(interaction),
                )
            )
            db.add(
                DbDecisionTrace(
                    interaction_id=interaction.interaction_id,
                    session_id=getattr(interaction, "session_id", None),
                    decision=_val(fd.decision),
                    overall_risk=float(fd.overall_risk),
                    trace=redacted,
                )
            )

    def get_trace(self, interaction_id: str):
        with self._factory() as db:
            row = db.get(DbDecisionTrace, interaction_id)
            return row.trace if row is not None else None

    def get_interaction(self, interaction_id: str):
        with self._factory() as db:
            row = db.get(DbInteraction, interaction_id)
            return row.payload if row is not None else None

    def order(self) -> list[str]:
        with self._factory() as db:
            return list(
                db.scalars(
                    select(DbDecisionTrace.interaction_id).order_by(
                        DbDecisionTrace.created_at.asc()
                    )
                )
            )


# ---------------------------------------------------------------------------


def _val(x: Any) -> str:
    return getattr(x, "value", None) or str(x)


def _safe_dump(obj: Any) -> dict:
    for method in ("redacted_dump", "model_dump"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return fn(mode="json") if method == "model_dump" else fn()
            except TypeError:
                return fn()
    return {}


@contextlib.contextmanager
def bound_factory(db: Session) -> Iterator[Callable[[], Any]]:
    """Adapt a single already-open Session into a ``session_factory``."""

    @contextlib.contextmanager
    def _factory() -> Iterator[Session]:
        yield db  # caller owns commit/rollback

    yield _factory
