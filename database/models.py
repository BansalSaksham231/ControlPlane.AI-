"""
SQLAlchemy 2.0 ORM models for ControlPlane.ai.

Design notes
------------
* Typed mappings (``Mapped`` / ``mapped_column``) — SQLAlchemy 2.0 style.
* Portable column types only (``String``, ``Integer``, ``Float``,
  ``DateTime``, ``JSON``, ``Text``) so the same models run on SQLite and
  PostgreSQL unchanged. ``JSON`` maps to ``JSONB`` automatically on
  PostgreSQL and to a TEXT-backed JSON column on SQLite.
* The full domain object is always stored as a JSON blob in ``payload`` /
  ``trace`` / ``state``; the promoted scalar columns exist purely for
  indexing and cheap filtering. The blob is the source of truth.
* ``DbDecisionTrace`` and ``DbGovernanceAction`` are **append-only** by
  contract (see ``block_mutation`` listener below) — mirroring the
  immutability guarantee of ``DecisionTrace`` and the append-only
  ``GovernanceStore``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every ControlPlane table."""


# ---------------------------------------------------------------------------
# mutable operational state
# ---------------------------------------------------------------------------


class DbInteraction(Base):
    """The governed interaction (prompt / response / application / metadata)."""

    __tablename__ = "cp_interactions"

    interaction_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, default=None)
    application: Mapped[str] = mapped_column(String(64), index=True, default="")

    prompt: Mapped[str] = mapped_column(Text, default="")
    response: Mapped[str] = mapped_column(Text, default="")
    context: Mapped[str] = mapped_column(Text, default="")

    # full redacted Interaction.model_dump() — source of truth
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class DbSessionState(Base):
    """
    Multi-turn session risk state: the cumulative risk plus the full
    ``ContextualSnapshot`` for a conversation. One row per ``session_id``,
    upserted after every turn.
    """

    __tablename__ = "cp_session_state"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    interaction_count: Mapped[int] = mapped_column(Integer, default=0)
    cumulative_risk: Mapped[float] = mapped_column(Float, default=0.0)
    critical_floor: Mapped[float] = mapped_column(Float, default=0.0)
    escalated: Mapped[bool] = mapped_column(default=False)

    # full SessionState.model_dump(mode="json") — source of truth
    state: Mapped[dict] = mapped_column(JSON, default=dict)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ---------------------------------------------------------------------------
# append-only audit + governance
# ---------------------------------------------------------------------------


class DbDecisionTrace(Base):
    """
    The immutable decision trace for one interaction. Written once, never
    updated (enforced by the ``block_mutation`` listener). Mirrors
    ``DecisionTrace.redacted_dump()``.
    """

    __tablename__ = "cp_decision_traces"

    interaction_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, default=None)

    decision: Mapped[str] = mapped_column(String(32), index=True, default="")
    overall_risk: Mapped[float] = mapped_column(Float, default=0.0)

    # full DecisionTrace.redacted_dump() — PII-safe, immutable
    trace: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class DbGovernanceAction(Base):
    """
    One human governance action. Append-only log (never updated / deleted).
    Mirrors ``investigation.schemas.GovernanceAction``; ``payload`` holds the
    full model dump, the scalar columns are for querying.
    """

    __tablename__ = "cp_governance_actions"

    # monotonic surrogate key -> deterministic chronological ordering
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    interaction_id: Mapped[str] = mapped_column(String(128), index=True)

    actor: Mapped[str] = mapped_column(String(128), default="reviewer")
    action: Mapped[str] = mapped_column(String(64), default="")
    comment: Mapped[str] = mapped_column(Text, default="")

    previous_status: Mapped[str] = mapped_column(String(32), default="OPEN")
    new_status: Mapped[str] = mapped_column(String(32), default="OPEN")

    original_decision: Mapped[str] = mapped_column(String(32), default="")
    reviewer_decision: Mapped[str | None] = mapped_column(String(32), default=None)

    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


# ---------------------------------------------------------------------------
# immutability guard for the append-only tables
# ---------------------------------------------------------------------------

_APPEND_ONLY = (DbDecisionTrace, DbGovernanceAction)


def _install_immutability_guards() -> None:
    for model in _APPEND_ONLY:

        @event.listens_for(model, "before_update", propagate=True)
        def _block_update(mapper, connection, target):  # noqa: ANN001
            raise RuntimeError(
                f"{type(target).__name__} is append-only; rows may not be updated."
            )

        @event.listens_for(model, "before_delete", propagate=True)
        def _block_delete(mapper, connection, target):  # noqa: ANN001
            raise RuntimeError(
                f"{type(target).__name__} is append-only; rows may not be deleted."
            )


_install_immutability_guards()
