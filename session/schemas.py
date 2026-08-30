"""Data contracts for session / multi-turn risk accumulation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CriticalEvent(BaseModel):
    """A turn that crossed a critical boundary (BLOCK / critical PII / severe toxicity)."""

    turn_index: int = Field(ge=1)
    interaction_id: str
    decision: str
    trigger: str                                # BLOCK | CRITICAL_PII | SEVERE_TOXICITY
    risk_at_event: float = Field(ge=0, le=1)


class ContextualSnapshot(BaseModel):
    """
    Structured, verified memory of the turns already processed in a session.

    The heavy detectors (claim extraction, NLI, PII regex, cost) run once per
    turn — on the newest turn only. Their *outputs* are folded into this
    snapshot so later turns inherit what earlier turns established without
    re-running anything: which PII entities were exposed, which reason codes
    recurred, which policy rules moved the tier, the peak risk seen on each
    dimension, and any non-decaying critical floor.
    """

    turns_recorded: int = 0

    # verified signals carried forward from earlier turns
    pii_entity_keys: list[str] = Field(default_factory=list)      # "<subtype>:<redacted>", sorted
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    tier_changing_rules: list[str] = Field(default_factory=list)  # distinct, sorted

    peak_performance_risk: float = Field(default=0.0, ge=0, le=1)
    peak_responsibility_risk: float = Field(default=0.0, ge=0, le=1)
    peak_cost_risk: float = Field(default=0.0, ge=0, le=1)

    # A critical violation sets this; it is NEVER decayed. Every subsequent
    # turn's adjusted risk is at least this value.
    critical_floor: float = Field(default=0.0, ge=0, le=1)
    critical_events: list[CriticalEvent] = Field(default_factory=list)

    @property
    def has_critical_history(self) -> bool:
        return bool(self.critical_events)


class SessionState(BaseModel):
    """Bounded, decaying view of one conversation's risk history + structured snapshot."""

    session_id: str
    interaction_count: int = 0

    recent_risks: list[float] = Field(default_factory=list)
    recent_decisions: list[str] = Field(default_factory=list)
    recent_interaction_ids: list[str] = Field(default_factory=list)

    cumulative_risk: float = Field(default=0.0, ge=0, le=1)
    high_risk_event_count: int = 0
    escalated: bool = False

    snapshot: ContextualSnapshot = Field(default_factory=ContextualSnapshot)

    last_updated: datetime | None = None


class SessionRiskContribution(BaseModel):
    """What the session history adds to the current turn's decision."""

    session_id: str
    interaction_count: int
    current_overall_risk: float = Field(ge=0, le=1)

    history_component: float = Field(ge=0, le=1)
    session_risk: float = Field(ge=0, le=1)
    adjusted_overall_risk: float = Field(ge=0, le=1)

    high_risk_events: int
    escalated: bool

    critical_floor: float = Field(default=0.0, ge=0, le=1)
    critical_floor_applied: bool = False
    has_critical_history: bool = False

    # Read-only copy of the session's contextual snapshot as it stood at the
    # START of this turn (turns 1..N-1). ``None`` for the first turn / a
    # stateless caller. Surfaced so an incident reviewer can see the full
    # multi-turn history that fed the decision.
    snapshot: ContextualSnapshot | None = None

    explanation: str
