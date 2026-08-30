"""
Data contracts for incident investigation & human governance.

Reuses (never duplicates) the existing schemas:
    monitoring.schemas.IncidentSummary
    decision.replay.IncidentReplay
    explainability.schemas.ExplainabilitySummary

The only NEW concepts here are the governance-workflow state
(``InvestigationStatus``), the explicit human actions
(``GovernanceActionType``), the immutable governance record
(``GovernanceAction``), and the wrapper (``IncidentInvestigation``).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decision.replay import IncidentReplay
from explainability.schemas import ExplainabilitySummary
from monitoring.schemas import IncidentSummary

__all__ = [
    "InvestigationStatus",
    "GovernanceActionType",
    "GovernanceAction",
    "InvestigationCounterfactual",
    "IncidentInvestigation",
    "InvestigationNotFound",
    "ACTION_TARGET_STATUS",
    "ACTIONS_REQUIRING_COMMENT",
    "ACTIONS_REQUIRING_REVIEWER_DECISION",
    "available_actions_for",
]


# ======================================================================
# governance workflow state  (NOT risk state)
# ======================================================================


class InvestigationStatus(str, Enum):
    """
    Where a governance workflow stands. This is workflow state only — it
    never changes the automated ControlPlane decision. A ``BLOCK`` that is
    ``CLOSED`` is still a ``BLOCK``.
    """

    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REVIEWED = "REVIEWED"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class GovernanceActionType(str, Enum):
    """Explicit human actions a reviewer can take on an incident."""

    ACKNOWLEDGE = "ACKNOWLEDGE"           # "I have seen this"
    APPROVE_DECISION = "APPROVE_DECISION"  # "I agree with ControlPlane's automated call"
    MODIFY_DECISION = "MODIFY_DECISION"    # "I would have chosen a different tier" (records both)
    REJECT_DECISION = "REJECT_DECISION"    # "ControlPlane's call was wrong"
    ESCALATE = "ESCALATE"                 # "needs a more senior reviewer"
    CLOSE = "CLOSE"                       # "governance workflow complete"


# which status an action moves the workflow into
ACTION_TARGET_STATUS: dict[GovernanceActionType, InvestigationStatus] = {
    GovernanceActionType.ACKNOWLEDGE: InvestigationStatus.ACKNOWLEDGED,
    GovernanceActionType.APPROVE_DECISION: InvestigationStatus.REVIEWED,
    GovernanceActionType.MODIFY_DECISION: InvestigationStatus.REVIEWED,
    GovernanceActionType.REJECT_DECISION: InvestigationStatus.REVIEWED,
    GovernanceActionType.ESCALATE: InvestigationStatus.ESCALATED,
    GovernanceActionType.CLOSE: InvestigationStatus.CLOSED,
}

ACTIONS_REQUIRING_COMMENT: frozenset[GovernanceActionType] = frozenset(
    {
        GovernanceActionType.MODIFY_DECISION,
        GovernanceActionType.REJECT_DECISION,
        GovernanceActionType.ESCALATE,
    }
)

ACTIONS_REQUIRING_REVIEWER_DECISION: frozenset[GovernanceActionType] = frozenset(
    {GovernanceActionType.MODIFY_DECISION}
)

# which actions are offered from each status (deterministic, ordered)
_AVAILABLE: dict[InvestigationStatus, tuple[GovernanceActionType, ...]] = {
    InvestigationStatus.OPEN: (
        GovernanceActionType.ACKNOWLEDGE,
        GovernanceActionType.APPROVE_DECISION,
        GovernanceActionType.MODIFY_DECISION,
        GovernanceActionType.REJECT_DECISION,
        GovernanceActionType.ESCALATE,
        GovernanceActionType.CLOSE,
    ),
    InvestigationStatus.ACKNOWLEDGED: (
        GovernanceActionType.APPROVE_DECISION,
        GovernanceActionType.MODIFY_DECISION,
        GovernanceActionType.REJECT_DECISION,
        GovernanceActionType.ESCALATE,
        GovernanceActionType.CLOSE,
    ),
    InvestigationStatus.REVIEWED: (
        GovernanceActionType.MODIFY_DECISION,
        GovernanceActionType.ESCALATE,
        GovernanceActionType.CLOSE,
    ),
    InvestigationStatus.ESCALATED: (
        GovernanceActionType.APPROVE_DECISION,
        GovernanceActionType.MODIFY_DECISION,
        GovernanceActionType.REJECT_DECISION,
        GovernanceActionType.CLOSE,
    ),
    InvestigationStatus.CLOSED: (),
}


def available_actions_for(status: InvestigationStatus) -> list[GovernanceActionType]:
    return list(_AVAILABLE[status])


# ======================================================================
# immutable governance record
# ======================================================================


class GovernanceAction(BaseModel):
    """
    One human governance action. Append-only; never mutated.

    ``original_decision`` is ControlPlane's automated decision at the time
    of the action (immutable). ``reviewer_decision`` is the tier the human
    would have chosen — a governance *outcome*, recorded alongside the
    automated decision and never substituted for it.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: str
    interaction_id: str
    timestamp: datetime
    actor: str = "reviewer"
    action: GovernanceActionType
    comment: str = ""

    previous_status: InvestigationStatus
    new_status: InvestigationStatus

    original_decision: str                     # ControlPlane automated decision (immutable)
    reviewer_decision: str | None = None       # human-proposed tier (NOT applied)


# ======================================================================
# counterfactual view  (explicitly a SIMULATION)
# ======================================================================


class InvestigationCounterfactual(BaseModel):
    """
    Result of an explicitly-triggered 'what if?' simulation, run by the
    existing ``simulation.engine``. It does NOT modify the stored decision
    or production policy. Free-text explanations are dropped so no raw PII
    can travel through this view.
    """

    model_config = ConfigDict(extra="forbid")

    simulated: bool = True
    interaction_id: str
    changed_fields: dict[str, Any] = Field(default_factory=dict)
    rejected_fields: list[str] = Field(default_factory=list)

    current_decision: str
    counterfactual_decision: str
    decision_changed: bool

    current_overall_risk: float = Field(ge=0, le=1)
    counterfactual_overall_risk: float = Field(ge=0, le=1)

    rules_removed: list[str] = Field(default_factory=list)
    rules_added: list[str] = Field(default_factory=list)
    reason_codes_removed: list[str] = Field(default_factory=list)
    reason_codes_added: list[str] = Field(default_factory=list)

    summary: str = ""
    note: str = (
        "SIMULATION — this counterfactual does not modify the stored ControlPlane "
        "decision or production policy configuration."
    )


# ======================================================================
# top-level investigation
# ======================================================================


class IncidentInvestigation(BaseModel):
    """The complete, auditable view of one incident + its governance state."""

    model_config = ConfigDict(extra="forbid")

    found: bool = True
    interaction_id: str

    # the operational incident header (None if the interaction is not
    # currently flagged as an incident — investigation still works)
    incident: IncidentSummary | None = None

    # read-only reconstruction (build_replay) + presentation (build_explanation)
    replay: IncidentReplay
    explanation: ExplainabilitySummary

    # immutable facts, surfaced for convenience
    original_decision: str                            # ControlPlane's automated call — NEVER changes
    requires_human_review: bool

    # governance workflow
    investigation_status: InvestigationStatus = InvestigationStatus.OPEN
    available_actions: list[GovernanceActionType] = Field(default_factory=list)
    governance_history: list[GovernanceAction] = Field(default_factory=list)
    latest_reviewer_decision: str | None = None       # from the most recent MODIFY_DECISION

    # The tier that is in effect after human governance: the latest reviewer
    # override if one exists, otherwise the immutable automated decision. This
    # is a *view* over the append-only governance track — the DecisionTrace and
    # ``original_decision`` are untouched.
    effective_governed_decision: str = ""
    is_overridden: bool = False

    notes: list[str] = Field(
        default_factory=lambda: [
            "Reconstructed from the stored DecisionTrace — no detector, decision "
            "engine, fusion, policy or verification pass was re-run.",
            "The automated ControlPlane decision is immutable. A governance "
            "action records the human outcome; it never overwrites the decision.",
            "Reviewer feedback is a governance signal, not ground truth — it does "
            "not change evaluation metrics or detector thresholds.",
            "All free text is the already-redacted text from the incident replay "
            "/ explainability summary; raw PII is never included.",
        ]
    )


class InvestigationNotFound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool = False
    interaction_id: str
    message: str
