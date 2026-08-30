"""Data contracts for the feedback loop."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from data.schemas import InterventionTier


class FeedbackOutcome(str, Enum):
    """What a human reviewer concluded about a ControlPlane decision."""

    APPROVED = "approved"      # system decision accepted as-is
    MODIFIED = "modified"      # reviewer changed the intervention tier
    REJECTED = "rejected"      # system decision judged wrong / overturned


class FeedbackRecord(BaseModel):
    feedback_id: str
    interaction_id: str

    system_decision: InterventionTier
    reviewer_decision: InterventionTier | None = None
    human_override: bool = False

    outcome: FeedbackOutcome
    actual_outcome: str | None = None
    reason: str = ""
    reviewer: str | None = None

    timestamp: datetime


class DecisionConfusionCell(BaseModel):
    system_decision: str
    reviewer_decision: str
    count: int


class FeedbackAggregate(BaseModel):
    total: int
    by_outcome: dict[str, int]
    by_system_decision: dict[str, int]

    override_count: int
    override_rate: float = Field(ge=0, le=1)
    approval_rate: float = Field(ge=0, le=1)

    # system tier vs reviewer tier, where a reviewer tier is present
    decision_confusion: list[DecisionConfusionCell]
    escalations: int      # reviewer chose a stricter tier
    de_escalations: int   # reviewer chose a more lenient tier
