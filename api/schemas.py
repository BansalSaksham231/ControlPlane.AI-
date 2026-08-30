"""Request / response models for the ControlPlane API.

These wrap the domain schemas with API-friendly defaults (e.g. a caller
need not send an interaction_id or timestamp). They never expose or
accept ground-truth / evaluation fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from data.schemas import (
    ActionType,
    Application,
    FinalDecision,
    InterventionTier,
    ModelName,
    UserType,
)


class CheckRequest(BaseModel):
    """One AI interaction submitted for a control-plane decision."""

    application: Application
    user_type: UserType = UserType.EXTERNAL_CUSTOMER
    model: ModelName = ModelName.GPT_4O_MINI
    session_id: str = Field(default="SESSION-ADHOC", min_length=1)

    prompt: str = ""
    context: str = ""
    response: str

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=1.0, gt=0)
    tool_calls: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)

    action_type: ActionType = ActionType.INFORMATION
    action_amount_inr: float = Field(default=0.0, ge=0)
    affected_entities: int = Field(default=1, ge=1)

    interaction_id: str | None = None
    timestamp: datetime | None = None

    include_trace: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "application": "customer_support",
                    "session_id": "SESSION-001",
                    "prompt": "What is the refund policy?",
                    "context": (
                        "Company policy allows customers to request a refund within 30 "
                        "business days of purchase, provided the item is unused."
                    ),
                    "response": (
                        "You are eligible for a refund within 30 business days of your "
                        "purchase, as long as the item is unused."
                    ),
                    "action_type": "information",
                },
                {
                    "application": "customer_support",
                    "session_id": "SESSION-002",
                    "prompt": "Confirm the contact details on file.",
                    "context": "Customer asked to confirm contact details for account ACC-227763.",
                    "response": (
                        "The contact details for account ACC-227763 are: Karan Mehta, "
                        "email karan.mehta@example-test.com, phone +91-940847221."
                    ),
                    "action_type": "information",
                    "include_trace": True,
                },
            ]
        }
    }


class CheckResponse(BaseModel):
    interaction_id: str
    decision: FinalDecision
    trace: dict[str, Any] | None = None


class SimulatePolicyRequest(BaseModel):
    """Run one interaction under several application policy profiles."""

    interaction: CheckRequest
    profiles: list[str] = Field(
        default_factory=lambda: [
            "customer_support",
            "internal_knowledge_assistant",
            "decision_support",
        ],
        min_length=1,
    )


class CounterfactualRequest(BaseModel):
    """Re-run one interaction with a few production-visible fields changed."""

    interaction: CheckRequest
    modified_fields: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"example": {"action_amount_inr": 100}},
    )


class FeedbackRequest(BaseModel):
    interaction_id: str
    reviewer_decision: InterventionTier | None = None
    outcome: str | None = None      # approved | modified | rejected
    actual_outcome: str | None = None
    reason: str = ""
    reviewer: str | None = None


class GovernanceActionRequest(BaseModel):
    """A human governance action on an incident investigation."""

    model_config = {"extra": "forbid"}

    action: str = Field(
        description="ACKNOWLEDGE | APPROVE_DECISION | MODIFY_DECISION | "
        "REJECT_DECISION | ESCALATE | CLOSE"
    )
    actor: str = "reviewer"
    comment: str = ""
    # required only for MODIFY_DECISION — the tier the reviewer would have chosen
    reviewer_decision: InterventionTier | None = None


class GovernanceOverrideRequest(BaseModel):
    """
    A reviewer decision on a specific interaction, recorded on the append-only
    governance track. The automated DecisionTrace is never mutated.
    """

    model_config = {"extra": "forbid"}

    interaction_id: str
    #: APPROVE(D) | MODIFY / MODIFIED | REJECT(ED)  (also accepts the long
    #: APPROVE_DECISION / MODIFY_DECISION / REJECT_DECISION forms)
    action_type: str
    #: required for MODIFY — the tier the reviewer would have chosen
    new_tier: InterventionTier | None = None
    justification: str = ""
    reviewer_id: str = "admin"


class InvestigationCounterfactualRequest(BaseModel):
    """'What if?' simulation inputs for an investigation (non-free-text fields only)."""

    model_config = {"extra": "forbid"}

    modified_fields: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"example": {"action_amount_inr": 100}},
    )


class AdaptiveApprovalRequest(BaseModel):
    """Human decision on an adaptive recommendation (approval gate)."""

    model_config = {"extra": "forbid"}

    actor: str = "reviewer"
    comment: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    detectors: list[str]
    checks_served: int
    active_sessions: int
