"""
Canonical data contracts for the ControlPlane.ai prototype.

All downstream modules (detectors, risk fusion, consequence engine,
policy engine, decision engine) must consume and produce these schemas.

This module is intentionally framework-agnostic:
- No FastAPI imports.
- No pandas / numpy / sklearn / ML library imports.
- No configuration loading.
- No detector implementation logic.

It defines data contracts only.

Production vs. evaluation boundary
-----------------------------------
``Interaction`` contains ONLY information a real-time ControlPlane
middleware can observe BEFORE detectors run. It must never contain
grounding_score, confidence, consequence factors, ground-truth labels,
or expected/final decisions — those are computed by later layers
(performance detector, responsibility detector, cost detector,
consequence engine, decision engine) and belong on ``EvaluationCase``
instead, not on the production interaction itself.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ==================================================
# ENUMS
# ==================================================


class Application(str, Enum):
    """The product surface an interaction originated from."""

    CUSTOMER_SUPPORT = "customer_support"
    INTERNAL_KNOWLEDGE_ASSISTANT = "internal_knowledge_assistant"
    DECISION_SUPPORT = "decision_support"


class UserType(str, Enum):
    """The category of end user involved in the interaction."""

    EXTERNAL_CUSTOMER = "external_customer"
    EMPLOYEE = "employee"
    MANAGER = "manager"
    OPERATIONS_AGENT = "operations_agent"


class ModelName(str, Enum):
    """The underlying AI model that produced the response."""

    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_FLASH = "gemini-flash"
    LOCAL_SMALL_MODEL = "local-small-model"


class ActionType(str, Enum):
    """The type of action the AI response represents or triggers."""

    INFORMATION = "information"
    REFUND = "refund"
    ACCOUNT_UPDATE = "account_update"
    ACCOUNT_CANCELLATION = "account_cancellation"
    EXTERNAL_COMMUNICATION = "external_communication"
    RECOMMENDATION = "recommendation"


class InterventionTier(str, Enum):
    """The set of possible oversight interventions the decision engine may select."""

    ALLOW = "ALLOW"
    ANNOTATE = "ANNOTATE"
    VERIFY = "VERIFY"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


class ResponseStatus(str, Enum):
    """The evaluated support status of a response relative to context/evidence."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class RiskDimension(str, Enum):
    """The top-level risk dimensions tracked by ControlPlane."""

    PERFORMANCE = "PERFORMANCE"
    RESPONSIBILITY = "RESPONSIBILITY"
    COST = "COST"


# ==================================================
# INTERACTION (production-visible only)
# ==================================================


class Interaction(BaseModel):
    """
    Raw AI interaction data available to ControlPlane at inference time.

    This model represents production input only. It must never contain
    ground-truth evaluation labels (e.g. ground_truth_hallucination,
    ground_truth_pii, ground_truth_bias, ground_truth_toxicity,
    ground_truth_cost, expected_decision), nor detector-computed fields
    (grounding_score, confidence, consequence factors). Those belong to
    the evaluation layer / detector outputs, not to detector input.
    """

    interaction_id: str = Field(min_length=1)
    timestamp: datetime

    application: Application
    user_type: UserType
    model: ModelName
    session_id: str

    prompt: str
    context: str
    response: str

    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(gt=0)
    tool_calls: int = Field(ge=0)
    retry_count: int = Field(ge=0)

    action_type: ActionType
    action_amount_inr: float = Field(ge=0)
    affected_entities: int = Field(ge=1)


# ==================================================
# DETECTOR RESULT
# ==================================================


class DetectorResult(BaseModel):
    """Generic output contract produced by any detector (performance, responsibility, cost)."""

    detector_name: str

    risk_score: float = Field(ge=0, le=1)

    status: str

    confidence: float = Field(ge=0, le=1)

    explanation: str

    latency_ms: float = Field(ge=0)

    evidence: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================================================
# CONSEQUENCE FACTORS
# ==================================================


class ConsequenceFactors(BaseModel):
    """
    Factors describing the consequence of an action if it were wrong,
    independent of the probability that it is wrong.
    """

    financial_impact: float = Field(ge=0, le=1)
    reversibility: float = Field(ge=0, le=1)
    sensitivity: float = Field(ge=0, le=1)
    blast_radius: float = Field(ge=0, le=1)
    action_automation: float = Field(ge=0, le=1)

    consequence_score: float = Field(ge=0, le=1)


# ==================================================
# EVALUATION CASE (evaluation-only, extends Interaction)
# ==================================================


class EvaluationCase(Interaction):
    """
    An evaluation-only record: a production ``Interaction`` extended with
    ground-truth labels, ground-truth risk scores, the expected policy
    outcome, consequence factors, and detector-produced placeholders.

    This model must never be handed to a detector as input — detectors
    consume ``Interaction`` only. ``EvaluationCase`` exists purely for
    scoring/benchmarking detector and decision-engine output against
    known-correct answers.
    """

    # --- ground-truth risk labels ---
    ground_truth_hallucination: bool
    ground_truth_pii: bool
    ground_truth_toxicity: bool
    ground_truth_bias: bool
    ground_truth_cost_anomaly: bool

    # --- ground-truth per-dimension risk scores ---
    ground_truth_performance_risk: float = Field(ge=0, le=1)
    ground_truth_responsibility_risk: float = Field(ge=0, le=1)
    ground_truth_cost_risk: float = Field(ge=0, le=1)

    # --- expected policy outcome ---
    human_review_expected: bool
    expected_decision: InterventionTier
    final_outcome: str

    # --- consequence factors (independent of likelihood of being wrong) ---
    financial_impact: float = Field(ge=0, le=1)
    reversibility: float = Field(ge=0, le=1)
    sensitivity: float = Field(ge=0, le=1)
    blast_radius: float = Field(ge=0, le=1)
    action_automation: float = Field(ge=0, le=1)
    consequence_score: float = Field(ge=0, le=1)

    # --- detector-produced placeholders (unset until detectors run) ---
    grounding_score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    # --- free-text evaluator commentary ---
    notes: str | None = None


# ==================================================
# FINAL DECISION
# ==================================================


class FinalDecision(BaseModel):
    """The final oversight decision produced by the decision engine."""

    performance_risk: float = Field(ge=0, le=1)
    responsibility_risk: float = Field(ge=0, le=1)
    cost_risk: float = Field(ge=0, le=1)

    consequence: ConsequenceFactors

    decision: InterventionTier

    overall_risk: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)

    explanation: str

    triggered_rules: list[str] = Field(default_factory=list)

    # Round 2 upgrade: canonical machine-readable reasons (see
    # ``common.reason_codes``). Optional / defaulted for backward
    # compatibility with any caller built against the original schema.
    reason_codes: list[str] = Field(default_factory=list)

    # Progressive Verification (Phase 2): whether this decision used the
    # FAST path (cheap checks only) or the DEEP path (full verification).
    # Defaults to "DEEP" to preserve compatibility with existing callers.
    verification_path: str = "DEEP"

    timestamp: datetime