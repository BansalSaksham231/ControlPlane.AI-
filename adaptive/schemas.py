"""Data contracts for the Adaptive Guardrails layer (Phase 10). All ``extra="forbid"``."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdaptiveConfig(_Model):
    """Local, validated thresholds for adaptive recommendation. PROVISIONAL."""

    # a pattern must reach this severity rank to trigger a recommendation
    min_pattern_incidents: int = Field(default=3, ge=1)
    # safety constraints used when running the counterfactual bridge
    minimum_recall: float = Field(default=0.90, ge=0, le=1)
    minimum_precision: float = Field(default=0.0, ge=0, le=1)
    # recommendations older than this many report-builds are EXPIRED (system: never)
    expire_after_builds: int | None = None


# ======================================================================
# recommendation
# ======================================================================


class RecommendationType(str, Enum):
    REVIEW_POLICY = "REVIEW_POLICY"
    REVIEW_VERIFICATION_THRESHOLD = "REVIEW_VERIFICATION_THRESHOLD"
    REVIEW_APPLICATION = "REVIEW_APPLICATION"
    REVIEW_DETECTOR = "REVIEW_DETECTOR"
    INVESTIGATE_DRIFT = "INVESTIGATE_DRIFT"
    NO_ACTION = "NO_ACTION"


class RecommendationStatus(str, Enum):
    DETECTED = "DETECTED"
    SIMULATED = "SIMULATED"
    RECOMMENDED_FOR_REVIEW = "RECOMMENDED_FOR_REVIEW"
    APPROVED = "APPROVED_FOR_EVALUATION"          # NEVER "APPLIED_TO_PRODUCTION"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RecommendationSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CounterfactualEvaluation(_Model):
    """
    Reuses ``calibration.select.ConfigurationSelection`` output. Safety is
    evaluated FIRST — a candidate that fails a safety constraint is never
    'compensated' by better latency / FAST rate.
    """

    current_configuration: dict[str, float] = Field(default_factory=dict)
    candidate_configuration: dict[str, float] | None = None

    current_decision_distribution: dict[str, int] = Field(default_factory=dict)
    candidate_decision_distribution: dict[str, int] = Field(default_factory=dict)

    current_recall: float | None = None
    candidate_recall: float | None = None
    current_precision: float | None = None
    candidate_precision: float | None = None
    current_false_positive_rate: float | None = None
    candidate_false_positive_rate: float | None = None
    current_missed_risk_rate: float | None = None
    candidate_missed_risk_rate: float | None = None

    current_fast_rate: float | None = None
    candidate_fast_rate: float | None = None
    current_deep_rate: float | None = None
    candidate_deep_rate: float | None = None
    current_human_review_rate: float | None = None
    candidate_human_review_rate: float | None = None
    current_average_latency_ms: float | None = None
    candidate_average_latency_ms: float | None = None

    safety_constraints: dict[str, float] = Field(default_factory=dict)
    safety_passed: bool = False
    safety_violations: list[str] = Field(default_factory=list)
    candidate_found: bool = False
    selection_reason: str = ""
    note: str = (
        "Safety constraints are evaluated FIRST via calibration.select. If no "
        "candidate satisfies them, no configuration change is recommended — "
        "efficiency never compensates for unsafe recall/precision."
    )


class ApprovalRecord(_Model):
    recommendation_id: str
    decision: str                                # "APPROVED_FOR_EVALUATION" | "REJECTED"
    actor: str = "reviewer"
    comment: str = ""
    sequence: int = 0
    disclaimer: str = (
        "Approval marks the candidate APPROVED_FOR_EVALUATION. It does NOT apply "
        "the candidate to production; config/settings.yaml is unchanged."
    )


class AdaptiveRecommendation(_Model):
    recommendation_id: str
    type: RecommendationType
    application: str | None = None
    severity: RecommendationSeverity
    trigger_patterns: list[str] = Field(default_factory=list)   # pattern ids / codes
    evidence: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    proposed_change: str

    current_configuration: dict[str, float] | None = None
    candidate_configuration: dict[str, float] | None = None
    simulation_result: CounterfactualEvaluation | None = None
    safety_constraints: dict[str, float] | None = None
    expected_tradeoff: str | None = None

    status: RecommendationStatus = RecommendationStatus.RECOMMENDED_FOR_REVIEW
    approval: ApprovalRecord | None = None
    points_to: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "RECOMMENDATION ONLY. Approval means APPROVED_FOR_EVALUATION, never applied "
        "to production. config/settings.yaml is never modified by this system."
    )


# ======================================================================
# top-level adaptive governance report
# ======================================================================


class AdaptiveGovernanceReport(_Model):
    generated_at: datetime | None = None
    config: AdaptiveConfig

    observation: str
    pattern_count: int = 0
    drift_signal_count: int = 0
    potential_drift_count: int = 0
    reviewer_override_count: int = 0

    top_patterns: list[dict[str, Any]] = Field(default_factory=list)
    drift_signals: list[dict[str, Any]] = Field(default_factory=list)
    reviewer_override_patterns: list[dict[str, Any]] = Field(default_factory=list)

    recommendations: list[AdaptiveRecommendation] = Field(default_factory=list)
    approved_for_evaluation_count: int = 0
    rejected_count: int = 0

    production_configuration_status: str = "UNCHANGED"
    notes: list[str] = Field(default_factory=list)
