"""
Data contracts for the Governance Intelligence layer (Phase 9).

Every model sets ``extra="forbid"``. Every field is an aggregate over
existing stored data — nothing here is fabricated, and reviewer feedback
is always named as a *governance signal*, never as ground truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _GovModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ======================================================================
# local, safe-default configuration (NOT config/settings.yaml)
# ======================================================================


class GovernanceConfig(_GovModel):
    """Thresholds for deterministic insight / trend generation. Local only."""

    high_risk_threshold: float = Field(default=0.50, ge=0, le=1)
    low_confidence_threshold: float = Field(default=0.45, ge=0, le=1)

    high_override_rate: float = Field(default=0.30, ge=0, le=1)
    high_human_review_rate: float = Field(default=0.35, ge=0, le=1)
    low_confidence_rate: float = Field(default=0.25, ge=0, le=1)
    rule_dominance_share: float = Field(default=0.50, ge=0, le=1)
    deep_routing_rate: float = Field(default=0.85, ge=0, le=1)
    risk_concentration_share: float = Field(default=0.50, ge=0, le=1)

    # an insight/trend needs at least this much supporting data
    min_application_volume: int = Field(default=10, ge=1)
    min_signal_count: int = Field(default=3, ge=1)

    # trend: |second_half - first_half| <= this  -> STABLE
    trend_stable_band: float = Field(default=0.05, ge=0, le=1)
    # a change is only labelled POTENTIAL_DRIFT above this magnitude
    potential_drift_magnitude: float = Field(default=0.20, ge=0, le=1)
    potential_drift_min_samples: int = Field(default=20, ge=1)


# ======================================================================
# Step 1 — governance overview
# ======================================================================


class TrafficMetrics(_GovModel):
    total_interactions: int = 0
    by_application: dict[str, int] = Field(default_factory=dict)
    by_action_type: dict[str, int] = Field(default_factory=dict)


class DecisionDistribution(_GovModel):
    allow: int = 0
    annotate: int = 0
    verify: int = 0
    human_review: int = 0
    block: int = 0

    allow_rate: float | None = None
    annotate_rate: float | None = None
    verify_rate: float | None = None
    human_review_rate: float | None = None
    block_rate: float | None = None
    human_oversight_rate: float | None = None      # (HUMAN_REVIEW + BLOCK) / total


class RiskDistribution(_GovModel):
    average_risk: float | None = None
    p50_risk: float | None = None
    p95_risk: float | None = None
    max_risk: float | None = None
    high_risk_rate: float | None = None            # risk >= high_risk_threshold
    high_risk_threshold: float


class ConfidenceDistribution(_GovModel):
    average_confidence: float | None = None
    low_confidence_rate: float | None = None
    low_confidence_threshold: float


class VerificationMetrics(_GovModel):
    fast_count: int = 0
    deep_count: int = 0
    fast_rate: float | None = None
    deep_rate: float | None = None
    average_verification_latency_ms: float | None = None
    average_total_latency_ms: float | None = None
    deep_trigger_reason_counts: dict[str, int] = Field(default_factory=dict)


class DetectorContributionMetrics(_GovModel):
    """Incident attribution by detector (not a causal claim)."""

    total_incidents: int = 0
    performance_driven_incidents: int = 0
    responsibility_driven_incidents: int = 0
    cost_driven_incidents: int = 0
    multi_risk_incidents: int = 0
    note: str = (
        "Attribution is by the incident's dominant risk dimension / triggers — "
        "an observational split, not a causal claim."
    )


class PolicyRuleMetric(_GovModel):
    rule: str
    fire_count: int = 0
    fire_rate: float | None = None                  # fired / total_interactions
    tier_changing_count: int = 0
    human_review_count: int = 0                     # fired and moved tier to HUMAN_REVIEW
    block_count: int = 0                            # fired and moved tier to BLOCK


class PolicyBehaviourMetrics(_GovModel):
    total_interactions: int = 0
    rules: list[PolicyRuleMetric] = Field(default_factory=list)


class HumanGovernanceMetrics(_GovModel):
    incidents_investigated: int = 0                 # interactions with >= 1 governance action
    open: int = 0
    acknowledged: int = 0
    reviewed: int = 0
    escalated: int = 0
    closed: int = 0
    action_counts: dict[str, int] = Field(default_factory=dict)


class ReviewerDisagreementMetrics(_GovModel):
    """
    Automated ``original_decision`` vs human ``reviewer_decision`` — kept
    strictly separate. ``reviewer_decision`` is a GOVERNANCE SIGNAL, not
    ground truth and not a correctness label.
    """

    reviewed_count: int = 0                         # MODIFY_DECISION + REJECT_DECISION actions
    override_count: int = 0                         # reviewer disagreed with the automated call
    override_rate: float | None = None
    reviewer_decision_distribution: dict[str, int] = Field(default_factory=dict)
    automated_to_reviewer_transitions: dict[str, int] = Field(default_factory=dict)
    note: str = (
        "reviewer_decision is a human governance signal, NOT ground truth. "
        "override_rate measures reviewer disagreement, not correctness."
    )


class GovernanceOverview(_GovModel):
    generated_at: datetime | None = None
    basis: str = "sequence-based"                   # "sequence-based" | "timestamped"
    config: GovernanceConfig

    traffic: TrafficMetrics
    decisions: DecisionDistribution
    risk: RiskDistribution
    confidence: ConfidenceDistribution
    verification: VerificationMetrics
    detector_contribution: DetectorContributionMetrics
    policy: PolicyBehaviourMetrics
    human_governance: HumanGovernanceMetrics
    reviewer_disagreement: ReviewerDisagreementMetrics

    notes: list[str] = Field(default_factory=list)


# ======================================================================
# Step 2 — application comparison
# ======================================================================


class ApplicationGovernanceMetrics(_GovModel):
    application: str
    volume: int = 0
    allow_rate: float | None = None
    verify_rate: float | None = None
    human_review_rate: float | None = None          # HUMAN_REVIEW tier only
    block_rate: float | None = None
    human_oversight_rate: float | None = None        # HUMAN_REVIEW + BLOCK
    average_risk: float | None = None
    p95_risk: float | None = None
    average_confidence: float | None = None
    low_confidence_rate: float | None = None
    fast_rate: float | None = None
    deep_rate: float | None = None
    incident_rate: float | None = None
    reviewer_override_rate: float | None = None      # governance signal, not ground truth
    average_latency_ms: float | None = None


class ApplicationComparison(_GovModel):
    applications: list[ApplicationGovernanceMetrics] = Field(default_factory=list)
    highest_risk: str | None = None
    highest_volume: str | None = None
    lowest_intervention: str | None = None           # lowest human_oversight_rate
    notes: list[str] = Field(default_factory=list)


# ======================================================================
# Step 3 — governance insights
# ======================================================================


class InsightSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    REVIEW_POLICY = "REVIEW_POLICY"
    REVIEW_THRESHOLD = "REVIEW_THRESHOLD"
    REVIEW_APPLICATION = "REVIEW_APPLICATION"
    REVIEW_DETECTOR = "REVIEW_DETECTOR"
    INVESTIGATE_INCIDENTS = "INVESTIGATE_INCIDENTS"
    NONE = "NONE"


class GovernanceInsight(_GovModel):
    code: str
    severity: InsightSeverity
    title: str
    explanation: str
    supporting_metrics: dict[str, Any] = Field(default_factory=dict)
    affected_applications: list[str] = Field(default_factory=list)
    affected_rules: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
    example_interaction_ids: list[str] = Field(default_factory=list)
    note: str = "A governance insight, not a truth claim."


# ======================================================================
# Step 4 — governance signals (feedback + reviewer overrides, unified)
# ======================================================================


class GovernanceSignalType(str, Enum):
    REVIEWER_OVERRIDE = "reviewer_override"
    FEEDBACK_MODIFIED = "feedback_modified"
    FEEDBACK_REJECTED = "feedback_rejected"
    FEEDBACK_APPROVED = "feedback_approved"


class GovernanceSignal(_GovModel):
    source: GovernanceSignalType
    interaction_id: str
    application: str | None = None
    automated_decision: str
    reviewer_outcome: str                            # reviewer tier or feedback outcome
    is_disagreement: bool
    timestamp: datetime | None = None
    comment: str = ""                                # reviewer/feedback text (never a raw PII span)


class GovernanceSignalSummary(_GovModel):
    signal_count: int = 0
    by_application: dict[str, int] = Field(default_factory=dict)
    by_decision: dict[str, int] = Field(default_factory=dict)      # automated decision
    by_signal_type: dict[str, int] = Field(default_factory=dict)
    override_rate: float | None = None               # disagreement signals / signal_count
    modification_rate: float | None = None
    rejection_rate: float | None = None
    note: str = (
        "Governance signals (reviewer overrides + reviewer feedback) are inputs "
        "for human analysis. They are NOT ground truth and are NOT used in any "
        "production decision."
    )


# ======================================================================
# Step 5 — calibration bridge / recommendations
# ======================================================================


class RecommendationType(str, Enum):
    REVIEW_POLICY = "REVIEW_POLICY"
    REVIEW_THRESHOLD = "REVIEW_THRESHOLD"
    REVIEW_APPLICATION = "REVIEW_APPLICATION"
    REVIEW_DETECTOR = "REVIEW_DETECTOR"
    NO_ACTION = "NO_ACTION"


class RecommendationDisposition(str, Enum):
    RECOMMENDED_FOR_EVALUATION = "RECOMMENDED_FOR_EVALUATION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NO_ACTION = "NO_ACTION"


class GovernanceRecommendation(_GovModel):
    recommendation_type: RecommendationType
    application: str | None = None
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    current_configuration: dict[str, float] | None = None
    candidate_configuration: dict[str, float] | None = None
    expected_tradeoff: str | None = None
    safety_constraints: dict[str, float] | None = None

    disposition: RecommendationDisposition = RecommendationDisposition.REVIEW_REQUIRED
    points_to: list[str] = Field(default_factory=list)   # e.g. ["calibration.sweep", "calibration.select"]
    disclaimer: str = "RECOMMENDATION ONLY — NOT APPLIED TO PRODUCTION."


# ======================================================================
# Step 6 — trends
# ======================================================================


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class TrendSignal(_GovModel):
    metric: str
    direction: TrendDirection
    magnitude: float = Field(ge=0)                   # abs(current - baseline)
    baseline: float | None = None
    current: float | None = None
    severity: InsightSeverity
    label: str = "TREND"                             # "TREND" | "SIGNAL" | "POTENTIAL_DRIFT"
    explanation: str


class GovernanceTrendReport(_GovModel):
    basis: str = "sequence-based (first half vs second half of ordered traces)"
    first_window_n: int = 0
    second_window_n: int = 0
    signals: list[TrendSignal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ======================================================================
# top-level report
# ======================================================================


class GovernanceIntelligenceReport(_GovModel):
    generated_at: datetime | None = None
    overview: GovernanceOverview
    application_comparison: ApplicationComparison
    signals: GovernanceSignalSummary
    signal_details: list[GovernanceSignal] = Field(default_factory=list)
    insights: list[GovernanceInsight] = Field(default_factory=list)
    trends: GovernanceTrendReport
    recommendations: list[GovernanceRecommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
