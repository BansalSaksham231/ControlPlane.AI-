"""Data contracts for Incident Intelligence (Phase 10). All ``extra="forbid"``."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ======================================================================
# local configuration  (NOT config/settings.yaml — never mutated at runtime)
# ======================================================================


class Phase10IncidentConfig(_Model):
    """
    Thresholds for deterministic incident grouping / pattern / drift
    classification. Local, validated, safe defaults. PROVISIONAL — chosen
    for sensible behaviour, not empirically tuned.
    """

    # a cluster is a "recurring" pattern at/above this incident count
    recurring_min_incidents: int = Field(default=3, ge=1)
    # a pattern needs at least this many incidents to be reported
    pattern_min_incidents: int = Field(default=3, ge=1)
    # low-confidence-pattern: cluster mean decision_confidence below this
    low_confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    # high-risk-pattern: incident overall_risk at/above this
    high_risk_threshold: float = Field(default=0.50, ge=0, le=1)
    # override pattern: reviewer disagreement rate for a transition at/above this
    high_override_rate: float = Field(default=0.30, ge=0, le=1)
    # deep-routing concentration: cluster / application DEEP share at/above this
    deep_routing_rate: float = Field(default=0.85, ge=0, le=1)
    # policy-rule dominance: a rule drives at/above this share of tier moves
    rule_dominance_share: float = Field(default=0.50, ge=0, le=1)
    # detector dominance: a dimension drives at/above this share of incidents
    detector_dominance_share: float = Field(default=0.50, ge=0, le=1)
    # risk concentration: one application holds at/above this share of high-risk
    risk_concentration_share: float = Field(default=0.50, ge=0, le=1)

    # drift: |recent - historical| <= this  -> STABLE
    drift_stable_band: float = Field(default=0.05, ge=0, le=1)
    # a change is only POTENTIAL_DRIFT above this magnitude
    drift_potential_magnitude: float = Field(default=0.15, ge=0, le=1)
    # ...and only with at least this many samples per window
    drift_min_window_samples: int = Field(default=15, ge=1)

    # incident surge: recent incident rate exceeds historical by this factor
    surge_factor: float = Field(default=1.5, ge=1.0)


# ======================================================================
# incident record  (PII-safe view of one flagged interaction)
# ======================================================================


class IncidentRecord(_Model):
    incident_id: str
    interaction_id: str
    timestamp: datetime
    application: str
    action_type: str

    decision: str
    overall_risk: float = Field(ge=0, le=1)
    decision_confidence: float = Field(ge=0, le=1)
    verification_path: str
    dominant_dimension: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    tier_changing_rules: list[str] = Field(default_factory=list)
    human_review_required: bool = False

    performance_risk: float = Field(ge=0, le=1)
    responsibility_risk: float = Field(ge=0, le=1)
    cost_risk: float = Field(ge=0, le=1)
    multi_risk: bool = False

    consequence_band: str
    criticality_band: str
    incident_severity: str
    incident_triggers: list[str] = Field(default_factory=list)

    # governance / feedback overlay (a governance signal — NOT ground truth)
    reviewer_signal: str | None = None            # e.g. "MODIFY_DECISION -> HUMAN_REVIEW"
    feedback_signal: str | None = None            # e.g. "modified" / "rejected" / "approved"

    signature: str                                # deterministic grouping key


# ======================================================================
# cluster
# ======================================================================


class IncidentCluster(_Model):
    cluster_id: str
    pattern_signature: str                        # human-readable
    incident_count: int = 0
    is_recurring: bool = False
    affected_applications: list[str] = Field(default_factory=list)
    representative_incidents: list[str] = Field(default_factory=list)   # interaction_ids
    average_risk: float | None = None
    max_risk: float | None = None
    average_confidence: float | None = None
    decisions: dict[str, int] = Field(default_factory=dict)
    verification_paths: dict[str, int] = Field(default_factory=dict)
    dominant_dimension: str | None = None
    dominant_reason_codes: list[str] = Field(default_factory=list)
    dominant_policy_rules: list[str] = Field(default_factory=list)
    reviewer_signal_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None


# ======================================================================
# pattern
# ======================================================================


class PatternType(str, Enum):
    HIGH_OVERRIDE_PATTERN = "HIGH_OVERRIDE_PATTERN"
    REPEATED_BLOCK_PATTERN = "REPEATED_BLOCK_PATTERN"
    REPEATED_VERIFY_PATTERN = "REPEATED_VERIFY_PATTERN"
    DEEP_ROUTING_CONCENTRATION = "DEEP_ROUTING_CONCENTRATION"
    LOW_CONFIDENCE_PATTERN = "LOW_CONFIDENCE_PATTERN"
    RISK_CONCENTRATION = "RISK_CONCENTRATION"
    POLICY_RULE_DOMINANCE = "POLICY_RULE_DOMINANCE"
    DETECTOR_DOMINANCE = "DETECTOR_DOMINANCE"
    INCIDENT_SURGE = "INCIDENT_SURGE"
    CROSS_APPLICATION_PATTERN = "CROSS_APPLICATION_PATTERN"


class PatternSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IncidentPattern(_Model):
    pattern_id: str
    type: PatternType
    severity: PatternSeverity
    applications: list[str] = Field(default_factory=list)
    incident_count: int = 0
    # confidence in the PATTERN DETECTION (sample size + effect size) —
    # NOT correctness of the underlying AI responses.
    detection_confidence: float = Field(ge=0, le=1)
    detection_confidence_note: str = (
        "Confidence that this operational pattern is real given the sample — "
        "NOT a statement about whether any AI response or decision was correct."
    )
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    affected_dimension: str | None = None
    affected_policy_rule: str | None = None
    representative_incidents: list[str] = Field(default_factory=list)
    cluster_ids: list[str] = Field(default_factory=list)
    recommended_next_step: str = "INVESTIGATE_INCIDENTS"


# ======================================================================
# attribution
# ======================================================================


class AttributionResult(_Model):
    pattern_id: str
    incident_count: int = 0
    dominant_dimension: str | None = None
    dimension_shares: dict[str, float] = Field(default_factory=dict)
    reason_code_shares: dict[str, float] = Field(default_factory=dict)
    decision_shares: dict[str, float] = Field(default_factory=dict)
    dominant_policy_rule: str | None = None
    policy_rule_shares: dict[str, float] = Field(default_factory=dict)
    reviewer_override_share: float = Field(default=0.0, ge=0, le=1)
    narrative: str
    disclaimer: str = (
        "Observed association / attribution, not causal proof. Percentages describe "
        "which factors were observed alongside the pattern, not what caused it."
    )


# ======================================================================
# drift
# ======================================================================


class DriftDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class DriftSignal(_Model):
    metric: str
    scope: str                                    # "global" | "application:<x>" | "detector:<x>" | "policy_rule:<x>"
    baseline: float | None = None
    recent: float | None = None
    delta: float | None = None
    direction: DriftDirection = DriftDirection.STABLE
    signal: str = "STABLE"                         # "STABLE" | "TREND" | "POTENTIAL_DRIFT"
    sample_sufficient: bool = True
    explanation: str


class DriftReport(_Model):
    basis: str = "sequence-based (historical half vs recent half of ordered traces)"
    historical_window_n: int = 0
    recent_window_n: int = 0
    potential_drift_count: int = 0
    signals: list[DriftSignal] = Field(default_factory=list)
    disclaimer: str = (
        "This is an operational drift signal, not proof of model degradation. "
        "No statistical-significance test is performed."
    )


# ======================================================================
# feedback / reviewer-override patterns
# ======================================================================


class ReviewerOverridePattern(_Model):
    pattern_id: str
    application: str
    original_decision: str
    reviewer_decision: str
    transition: str                               # "BLOCK -> HUMAN_REVIEW"
    count: int = 0
    override_rate: float = Field(ge=0, le=1)       # for this (app, original) pair
    affected_reason_codes: list[str] = Field(default_factory=list)
    affected_policy_rules: list[str] = Field(default_factory=list)
    representative_incidents: list[str] = Field(default_factory=list)
    note: str = (
        "Reviewer disagreement is an operational governance signal, NOT evidence "
        "that the automated decision was incorrect."
    )


# ======================================================================
# top-level report
# ======================================================================


class IncidentIntelligenceReport(_Model):
    generated_at: datetime | None = None
    config: Phase10IncidentConfig
    total_incidents: int = 0
    incidents: list[IncidentRecord] = Field(default_factory=list)
    clusters: list[IncidentCluster] = Field(default_factory=list)
    patterns: list[IncidentPattern] = Field(default_factory=list)
    attributions: list[AttributionResult] = Field(default_factory=list)
    reviewer_override_patterns: list[ReviewerOverridePattern] = Field(default_factory=list)
    drift: DriftReport
    notes: list[str] = Field(default_factory=list)
