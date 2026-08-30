"""
Pydantic data contracts for the enterprise observability layer.

Every field is an aggregate over real ``DecisionTrace`` records. Nothing
here is fabricated; a metric whose denominator is zero is ``None``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# ==================================================
# WINDOW
# ==================================================


class MonitoringWindow(BaseModel):
    """The (optional) time window a report was computed over."""

    start_time: datetime | None = None
    end_time: datetime | None = None
    applied: bool = False               # True when at least one bound was supplied
    traces_in_window: int = 0
    traces_excluded_by_window: int = 0


# ==================================================
# TRAFFIC / DECISIONS
# ==================================================


class DecisionMetrics(BaseModel):
    """Distribution of final intervention tiers."""

    allow_count: int = 0
    annotate_count: int = 0
    verify_count: int = 0
    human_review_count: int = 0
    block_count: int = 0

    human_review_rate: float | None = None     # HUMAN_REVIEW tier / total
    block_rate: float | None = None            # BLOCK tier / total
    human_oversight_rate: float | None = None  # (HUMAN_REVIEW + BLOCK) / total


class RiskStats(BaseModel):
    """Summary statistics for ``overall_risk`` across the window."""

    mean_overall_risk: float | None = None
    p50_overall_risk: float | None = None
    p95_overall_risk: float | None = None
    max_overall_risk: float | None = None


class ConfidenceMetrics(BaseModel):
    mean_decision_confidence: float | None = None
    low_confidence_count: int = 0
    low_confidence_rate: float | None = None
    low_confidence_threshold: float


class LatencyStats(BaseModel):
    mean_total_latency_ms: float | None = None
    p50_total_latency_ms: float | None = None
    p95_total_latency_ms: float | None = None
    max_total_latency_ms: float | None = None


# ==================================================
# RISK DISTRIBUTION
# ==================================================


class RiskBucket(BaseModel):
    bucket_name: str
    min_risk: float
    max_risk: float
    count: int = 0
    percentage: float | None = None            # % of window traffic in this bucket


class RiskDistribution(BaseModel):
    buckets: list[RiskBucket] = Field(default_factory=list)
    total: int = 0


# ==================================================
# VERIFICATION (FAST vs DEEP)
# ==================================================


class VerificationMetrics(BaseModel):
    fast_count: int = 0
    deep_count: int = 0
    unknown_count: int = 0
    fast_rate: float | None = None
    deep_rate: float | None = None

    mean_total_latency_fast_ms: float | None = None
    mean_total_latency_deep_ms: float | None = None
    # From VerificationReport.total_verification_latency_ms, where present.
    mean_verification_latency_fast_ms: float | None = None
    mean_verification_latency_deep_ms: float | None = None

    deep_trigger_reason_counts: dict[str, int] = Field(default_factory=dict)


class ApplicationVerificationSplit(BaseModel):
    application: str
    fast_count: int = 0
    deep_count: int = 0
    fast_rate: float | None = None
    deep_rate: float | None = None
    mean_total_latency_fast_ms: float | None = None
    mean_total_latency_deep_ms: float | None = None


# ==================================================
# DETECTOR HEALTH
# ==================================================


class DetectorHealth(BaseModel):
    detector: str                              # performance / responsibility / cost
    invocation_count: int = 0
    mean_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    max_latency_ms: float | None = None
    # The trace structure does not record detector errors, so this is
    # deliberately ``None`` — never a fabricated ``0``.
    error_count: int | None = None


# ==================================================
# APPLICATION / MODEL BREAKDOWN
# ==================================================


class ApplicationMetrics(BaseModel):
    application: str
    interactions: int = 0
    mean_risk: float | None = None
    p95_risk: float | None = None
    deep_rate: float | None = None
    human_review_rate: float | None = None
    block_rate: float | None = None
    mean_latency_ms: float | None = None
    low_confidence_rate: float | None = None


class ModelMetrics(BaseModel):
    model: str
    interaction_count: int = 0
    mean_risk: float | None = None
    deep_rate: float | None = None
    human_review_rate: float | None = None
    block_rate: float | None = None
    mean_latency_ms: float | None = None
    # Differences between models are an OBSERVED ASSOCIATION over this
    # traffic sample, not evidence that the model causes the risk.
    interpretation: str = "observed association only — not a causal claim"


# ==================================================
# POLICY / REASON-CODE MONITORING
# ==================================================


class RuleMetrics(BaseModel):
    rule: str
    fired_count: int                            # only rule_trace entries with fired == True


class ReasonCodeMetrics(BaseModel):
    reason_code: str
    count: int


# ==================================================
# ATTENTION SIGNALS
# ==================================================


class AttentionSignal(BaseModel):
    """
    A deterministic operational attention signal — NOT a statistically
    detected anomaly. It flags that an aggregate has crossed a
    configurable threshold and may warrant a human look.
    """

    code: str
    severity: str                               # "WARNING" | "CRITICAL"
    observed_value: float
    threshold: float
    explanation: str


# ==================================================
# TREND
# ==================================================


class TrendBucket(BaseModel):
    bucket_start: datetime
    interaction_count: int = 0
    mean_risk: float | None = None
    deep_rate: float | None = None
    human_review_rate: float | None = None
    block_rate: float | None = None


# ==================================================
# DATA QUALITY
# ==================================================


class DataQualityReport(BaseModel):
    total_records_seen: int = 0
    valid_records: int = 0
    invalid_records_skipped: int = 0
    # field name -> number of valid traces missing / defaulting it
    missing_trace_fields: dict[str, int] = Field(default_factory=dict)
    # human-readable reasons records were excluded
    exclusion_reasons: list[str] = Field(default_factory=list)


# ==================================================
# TOP-LEVEL REPORT
# ==================================================


class MonitoringReport(BaseModel):
    generated_at: datetime | None = None
    window: MonitoringWindow

    total_interactions: int = 0

    decisions: DecisionMetrics
    risk: RiskStats
    confidence: ConfidenceMetrics
    latency: LatencyStats
    verification: VerificationMetrics

    risk_distribution: RiskDistribution

    reason_codes: list[ReasonCodeMetrics] = Field(default_factory=list)
    policy_rules: list[RuleMetrics] = Field(default_factory=list)

    applications: list[ApplicationMetrics] = Field(default_factory=list)
    models: list[ModelMetrics] = Field(default_factory=list)
    detectors: list[DetectorHealth] = Field(default_factory=list)

    verification_by_application: list[ApplicationVerificationSplit] = Field(
        default_factory=list
    )

    attention_signals: list[AttentionSignal] = Field(default_factory=list)
    trend_granularity: str = "hourly"
    trend: list[TrendBucket] = Field(default_factory=list)

    data_quality: DataQualityReport

    notes: list[str] = Field(default_factory=list)


# ======================================================================
# PHASE 8 — OPERATIONAL RISK MONITORING + INCIDENT INTELLIGENCE
# ======================================================================
# These models sit on top of the Phase-5 aggregation. They add incident
# intelligence, first-half/second-half trend direction, a recent-vs-
# historical operational-shift signal, and a governance feedback summary.
# Every field is derived from real DecisionTrace / FeedbackRecord data —
# never from ground truth, and never by re-running the pipeline.


class MonitoringConfig(BaseModel):
    """
    Local monitoring thresholds with safe defaults. Kept OUT of
    ``config/settings.yaml`` so tuning the monitor never touches
    production decision behaviour.
    """

    model_config = ConfigDict(extra="forbid")

    low_confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    high_consequence_threshold: float = Field(default=0.60, ge=0, le=1)
    high_criticality_threshold: float = Field(default=0.60, ge=0, le=1)
    elevated_risk_threshold: float = Field(default=0.50, ge=0, le=1)
    critical_risk_threshold: float = Field(default=0.85, ge=0, le=1)
    detector_high_risk_threshold: float = Field(default=0.50, ge=0, le=1)

    # |second_half - first_half| <= this  -> STABLE
    trend_stable_band: float = Field(default=0.05, ge=0, le=1)
    # last <fraction> of chronologically-ordered traces = the "recent" window
    shift_recent_fraction: float = Field(default=0.25, gt=0, lt=1)
    # |recent - historical| <= this  -> "flat"
    shift_flat_band: float = Field(default=0.03, ge=0, le=1)

    risk_buckets: list[tuple[str, float, float]] = Field(
        default_factory=lambda: [
            ("LOW", 0.0, 0.20),
            ("MODERATE", 0.20, 0.50),
            ("HIGH", 0.50, 0.75),
            ("CRITICAL", 0.75, 1.01),
        ]
    )


class MonitoringSnapshot(BaseModel):
    """A single flat point-in-time picture of the monitored traffic."""

    model_config = ConfigDict(extra="forbid")

    total_interactions: int = 0

    allow_count: int = 0
    annotate_count: int = 0
    verify_count: int = 0
    human_review_count: int = 0
    block_count: int = 0

    allow_rate: float = Field(default=0.0, ge=0, le=1)
    annotate_rate: float = Field(default=0.0, ge=0, le=1)
    verify_rate: float = Field(default=0.0, ge=0, le=1)
    human_review_rate: float = Field(default=0.0, ge=0, le=1)
    block_rate: float = Field(default=0.0, ge=0, le=1)

    average_risk: float = Field(default=0.0, ge=0, le=1)
    p95_risk: float = Field(default=0.0, ge=0, le=1)

    average_confidence: float = Field(default=0.0, ge=0, le=1)
    low_confidence_rate: float = Field(default=0.0, ge=0, le=1)

    fast_path_rate: float = Field(default=0.0, ge=0, le=1)
    deep_path_rate: float = Field(default=0.0, ge=0, le=1)

    average_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)

    average_cost_risk: float = Field(default=0.0, ge=0, le=1)
    high_consequence_rate: float = Field(default=0.0, ge=0, le=1)
    high_criticality_rate: float = Field(default=0.0, ge=0, le=1)
    multi_risk_rate: float = Field(default=0.0, ge=0, le=1)


class ApplicationRiskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application: str
    interaction_count: int = 0
    average_risk: float = Field(default=0.0, ge=0, le=1)
    p95_risk: float = Field(default=0.0, ge=0, le=1)
    average_confidence: float = Field(default=0.0, ge=0, le=1)
    decision_distribution: dict[str, int] = Field(default_factory=dict)
    human_review_rate: float = Field(default=0.0, ge=0, le=1)
    block_rate: float = Field(default=0.0, ge=0, le=1)
    fast_path_rate: float = Field(default=0.0, ge=0, le=1)
    deep_path_rate: float = Field(default=0.0, ge=0, le=1)
    high_consequence_rate: float = Field(default=0.0, ge=0, le=1)
    high_criticality_rate: float = Field(default=0.0, ge=0, le=1)
    dominant_risk_dimension: str | None = None


class DetectorRiskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: str                                    # performance / responsibility / cost
    interaction_coverage: int = 0
    average_risk: float = Field(default=0.0, ge=0, le=1)
    high_risk_rate: float = Field(default=0.0, ge=0, le=1)
    mean_weighted_contribution: float | None = None   # from fusion.risk_breakdown
    dominant_dimension_count: int = 0


class ReasonCodeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str
    count: int = 0
    share_of_interventions: float = Field(default=0.0, ge=0, le=1)


class VerificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fast_count: int = 0
    deep_count: int = 0
    fast_rate: float = Field(default=0.0, ge=0, le=1)
    deep_rate: float = Field(default=0.0, ge=0, le=1)

    deep_trigger_reason_counts: dict[str, int] = Field(default_factory=dict)

    average_fast_latency_ms: float | None = None
    average_deep_latency_ms: float | None = None
    average_total_verification_latency_ms: float | None = None
    p95_total_verification_latency_ms: float | None = None

    # --- deterministic semantic bypass (Phase: cascade router) ---------------
    # DEEP interactions where claim extraction / TF-IDF / NLI were skipped
    # because a deterministic hard boundary (critical outbound PII) will be
    # blocked by the policy layer regardless of the grounding score.
    semantic_bypass_count: int = 0
    # share of DEEP interactions that were bypassed
    semantic_bypass_rate_of_deep: float = Field(default=0.0, ge=0, le=1)
    # rough compute saved: bypass_count x the mean full-DEEP verification cost
    # observed on the non-bypassed DEEP interactions. ``None`` if unmeasurable.
    estimated_bypass_compute_saved_ms: float | None = None


class MultiTurnSummary(BaseModel):
    """
    Multi-turn session accumulation, aggregated from ``trace.session``. Traces
    with no session block (single-turn / stateless callers) are ignored.
    """

    model_config = ConfigDict(extra="forbid")

    total_sessions: int = 0
    multi_turn_sessions: int = 0                       # sessions with >= 2 recorded turns
    sessions_hitting_critical_floor: int = 0
    # of the multi-turn sessions, the share that hit the non-decaying critical
    # floor (a BLOCK / critical-PII / severe-toxicity turn forcing elevated
    # scrutiny on every later turn). ``None`` when there are no multi-turn sessions.
    critical_floor_session_rate: float | None = Field(default=None, ge=0, le=1)
    critical_floor_events: int = 0                     # total critical events across all sessions


class IncidentSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


class IncidentSummary(BaseModel):
    """
    One interaction flagged for operational attention. Contains only
    structured, already-safe fields from the trace — no claim text, no
    response text, no ``matched_text``.
    """

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    application: str
    timestamp: datetime
    action_type: str
    decision: str
    overall_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    dominant_dimension: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    verification_path: str
    consequence_score: float = Field(ge=0, le=1)
    criticality: float = Field(ge=0, le=1)
    requires_human_review: bool = False

    severity: IncidentSeverity
    triggers: list[str] = Field(default_factory=list)
    severity_rationale: str = ""


class IncidentDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 0
    incident_rate: float = Field(default=0.0, ge=0, le=1)
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_application: dict[str, int] = Field(default_factory=dict)
    by_trigger: dict[str, int] = Field(default_factory=dict)
    incident_definition: str = ""


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class MetricTrend(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    first_half_value: float | None = None
    second_half_value: float | None = None
    delta: float | None = None
    direction: TrendDirection = TrendDirection.STABLE
    first_half_n: int = 0
    second_half_n: int = 0


class TrendAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordered_by_timestamp: bool = True
    stable_band: float = Field(default=0.05, ge=0, le=1)
    metrics: list[MetricTrend] = Field(default_factory=list)
    method: str = (
        "deterministic first-half vs second-half comparison of "
        "chronologically-ordered traces; no statistical-significance claim"
    )


class OperationalShift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_value: float | None = None      # historical window
    recent_value: float | None = None
    delta: float | None = None
    direction: str = "flat"                   # "up" | "down" | "flat"
    baseline_n: int = 0
    recent_n: int = 0


class OperationalShiftReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_fraction: float = Field(ge=0, le=1)
    recent_window_size: int = 0
    baseline_window_size: int = 0
    flat_band: float = Field(ge=0, le=1)
    shifts: list[OperationalShift] = Field(default_factory=list)
    disclaimer: str = (
        "operational_shift compares a recent vs a historical window of "
        "ALREADY-DECIDED traces. It is NOT AI/model-drift detection — no "
        "model output distribution is measured."
    )


class MonitoredFeedbackSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_available: bool = False
    feedback_count: int = 0
    interactions_with_feedback: int = 0
    approved: int = 0
    modified: int = 0
    rejected: int = 0
    override_count: int = 0
    override_rate: float | None = None
    approval_rate: float | None = None
    note: str = (
        "Feedback is a human governance signal, not ground truth. Override "
        "rate measures reviewer disagreement with ControlPlane, not "
        "correctness of the AI or the decision."
    )


class OperationalMonitoringReport(BaseModel):
    """
    Phase 8 top-level report. (Named distinctly from the Phase-5
    ``MonitoringReport`` it builds on, to avoid a breaking rename.)
    """

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime | None = None
    total_interactions: int = 0
    config: MonitoringConfig

    snapshot: MonitoringSnapshot
    risk_distribution: RiskDistribution
    applications: list[ApplicationRiskSummary] = Field(default_factory=list)
    detectors: list[DetectorRiskSummary] = Field(default_factory=list)
    reason_codes: list[ReasonCodeSummary] = Field(default_factory=list)
    verification: VerificationSummary
    multi_turn: MultiTurnSummary = Field(default_factory=MultiTurnSummary)

    incidents: list[IncidentSummary] = Field(default_factory=list)
    incident_digest: IncidentDigest

    trend: TrendAnalysis
    operational_shift: OperationalShiftReport
    feedback: MonitoredFeedbackSummary

    data_quality: DataQualityReport
    notes: list[str] = Field(default_factory=list)
