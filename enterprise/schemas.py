"""Data contracts for the Enterprise Command Center (Phase 11). All ``extra="forbid"``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Ent(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ======================================================================
# A. executive KPI strip
# ======================================================================


class ExecutiveKpiStrip(_Ent):
    has_data: bool = False
    total_interactions: int = 0
    allow_rate: float = Field(default=0.0, ge=0, le=1)
    annotate_rate: float = Field(default=0.0, ge=0, le=1)
    verify_rate: float = Field(default=0.0, ge=0, le=1)
    human_review_rate: float = Field(default=0.0, ge=0, le=1)
    block_rate: float = Field(default=0.0, ge=0, le=1)
    average_risk: float = Field(default=0.0, ge=0, le=1)
    average_confidence: float = Field(default=0.0, ge=0, le=1)
    fast_rate: float = Field(default=0.0, ge=0, le=1)
    deep_rate: float = Field(default=0.0, ge=0, le=1)
    incident_rate: float = Field(default=0.0, ge=0, le=1)
    override_rate: float = Field(default=0.0, ge=0, le=1)
    potential_drift_count: int = 0
    active_recommendations: int = 0


# ======================================================================
# B. risk posture
# ======================================================================


class RiskPosture(_Ent):
    performance_average: float = Field(default=0.0, ge=0, le=1)
    responsibility_average: float = Field(default=0.0, ge=0, le=1)
    cost_average: float = Field(default=0.0, ge=0, le=1)
    overall_average: float = Field(default=0.0, ge=0, le=1)
    performance_high_risk_count: int = 0
    responsibility_high_risk_count: int = 0
    cost_high_risk_count: int = 0
    overall_high_risk_count: int = 0
    high_risk_threshold: float = Field(ge=0, le=1)
    dominant_dimension: str | None = None
    risk_trend: str | None = None                    # "increasing" | "decreasing" | "stable" | None


# ======================================================================
# C. application risk matrix
# ======================================================================


class ApplicationPostureRow(_Ent):
    application: str
    interactions: int = 0
    average_risk: float = Field(default=0.0, ge=0, le=1)
    high_risk_rate: float = Field(default=0.0, ge=0, le=1)
    human_review_rate: float = Field(default=0.0, ge=0, le=1)
    block_rate: float = Field(default=0.0, ge=0, le=1)
    fast_rate: float = Field(default=0.0, ge=0, le=1)
    deep_rate: float = Field(default=0.0, ge=0, le=1)
    dominant_risk_dimension: str | None = None
    incident_count: int = 0
    override_rate: float | None = None
    posture: str = "LOW"                              # LOW | MODERATE | HIGH  (band label, not a score)
    posture_rationale: str = ""
    recommended_posture: str | None = None            # adaptive recommendation type, if any
    open_recommendations: list[str] = Field(default_factory=list)


# ======================================================================
# D. risk heatmap
# ======================================================================


class HeatmapCell(_Ent):
    application: str
    dimension: str
    value: float | None = None                       # None renders as "N/A"


class RiskHeatmap(_Ent):
    applications: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    cells: list[HeatmapCell] = Field(default_factory=list)
    note: str = (
        "Cell = mean of the corresponding stored risk field over that "
        "application's traces. No new risk formula. 'N/A' where unavailable."
    )


# ======================================================================
# live decision / intervention feed
# ======================================================================


class LiveDecisionRow(_Ent):
    interaction_id: str
    application: str
    decision: str
    overall_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    verification_path: str
    dominant_dimension: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    timestamp: datetime
    source: str = "STORED_TRACE"                      # never "SIMULATION"


# ======================================================================
# executive summary / story mode
# ======================================================================


class ApplicationPostureBadge(_Ent):
    application: str
    posture: str


class ExecutiveSummary(_Ent):
    has_data: bool = False
    ai_systems_monitored: int = 0
    interactions_evaluated: int = 0
    high_risk_interactions: int = 0
    human_oversight_count: int = 0
    potential_drift_signals: int = 0
    open_governance_recommendations: int = 0
    application_posture: list[ApplicationPostureBadge] = Field(default_factory=list)
    top_risk_dimension: str | None = None
    top_governance_issue: str | None = None
    recommended_action: str | None = None
    safety_status: str = "No production changes are automatically applied."


# ======================================================================
# governance audit timeline
# ======================================================================


class TimelineEvent(_Ent):
    order: int
    event_type: str                                  # DECISION | INCIDENT | REVIEWER_FEEDBACK | PATTERN | DRIFT | RECOMMENDATION | COUNTERFACTUAL | APPROVAL
    timestamp: datetime | None = None
    entity: str
    description: str
    timestamp_note: str = ""


class GovernanceTimeline(_Ent):
    events: list[TimelineEvent] = Field(default_factory=list)
    note: str = (
        "Built from stored objects. Where a chronological timestamp is unavailable "
        "the event is displayed in causal workflow order only."
    )


# ======================================================================
# what-if / policy playground
# ======================================================================


class WhatIfMetricRow(_Ent):
    metric: str
    current: float | None = None
    candidate: float | None = None
    direction: str = "flat"                           # "up" | "down" | "flat"


class WhatIfResult(_Ent):
    application: str | None = None
    control: str
    current_value: float | None = None
    candidate_value: float | None = None
    current_configuration: dict[str, float] = Field(default_factory=dict)
    candidate_configuration: dict[str, float] | None = None
    metrics: list[WhatIfMetricRow] = Field(default_factory=list)
    current_decision_distribution: dict[str, int] = Field(default_factory=dict)
    candidate_decision_distribution: dict[str, int] = Field(default_factory=dict)
    safety_status: str = "NOT_RUN"                    # PASS | FAIL | NO_CANDIDATE | NOT_RUN
    safety_constraints: dict[str, float] = Field(default_factory=dict)
    safety_violations: list[str] = Field(default_factory=list)
    interpretation: str = ""
    disclaimer: str = (
        "SIMULATION over the synthetic evaluation set (calibration.sweep + "
        "calibration.select). Safety is evaluated FIRST. This does NOT modify the "
        "stored decision or production configuration."
    )


# ======================================================================
# technical architecture view
# ======================================================================


class ArchitectureStage(_Ent):
    stage: str
    module: str
    description: str


class TechnicalArchitecture(_Ent):
    stages: list[ArchitectureStage] = Field(default_factory=list)
    note: str = "Visual explanation only — nothing in this diagram is executed."


# ======================================================================
# one-click executive demo
# ======================================================================


class DemoStep(_Ent):
    step: int
    title: str
    detail: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class EnterpriseDemoResult(_Ent):
    generated_at: datetime | None = None
    steps: list[DemoStep] = Field(default_factory=list)
    interactions: int = 0
    incidents: int = 0
    patterns: int = 0
    potential_drift: int = 0
    top_pattern: str | None = None
    top_recommendation: str | None = None
    counterfactual_safety: str = "NOT_RUN"            # PASS | FAIL | NO_CANDIDATE | NOT_RUN
    approval_status: str | None = None
    production_configuration_status: str = "UNCHANGED"
    notes: list[str] = Field(default_factory=list)


# ======================================================================
# top-level command-center view
# ======================================================================


class CommandCenterView(_Ent):
    generated_at: datetime | None = None
    kpi: ExecutiveKpiStrip
    risk_posture: RiskPosture
    application_posture: list[ApplicationPostureRow] = Field(default_factory=list)
    heatmap: RiskHeatmap
    recent_decisions: list[LiveDecisionRow] = Field(default_factory=list)
    executive_summary: ExecutiveSummary
    notes: list[str] = Field(default_factory=list)
