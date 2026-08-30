"""Data contracts for the Calibration Advisor (offline / evaluation only)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# The config path each sweepable threshold_type maps to.
THRESHOLD_CONFIG_PATH: dict[str, tuple[str, ...]] = {
    "deep_verification_risk": ("verification", "deep_verification_risk_threshold"),
    "fast_path_min_confidence": ("verification", "fast_path_min_confidence"),
    "deep_verification_consequence": ("verification", "deep_verification_consequence_threshold"),
    "deep_verification_criticality": ("verification", "deep_verification_criticality_threshold"),
    "disagreement_trigger": ("verification", "disagreement_trigger"),
}


class CalibrationMetrics(BaseModel):
    """
    Metrics for one simulated configuration over the evaluation dataset.

    A metric whose denominator is zero is ``None`` — never an artificial ``0``.
    """

    n_cases: int
    n_risky: int          # cases with any ground-truth risk flag
    n_clean: int

    # verification workload
    deep_verification_rate: float = Field(ge=0, le=1)
    fast_verification_rate: float = Field(ge=0, le=1)

    # intervention quality vs "any ground-truth risk"  (intervention = decision != ALLOW)
    intervention_recall: float | None = None
    intervention_precision: float | None = None
    intervention_f1: float | None = None
    false_positive_rate: float | None = None      # intervening on genuinely-clean cases
    false_negative_rate: float | None = None      # risky cases NOT intervened on

    human_review_rate: float = Field(ge=0, le=1)
    block_rate: float = Field(ge=0, le=1)
    abstention_rate: float = Field(ge=0, le=1)     # performance status == UNVERIFIED

    # latency: recombined from ONE-TIME MEASURED per-pass timings.
    mean_verification_latency_ms: float
    mean_total_pipeline_latency_ms: float
    total_deep_workload_ms: float                  # sum of measured deep-pass time over DEEP cases


class ThresholdPoint(BaseModel):
    threshold_type: str
    threshold_value: float
    metrics: CalibrationMetrics
    satisfies_safety_constraints: bool
    constraint_violations: list[str] = Field(default_factory=list)


class ThresholdSweep(BaseModel):
    threshold_type: str
    config_path: str
    baseline_value: float
    points: list[ThresholdPoint]
    safety_constraints: dict[str, float]
    tradeoff_note: str


class CalibrationRecommendation(BaseModel):
    status: str                                   # "RECOMMENDATION" | "NO_SAFE_OPERATING_POINT"
    threshold_type: str | None = None
    config_path: str | None = None
    current_value: float | None = None
    recommended_value: float | None = None

    current_metrics: CalibrationMetrics | None = None
    recommended_metrics: CalibrationMetrics | None = None

    objective: str
    explanation: str


class CounterfactualChange(BaseModel):
    recall_delta: float | None = None
    precision_delta: float | None = None
    fpr_delta: float | None = None
    human_review_delta: float | None = None
    deep_verification_delta: float | None = None
    mean_verification_latency_delta_ms: float | None = None
    deep_workload_delta_ms: float | None = None


class CounterfactualAnalysis(BaseModel):
    threshold_type: str
    config_path: str
    baseline_value: float
    counterfactual_value: float
    baseline_metrics: CalibrationMetrics
    counterfactual_metrics: CalibrationMetrics
    changes: CounterfactualChange
    summary: str


class GridPoint(BaseModel):
    risk_threshold: float
    confidence_threshold: float
    metrics: CalibrationMetrics
    satisfies_safety_constraints: bool


class GridExperiment(BaseModel):
    risk_values: list[float]
    confidence_values: list[float]
    points: list[GridPoint]
    best_points: list[GridPoint]
    note: str


class CalibrationReport(BaseModel):
    generated_at: datetime | None = None
    n_cases: int
    dataset: str = "synthetic evaluation set (data/generated/evaluation_cases.csv)"

    baseline_config: dict[str, float]
    current_operating_point: CalibrationMetrics

    sweeps: list[ThresholdSweep]
    recommendation: CalibrationRecommendation
    grid: GridExperiment
    counterfactuals: list[CounterfactualAnalysis]

    notes: list[str] = Field(default_factory=list)
