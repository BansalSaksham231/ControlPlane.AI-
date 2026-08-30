"""Data contracts for the Cost / Operational Risk Detector."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CostBreakdown(BaseModel):
    """Transparent per-component cost estimate (prototype rates, not real billing)."""

    input_cost_inr: float = Field(ge=0)
    output_cost_inr: float = Field(ge=0)
    tool_cost_inr: float = Field(ge=0)
    retry_cost_inr: float = Field(ge=0)
    total_cost_inr: float = Field(ge=0)


class CostAnomalyIndicator(BaseModel):
    """One dimension compared against its baseline."""

    dimension: str
    observed: float
    baseline: float
    ratio: float = Field(ge=0)
    threshold: float = Field(ge=0)
    triggered: bool
    explanation: str


class CostResult(BaseModel):
    """Full output of the Cost Detector for one interaction."""

    estimated_cost_inr: float = Field(ge=0)
    cost_breakdown: CostBreakdown

    cost_risk: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    anomaly_indicators: list[CostAnomalyIndicator] = Field(default_factory=list)
    triggered_dimensions: list[str] = Field(default_factory=list)

    # --- Round 2 upgrade: efficiency + typed anomalies ---
    # 1.0 = as efficient as the baseline; lower = wasted spend (retries,
    # tool overhead, oversized output).
    cost_efficiency_score: float = Field(default=1.0, ge=0, le=1)
    # Cost attributable to producing one *successful* response (retries
    # inflate this because the earlier attempts were wasted).
    cost_per_success_inr: float = Field(default=0.0, ge=0)
    # Fraction of spend that went to retries.
    retry_inefficiency: float = Field(default=0.0, ge=0, le=1)
    # Named anomaly types: TOKEN_SPIKE, RETRY_SPIKE, TOOL_LOOP,
    # LATENCY_SPIKE, COST_PER_SUCCESS_SPIKE.
    anomaly_types: list[str] = Field(default_factory=list)

    baseline_source: str = "static"

    explanation: str
    latency_ms: float = Field(ge=0)
