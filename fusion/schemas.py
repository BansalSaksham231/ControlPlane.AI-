"""Data contracts for the Risk Fusion Engine."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DimensionContribution(BaseModel):
    dimension: str
    risk: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class FusionResult(BaseModel):
    """Cross-dimension risk fusion output."""

    performance_risk: float = Field(ge=0, le=1)
    responsibility_risk: float = Field(ge=0, le=1)
    cost_risk: float = Field(ge=0, le=1)

    overall_risk: float = Field(ge=0, le=1)
    dominant_dimension: str
    dominant_risk: float = Field(ge=0, le=1)

    risk_breakdown: list[DimensionContribution]
    # How much ControlPlane trusts THIS fused risk number (distinct from
    # the risk itself). Driven by detector confidence + agreement.
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(default=0.0, ge=0, le=1)
    # True when several dimensions are elevated simultaneously.
    multi_risk: bool = False

    weighted_only_risk: float = Field(ge=0, le=1)
    severity_rule_applied: bool
    severity_floor_applied: bool

    explanation: str
