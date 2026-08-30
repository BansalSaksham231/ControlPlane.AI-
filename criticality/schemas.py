"""Data contracts for the Claim / Action Criticality engine."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriticalityFactor(BaseModel):
    factor: str
    value: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=1)
    band: str  # LOW / MEDIUM / HIGH


class ClaimCriticality(BaseModel):
    claim: str
    criticality: float = Field(ge=0, le=1)
    signals: list[str] = Field(default_factory=list)


class CriticalityAssessment(BaseModel):
    """
    "How much does it matter if this response is wrong?"

    Derived only from production-visible action metadata and the response
    text — never from ground truth.
    """

    action_criticality: float = Field(ge=0, le=1)
    band: str  # low / moderate / high

    factors: list[CriticalityFactor]
    dominant_factors: list[str]

    claim_criticalities: list[ClaimCriticality] = Field(default_factory=list)
    max_claim_criticality: float = Field(default=0.0, ge=0, le=1)

    reason_codes: list[str] = Field(default_factory=list)
    explanation: str
