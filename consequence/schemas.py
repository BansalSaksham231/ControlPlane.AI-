"""Data contracts for the Consequence Engine."""

from __future__ import annotations

from pydantic import BaseModel, Field

from data.schemas import ConsequenceFactors

__all__ = ["ConsequenceFactors", "ConsequenceContribution", "ConsequenceAssessment"]


class ConsequenceContribution(BaseModel):
    """How much one factor contributed to the weighted consequence score."""

    factor: str
    value: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float = Field(ge=0, le=1)


class ConsequenceAssessment(BaseModel):
    """
    Consequence severity for an interaction — how bad the outcome would be
    *if the AI is wrong*, independent of how likely it is to be wrong.
    """

    factors: ConsequenceFactors
    consequence_score: float = Field(ge=0, le=1)
    severity_band: str
    contributions: list[ConsequenceContribution]
    dominant_factors: list[str]
    explanation: str
