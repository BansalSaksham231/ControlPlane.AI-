"""Data contracts for progressive verification."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VerificationPath(str, Enum):
    FAST = "FAST"
    DEEP = "DEEP"


# Canonical reasons a response was escalated to deep verification.
DEEP_TRIGGER_REASONS = (
    "HIGH_PRELIMINARY_RISK",
    "LOW_CONFIDENCE",
    "HIGH_CONSEQUENCE",
    "HIGH_CRITICALITY",
    "DETECTOR_DISAGREEMENT",
    "MISSING_EVIDENCE",
    # A deterministic hard boundary (e.g. critical outbound PII) that the policy
    # layer will adjudicate regardless of the grounding score. When present the
    # semantic verification work can be skipped — see ``semantics_bypassed``.
    "DETERMINISTIC_HARD_BOUNDARY",
    # Tier 1: the ML-driven cascade router predicted this interaction is
    # semantically complex enough to warrant DEEP verification, even though no
    # deterministic Tier-0.5 gate fired.
    "TIER1_CASCADE_COMPLEXITY",
)


class DisagreementBreakdown(BaseModel):
    """Transparent components of the detector-disagreement / uncertainty score."""

    risk_spread: float = Field(ge=0, le=1)          # pstdev of the 3 dimension risks (rescaled)
    weak_evidence: float = Field(ge=0, le=1)        # 1 - evidence_quality
    neutral_rate: float = Field(ge=0, le=1)         # fraction of claims that were NEUTRAL
    missing_evidence: float = Field(ge=0, le=1)     # 1.0 if no usable evidence retrieved
    low_confidence: float = Field(ge=0, le=1)       # how far preliminary confidence is below 0.6
    score: float = Field(ge=0, le=1)               # weighted blend of the above


class VerificationReport(BaseModel):
    """
    Describes HOW an interaction was verified — the core artefact of the
    Adaptive Guard.
    """

    verification_path: VerificationPath
    deep_trigger_reasons: list[str] = Field(default_factory=list)
    reason_for_deep_verification: str = ""       # human-readable join of the above
    deep_was_forced: bool = False                # deep entered despite a clean-looking response

    preliminary_risk: float = Field(ge=0, le=1)
    preliminary_confidence: float = Field(ge=0, le=1)
    final_risk: float = Field(ge=0, le=1)
    final_confidence: float = Field(ge=0, le=1)

    disagreement_score: float = Field(ge=0, le=1)
    disagreement_breakdown: DisagreementBreakdown

    evidence_available: bool

    # Measured wall-clock time (never fabricated).
    fast_path_latency_ms: float = Field(ge=0)
    deep_path_latency_ms: float = Field(ge=0)
    total_verification_latency_ms: float = Field(ge=0)

    shallow_top_k: int = Field(ge=1)
    deep_top_k: int = Field(ge=1)

    # Deterministic semantic bypass: a hard boundary (critical outbound PII)
    # meant this interaction is decided by the responsibility override, so
    # claim extraction / TF-IDF retrieval / NLI were skipped. The interaction
    # still passes through the ResponsibilityDetector and the PolicyEngine.
    semantics_bypassed: bool = False
    bypass_reason: str = ""

    # Tiered Cascade Router telemetry. Recorded on every routed interaction so
    # the cascade's verdict can be compared against the safety router's actual
    # FAST/DEEP call on real traffic (champion/challenger). The cascade does not
    # override the safety router's decision — see verification/router.py.
    cascade_verdict: str = ""                       # FAST_ALLOW | ROUTE_TO_DEEP | ""
    cascade_reason_code: str = ""
    cascade_tier: str = ""
    predicted_complexity_score: float | None = None
    cascade_threshold: float | None = None
    cascade_agrees_with_router: bool | None = None  # cascade DEEP == router DEEP?

    explanation: str
