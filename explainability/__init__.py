"""
ControlPlane.ai — Explainability Foundation (Phase 6).

A presentation layer that turns a stored :class:`decision.schemas.DecisionTrace`
into a single, UI-ready :class:`ExplainabilitySummary`.

Step 1 shipped the contract (``explainability.schemas``). Step 2 adds the
presentation adapter ``explainability.builder.build_explanation`` —
``DecisionTrace -> ExplainabilitySummary``. Neither runs a detector, computes
a new risk/confidence score, reads ground truth, or carries raw PII;
free-text fields come from the already-redacted incident replay.
"""

from explainability.builder import build_explanation

from explainability.schemas import (
    ConfidenceSummary,
    ConsequenceExplanation,
    ConsequenceFactorRow,
    CounterfactualExplanation,
    CriticalityExplanation,
    DecisionPathExplanation,
    EvidenceExplanation,
    ExplainabilitySummary,
    HumanReviewExplanation,
    PolicyRuleExplanation,
    RiskDimensionExplanation,
    RiskSummary,
    VerificationExplanation,
)

__all__ = [
    "build_explanation",
    "ExplainabilitySummary",
    "RiskSummary",
    "ConfidenceSummary",
    "RiskDimensionExplanation",
    "EvidenceExplanation",
    "ConsequenceExplanation",
    "ConsequenceFactorRow",
    "CriticalityExplanation",
    "PolicyRuleExplanation",
    "DecisionPathExplanation",
    "VerificationExplanation",
    "HumanReviewExplanation",
    "CounterfactualExplanation",
]
