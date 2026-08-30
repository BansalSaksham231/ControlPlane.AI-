"""
ControlPlane.ai — Closed-Loop Adaptive Guardrails (Phase 10, Component 6-9).

Turns incident patterns + drift + governance signals into **safe,
human-approved recommendations**:

    incident patterns / drift / governance signals
        -> AdaptiveRecommendation (RECOMMENDED_FOR_REVIEW)
        -> counterfactual simulation (reuses calibration.sweep + calibration.select)
        -> safety-constrained evaluation (safety FIRST)
        -> HUMAN APPROVAL GATE
        -> APPROVED_FOR_EVALUATION            (never "DEPLOYED" / "APPLIED")
    production configuration REMAINS UNCHANGED.

Non-negotiable safety properties (enforced by tests)
---------------------------------------------------
* The adaptive layer NEVER writes the production configuration file and
  contains no YAML writer / file-write / config mutation.
* Approval produces ``APPROVED_FOR_EVALUATION`` — there is no
  auto-deployment path.
* It re-runs no detector / decision engine. Counterfactuals are the
  EXISTING ``calibration``/``simulation`` machinery, run only on request.
* It reads no ground truth and leaks no PII.
* Deterministic: recommendation IDs are content hashes; ordering is stable.
"""

from adaptive.schemas import (
    AdaptiveConfig,
    AdaptiveGovernanceReport,
    AdaptiveRecommendation,
    ApprovalRecord,
    CounterfactualEvaluation,
    RecommendationStatus,
    RecommendationType,
)
from adaptive.approval import ApprovalStore
from adaptive.service import AdaptiveGovernanceService

__all__ = [
    "AdaptiveGovernanceService",
    "ApprovalStore",
    "AdaptiveRecommendation",
    "AdaptiveGovernanceReport",
    "CounterfactualEvaluation",
    "ApprovalRecord",
    "RecommendationType",
    "RecommendationStatus",
    "AdaptiveConfig",
]
