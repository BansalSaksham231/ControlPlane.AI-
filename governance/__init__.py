"""
ControlPlane.ai — Governance Intelligence & Closed-Loop Monitoring (Phase 9).

An ANALYTICS / GOVERNANCE layer that closes the PS loop:

    traffic -> detection -> decision -> monitoring -> incident -> human review
      -> governance signal -> analysis -> calibration RECOMMENDATION

It answers operational-governance questions ("are we over- or under-flagging?",
"which applications generate the most risk?", "are reviewers overriding us?",
"should a policy or threshold be reviewed?") from data that ALREADY exists:

    DecisionTrace          (stored decisions)
    GovernanceAction       (human review outcomes, from investigation/)
    FeedbackRecord         (reviewer feedback, from feedback/)
    CalibrationSweepReport / ConfigurationSelection   (from calibration/)

Hard boundaries (enforced by tests)
-----------------------------------
* It is a READ-ONLY analytics pass. It NEVER re-runs a detector,
  ``DecisionEngine.evaluate``, fusion, policy or verification.
* It NEVER imports the evaluation package and NEVER reads any
  ground-truth or evaluation-only label.
* Reviewer feedback is a GOVERNANCE SIGNAL, never ground truth — the code
  and schemas say ``reviewer_override`` / ``governance_signal`` /
  ``reviewer_disagreement``, never a correctness label.
* Recommendations are RECOMMENDATION-ONLY. Nothing here writes the
  production configuration file; the disposition is
  ``RECOMMENDED_FOR_EVALUATION`` / ``REVIEW_REQUIRED`` — never ``APPLIED``.
* No raw PII: only structured fields + already-redacted / reviewer-authored
  text.
* Deterministic for a given (traces, governance actions, feedback) set.
"""

from governance.report import build_governance_report
from governance.schemas import (
    ApplicationComparison,
    GovernanceIntelligenceReport,
    GovernanceInsight,
    GovernanceOverview,
    GovernanceRecommendation,
    GovernanceSignal,
    GovernanceSignalSummary,
    GovernanceTrendReport,
)

__all__ = [
    "build_governance_report",
    "GovernanceIntelligenceReport",
    "GovernanceOverview",
    "ApplicationComparison",
    "GovernanceInsight",
    "GovernanceSignal",
    "GovernanceSignalSummary",
    "GovernanceTrendReport",
    "GovernanceRecommendation",
]
