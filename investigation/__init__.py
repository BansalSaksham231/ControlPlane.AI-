"""
ControlPlane.ai — Enterprise Incident Investigation & Governance (Phase 8, Step 3).

This package turns an operational incident into a full, auditable
investigation and records the **human governance** response.

    Command Center incident
        -> InvestigationService.investigate(interaction_id)
             = stored DecisionTrace
             + build_replay(trace)          (Incident Replay — reconstruction)
             + build_explanation(trace)     (Explainability — presentation)
             + governance history
        -> IncidentInvestigation

Hard boundaries (enforced by tests)
-----------------------------------
* Investigation is READ-ONLY reconstruction. ``investigate`` NEVER calls a
  detector, ``DecisionEngine.evaluate``, ``VerificationRouter.route``,
  ``PolicyEngine.decide``, ``RiskFusionEngine.fuse_scores``,
  ``ConsequenceEngine.assess`` or ``CriticalityEngine.assess``.
* It NEVER reads ground truth / evaluation labels and never imports the
  ``evaluation`` package.
* It NEVER exposes raw PII (no ``matched_text``, no unredacted response) —
  free text comes from the already-redacted replay / explanation.
* A human governance action (ACKNOWLEDGE / APPROVE / MODIFY / REJECT /
  ESCALATE / CLOSE) records the reviewer's outcome. It does **not**
  overwrite ``trace.final_decision.decision`` — the automated ControlPlane
  decision is immutable.
* Counterfactuals are explicitly-labelled SIMULATIONS run by the existing
  ``simulation.engine``; they never change the stored decision or
  production policy, and reviewer feedback is a governance signal, not
  ground truth.
"""

from investigation.schemas import (
    GovernanceAction,
    GovernanceActionType,
    IncidentInvestigation,
    InvestigationCounterfactual,
    InvestigationNotFound,
    InvestigationStatus,
)
from investigation.service import GovernanceStore, InvestigationService

__all__ = [
    "InvestigationService",
    "GovernanceStore",
    "IncidentInvestigation",
    "InvestigationNotFound",
    "InvestigationStatus",
    "GovernanceAction",
    "GovernanceActionType",
    "InvestigationCounterfactual",
]
