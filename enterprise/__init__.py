"""
ControlPlane.ai — Enterprise Command Center & Judge-Ready Demonstration (Phase 11).

A **read-only presentation / orchestration layer**. It adds no risk formula,
no decision logic and no simulation framework — it assembles judge-facing
views from the reports the earlier phases already produce:

    monitoring.OperationalMonitoringReport      (Phase 8)
    governance.GovernanceIntelligenceReport     (Phase 9)
    incident.IncidentIntelligenceReport         (Phase 10)
    adaptive.AdaptiveGovernanceReport           (Phase 10)
    + stored DecisionTrace / GovernanceAction / FeedbackRecord

Guarantees (enforced by tests)
------------------------------
* Never re-runs a detector / DecisionEngine / fusion / policy / verification.
* Never imports the evaluation package; never reads ground truth.
* Never writes the production configuration file; there is NO deployment / apply path.
* No raw PII: free text comes from the already-redacted replay / explanation.
* Deterministic for a given stored state.
* Expensive operations (calibration sweep / counterfactual) run ONLY when a
  user explicitly asks — never on a plain dashboard load.
"""

from enterprise.schemas import (
    CommandCenterView,
    EnterpriseDemoResult,
    ExecutiveSummary,
    GovernanceTimeline,
    RiskHeatmap,
    TechnicalArchitecture,
    WhatIfResult,
)
from enterprise.service import EnterpriseService

__all__ = [
    "EnterpriseService",
    "CommandCenterView",
    "ExecutiveSummary",
    "RiskHeatmap",
    "GovernanceTimeline",
    "WhatIfResult",
    "EnterpriseDemoResult",
    "TechnicalArchitecture",
]
