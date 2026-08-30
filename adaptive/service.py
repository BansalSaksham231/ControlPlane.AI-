"""
AdaptiveGovernanceService — orchestrates incident intelligence + the
adaptive recommendation engine + the human approval gate over a
``ControlPlaneService``'s stored state.

Read-only w.r.t. the pipeline and production configuration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from adaptive.approval import ApprovalStore
from adaptive.report import build_adaptive_governance_report
from adaptive.schemas import AdaptiveConfig, AdaptiveRecommendation
from incident.report import build_incident_intelligence
from incident.schemas import Phase10IncidentConfig

__all__ = ["AdaptiveGovernanceService", "RecommendationNotFound"]


class RecommendationNotFound(KeyError):
    pass


class AdaptiveGovernanceService:
    def __init__(
        self,
        control_plane: Any,
        approval_store: ApprovalStore | None = None,
        *,
        incident_config: Phase10IncidentConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
    ) -> None:
        self._cp = control_plane
        self.approvals = approval_store or ApprovalStore()
        self._incident_config = incident_config or Phase10IncidentConfig()
        self._adaptive_config = adaptive_config or AdaptiveConfig()

    # ------------------------------------------------------------------

    def incident_intelligence(self):
        return build_incident_intelligence(
            self._cp.all_traces(),
            self._cp.governance.get_all_actions(),
            self._cp.feedback.all(),
            config=self._incident_config,
        )

    def report(
        self, *, with_counterfactual: bool = False, generated_at: datetime | None = None
    ):
        intelligence = self.incident_intelligence()
        selection = None
        if with_counterfactual:
            from adaptive.counterfactual import run_threshold_counterfactual

            selection = run_threshold_counterfactual(
                self._cp._config,
                minimum_recall=self._adaptive_config.minimum_recall,
                minimum_precision=self._adaptive_config.minimum_precision,
            )
        return build_adaptive_governance_report(
            intelligence,
            config=self._adaptive_config,
            calibration_selection=selection,
            approval_store=self.approvals,
            generated_at=generated_at,
        )

    def recommendations(
        self, *, with_counterfactual: bool = False
    ) -> list[AdaptiveRecommendation]:
        return self.report(with_counterfactual=with_counterfactual).recommendations

    def get_recommendation(
        self, recommendation_id: str, *, with_counterfactual: bool = False
    ) -> AdaptiveRecommendation:
        for rec in self.recommendations(with_counterfactual=with_counterfactual):
            if rec.recommendation_id == recommendation_id:
                return rec
        raise RecommendationNotFound(recommendation_id)

    # ------------------------------------------------------------------
    # human approval gate — records APPROVED_FOR_EVALUATION, never applies

    def approve(
        self, recommendation_id: str, *, actor: str = "reviewer", comment: str = ""
    ) -> AdaptiveRecommendation:
        self.get_recommendation(recommendation_id)  # 404 if unknown
        self.approvals.approve(recommendation_id, actor=actor, comment=comment)
        return self.get_recommendation(recommendation_id)

    def reject(
        self, recommendation_id: str, *, actor: str = "reviewer", comment: str = ""
    ) -> AdaptiveRecommendation:
        self.get_recommendation(recommendation_id)
        self.approvals.reject(recommendation_id, actor=actor, comment=comment)
        return self.get_recommendation(recommendation_id)
