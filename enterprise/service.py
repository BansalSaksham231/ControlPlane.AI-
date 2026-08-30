"""
EnterpriseService — read-only orchestration over a ``ControlPlaneService``.

Assembles the judge-facing views (Command Center, Executive Summary,
Governance Timeline, What-If, Enterprise Demo) from the reports the
earlier phases already produce. Nothing here re-runs the pipeline, reads
ground truth, or writes configuration.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from enterprise.demo import run_enterprise_demo
from enterprise.schemas import CommandCenterView, GovernanceTimeline, WhatIfResult
from enterprise.timeline import build_governance_timeline
from enterprise.views import (
    build_application_posture,
    build_executive_summary,
    build_heatmap,
    build_kpi_strip,
    build_risk_posture,
    build_technical_architecture,
    recent_decisions,
    whatif_from_selection,
)

__all__ = ["EnterpriseService"]

# EXISTING calibration controls the What-If playground may explore.
_WHATIF_CONTROLS = ("deep_verification_risk_threshold", "fast_path_min_confidence")


class EnterpriseService:
    def __init__(self, control_plane: Any) -> None:
        self._cp = control_plane

    # ------------------------------------------------------------------
    # cheap, no-calibration bundle used by every dashboard view

    def _bundle(self):
        return {
            "monitoring": self._cp.get_operational_monitoring(),
            "governance": self._cp.governance_report(),
            "incident": self._cp.incident_intelligence(),
            "adaptive": self._cp.adaptive_report(),   # with_counterfactual=False
            "traces": self._cp.all_traces(),
        }

    # ------------------------------------------------------------------

    def command_center(self, *, generated_at: datetime | None = None) -> CommandCenterView:
        b = self._bundle()
        posture = build_application_posture(
            b["traces"], b["monitoring"], b["incident"], b["adaptive"], b["governance"]
        )
        return CommandCenterView(
            generated_at=generated_at,
            kpi=build_kpi_strip(b["monitoring"], b["governance"], b["incident"], b["adaptive"]),
            risk_posture=build_risk_posture(b["traces"], b["monitoring"]),
            application_posture=posture,
            heatmap=build_heatmap(b["traces"]),
            recent_decisions=recent_decisions(b["traces"]),
            executive_summary=build_executive_summary(
                b["monitoring"], b["governance"], b["incident"], b["adaptive"], posture
            ),
            notes=[
                "Read-only presentation of already-computed reports. No detector / "
                "decision engine / verification pass is re-run for this view.",
                "Posture (LOW / MODERATE / HIGH) is a band label over existing metrics, "
                "not a new risk score.",
                "No ground truth is read; reviewer feedback is a governance signal.",
            ],
        )

    def application_posture(self):
        b = self._bundle()
        return build_application_posture(
            b["traces"], b["monitoring"], b["incident"], b["adaptive"], b["governance"]
        )

    def executive_summary(self):
        return self.command_center().executive_summary

    def technical_architecture(self):
        return build_technical_architecture()

    def governance_timeline(
        self, *, focus_interaction_id: str | None = None
    ) -> GovernanceTimeline:
        return build_governance_timeline(
            self._cp.all_traces(),
            self._cp.governance.get_all_actions(),
            self._cp.feedback.all(),
            self._cp.incident_intelligence(),
            self._cp.adaptive_report(),
            focus_interaction_id=focus_interaction_id,
        )

    # ------------------------------------------------------------------
    # explicit, user-triggered simulation only

    def whatif(
        self,
        *,
        application: str | None = None,
        control: str = "deep_verification_risk_threshold",
        minimum_recall: float = 0.90,
        minimum_precision: float = 0.0,
    ) -> WhatIfResult:
        if control not in _WHATIF_CONTROLS:
            raise ValueError(
                f"Unknown control '{control}'. Allowed: {_WHATIF_CONTROLS}"
            )
        from adaptive.counterfactual import run_threshold_counterfactual

        selection = run_threshold_counterfactual(
            self._cp._config,
            minimum_recall=minimum_recall,
            minimum_precision=minimum_precision,
        )
        return whatif_from_selection(selection, application=application, control=control)

    def run_demo(
        self, *, with_counterfactual: bool = True, generated_at: datetime | None = None
    ):
        return run_enterprise_demo(
            self._cp, with_counterfactual=with_counterfactual, generated_at=generated_at
        )
