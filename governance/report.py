"""
Assemble the full :class:`GovernanceIntelligenceReport` from stored data.

Pure orchestration over the governance sub-modules — no pipeline re-run,
no ground truth, no config writes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from decision.schemas import DecisionTrace
from governance.analytics import build_overview, compare_applications
from governance.insights import generate_insights
from governance.recommendations import build_recommendations
from governance.schemas import GovernanceConfig, GovernanceIntelligenceReport
from governance.signals import collect_signals, summarize_signals
from governance.trends import build_trends

__all__ = ["build_governance_report"]


def build_governance_report(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    feedback_records: list[Any],
    *,
    config: GovernanceConfig | None = None,
    calibration_selection: Any | None = None,
    generated_at: datetime | None = None,
) -> GovernanceIntelligenceReport:
    config = config or GovernanceConfig()

    overview = build_overview(
        traces, governance_actions, config=config, generated_at=generated_at
    )
    comparison = compare_applications(traces, governance_actions, config=config)
    signal_details = collect_signals(traces, governance_actions, feedback_records)
    signal_summary = summarize_signals(signal_details)
    insights = generate_insights(overview, comparison, traces, config=config)
    trends = build_trends(traces, governance_actions, config=config)
    recommendations = build_recommendations(
        overview,
        comparison,
        signal_summary,
        config=config,
        calibration_selection=calibration_selection,
    )

    return GovernanceIntelligenceReport(
        generated_at=generated_at,
        overview=overview,
        application_comparison=comparison,
        signals=signal_summary,
        signal_details=signal_details,
        insights=insights,
        trends=trends,
        recommendations=recommendations,
        notes=[
            "Closed-loop governance view: traffic -> detection -> decision -> "
            "monitoring -> incident -> human review -> governance signal -> "
            "analysis -> calibration RECOMMENDATION.",
            "Read-only. No detector / decision engine / policy / verification pass "
            "was re-run; no ground truth was read; config/settings.yaml was not touched.",
            "Reviewer feedback is a governance signal, not ground truth. "
            "Recommendations are RECOMMENDED_FOR_EVALUATION / REVIEW_REQUIRED — never applied.",
        ],
    )
