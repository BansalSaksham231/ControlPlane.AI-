"""
Deterministic sequence-based trend analysis.

Splits the chronologically-ordered traces into a first half and a second
half and compares governance metrics. Where traces carry distinct
timestamps the split is time-ordered; otherwise it is sequence-ordered
and labelled as such. Ordinary variation is a ``TREND`` — the
``POTENTIAL_DRIFT`` label is only used for large moves backed by enough
samples. Nothing here is called "drift" outright.
"""

from __future__ import annotations

from typing import Any

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from governance.analytics import incidents_for, ts_key
from governance.schemas import (
    GovernanceConfig,
    GovernanceTrendReport,
    InsightSeverity,
    TrendDirection,
    TrendSignal,
)

__all__ = ["build_trends"]

_HUMAN = (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK)
_DISAGREEMENT_ACTIONS = ("MODIFY_DECISION", "REJECT_DECISION")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def build_trends(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    *,
    config: GovernanceConfig | None = None,
) -> GovernanceTrendReport:
    config = config or GovernanceConfig()
    ordered = sorted(traces, key=lambda t: (ts_key(t.timestamp), t.interaction_id))
    n = len(ordered)
    timestamped = len({ts_key(t.timestamp) for t in ordered}) > 1
    basis = (
        "timestamped (first half vs second half of time-ordered traces)"
        if timestamped
        else "sequence-based (first half vs second half of ordered traces)"
    )

    if n < 4:
        return GovernanceTrendReport(
            basis=basis,
            first_window_n=n,
            second_window_n=0,
            signals=[],
            notes=["Not enough traces for a first-half / second-half comparison."],
        )

    mid = n // 2
    first, second = ordered[:mid], ordered[mid:]
    first_ids = {t.interaction_id for t in first}
    second_ids = {t.interaction_id for t in second}
    incidents = incidents_for(ordered, config)

    override_ids = {
        act.interaction_id
        for act in governance_actions
        if act.action.value in _DISAGREEMENT_ACTIONS
        and (
            act.action.value == "REJECT_DECISION"
            or (act.reviewer_decision is not None and act.reviewer_decision != act.original_decision)
        )
    }

    def _metrics(group: list[DecisionTrace], ids: set[str]) -> dict[str, float | None]:
        k = len(group)
        return {
            "average_risk": _mean([t.final_decision.overall_risk for t in group]),
            "average_confidence": _mean([t.final_decision.decision_confidence for t in group]),
            "human_oversight_rate": _rate(
                sum(1 for t in group if t.final_decision.decision in _HUMAN), k
            ),
            "deep_rate": _rate(
                sum(1 for t in group if (t.verification_path or "DEEP").upper() == "DEEP"), k
            ),
            "incident_rate": _rate(
                sum(1 for t in group if t.interaction_id in incidents), k
            ),
            "reviewer_override_rate": _rate(
                len(ids & override_ids), k
            ),
        }

    m1 = _metrics(first, first_ids)
    m2 = _metrics(second, second_ids)
    band = config.trend_stable_band

    _lower_is_better = {"average_risk", "human_oversight_rate", "deep_rate",
                        "incident_rate", "reviewer_override_rate"}
    signals: list[TrendSignal] = []
    for metric in (
        "average_risk", "average_confidence", "human_oversight_rate",
        "deep_rate", "incident_rate", "reviewer_override_rate",
    ):
        a, b = m1[metric], m2[metric]
        if a is None or b is None:
            continue
        delta = round(b - a, 6)
        magnitude = abs(delta)
        if magnitude <= band:
            direction = TrendDirection.STABLE
        elif delta > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # cautious labelling
        big = (
            magnitude >= config.potential_drift_magnitude
            and len(first) >= config.potential_drift_min_samples
            and len(second) >= config.potential_drift_min_samples
        )
        if direction is TrendDirection.STABLE:
            label, severity = "TREND", InsightSeverity.INFO
        elif big:
            label = "POTENTIAL_DRIFT"
            severity = InsightSeverity.HIGH
        elif magnitude >= band * 2:
            label, severity = "SIGNAL", InsightSeverity.MEDIUM
        else:
            label, severity = "TREND", InsightSeverity.LOW

        worsening = metric in _lower_is_better and direction is TrendDirection.INCREASING
        if metric == "average_confidence" and direction is TrendDirection.DECREASING:
            worsening = True
        if severity is InsightSeverity.MEDIUM and not worsening:
            severity = InsightSeverity.LOW

        signals.append(
            TrendSignal(
                metric=metric,
                direction=direction,
                magnitude=magnitude,
                baseline=a,
                current=b,
                severity=severity,
                label=label,
                explanation=(
                    f"{metric.replace('_', ' ')} moved from {a:.3f} (first half, "
                    f"n={len(first)}) to {b:.3f} (second half, n={len(second)}); "
                    f"{direction.value} by {magnitude:.3f}. "
                    + (
                        "This is a large move over enough samples — worth a closer look."
                        if label == "POTENTIAL_DRIFT"
                        else "Ordinary variation unless it persists."
                    )
                ),
            )
        )

    return GovernanceTrendReport(
        basis=basis,
        first_window_n=len(first),
        second_window_n=len(second),
        signals=signals,
        notes=[
            "First-half / second-half comparison only — no statistical-significance "
            "claim. 'POTENTIAL_DRIFT' is used sparingly and never means confirmed drift.",
        ],
    )
