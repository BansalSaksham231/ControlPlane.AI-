"""
Counterfactual adaptation — REUSES ``calibration.sweep`` + ``calibration.select``.

There is no second simulation framework here. This module only maps an
existing ``calibration.select.ConfigurationSelection`` onto the
:class:`CounterfactualEvaluation` view. Safety is evaluated FIRST (by
``calibration.select`` itself).
"""

from __future__ import annotations

from typing import Any

from adaptive.schemas import CounterfactualEvaluation

__all__ = ["run_threshold_counterfactual", "selection_to_evaluation"]


def run_threshold_counterfactual(
    config: dict[str, Any] | None = None,
    *,
    minimum_recall: float = 0.90,
    minimum_precision: float = 0.0,
):
    """Execute the existing calibration sweep + safety-constrained selection."""
    from governance.recommendations import run_calibration_bridge

    return run_calibration_bridge(
        config, minimum_recall=minimum_recall, minimum_precision=minimum_precision
    )


def selection_to_evaluation(selection: Any) -> CounterfactualEvaluation:
    """Map a ConfigurationSelection onto a CounterfactualEvaluation."""
    base = selection.baseline_result
    sel = selection.selected_result
    passed = selection.status == "SELECTED" and sel is not None

    ev = CounterfactualEvaluation(
        current_configuration={k: round(v, 4) for k, v in base.resolved_thresholds.items()},
        candidate_configuration=(
            {k: round(v, 4) for k, v in sel.resolved_thresholds.items()} if sel else None
        ),
        current_decision_distribution=dict(base.decision_counts),
        candidate_decision_distribution=dict(sel.decision_counts) if sel else {},
        current_recall=base.safety.recall,
        candidate_recall=sel.safety.recall if sel else None,
        current_precision=base.safety.precision,
        candidate_precision=sel.safety.precision if sel else None,
        current_false_positive_rate=base.safety.false_positive_rate,
        candidate_false_positive_rate=sel.safety.false_positive_rate if sel else None,
        current_missed_risk_rate=base.safety.missed_risk_rate,
        candidate_missed_risk_rate=sel.safety.missed_risk_rate if sel else None,
        current_fast_rate=base.efficiency.fast_path_rate,
        candidate_fast_rate=sel.efficiency.fast_path_rate if sel else None,
        current_deep_rate=base.efficiency.deep_path_rate,
        candidate_deep_rate=sel.efficiency.deep_path_rate if sel else None,
        current_human_review_rate=base.efficiency.human_review_rate,
        candidate_human_review_rate=sel.efficiency.human_review_rate if sel else None,
        current_average_latency_ms=round(base.efficiency.average_latency_ms, 3),
        candidate_average_latency_ms=(
            round(sel.efficiency.average_latency_ms, 3) if sel else None
        ),
        safety_constraints=selection.safety_constraints.as_dict(),
        safety_passed=passed,
        safety_violations=list(selection.baseline_violations) if not passed else [],
        candidate_found=selection.eligible_candidate_count > 0,
        selection_reason=selection.selection_reason,
    )
    return ev


def expected_tradeoff_text(ev: CounterfactualEvaluation) -> str:
    if not ev.safety_passed or ev.candidate_configuration is None:
        return (
            "No safe candidate configuration was found — no threshold change is "
            "recommended. " + ev.selection_reason
        )
    return (
        f"recall {ev.current_recall:.2f} -> {ev.candidate_recall:.2f}; "
        f"precision {ev.current_precision:.2f} -> {ev.candidate_precision:.2f}; "
        f"missed-risk {ev.current_missed_risk_rate:.2f} -> {ev.candidate_missed_risk_rate:.2f}; "
        f"FAST {ev.current_fast_rate:.0%} -> {ev.candidate_fast_rate:.0%}; "
        f"DEEP {ev.current_deep_rate:.0%} -> {ev.candidate_deep_rate:.0%}; "
        f"human-review {ev.current_human_review_rate:.2f} -> {ev.candidate_human_review_rate:.2f}; "
        f"latency {ev.current_average_latency_ms:.1f}ms -> {ev.candidate_average_latency_ms:.1f}ms"
    )
