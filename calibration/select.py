"""
calibration/select.py — safety-constrained configuration selection (Phase 7, Step 2).

THE QUESTION THIS ANSWERS
    "Among the threshold configurations that satisfy explicit SAFETY
     requirements, which one is operationally most efficient?"

It is not an optimiser. Safety is a hard gate applied first; only the
candidates that pass are then ranked on a single, explicit efficiency
objective.

WHAT IT OPERATES ON
    A :class:`calibration.sweep.CalibrationSweepReport` produced by
    ``sweep_thresholds(...)`` — i.e. real pipeline outputs over the
    150-case synthetic evaluation set. This module adds no metric
    formulas (it reads the ``SafetyMetrics`` / ``EfficiencyMetrics``
    already on each ``CalibrationResult``) and runs no pipeline.

GOVERNANCE BOUNDARY
    The output is evidence, never an action. A selected candidate is
    "Recommended for evaluation" — it is NOT written to
    ``config/settings.yaml`` and NOT applied to production. The BASELINE
    (current production config) is reported separately and is never
    silently replaced. If nothing satisfies the safety constraints the
    result says so; the constraints are never relaxed.

    The synthetic dataset means a selection is "best among the tested
    candidates under the stated constraints on the synthetic evaluation
    set" — not proof of production optimality.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from calibration.sweep import (
    CalibrationResult,
    CalibrationSweepReport,
    CandidateConfig,
)

__all__ = [
    "SafetyConstraints",
    "EfficiencyObjective",
    "CandidateEvaluation",
    "ConfigurationSelection",
    "select_configuration",
]


# ======================================================================
# inputs
# ======================================================================


class SafetyConstraints(BaseModel):
    """
    Explicit, hard safety requirements a candidate MUST satisfy before it
    is even considered on efficiency. Every field is a probability/rate in
    [0, 1]. Only metrics that exist on ``CalibrationResult.safety`` /
    ``.efficiency`` are accepted.
    """

    model_config = ConfigDict(extra="forbid")

    minimum_recall: float = Field(ge=0, le=1)
    minimum_precision: float = Field(ge=0, le=1)

    maximum_false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_missed_risk_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_human_review_rate: float | None = Field(default=None, ge=0, le=1)

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class EfficiencyObjective(str, Enum):
    """How eligible (safety-passing) candidates are ranked."""

    MIN_HUMAN_REVIEW = "MIN_HUMAN_REVIEW"
    MAX_FAST_PATH = "MAX_FAST_PATH"
    MIN_LATENCY = "MIN_LATENCY"


# ======================================================================
# output
# ======================================================================


class CandidateEvaluation(BaseModel):
    """One candidate, whether it passed the safety gate, and why/why not."""

    resolved_thresholds: dict[str, float]
    eligible: bool
    violations: list[str] = Field(default_factory=list)
    result: CalibrationResult


class ConfigurationSelection(BaseModel):
    status: str                       # "SELECTED" | "NO_ELIGIBLE_CANDIDATE"
    disposition: str = (
        "Recommended for evaluation — NOT applied to production. The baseline "
        "remains the production configuration."
    )

    objective: EfficiencyObjective
    safety_constraints: SafetyConstraints

    total_candidate_count: int
    eligible_candidate_count: int

    # BASELINE (current production config) — always reported, never replaced.
    baseline_result: CalibrationResult
    baseline_eligible: bool
    baseline_violations: list[str] = Field(default_factory=list)

    # SELECTED CANDIDATE — None when nothing passes the safety gate.
    selected_configuration: CandidateConfig | None = None
    selected_result: CalibrationResult | None = None

    selection_reason: str
    candidate_evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "ConfigurationSelection":
        if self.status == "SELECTED":
            assert self.selected_configuration is not None
            assert self.selected_result is not None
        else:
            assert self.selected_configuration is None
            assert self.selected_result is None
        return self


# ======================================================================
# constraint checking
# ======================================================================


def _violations(result: CalibrationResult, c: SafetyConstraints) -> list[str]:
    s, e = result.safety, result.efficiency
    out: list[str] = []
    if s.recall < c.minimum_recall:
        out.append(f"recall {s.recall:.2f} < minimum_recall {c.minimum_recall:.2f}")
    if s.precision < c.minimum_precision:
        out.append(
            f"precision {s.precision:.2f} < minimum_precision {c.minimum_precision:.2f}"
        )
    if (
        c.maximum_false_positive_rate is not None
        and s.false_positive_rate > c.maximum_false_positive_rate
    ):
        out.append(
            f"false_positive_rate {s.false_positive_rate:.2f} > "
            f"maximum_false_positive_rate {c.maximum_false_positive_rate:.2f}"
        )
    if (
        c.maximum_missed_risk_rate is not None
        and s.missed_risk_rate > c.maximum_missed_risk_rate
    ):
        out.append(
            f"missed_risk_rate {s.missed_risk_rate:.2f} > "
            f"maximum_missed_risk_rate {c.maximum_missed_risk_rate:.2f}"
        )
    if (
        c.maximum_human_review_rate is not None
        and e.human_review_rate > c.maximum_human_review_rate
    ):
        out.append(
            f"human_review_rate {e.human_review_rate:.2f} > "
            f"maximum_human_review_rate {c.maximum_human_review_rate:.2f}"
        )
    return out


# ======================================================================
# ranking
# ======================================================================

# Deterministic ordering. The winner is the ``min`` of this key over the
# candidates that already passed the safety gate:
#
#   1. primary objective
#        MIN_HUMAN_REVIEW -> efficiency.human_review_rate           (ascending)
#        MAX_FAST_PATH    -> -efficiency.fast_path_rate             (i.e. highest first)
#        MIN_LATENCY      -> efficiency.average_latency_ms          (ascending)
#   2. higher recall           (safety-favouring tie-break)   -> -safety.recall
#   3. higher precision                                       -> -safety.precision
#   4. lower human-review rate                                -> efficiency.human_review_rate
#   5. lower measured latency                                 -> efficiency.average_latency_ms
#   6. original sweep order (Cartesian-product index)         -> index
#
# Step 6 guarantees total determinism regardless of dict/set iteration and
# regardless of measured-latency jitter between cache builds.


def _primary(result: CalibrationResult, objective: EfficiencyObjective) -> float:
    e = result.efficiency
    if objective is EfficiencyObjective.MIN_HUMAN_REVIEW:
        return e.human_review_rate
    if objective is EfficiencyObjective.MAX_FAST_PATH:
        return -e.fast_path_rate
    return e.average_latency_ms  # MIN_LATENCY


def _sort_key(index: int, result: CalibrationResult, objective: EfficiencyObjective):
    s, e = result.safety, result.efficiency
    return (
        round(_primary(result, objective), 6),
        round(-s.recall, 6),
        round(-s.precision, 6),
        round(e.human_review_rate, 6),
        round(e.average_latency_ms, 6),
        index,
    )


# ======================================================================
# selection
# ======================================================================


def select_configuration(
    sweep: CalibrationSweepReport,
    constraints: SafetyConstraints,
    objective: EfficiencyObjective = EfficiencyObjective.MIN_HUMAN_REVIEW,
) -> ConfigurationSelection:
    """
    Filter the sweep's candidates by ``constraints`` (SAFETY FIRST), then
    rank the survivors by ``objective`` and return the winner as a
    *recommendation for evaluation* — never as an applied change.
    """
    objective = EfficiencyObjective(objective)

    baseline_violations = _violations(sweep.baseline, constraints)
    baseline_eligible = not baseline_violations

    evaluations: list[CandidateEvaluation] = []
    eligible: list[tuple[int, CalibrationResult]] = []
    for index, result in enumerate(sweep.results):
        v = _violations(result, constraints)
        evaluations.append(
            CandidateEvaluation(
                resolved_thresholds=result.resolved_thresholds,
                eligible=not v,
                violations=v,
                result=result,
            )
        )
        if not v:
            eligible.append((index, result))

    common = dict(
        objective=objective,
        safety_constraints=constraints,
        total_candidate_count=len(sweep.results),
        eligible_candidate_count=len(eligible),
        baseline_result=sweep.baseline,
        baseline_eligible=baseline_eligible,
        baseline_violations=baseline_violations,
        candidate_evaluations=evaluations,
        notes=[
            "Safety constraints are a hard gate applied BEFORE the efficiency "
            "objective. A candidate is never chosen for efficiency alone.",
            "The selected candidate is 'Recommended for evaluation' only. It is "
            "NOT written to config/settings.yaml and NOT applied to production.",
            "Results are for the synthetic 150-case evaluation set: the winner "
            "is 'best among the tested candidates under the stated constraints "
            "on the synthetic evaluation set', not proven production-optimal.",
        ],
    )

    if not eligible:
        return ConfigurationSelection(
            status="NO_ELIGIBLE_CANDIDATE",
            selection_reason=_no_selection_reason(sweep, constraints, baseline_eligible),
            **common,
        )

    winner_index, winner = min(
        eligible, key=lambda pair: _sort_key(pair[0], pair[1], objective)
    )
    return ConfigurationSelection(
        status="SELECTED",
        selected_configuration=winner.configuration,
        selected_result=winner,
        selection_reason=_selection_reason(
            sweep, winner, objective, len(eligible), baseline_eligible
        ),
        **common,
    )


# ---------------------------------------------------------------- reasons


def _objective_phrase(objective: EfficiencyObjective) -> str:
    return {
        EfficiencyObjective.MIN_HUMAN_REVIEW: "the lowest human-review rate",
        EfficiencyObjective.MAX_FAST_PATH: "the highest FAST-path rate",
        EfficiencyObjective.MIN_LATENCY: "the lowest measured average latency",
    }[objective]


def _objective_values(result: CalibrationResult, objective: EfficiencyObjective) -> str:
    e = result.efficiency
    return {
        EfficiencyObjective.MIN_HUMAN_REVIEW: f"human-review rate {e.human_review_rate:.3f}",
        EfficiencyObjective.MAX_FAST_PATH: f"FAST-path rate {e.fast_path_rate:.3f}",
        EfficiencyObjective.MIN_LATENCY: f"average latency {e.average_latency_ms:.2f} ms",
    }[objective]


def _selection_reason(
    sweep: CalibrationSweepReport,
    winner: CalibrationResult,
    objective: EfficiencyObjective,
    eligible_count: int,
    baseline_eligible: bool,
) -> str:
    c_ok = (
        "The baseline (current production config) also satisfies these safety "
        "constraints."
        if baseline_eligible
        else "NOTE: the baseline (current production config) does NOT satisfy "
        "these safety constraints."
    )
    b, w = sweep.baseline, winner
    improves = _primary(w, objective) < _primary(b, objective) - 1e-9
    vs_baseline = (
        f"Versus the baseline it improves {_objective_phrase(objective)} "
        f"({_objective_values(b, objective)} -> {_objective_values(w, objective)}), "
        f"while keeping recall {w.safety.recall:.2f} (baseline {b.safety.recall:.2f}) "
        f"and precision {w.safety.precision:.2f} (baseline {b.safety.precision:.2f})."
        if improves
        else (
            f"No tested candidate improves on the baseline for {_objective_phrase(objective)} "
            f"({_objective_values(b, objective)}); the selected candidate matches or is "
            f"the closest safety-passing option ({_objective_values(w, objective)})."
        )
    )
    return (
        f"{eligible_count} of {len(sweep.results)} tested candidates satisfy the safety "
        f"constraints. Among those, the selected candidate has {_objective_phrase(objective)} "
        f"({_objective_values(w, objective)}); ties are broken by higher recall, then higher "
        f"precision, then lower human-review rate, then lower latency, then original sweep "
        f"order. Resolved thresholds: {w.resolved_thresholds}. {vs_baseline} {c_ok} "
        "This is best among the tested candidates under the stated constraints on the "
        "synthetic evaluation set — not proof of production optimality."
    )


def _no_selection_reason(
    sweep: CalibrationSweepReport,
    constraints: SafetyConstraints,
    baseline_eligible: bool,
) -> str:
    if not sweep.results:
        return (
            "No candidate configurations were supplied to select from. The safety "
            "constraints were NOT relaxed."
        )
    fails: dict[str, int] = {}
    for result in sweep.results:
        for v in _violations(result, constraints):
            key = v.split()[0]
            fails[key] = fails.get(key, 0) + 1
    breakdown = ", ".join(f"{k}: {n}" for k, n in sorted(fails.items()))
    base = (
        "The baseline satisfies these constraints even though no swept candidate does."
        if baseline_eligible
        else "The baseline also fails these constraints."
    )
    return (
        f"None of the {len(sweep.results)} tested candidates satisfy the safety "
        f"constraints ({constraints.as_dict()}). Failing metric counts: {breakdown}. "
        f"{base} The safety constraints were NOT relaxed; no configuration is recommended."
    )


# ---------------------------------------------------------------- summary


def format_selection(selection: ConfigurationSelection) -> str:
    lines = [
        "Safety-constrained configuration selection",
        "-----------------------------------------",
        "",
        f"Objective          : {selection.objective.value}",
        f"Safety constraints : {selection.safety_constraints.as_dict()}",
        f"Candidates         : {selection.total_candidate_count} tested, "
        f"{selection.eligible_candidate_count} eligible",
        "",
        "BASELINE (current production config):",
        f"  eligible: {selection.baseline_eligible}"
        + ("" if selection.baseline_eligible else f"  ({'; '.join(selection.baseline_violations)})"),
        f"  recall {selection.baseline_result.safety.recall:.2f} · "
        f"precision {selection.baseline_result.safety.precision:.2f} · "
        f"FPR {selection.baseline_result.safety.false_positive_rate:.2f} · "
        f"missed-risk {selection.baseline_result.safety.missed_risk_rate:.2f}",
        f"  FAST {selection.baseline_result.efficiency.fast_path_rate * 100:.1f}% · "
        f"human-review {selection.baseline_result.efficiency.human_review_rate * 100:.1f}% · "
        f"avg latency {selection.baseline_result.efficiency.average_latency_ms:.2f} ms",
        "",
    ]
    if selection.status == "SELECTED":
        r = selection.selected_result
        lines += [
            "SELECTED CANDIDATE (recommended for evaluation, NOT applied):",
            f"  thresholds: {r.resolved_thresholds}",
            f"  recall {r.safety.recall:.2f} · precision {r.safety.precision:.2f} · "
            f"FPR {r.safety.false_positive_rate:.2f} · missed-risk {r.safety.missed_risk_rate:.2f}",
            f"  FAST {r.efficiency.fast_path_rate * 100:.1f}% · "
            f"human-review {r.efficiency.human_review_rate * 100:.1f}% · "
            f"avg latency {r.efficiency.average_latency_ms:.2f} ms",
        ]
    else:
        lines.append(f"SELECTED CANDIDATE : none ({selection.status})")
    lines += ["", "Reason:", "  " + selection.selection_reason]
    return "\n".join(lines)


__all__.append("format_selection")


if __name__ == "__main__":  # pragma: no cover
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    from settings import load_settings
    from calibration.sweep import sweep_thresholds

    cfg = load_settings()
    cc = cfg.get("calibration", {})
    report = sweep_thresholds(
        risk_thresholds=[float(v) for v in cc.get("grid_risk", [0.35, 0.60, 0.85])],
        confidence_thresholds=[float(v) for v in cc.get("grid_confidence", [0.30, 0.60, 0.90])],
        config=cfg,
    )
    selection = select_configuration(
        report,
        SafetyConstraints(minimum_recall=0.95, minimum_precision=0.0),
        EfficiencyObjective.MIN_HUMAN_REVIEW,
    )
    print(format_selection(selection))
