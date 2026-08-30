"""
calibration/sweep.py — deterministic threshold-sweep foundation (Phase 7, Step 1).

WHY CALIBRATION EXISTS
    Several ControlPlane thresholds are provisional. Every value in the
    ``verification:`` section of ``config/settings.yaml`` is explicitly
    marked *PROVISIONAL — chosen for sensible behaviour, NOT empirically
    tuned*. This module answers the governance question **"why were the
    current thresholds chosen?"** by evaluating explicit candidate
    threshold configurations against the existing synthetic evaluation
    set and reporting, per candidate, a **safety** profile and a
    separate **efficiency** profile.

WHAT THRESHOLDS IT EVALUATES
    Only the FAST/DEEP verification-router thresholds that already exist
    in the ``verification:`` config section, all probabilities in [0, 1]:

        deep_verification_risk_threshold        (FAST is only allowed at/below this)
        fast_path_max_risk                      (mirror of the above — same boundary)
        fast_path_min_confidence
        low_risk_floor
        disagreement_trigger
        deep_verification_consequence_threshold
        deep_verification_criticality_threshold
        deep_verification_extreme_factor

    No configuration key is invented. Thresholds that cannot be swept
    without changing production *logic* (``shallow_top_k`` / ``deep_top_k``
    are structural, not probabilities; the ``policy:`` risk-band cutoffs
    belong to a different subsystem) are deliberately left out.

WHY CALIBRATION IS SEPARATED FROM PRODUCTION
    Calibration is an EVALUATION operation. It MAY read ``EvaluationCase``
    ground-truth labels (that is its whole job); the production pipeline
    must never see them. It runs the REAL pipeline once through
    :class:`calibration.advisor.CalibrationCache`, then recombines the
    cached detector outputs under each candidate threshold set — it never
    re-runs a detector, never runs a second decision algorithm, and never
    mutates ``config/settings.yaml``. No production module imports this
    package (enforced by a source-level test).

HOW TO READ SAFETY vs EFFICIENCY  (never fused into one score)
    SAFETY      — did we catch the genuinely-risky cases without
                  over-flagging?  ``recall`` / ``precision`` / ``f1`` /
                  ``accuracy`` up, ``false_positive_rate`` /
                  ``missed_risk_rate`` down = safer.
    EFFICIENCY  — what did that safety cost operationally?
                  ``fast_path_rate`` up and ``deep_path_rate`` /
                  ``human_review_rate`` / ``average_latency_ms`` /
                  ``p95_latency_ms`` down = cheaper.
    A safer configuration is usually a more expensive one; this tool
    shows the trade, it does not pick a winner.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from calibration.advisor import CalibrationAdvisor, CalibrationCache
from data.schemas import Interaction, InterventionTier
from evaluation.evaluation import _EVAL_TS
from evaluation.metrics import binary_metrics, rate

__all__ = [
    "CandidateConfig",
    "SafetyMetrics",
    "EfficiencyMetrics",
    "CalibrationResult",
    "CalibrationSweepReport",
    "evaluate_candidate",
    "sweep_thresholds",
    "format_summary",
]

_VERIFICATION_SECTION = "verification"

# axis name -> function producing the CandidateConfig field updates for one value
_AXES: dict[str, Any] = {
    "risk_thresholds": lambda v: {
        # the FAST/DEEP risk boundary is two mirrored keys
        "deep_verification_risk_threshold": v,
        "fast_path_max_risk": v,
    },
    "confidence_thresholds": lambda v: {"fast_path_min_confidence": v},
    "disagreement_thresholds": lambda v: {"disagreement_trigger": v},
    "low_risk_floors": lambda v: {"low_risk_floor": v},
    "consequence_thresholds": lambda v: {"deep_verification_consequence_threshold": v},
    "criticality_thresholds": lambda v: {"deep_verification_criticality_threshold": v},
    "extreme_factor_thresholds": lambda v: {"deep_verification_extreme_factor": v},
}


# ======================================================================
# candidate configuration
# ======================================================================


class CandidateConfig(BaseModel):
    """
    One candidate set of verification thresholds. Every field is optional;
    ``None`` means "keep the value currently in ``config/settings.yaml``".
    All fields are probabilities and are validated to ``0 <= x <= 1``.
    """

    model_config = ConfigDict(extra="forbid")

    deep_verification_risk_threshold: float | None = Field(default=None, ge=0, le=1)
    fast_path_max_risk: float | None = Field(default=None, ge=0, le=1)
    fast_path_min_confidence: float | None = Field(default=None, ge=0, le=1)
    low_risk_floor: float | None = Field(default=None, ge=0, le=1)
    disagreement_trigger: float | None = Field(default=None, ge=0, le=1)
    deep_verification_consequence_threshold: float | None = Field(default=None, ge=0, le=1)
    deep_verification_criticality_threshold: float | None = Field(default=None, ge=0, le=1)
    deep_verification_extreme_factor: float | None = Field(default=None, ge=0, le=1)

    def set_fields(self) -> dict[str, float]:
        """Only the thresholds this candidate actually overrides."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def to_overrides(self) -> dict[str, Any]:
        """A deep-merge override dict for the decision engine config."""
        fields = self.set_fields()
        return {_VERIFICATION_SECTION: fields} if fields else {}

    def resolved_thresholds(self, config: dict[str, Any]) -> dict[str, float]:
        """The effective value of every tunable threshold under this candidate."""
        base = config[_VERIFICATION_SECTION]
        overrides = self.set_fields()
        return {
            name: float(overrides.get(name, base[name]))
            for name in self.model_dump()
        }


# ======================================================================
# result models  (safety kept structurally separate from efficiency)
# ======================================================================


class SafetyMetrics(BaseModel):
    """Did ControlPlane catch the genuinely-risky cases without over-flagging?"""

    evaluation_count: int
    risky_count: int
    clean_count: int

    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    false_positive_rate: float = Field(ge=0, le=1)
    missed_risk_rate: float = Field(ge=0, le=1)   # risky cases that were ALLOWed


class EfficiencyMetrics(BaseModel):
    """What did that safety cost operationally?"""

    allow_rate: float = Field(ge=0, le=1)
    annotate_rate: float = Field(ge=0, le=1)
    verify_rate: float = Field(ge=0, le=1)
    human_review_rate: float = Field(ge=0, le=1)
    block_rate: float = Field(ge=0, le=1)

    fast_path_rate: float = Field(ge=0, le=1)
    deep_path_rate: float = Field(ge=0, le=1)

    average_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    latency_basis: str = (
        "recombined from one-time MEASURED per-pass timings (CalibrationCache); "
        "not fabricated, and it varies between cache builds"
    )


class CalibrationResult(BaseModel):
    """The evaluation of one candidate configuration."""

    configuration: CandidateConfig
    resolved_thresholds: dict[str, float]
    evaluation_count: int
    decision_counts: dict[str, int]
    safety: SafetyMetrics
    efficiency: EfficiencyMetrics


class CalibrationSweepReport(BaseModel):
    """A deterministic sweep over explicit candidate threshold values."""

    generated_at: datetime | None = None
    evaluation_count: int
    candidate_count: int
    swept_thresholds: list[str]
    baseline: CalibrationResult
    results: list[CalibrationResult]
    notes: list[str] = Field(default_factory=list)


# ======================================================================
# evaluation
# ======================================================================


def _nearest_rank_p95(values: Sequence[float]) -> float:
    """Deterministic nearest-rank 95th percentile; 0.0 for an empty list."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 4)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return round(ordered[idx], 4)


def evaluate_candidate(
    candidate: CandidateConfig,
    *,
    cache: CalibrationCache,
    config: dict[str, Any],
) -> CalibrationResult:
    """
    Run every cached evaluation case through the REAL pipeline with this
    candidate's thresholds applied, then aggregate a safety profile and an
    efficiency profile. No detector is re-run; ``config/settings.yaml`` is
    not touched (the engine gets a deep-merged copy).
    """
    # CalibrationAdvisor already owns the "cached DecisionEngine" adapter —
    # reuse it rather than copying the wiring or the decision engine.
    advisor = CalibrationAdvisor(config=config, cache=cache)
    engine = advisor._engine_for(candidate.to_overrides())

    y_true: list[bool] = []
    y_pred: list[bool] = []
    tiers: list[str] = []
    deep = 0
    latencies: list[float] = []

    for rec in cache.records:
        trace = engine.evaluate(
            rec.interaction, timestamp=_EVAL_TS, record_session=False
        )
        decision = trace.final_decision.decision
        is_deep = trace.verification_path == "DEEP"

        y_true.append(bool(rec.gt_any))
        y_pred.append(decision != InterventionTier.ALLOW)
        tiers.append(decision.value)
        if is_deep:
            deep += 1
        # measured per-pass timings recombined for this routing outcome
        latencies.append(
            rec.fast_path_ms
            + (rec.deep_extra_ms if is_deep else 0.0)
            + rec.downstream_ms
        )

    n = len(cache.records)
    bm = binary_metrics(y_true, y_pred)
    counts = {tier.value: tiers.count(tier.value) for tier in InterventionTier}

    safety = SafetyMetrics(
        evaluation_count=n,
        risky_count=sum(y_true),
        clean_count=n - sum(y_true),
        accuracy=bm["accuracy"],
        precision=bm["precision"],
        recall=bm["recall"],
        f1=bm["f1"],
        false_positive_rate=bm["false_positive_rate"],
        missed_risk_rate=bm["false_negative_rate"],
    )
    efficiency = EfficiencyMetrics(
        allow_rate=rate(counts["ALLOW"], n),
        annotate_rate=rate(counts["ANNOTATE"], n),
        verify_rate=rate(counts["VERIFY"], n),
        human_review_rate=rate(counts["HUMAN_REVIEW"], n),
        block_rate=rate(counts["BLOCK"], n),
        fast_path_rate=rate(n - deep, n),
        deep_path_rate=rate(deep, n),
        average_latency_ms=round(sum(latencies) / n, 4) if n else 0.0,
        p95_latency_ms=_nearest_rank_p95(latencies),
    )
    return CalibrationResult(
        configuration=candidate,
        resolved_thresholds=candidate.resolved_thresholds(config),
        evaluation_count=n,
        decision_counts=counts,
        safety=safety,
        efficiency=efficiency,
    )


# ======================================================================
# sweep
# ======================================================================


def _candidate_grid(axis_values: dict[str, Sequence[float]]) -> list[CandidateConfig]:
    """Cartesian product of the supplied axes -> explicit CandidateConfig list."""
    provided = [(name, list(values)) for name, values in axis_values.items() if values]
    if not provided:
        return []
    names = [name for name, _ in provided]
    value_lists = [values for _, values in provided]

    candidates: list[CandidateConfig] = []
    for combo in itertools.product(*value_lists):
        fields: dict[str, float] = {}
        for name, value in zip(names, combo):
            fields.update(_AXES[name](value))
        candidates.append(CandidateConfig(**fields))
    return candidates


def sweep_thresholds(
    *,
    risk_thresholds: Sequence[float] | None = None,
    confidence_thresholds: Sequence[float] | None = None,
    disagreement_thresholds: Sequence[float] | None = None,
    low_risk_floors: Sequence[float] | None = None,
    consequence_thresholds: Sequence[float] | None = None,
    criticality_thresholds: Sequence[float] | None = None,
    extreme_factor_thresholds: Sequence[float] | None = None,
    config: dict[str, Any] | None = None,
    dataset: list[tuple[Interaction, dict[str, Any]]] | None = None,
    cache: CalibrationCache | None = None,
    generated_at: datetime | None = None,
) -> CalibrationSweepReport:
    """
    Deterministically evaluate every combination of the supplied candidate
    threshold values against the synthetic evaluation set.

    Each ``*_thresholds`` argument is an explicit list of values for one
    axis; the sweep is their Cartesian product (unspecified axes are left
    at the current config value). Pass a shared ``cache`` to make repeated
    sweeps identical down to the measured-latency figures.

    Returns a :class:`CalibrationSweepReport` — never writes config.
    """
    from settings import load_settings

    config = config if config is not None else load_settings()
    cache = cache if cache is not None else CalibrationCache(config, dataset)

    axis_values = {
        "risk_thresholds": risk_thresholds,
        "confidence_thresholds": confidence_thresholds,
        "disagreement_thresholds": disagreement_thresholds,
        "low_risk_floors": low_risk_floors,
        "consequence_thresholds": consequence_thresholds,
        "criticality_thresholds": criticality_thresholds,
        "extreme_factor_thresholds": extreme_factor_thresholds,
    }
    candidates = _candidate_grid(
        {k: v for k, v in axis_values.items() if v is not None}
    )

    baseline = evaluate_candidate(CandidateConfig(), cache=cache, config=config)
    results = [
        evaluate_candidate(c, cache=cache, config=config) for c in candidates
    ]

    swept: set[str] = set()
    for c in candidates:
        swept.update(c.set_fields())

    return CalibrationSweepReport(
        generated_at=generated_at,
        evaluation_count=len(cache.records),
        candidate_count=len(candidates),
        swept_thresholds=sorted(swept),
        baseline=baseline,
        results=results,
        notes=[
            "OFFLINE evaluation. Detectors ran ONCE (CalibrationCache); each "
            "candidate recombines those cached outputs — no re-inference, no "
            "second decision algorithm.",
            "config/settings.yaml is never modified. A promising candidate is "
            "an input to human config governance, not an auto-tune.",
            "Safety metrics (recall / precision / f1 / accuracy / FPR / "
            "missed_risk_rate) and efficiency metrics (FAST/DEEP/human-review "
            "rates, latency) are reported separately and never fused.",
            "Ground-truth 'any risk' label = any of ground_truth_"
            "{hallucination,pii,toxicity,bias,cost_anomaly}; prediction = "
            "decision != ALLOW.",
            "Latency figures are recombined from one-time MEASURED per-pass "
            "timings and vary between cache builds; decision and safety "
            "metrics are fully deterministic.",
        ],
    )


# ======================================================================
# human-readable summary
# ======================================================================


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_summary(report: CalibrationSweepReport) -> str:
    lines = [
        "Calibration sweep",
        "------------------",
        "",
        f"Candidates: {report.candidate_count}",
        f"Evaluation cases: {report.evaluation_count}",
        f"Swept thresholds: {', '.join(report.swept_thresholds) or '(none)'}",
        "",
        "Baseline (current config):",
        f"  recall {report.baseline.safety.recall:.2f} · "
        f"precision {report.baseline.safety.precision:.2f} · "
        f"FPR {report.baseline.safety.false_positive_rate:.2f} · "
        f"missed-risk {report.baseline.safety.missed_risk_rate:.2f}",
        f"  FAST {_fmt_pct(report.baseline.efficiency.fast_path_rate)} · "
        f"DEEP {_fmt_pct(report.baseline.efficiency.deep_path_rate)} · "
        f"human-review {_fmt_pct(report.baseline.efficiency.human_review_rate)} · "
        f"avg latency {report.baseline.efficiency.average_latency_ms:.2f} ms",
    ]
    if report.results:
        best_recall = max(report.results, key=lambda r: r.safety.recall)
        low_hr = min(report.results, key=lambda r: r.efficiency.human_review_rate)
        fast_lo = min(r.efficiency.fast_path_rate for r in report.results)
        fast_hi = max(r.efficiency.fast_path_rate for r in report.results)
        lines += [
            "",
            "Across candidates:",
            f"  Best recall        : {best_recall.safety.recall:.2f} at "
            f"{best_recall.resolved_thresholds}",
            f"  Lowest human-review: {_fmt_pct(low_hr.efficiency.human_review_rate)} at "
            f"{low_hr.resolved_thresholds}",
            f"  FAST-path range    : {_fmt_pct(fast_lo)} → {_fmt_pct(fast_hi)}",
        ]
    lines += ["", "(No 'best configuration' is declared — this is an analysis tool.)"]
    return "\n".join(lines)


def _default_axes(config: dict[str, Any]) -> dict[str, list[float]]:
    """A tiny demo grid drawn from the existing calibration config."""
    cc = config.get("calibration", {})
    return {
        "risk_thresholds": [float(v) for v in cc.get("grid_risk", [0.25, 0.35, 0.45])],
        "confidence_thresholds": [
            float(v) for v in cc.get("grid_confidence", [0.60, 0.70, 0.80])
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    from settings import load_settings

    cfg = load_settings()
    axes = _default_axes(cfg)
    report = sweep_thresholds(
        risk_thresholds=axes["risk_thresholds"],
        confidence_thresholds=axes["confidence_thresholds"],
        config=cfg,
    )
    print(format_summary(report))
