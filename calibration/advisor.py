"""
Calibration Advisor — OFFLINE / EVALUATION ONLY.

Simulates alternative threshold configurations against the synthetic
evaluation dataset and reports the safety-vs-cost tradeoff. It is a
*consumer* of the evaluation infrastructure — it reuses
``evaluation.evaluation.load_evaluation_dataset`` / ``build_engine`` and
``evaluation.metrics``.

Architecture / data boundary
----------------------------
    EvaluationCase (ground truth)              PRODUCTION (no ground truth)
        |                                          Interaction
    CalibrationAdvisor                                 |
        |  -- precompute detector outputs ONCE --   detectors
    threshold experiments  (recombine, no re-run)      |
        |                                          verification -> ... -> decision
    metrics / recommendation

The advisor NEVER mutates ``config/settings.yaml`` (every experiment
runs on a deep-merged copy) and is NEVER imported by any production
module.
"""

from __future__ import annotations

import copy
from statistics import fmean
from typing import Any

from common.timing import Stopwatch
from calibration.schemas import (
    THRESHOLD_CONFIG_PATH,
    CalibrationMetrics,
    CalibrationRecommendation,
    CalibrationReport,
    CounterfactualAnalysis,
    CounterfactualChange,
    GridExperiment,
    GridPoint,
    ThresholdPoint,
    ThresholdSweep,
)
from data.schemas import Interaction, InterventionTier
from decision.engine import DecisionEngine
from evaluation.evaluation import _EVAL_TS, build_engine, load_evaluation_dataset
from settings import load_settings
from verification.backend import VerificationBackend
from verification.router import VerificationRouter

ADVISOR_NAME = "calibration_advisor"

_RISKY_GT_KEYS = (
    "ground_truth_hallucination",
    "ground_truth_pii",
    "ground_truth_toxicity",
    "ground_truth_bias",
    "ground_truth_cost_anomaly",
)


# ------------------------------------------------------------------ helpers


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _frange(spec: list[float]) -> list[float]:
    lo, hi, step = float(spec[0]), float(spec[1]), float(spec[2])
    values: list[float] = []
    current = lo
    # avoid float drift accumulating
    n = int(round((hi - lo) / step)) + 1
    for i in range(n):
        values.append(round(lo + i * step, 4))
    return [v for v in values if v <= hi + 1e-9]


def _override_for(threshold_type: str, value: float) -> dict:
    if threshold_type == "deep_verification_risk":
        # the FAST/DEEP risk boundary is two mirrored keys
        return {
            "verification": {
                "deep_verification_risk_threshold": value,
                "fast_path_max_risk": value,
            }
        }
    section, key = THRESHOLD_CONFIG_PATH[threshold_type]
    return {section: {key: value}}


def _ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


# ---------------------------------------------- cached detector views


class _CachedPerf:
    """Returns the pre-computed PerformanceResult for the requested depth."""

    def __init__(self, shallow: dict, deep: dict) -> None:
        self._shallow = shallow
        self._deep = deep

    def detect(self, interaction, context=None, *, depth: str = "deep"):
        table = self._shallow if depth == "shallow" else self._deep
        return table[interaction.interaction_id]


class _CachedDetector:
    """Returns the pre-computed result for responsibility / cost."""

    def __init__(self, by_id: dict) -> None:
        self._by_id = by_id

    def detect(self, interaction, *args, **kwargs):
        return self._by_id[interaction.interaction_id]


class _CachedVerifierBackend(VerificationBackend):
    name = "cached"

    def __init__(self, perf_view: _CachedPerf) -> None:
        self._perf = perf_view

    def verify(
        self,
        interaction,
        *,
        depth: str = "deep",
        bypass_semantics: bool = False,
        bypass_reason: str = "",
    ):
        if bypass_semantics:
            from verification.backend import _bypass_result

            return _bypass_result(bypass_reason or "deterministic hard boundary")
        return self._perf.detect(interaction, depth=depth)


# ---------------------------------------------- per-interaction record


class _Record:
    __slots__ = (
        "interaction", "gt_any", "gt_clean",
        "fast_path_ms", "deep_extra_ms", "downstream_ms",
    )

    def __init__(self, interaction, gt_any, fast_path_ms, deep_extra_ms, downstream_ms):
        self.interaction = interaction
        self.gt_any = gt_any
        self.gt_clean = not gt_any
        self.fast_path_ms = fast_path_ms
        self.deep_extra_ms = deep_extra_ms
        self.downstream_ms = downstream_ms


# ------------------------------------------------------------------ cache


class CalibrationCache:
    """
    One-time precompute: every detector output + measured per-pass
    latencies + ground-truth flag for every evaluation case. Threshold
    experiments recombine these; detectors are never re-run.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset: list[tuple[Interaction, dict[str, Any]]] | None = None,
    ) -> None:
        self._config = config if config is not None else load_settings()
        engine = build_engine(self._config)  # fitted cost baseline

        self.shallow_perf: dict[str, Any] = {}
        self.deep_perf: dict[str, Any] = {}
        self.responsibility: dict[str, Any] = {}
        self.cost: dict[str, Any] = {}
        self.records: list[_Record] = []

        rows = dataset if dataset is not None else load_evaluation_dataset(self._config)
        for interaction, ground_truth in rows:
            iid = interaction.interaction_id
            watch = Stopwatch()
            with watch.stage("resp"):
                resp = engine.responsibility.detect(interaction)
            with watch.stage("cost"):
                cost = engine.cost.detect(interaction)
            with watch.stage("shallow"):
                shallow = engine.performance.detect(interaction, depth="shallow")
            with watch.stage("deep"):
                deep = engine.performance.detect(interaction, depth="deep")

            self.responsibility[iid] = resp
            self.cost[iid] = cost
            self.shallow_perf[iid] = shallow
            self.deep_perf[iid] = deep

            with watch.stage("downstream"):
                crit = engine.criticality.assess(interaction, deep)
                crit_input = max(crit.action_criticality, crit.max_claim_criticality)
                amplified = engine.criticality.amplify_performance_risk(
                    deep.performance_risk, crit_input
                )
                fused = engine.fusion.fuse_scores(
                    amplified, resp.overall_responsibility_risk, cost.cost_risk,
                    performance_confidence=deep.confidence,
                    responsibility_confidence=resp.confidence,
                    cost_confidence=cost.confidence,
                )
                engine.consequence.assess(interaction)
                _ = fused

            gt_any = any(bool(ground_truth.get(k)) for k in _RISKY_GT_KEYS)
            self.records.append(
                _Record(
                    interaction=interaction,
                    gt_any=gt_any,
                    fast_path_ms=watch.get("resp") + watch.get("cost") + watch.get("shallow"),
                    deep_extra_ms=watch.get("deep"),
                    downstream_ms=watch.get("downstream"),
                )
            )

    # -- views for a cached DecisionEngine --

    def perf_view(self) -> _CachedPerf:
        return _CachedPerf(self.shallow_perf, self.deep_perf)

    def resp_view(self) -> _CachedDetector:
        return _CachedDetector(self.responsibility)

    def cost_view(self) -> _CachedDetector:
        return _CachedDetector(self.cost)


# ------------------------------------------------------------------ advisor


class CalibrationAdvisor:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache: CalibrationCache | None = None,
    ) -> None:
        self._config = config if config is not None else load_settings()
        self._cc = self._config["calibration"]
        self._cache = cache or CalibrationCache(self._config)
        self._current_metrics: CalibrationMetrics | None = None
        # memoise per-config metrics and sweeps (deterministic given the cache)
        self._metrics_cache: dict[str, CalibrationMetrics] = {}
        self._sweep_cache: dict[str, ThresholdSweep] = {}

    # -------------------------------------------------- core

    def _engine_for(self, overrides: dict) -> DecisionEngine:
        cfg = _deep_merge(self._config, overrides) if overrides else self._config
        engine = DecisionEngine(cfg)
        perf_view = self._cache.perf_view()
        engine.performance = perf_view
        engine.responsibility = self._cache.resp_view()
        engine.cost = self._cache.cost_view()
        engine.verification_router = VerificationRouter(
            cfg,
            responsibility=engine.responsibility,
            cost=engine.cost,
            criticality=engine.criticality,
            consequence=engine.consequence,
            fusion=engine.fusion,
            backend=_CachedVerifierBackend(perf_view),
        )
        return engine

    def evaluate_config(self, overrides: dict) -> CalibrationMetrics:
        import json

        key = json.dumps(overrides, sort_keys=True)
        cached = self._metrics_cache.get(key)
        if cached is not None:
            return cached
        metrics = self._evaluate_config(overrides)
        self._metrics_cache[key] = metrics
        return metrics

    def _evaluate_config(self, overrides: dict) -> CalibrationMetrics:
        engine = self._engine_for(overrides)
        n = len(self._cache.records)
        deep = tp = fp = fn = tn = hr = block = abst = 0
        verif_lat: list[float] = []
        total_lat: list[float] = []
        deep_workload = 0.0

        for rec in self._cache.records:
            trace = engine.evaluate(
                rec.interaction, timestamp=_EVAL_TS, record_session=False
            )
            decision = trace.final_decision.decision
            intervened = decision != InterventionTier.ALLOW
            is_deep = trace.verification_path == "DEEP"

            if is_deep:
                deep += 1
                deep_workload += rec.deep_extra_ms
            if rec.gt_any and intervened:
                tp += 1
            elif rec.gt_clean and intervened:
                fp += 1
            elif rec.gt_any and not intervened:
                fn += 1
            else:
                tn += 1
            if decision in (InterventionTier.HUMAN_REVIEW, InterventionTier.BLOCK):
                hr += 1
            if decision == InterventionTier.BLOCK:
                block += 1
            if trace.performance.status.value == "UNVERIFIED":
                abst += 1

            lat = rec.fast_path_ms + (rec.deep_extra_ms if is_deep else 0.0)
            verif_lat.append(lat)
            total_lat.append(lat + rec.downstream_ms)

        recall = _ratio(tp, tp + fn)
        precision = _ratio(tp, tp + fp)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision and recall and (precision + recall)
            else None
        )
        fpr = _ratio(fp, fp + tn)
        fnr = _ratio(fn, fn + tp)
        return CalibrationMetrics(
            n_cases=n,
            n_risky=sum(1 for r in self._cache.records if r.gt_any),
            n_clean=sum(1 for r in self._cache.records if r.gt_clean),
            deep_verification_rate=round(deep / n, 4),
            fast_verification_rate=round((n - deep) / n, 4),
            intervention_recall=None if recall is None else round(recall, 4),
            intervention_precision=None if precision is None else round(precision, 4),
            intervention_f1=f1,
            false_positive_rate=None if fpr is None else round(fpr, 4),
            false_negative_rate=None if fnr is None else round(fnr, 4),
            human_review_rate=round(hr / n, 4),
            block_rate=round(block / n, 4),
            abstention_rate=round(abst / n, 4),
            mean_verification_latency_ms=round(fmean(verif_lat), 4),
            mean_total_pipeline_latency_ms=round(fmean(total_lat), 4),
            total_deep_workload_ms=round(deep_workload, 4),
        )

    def current_operating_point(self) -> CalibrationMetrics:
        if self._current_metrics is None:
            self._current_metrics = self.evaluate_config({})
        return self._current_metrics

    # -------------------------------------------------- sweeps

    def _baseline_value(self, threshold_type: str) -> float:
        section, key = THRESHOLD_CONFIG_PATH[threshold_type]
        return float(self._config[section][key])

    def _sweep_values(self, threshold_type: str) -> list[float]:
        key = {
            "deep_verification_risk": "risk_sweep",
            "fast_path_min_confidence": "confidence_sweep",
            "deep_verification_consequence": "consequence_sweep",
            "deep_verification_criticality": "criticality_sweep",
            "disagreement_trigger": "disagreement_sweep",
        }[threshold_type]
        return _frange(self._cc[key])

    def _constraints(self) -> dict[str, float]:
        return {
            "target_recall": float(self._cc["target_recall"]),
            "max_false_positive_rate": float(self._cc["max_false_positive_rate"]),
            "max_human_review_rate": float(self._cc["max_human_review_rate"]),
        }

    def _check_safety(self, m: CalibrationMetrics) -> list[str]:
        c = self._constraints()
        violations: list[str] = []
        if m.intervention_recall is None:
            violations.append("recall could not be computed (no risky cases)")
        elif m.intervention_recall < c["target_recall"]:
            violations.append(
                f"recall {m.intervention_recall:.2f} < target {c['target_recall']:.2f}"
            )
        if m.false_positive_rate is not None and m.false_positive_rate > c["max_false_positive_rate"]:
            violations.append(
                f"false-positive rate {m.false_positive_rate:.2f} > max "
                f"{c['max_false_positive_rate']:.2f}"
            )
        if m.human_review_rate > c["max_human_review_rate"]:
            violations.append(
                f"human-review rate {m.human_review_rate:.2f} > max "
                f"{c['max_human_review_rate']:.2f}"
            )
        return violations

    def sweep(
        self, threshold_type: str, values: list[float] | None = None
    ) -> ThresholdSweep:
        if threshold_type not in THRESHOLD_CONFIG_PATH:
            raise ValueError(f"unknown threshold_type: {threshold_type}")
        values = values if values is not None else self._sweep_values(threshold_type)
        cache_key = f"{threshold_type}:{values}"
        if cache_key in self._sweep_cache:
            return self._sweep_cache[cache_key]
        section, key = THRESHOLD_CONFIG_PATH[threshold_type]

        points: list[ThresholdPoint] = []
        for value in values:
            metrics = self.evaluate_config(_override_for(threshold_type, value))
            violations = self._check_safety(metrics)
            points.append(
                ThresholdPoint(
                    threshold_type=threshold_type,
                    threshold_value=value,
                    metrics=metrics,
                    satisfies_safety_constraints=not violations,
                    constraint_violations=violations,
                )
            )
        sweep = ThresholdSweep(
            threshold_type=threshold_type,
            config_path=f"{section}.{key}",
            baseline_value=self._baseline_value(threshold_type),
            points=points,
            safety_constraints=self._constraints(),
            tradeoff_note=(
                "Lower threshold -> more DEEP verification -> more measured compute/"
                "latency, potentially better risk detection. Higher threshold -> more "
                "FAST verification -> lower measured latency, potentially greater "
                "missed-risk exposure. Latency figures are recombined from one-time "
                "MEASURED per-pass timings; DEEP workload is relative measured compute."
            ),
        )
        self._sweep_cache[cache_key] = sweep
        return sweep

    # -------------------------------------------------- recommendation

    def recommend(
        self, threshold_type: str = "deep_verification_risk"
    ) -> CalibrationRecommendation:
        sweep = self.sweep(threshold_type)
        objective = str(self._cc["objective"])
        current_value = self._baseline_value(threshold_type)
        current = self.current_operating_point()
        section, key = THRESHOLD_CONFIG_PATH[threshold_type]

        safe = [p for p in sweep.points if p.satisfies_safety_constraints]
        if not safe:
            return CalibrationRecommendation(
                status="NO_SAFE_OPERATING_POINT",
                threshold_type=threshold_type,
                config_path=f"{section}.{key}",
                current_value=current_value,
                current_metrics=current,
                objective=objective,
                explanation=self._explain_no_safe(sweep),
            )

        if objective == "max_f1":
            best = max(safe, key=lambda p: (p.metrics.intervention_f1 or 0.0, p.threshold_value))
        else:  # min_deep_workload — least DEEP verification, break ties toward higher threshold
            best = min(safe, key=lambda p: (p.metrics.deep_verification_rate, -p.threshold_value))

        return CalibrationRecommendation(
            status="RECOMMENDATION",
            threshold_type=threshold_type,
            config_path=f"{section}.{key}",
            current_value=current_value,
            recommended_value=best.threshold_value,
            current_metrics=current,
            recommended_metrics=best.metrics,
            objective=objective,
            explanation=self._explain_recommendation(best, current_value, current, objective),
        )

    def _explain_recommendation(self, best, current_value, current, objective) -> str:
        m = best.metrics
        c = self._constraints()
        lines = [
            f"Recommended {best.threshold_type} = {best.threshold_value:.2f} "
            f"(current {current_value:.2f}). Objective: {objective}.",
            "It satisfies every safety constraint:",
            f"  - intervention recall {m.intervention_recall:.2f} >= target {c['target_recall']:.2f}",
        ]
        if m.false_positive_rate is not None:
            lines.append(
                f"  - false-positive rate {m.false_positive_rate:.2f} <= max "
                f"{c['max_false_positive_rate']:.2f}"
            )
        lines.append(
            f"  - human-review rate {m.human_review_rate:.2f} <= max "
            f"{c['max_human_review_rate']:.2f}"
        )
        lines.append(
            f"Among safe points it has the lowest DEEP-verification workload: "
            f"{m.deep_verification_rate:.0%} of interactions go DEEP "
            f"(current {current.deep_verification_rate:.0%}), "
            f"mean verification latency {m.mean_verification_latency_ms:.2f} ms "
            f"(current {current.mean_verification_latency_ms:.2f} ms)."
        )
        rd = _delta(m.intervention_recall, current.intervention_recall)
        fd = _delta(m.false_positive_rate, current.false_positive_rate)
        if rd is not None:
            lines.append(
                f"Versus the current operating point: recall {rd:+.2f}, "
                f"FPR {fd:+.2f} (if computable), DEEP workload "
                f"{m.deep_verification_rate - current.deep_verification_rate:+.2f}."
            )
        return "\n".join(lines)

    def _explain_no_safe(self, sweep: ThresholdSweep) -> str:
        c = self._constraints()
        recall_fail = sum(1 for p in sweep.points if any("recall" in v for v in p.constraint_violations))
        fpr_fail = sum(1 for p in sweep.points if any("false-positive" in v for v in p.constraint_violations))
        hr_fail = sum(1 for p in sweep.points if any("human-review" in v for v in p.constraint_violations))
        return (
            f"No operating point over the {sweep.threshold_type} sweep satisfies all "
            f"safety constraints (target_recall {c['target_recall']:.2f}, "
            f"max_false_positive_rate {c['max_false_positive_rate']:.2f}, "
            f"max_human_review_rate {c['max_human_review_rate']:.2f}). "
            f"Of {len(sweep.points)} candidates: {recall_fail} miss the recall target, "
            f"{fpr_fail} exceed the FPR limit, {hr_fail} exceed the human-review limit. "
            "The safety constraints were NOT relaxed."
        )

    # -------------------------------------------------- counterfactual

    def counterfactual(
        self,
        threshold_type: str,
        baseline_value: float,
        counterfactual_value: float,
    ) -> CounterfactualAnalysis:
        section, key = THRESHOLD_CONFIG_PATH[threshold_type]
        base = self.evaluate_config(_override_for(threshold_type, baseline_value))
        cf = self.evaluate_config(_override_for(threshold_type, counterfactual_value))
        changes = CounterfactualChange(
            recall_delta=_delta(cf.intervention_recall, base.intervention_recall),
            precision_delta=_delta(cf.intervention_precision, base.intervention_precision),
            fpr_delta=_delta(cf.false_positive_rate, base.false_positive_rate),
            human_review_delta=_delta(cf.human_review_rate, base.human_review_rate),
            deep_verification_delta=_delta(cf.deep_verification_rate, base.deep_verification_rate),
            mean_verification_latency_delta_ms=_delta(
                cf.mean_verification_latency_ms, base.mean_verification_latency_ms
            ),
            deep_workload_delta_ms=_delta(cf.total_deep_workload_ms, base.total_deep_workload_ms),
        )
        direction = "raising" if counterfactual_value > baseline_value else "lowering"
        summary = (
            f"{direction.capitalize()} {threshold_type} from {baseline_value:.2f} to "
            f"{counterfactual_value:.2f} changes DEEP verification by "
            f"{changes.deep_verification_delta:+.2f}, recall by "
            f"{(changes.recall_delta if changes.recall_delta is not None else 0):+.2f}, "
            f"FPR by {(changes.fpr_delta if changes.fpr_delta is not None else 0):+.2f}, "
            f"human-review rate by {changes.human_review_delta:+.2f}, and mean "
            f"verification latency by {changes.mean_verification_latency_delta_ms:+.2f} ms "
            "(recombined from measured per-pass timings)."
        )
        return CounterfactualAnalysis(
            threshold_type=threshold_type,
            config_path=f"{section}.{key}",
            baseline_value=baseline_value,
            counterfactual_value=counterfactual_value,
            baseline_metrics=base,
            counterfactual_metrics=cf,
            changes=changes,
            summary=summary,
        )

    # -------------------------------------------------- 2-D grid

    def grid(self) -> GridExperiment:
        risk_values = [float(v) for v in self._cc["grid_risk"]]
        conf_values = [float(v) for v in self._cc["grid_confidence"]]
        top_n = int(self._cc.get("grid_top_n", 3))
        objective = str(self._cc["objective"])

        points: list[GridPoint] = []
        for r in risk_values:
            for conf in conf_values:
                metrics = self.evaluate_config(
                    {
                        "verification": {
                            "deep_verification_risk_threshold": r,
                            "fast_path_max_risk": r,
                            "fast_path_min_confidence": conf,
                        }
                    }
                )
                points.append(
                    GridPoint(
                        risk_threshold=r,
                        confidence_threshold=conf,
                        metrics=metrics,
                        satisfies_safety_constraints=not self._check_safety(metrics),
                    )
                )
        safe = [p for p in points if p.satisfies_safety_constraints]
        if objective == "max_f1":
            safe.sort(key=lambda p: -(p.metrics.intervention_f1 or 0.0))
        else:
            safe.sort(key=lambda p: (p.metrics.deep_verification_rate, -p.risk_threshold))

        # Do the two axes actually move the outcome on this dataset?
        deep_by_risk = {
            r: {p.metrics.deep_verification_rate for p in points if p.risk_threshold == r}
            for r in risk_values
        }
        interact = any(len(v) > 1 for v in deep_by_risk.values())
        note = (
            "The two axes interact on this dataset: at a fixed risk threshold, "
            "requiring higher confidence for FAST pushes more borderline interactions "
            "to DEEP."
            if interact
            else "On this dataset the two axes are largely independent in the grid "
            "range tested — routing is dominated by other triggers "
            "(disagreement / missing-evidence / extreme consequence factor)."
        )
        return GridExperiment(
            risk_values=risk_values,
            confidence_values=conf_values,
            points=points,
            best_points=safe[:top_n],
            note=note + " Only safe grid points are ranked.",
        )

    # -------------------------------------------------- full report

    def report(self, generated_at=None) -> CalibrationReport:
        baseline_cfg = {
            "deep_verification_risk_threshold": self._baseline_value("deep_verification_risk"),
            "fast_path_min_confidence": self._baseline_value("fast_path_min_confidence"),
            "deep_verification_consequence_threshold": self._baseline_value(
                "deep_verification_consequence"
            ),
            "deep_verification_criticality_threshold": self._baseline_value(
                "deep_verification_criticality"
            ),
            "disagreement_trigger": self._baseline_value("disagreement_trigger"),
        }
        sweeps = [
            self.sweep("deep_verification_risk"),
            self.sweep("fast_path_min_confidence"),
            self.sweep("deep_verification_consequence"),
            self.sweep("deep_verification_criticality"),
        ]
        recommendation = self.recommend("deep_verification_risk")
        grid = self.grid()
        base_r = baseline_cfg["deep_verification_risk_threshold"]
        counterfactuals = [
            self.counterfactual("deep_verification_risk", base_r, round(base_r - 0.15, 4)),
            self.counterfactual("deep_verification_risk", base_r, round(base_r + 0.20, 4)),
        ]
        return CalibrationReport(
            generated_at=generated_at,
            n_cases=len(self._cache.records),
            baseline_config=baseline_cfg,
            current_operating_point=self.current_operating_point(),
            sweeps=sweeps,
            recommendation=recommendation,
            grid=grid,
            counterfactuals=counterfactuals,
            notes=[
                "OFFLINE experiment. Detectors were run ONCE; every threshold point "
                "recombines those cached outputs — no re-inference.",
                "Latency values are recombined from one-time MEASURED per-pass timings; "
                "DEEP workload is relative measured compute, not a production SLA.",
                "This report does NOT change any production threshold. Applying a "
                "recommendation is a human config-governance decision.",
                "Metrics are computed against synthetic ground truth; a real enterprise "
                "traffic mix would shift the operating point.",
            ],
        )


def run_calibration(config: dict[str, Any] | None = None) -> CalibrationReport:
    """Convenience: build the advisor and produce a full report."""
    from datetime import datetime, timezone

    return CalibrationAdvisor(config=config).report(
        generated_at=datetime.now(timezone.utc)
    )


if __name__ == "__main__":  # pragma: no cover
    import json
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    report = run_calibration()
    with open("calibration_report.json", "w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)

    m = report.current_operating_point
    rec = report.recommendation
    rm = rec.recommended_metrics
    grid_best = report.grid.best_points[0] if report.grid.best_points else None

    print("CONTROLPLANE CALIBRATION ADVISOR")
    print(f"\nEvaluation cases: {report.n_cases}")
    print("\nCurrent operating point:")
    print(f"  deep verification : {m.deep_verification_rate:.1%}")
    print(f"  intervention recall: {m.intervention_recall}")
    print(f"  false-positive rate: {m.false_positive_rate}")
    print(f"  human review rate : {m.human_review_rate:.1%}")
    print(f"  mean verification latency: {m.mean_verification_latency_ms:.2f} ms "
          "(recombined from measured per-pass timings)")

    if rec.status == "RECOMMENDATION":
        print("\nRecommended operating point (1-D, objective "
              f"'{rec.objective}'):")
        print(f"  {rec.config_path}: {rec.current_value:.2f} -> {rec.recommended_value:.2f}")
        print("\nImpact vs current:")
        print(f"  DEEP workload     : {rm.deep_verification_rate - m.deep_verification_rate:+.1%}")
        if rm.intervention_recall is not None and m.intervention_recall is not None:
            print(f"  recall            : {rm.intervention_recall - m.intervention_recall:+.2f}")
        if rm.false_positive_rate is not None and m.false_positive_rate is not None:
            print(f"  FPR               : {rm.false_positive_rate - m.false_positive_rate:+.2f}")
        print(f"  human review rate : {rm.human_review_rate - m.human_review_rate:+.2f}")
        print(f"  mean verification latency: "
              f"{rm.mean_verification_latency_ms - m.mean_verification_latency_ms:+.2f} ms")
        if grid_best is not None:
            print("\n2-D grid optimum (risk x fast-path confidence):")
            print(f"  risk={grid_best.risk_threshold:.2f}, confidence={grid_best.confidence_threshold:.2f}"
                  f"  ->  DEEP {grid_best.metrics.deep_verification_rate:.1%}"
                  f" (recall {grid_best.metrics.intervention_recall})")
        print("\nRecommendation:")
        print("  " + rec.explanation.replace("\n", "\n  "))
    else:
        print(f"\nRecommendation: {rec.status}")
        print("  " + rec.explanation.replace("\n", "\n  "))

    print("\n[calibration_report.json written]")
