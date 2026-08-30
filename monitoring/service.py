"""
Enterprise observability service.

``MonitoringService.report(traces)`` turns a collection of real
``DecisionTrace`` records into a single :class:`MonitoringReport`. It is a
pure aggregation pass — O(number_of_traces) — and never re-runs a
detector, the decision engine, or verification, and never reads ground
truth.

Run ``python -m monitoring.service`` to execute the real pipeline over the
demo scenarios + a sample of synthetic production traffic and print the
operational monitor summary.
"""

from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from monitoring.metrics import (
    bucket_index,
    count_by,
    max_or_none,
    mean_or_none,
    percentile_or_none,
    rate_or_none,
    truncate_timestamp,
)
from monitoring.schemas import (
    ApplicationMetrics,
    ApplicationVerificationSplit,
    AttentionSignal,
    ConfidenceMetrics,
    DataQualityReport,
    DecisionMetrics,
    DetectorHealth,
    LatencyStats,
    ModelMetrics,
    MonitoringReport,
    MonitoringWindow,
    ReasonCodeMetrics,
    RiskBucket,
    RiskDistribution,
    RiskStats,
    RuleMetrics,
    TrendBucket,
    VerificationMetrics,
)
from settings import load_settings

_FAST = "FAST"
_DEEP = "DEEP"


class MonitoringService:
    """Aggregates real ``DecisionTrace`` records into a ``MonitoringReport``."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config if config is not None else load_settings()
        mon = dict(self._config.get("monitoring", {}) or {})

        raw_buckets = mon.get("risk_buckets") or [
            {"name": "LOW", "min": 0.0, "max": 0.2},
            {"name": "MODERATE", "min": 0.2, "max": 0.5},
            {"name": "HIGH", "min": 0.5, "max": 0.75},
            {"name": "CRITICAL", "min": 0.75, "max": 1.01},
        ]
        self._buckets = [
            (str(b["name"]), float(b["min"]), float(b["max"])) for b in raw_buckets
        ]
        self._low_confidence_threshold = float(mon.get("low_confidence_threshold", 0.45))
        self._percentiles = [int(p) for p in (mon.get("percentiles") or [50, 95])]
        self._trend_granularity = str(mon.get("trend_granularity", "hourly"))
        self._signal_cfg = dict(mon.get("attention_signals", {}) or {})
        self._critical_multiplier = float(
            self._signal_cfg.get("critical_multiplier", 1.5)
        )
        # buckets whose midpoint is "elevated" — used for risk concentration.
        self._elevated_bucket_names = {
            name for name, _lo, hi in self._buckets if hi > 0.5
        }

    # ------------------------------------------------------------------
    # public entry point

    def report(
        self,
        traces: Iterable[Any],
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        generated_at: datetime | None = None,
        trend_granularity: str | None = None,
    ) -> MonitoringReport:
        granularity = trend_granularity or self._trend_granularity

        valid, quality = self._validate(traces)
        in_window, excluded = self._apply_window(valid, start_time, end_time)

        window = MonitoringWindow(
            start_time=start_time,
            end_time=end_time,
            applied=start_time is not None or end_time is not None,
            traces_in_window=len(in_window),
            traces_excluded_by_window=excluded,
        )

        if not in_window:
            return self._empty_report(window, quality, generated_at, granularity)

        n = len(in_window)
        decisions = self._decision_metrics(in_window, n)
        risk = self._risk_stats(in_window)
        confidence = self._confidence_metrics(in_window, n)
        latency = self._latency_stats(in_window)
        verification = self._verification_metrics(in_window, n)
        risk_distribution = self._risk_distribution(in_window, n)
        reason_codes = self._reason_code_metrics(in_window)
        policy_rules = self._policy_rule_metrics(in_window)
        applications = self._application_metrics(in_window)
        models = self._model_metrics(in_window)
        detectors = self._detector_health(in_window)
        verification_by_application = self._verification_by_application(in_window)
        trend = self._trend(in_window, granularity)
        attention_signals = self._attention_signals(
            decisions, confidence, latency, verification, risk_distribution, n
        )

        return MonitoringReport(
            generated_at=generated_at,
            window=window,
            total_interactions=n,
            decisions=decisions,
            risk=risk,
            confidence=confidence,
            latency=latency,
            verification=verification,
            risk_distribution=risk_distribution,
            reason_codes=reason_codes,
            policy_rules=policy_rules,
            applications=applications,
            models=models,
            detectors=detectors,
            verification_by_application=verification_by_application,
            attention_signals=attention_signals,
            trend_granularity=granularity,
            trend=trend,
            data_quality=quality,
            notes=[
                "All metrics are aggregated from real DecisionTrace records; "
                "no detector, engine or verification pass was re-run.",
                "Monitoring observes production decisions — it does not evaluate "
                "whether those decisions were correct (no ground truth is read).",
                "Model-level differences are an observed association over this "
                "traffic sample, not a causal claim.",
                "Detector error_count is None because the trace structure does "
                "not record detector errors.",
                "Attention signals are deterministic operational attention "
                "signals (threshold crossings), not statistically detected "
                "anomalies.",
            ],
        )

    # ------------------------------------------------------------------
    # validation / data quality

    def _validate(self, traces: Iterable[Any]) -> tuple[list[DecisionTrace], DataQualityReport]:
        valid: list[DecisionTrace] = []
        seen = 0
        skipped = 0
        reasons: list[str] = []

        for idx, item in enumerate(traces):
            seen += 1
            if isinstance(item, DecisionTrace):
                valid.append(item)
                continue
            try:
                valid.append(DecisionTrace.model_validate(item))
            except Exception as exc:  # noqa: BLE001 — we want to report, not raise
                skipped += 1
                kind = type(item).__name__
                reasons.append(
                    f"record #{idx}: could not be parsed as a DecisionTrace "
                    f"(input type {kind}): {type(exc).__name__}"
                )

        missing: dict[str, int] = defaultdict(int)
        for trace in valid:
            if trace.model is None:
                missing["model"] += 1
            if trace.verification is None:
                missing["verification"] += 1
            if not trace.policy.rule_trace:
                missing["policy.rule_trace"] += 1
            if not trace.decision_path:
                missing["decision_path"] += 1

        quality = DataQualityReport(
            total_records_seen=seen,
            valid_records=len(valid),
            invalid_records_skipped=skipped,
            missing_trace_fields=dict(missing),
            exclusion_reasons=reasons,
        )
        return valid, quality

    @staticmethod
    def _apply_window(
        traces: list[DecisionTrace],
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> tuple[list[DecisionTrace], int]:
        if start_time is None and end_time is None:
            return list(traces), 0
        kept: list[DecisionTrace] = []
        excluded = 0
        for trace in traces:
            ts = trace.timestamp
            if start_time is not None and ts < start_time:
                excluded += 1
                continue
            if end_time is not None and ts > end_time:
                excluded += 1
                continue
            kept.append(trace)
        return kept, excluded

    # ------------------------------------------------------------------
    # sections

    @staticmethod
    def _decision_metrics(traces: list[DecisionTrace], n: int) -> DecisionMetrics:
        counts = count_by(t.final_decision.decision.value for t in traces)
        allow = counts.get("ALLOW", 0)
        annotate = counts.get("ANNOTATE", 0)
        verify = counts.get("VERIFY", 0)
        human = counts.get("HUMAN_REVIEW", 0)
        block = counts.get("BLOCK", 0)
        return DecisionMetrics(
            allow_count=allow,
            annotate_count=annotate,
            verify_count=verify,
            human_review_count=human,
            block_count=block,
            human_review_rate=rate_or_none(human, n),
            block_rate=rate_or_none(block, n),
            human_oversight_rate=rate_or_none(human + block, n),
        )

    @staticmethod
    def _risk_stats(traces: list[DecisionTrace]) -> RiskStats:
        risks = [t.final_decision.overall_risk for t in traces]
        return RiskStats(
            mean_overall_risk=mean_or_none(risks),
            p50_overall_risk=percentile_or_none(risks, 50),
            p95_overall_risk=percentile_or_none(risks, 95),
            max_overall_risk=max_or_none(risks),
        )

    def _confidence_metrics(self, traces: list[DecisionTrace], n: int) -> ConfidenceMetrics:
        confidences = [t.final_decision.decision_confidence for t in traces]
        low = sum(1 for c in confidences if c < self._low_confidence_threshold)
        return ConfidenceMetrics(
            mean_decision_confidence=mean_or_none(confidences),
            low_confidence_count=low,
            low_confidence_rate=rate_or_none(low, n),
            low_confidence_threshold=self._low_confidence_threshold,
        )

    @staticmethod
    def _latency_stats(traces: list[DecisionTrace]) -> LatencyStats:
        lat = [t.latency_ms for t in traces]
        return LatencyStats(
            mean_total_latency_ms=mean_or_none(lat),
            p50_total_latency_ms=percentile_or_none(lat, 50),
            p95_total_latency_ms=percentile_or_none(lat, 95),
            max_total_latency_ms=max_or_none(lat),
        )

    @staticmethod
    def _path(trace: DecisionTrace) -> str:
        return (trace.verification_path or _DEEP).upper()

    def _verification_metrics(self, traces: list[DecisionTrace], n: int) -> VerificationMetrics:
        fast = [t for t in traces if self._path(t) == _FAST]
        deep = [t for t in traces if self._path(t) == _DEEP]
        unknown = [t for t in traces if self._path(t) not in (_FAST, _DEEP)]

        trigger_counts: dict[str, int] = {}
        for reason, count in count_by(
            reason
            for t in traces
            if t.verification is not None
            for reason in t.verification.deep_trigger_reasons
        ).items():
            trigger_counts[reason] = count

        return VerificationMetrics(
            fast_count=len(fast),
            deep_count=len(deep),
            unknown_count=len(unknown),
            fast_rate=rate_or_none(len(fast), n),
            deep_rate=rate_or_none(len(deep), n),
            mean_total_latency_fast_ms=mean_or_none([t.latency_ms for t in fast]),
            mean_total_latency_deep_ms=mean_or_none([t.latency_ms for t in deep]),
            mean_verification_latency_fast_ms=mean_or_none(
                [
                    t.verification.total_verification_latency_ms
                    for t in fast
                    if t.verification is not None
                ]
            ),
            mean_verification_latency_deep_ms=mean_or_none(
                [
                    t.verification.total_verification_latency_ms
                    for t in deep
                    if t.verification is not None
                ]
            ),
            deep_trigger_reason_counts=trigger_counts,
        )

    def _risk_distribution(self, traces: list[DecisionTrace], n: int) -> RiskDistribution:
        bounds = [(lo, hi) for _name, lo, hi in self._buckets]
        counts = [0] * len(self._buckets)
        for trace in traces:
            counts[bucket_index(trace.final_decision.overall_risk, bounds)] += 1
        buckets = [
            RiskBucket(
                bucket_name=name,
                min_risk=lo,
                max_risk=hi,
                count=count,
                percentage=(
                    None if n == 0 else round(100.0 * count / n, 4)
                ),
            )
            for (name, lo, hi), count in zip(self._buckets, counts)
        ]
        return RiskDistribution(buckets=buckets, total=sum(counts))

    @staticmethod
    def _reason_code_metrics(traces: list[DecisionTrace]) -> list[ReasonCodeMetrics]:
        counts = count_by(
            code for t in traces for code in t.final_decision.reason_codes
        )
        return [ReasonCodeMetrics(reason_code=c, count=n) for c, n in counts.items()]

    @staticmethod
    def _policy_rule_metrics(traces: list[DecisionTrace]) -> list[RuleMetrics]:
        counts = count_by(
            entry.rule
            for t in traces
            for entry in t.policy.rule_trace
            if entry.fired
        )
        return [RuleMetrics(rule=r, fired_count=n) for r, n in counts.items()]

    def _application_metrics(self, traces: list[DecisionTrace]) -> list[ApplicationMetrics]:
        by_app: dict[str, list[DecisionTrace]] = defaultdict(list)
        for trace in traces:
            by_app[trace.application].append(trace)

        out: list[ApplicationMetrics] = []
        for app in sorted(by_app):
            rows = by_app[app]
            k = len(rows)
            risks = [t.final_decision.overall_risk for t in rows]
            out.append(
                ApplicationMetrics(
                    application=app,
                    interactions=k,
                    mean_risk=mean_or_none(risks),
                    p95_risk=percentile_or_none(risks, 95),
                    deep_rate=rate_or_none(
                        sum(1 for t in rows if self._path(t) == _DEEP), k
                    ),
                    human_review_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.HUMAN_REVIEW
                        ),
                        k,
                    ),
                    block_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.BLOCK
                        ),
                        k,
                    ),
                    mean_latency_ms=mean_or_none([t.latency_ms for t in rows]),
                    low_confidence_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision_confidence
                            < self._low_confidence_threshold
                        ),
                        k,
                    ),
                )
            )
        return out

    def _model_metrics(self, traces: list[DecisionTrace]) -> list[ModelMetrics]:
        by_model: dict[str, list[DecisionTrace]] = defaultdict(list)
        for trace in traces:
            by_model[trace.model or "unknown"].append(trace)

        out: list[ModelMetrics] = []
        for model in sorted(by_model):
            rows = by_model[model]
            k = len(rows)
            out.append(
                ModelMetrics(
                    model=model,
                    interaction_count=k,
                    mean_risk=mean_or_none(
                        [t.final_decision.overall_risk for t in rows]
                    ),
                    deep_rate=rate_or_none(
                        sum(1 for t in rows if self._path(t) == _DEEP), k
                    ),
                    human_review_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.HUMAN_REVIEW
                        ),
                        k,
                    ),
                    block_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.BLOCK
                        ),
                        k,
                    ),
                    mean_latency_ms=mean_or_none([t.latency_ms for t in rows]),
                )
            )
        return out

    @staticmethod
    def _detector_health(traces: list[DecisionTrace]) -> list[DetectorHealth]:
        specs = {
            "performance": [t.performance.latency.total_ms for t in traces],
            "responsibility": [t.responsibility.latency_ms for t in traces],
            "cost": [t.cost.latency_ms for t in traces],
        }
        out: list[DetectorHealth] = []
        for name, lat in specs.items():
            out.append(
                DetectorHealth(
                    detector=name,
                    invocation_count=len(lat),
                    mean_latency_ms=mean_or_none(lat),
                    p95_latency_ms=percentile_or_none(lat, 95),
                    max_latency_ms=max_or_none(lat),
                    error_count=None,
                )
            )
        return out

    def _verification_by_application(
        self, traces: list[DecisionTrace]
    ) -> list[ApplicationVerificationSplit]:
        by_app: dict[str, list[DecisionTrace]] = defaultdict(list)
        for trace in traces:
            by_app[trace.application].append(trace)

        out: list[ApplicationVerificationSplit] = []
        for app in sorted(by_app):
            rows = by_app[app]
            k = len(rows)
            fast = [t for t in rows if self._path(t) == _FAST]
            deep = [t for t in rows if self._path(t) == _DEEP]
            out.append(
                ApplicationVerificationSplit(
                    application=app,
                    fast_count=len(fast),
                    deep_count=len(deep),
                    fast_rate=rate_or_none(len(fast), k),
                    deep_rate=rate_or_none(len(deep), k),
                    mean_total_latency_fast_ms=mean_or_none(
                        [t.latency_ms for t in fast]
                    ),
                    mean_total_latency_deep_ms=mean_or_none(
                        [t.latency_ms for t in deep]
                    ),
                )
            )
        return out

    def _trend(self, traces: list[DecisionTrace], granularity: str) -> list[TrendBucket]:
        buckets: dict[datetime, list[DecisionTrace]] = defaultdict(list)
        for trace in traces:
            buckets[truncate_timestamp(trace.timestamp, granularity)].append(trace)

        out: list[TrendBucket] = []
        for start in sorted(buckets):
            rows = buckets[start]
            k = len(rows)
            out.append(
                TrendBucket(
                    bucket_start=start,
                    interaction_count=k,
                    mean_risk=mean_or_none(
                        [t.final_decision.overall_risk for t in rows]
                    ),
                    deep_rate=rate_or_none(
                        sum(1 for t in rows if self._path(t) == _DEEP), k
                    ),
                    human_review_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.HUMAN_REVIEW
                        ),
                        k,
                    ),
                    block_rate=rate_or_none(
                        sum(
                            1
                            for t in rows
                            if t.final_decision.decision is InterventionTier.BLOCK
                        ),
                        k,
                    ),
                )
            )
        return out

    # ------------------------------------------------------------------
    # attention signals

    def _attention_signals(
        self,
        decisions: DecisionMetrics,
        confidence: ConfidenceMetrics,
        latency: LatencyStats,
        verification: VerificationMetrics,
        risk_distribution: RiskDistribution,
        n: int,
    ) -> list[AttentionSignal]:
        if n == 0:
            return []

        cfg = self._signal_cfg
        signals: list[AttentionSignal] = []

        def _high(code: str, observed: float | None, key: str, explanation: str) -> None:
            if observed is None or key not in cfg:
                return
            threshold = float(cfg[key])
            if observed > threshold:
                severity = (
                    "CRITICAL"
                    if observed >= threshold * self._critical_multiplier
                    else "WARNING"
                )
                signals.append(
                    AttentionSignal(
                        code=code,
                        severity=severity,
                        observed_value=round(observed, 6),
                        threshold=threshold,
                        explanation=explanation,
                    )
                )

        _high(
            "HIGH_BLOCK_RATE",
            decisions.block_rate,
            "high_block_rate",
            f"{_pct(decisions.block_rate)} of traffic was BLOCKed.",
        )
        _high(
            "HIGH_HUMAN_REVIEW_RATE",
            decisions.human_review_rate,
            "high_human_review_rate",
            f"{_pct(decisions.human_review_rate)} of traffic was routed to HUMAN_REVIEW.",
        )
        _high(
            "HIGH_DEEP_VERIFICATION_RATE",
            verification.deep_rate,
            "high_deep_verification_rate",
            f"{_pct(verification.deep_rate)} of traffic used DEEP verification; "
            f"progressive verification is saving little compute at this mix.",
        )
        _high(
            "HIGH_P95_LATENCY",
            latency.p95_total_latency_ms,
            "high_p95_latency_ms",
            f"p95 total pipeline latency is "
            f"{latency.p95_total_latency_ms:.2f} ms.",
        )

        concentration = None
        if risk_distribution.total:
            elevated = sum(
                b.count
                for b in risk_distribution.buckets
                if b.bucket_name in self._elevated_bucket_names
            )
            concentration = elevated / risk_distribution.total
        _high(
            "HIGH_RISK_CONCENTRATION",
            concentration,
            "high_risk_concentration",
            f"{_pct(concentration)} of traffic sits in an elevated risk bucket "
            f"({sorted(self._elevated_bucket_names)}).",
        )

        # low-confidence signal (fires on the low side)
        mean_conf = confidence.mean_decision_confidence
        if mean_conf is not None and "low_mean_confidence" in cfg:
            threshold = float(cfg["low_mean_confidence"])
            if mean_conf < threshold:
                severity = (
                    "CRITICAL"
                    if mean_conf <= threshold / self._critical_multiplier
                    else "WARNING"
                )
                signals.append(
                    AttentionSignal(
                        code="LOW_MEAN_CONFIDENCE",
                        severity=severity,
                        observed_value=round(mean_conf, 6),
                        threshold=threshold,
                        explanation=(
                            f"Mean decision confidence is {mean_conf:.2f}; "
                            f"the pipeline is frequently unsure of its own calls."
                        ),
                    )
                )

        signals.sort(key=lambda s: (s.severity != "CRITICAL", s.code))
        return signals

    # ------------------------------------------------------------------

    def _empty_report(
        self,
        window: MonitoringWindow,
        quality: DataQualityReport,
        generated_at: datetime | None,
        granularity: str,
    ) -> MonitoringReport:
        buckets = [
            RiskBucket(bucket_name=name, min_risk=lo, max_risk=hi, count=0, percentage=None)
            for name, lo, hi in self._buckets
        ]
        return MonitoringReport(
            generated_at=generated_at,
            window=window,
            total_interactions=0,
            decisions=DecisionMetrics(),
            risk=RiskStats(),
            confidence=ConfidenceMetrics(
                low_confidence_threshold=self._low_confidence_threshold
            ),
            latency=LatencyStats(),
            verification=VerificationMetrics(),
            risk_distribution=RiskDistribution(buckets=buckets, total=0),
            reason_codes=[],
            policy_rules=[],
            applications=[],
            models=[],
            detectors=[
                DetectorHealth(detector=name, invocation_count=0, error_count=None)
                for name in ("performance", "responsibility", "cost")
            ],
            verification_by_application=[],
            attention_signals=[],
            trend_granularity=granularity,
            trend=[],
            data_quality=quality,
            notes=["No traces in scope — empty monitoring report."],
        )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ======================================================================
# real execution
# ======================================================================


def collect_operational_traces(
    config: dict[str, Any] | None = None,
    *,
    traffic_sample: int = 300,
    include_demo_scenarios: bool = True,
) -> list[DecisionTrace]:
    """
    Run the REAL pipeline and return the resulting ``DecisionTrace``
    collection — the operational traffic monitoring consumes.

    Source = the demo scenarios (A-J) + a head sample of the synthetic
    *production-traffic* dataset. The synthetic **evaluation** dataset
    (which carries ground-truth labels) is deliberately NOT used here.
    All timestamps are taken from the interactions themselves, so the
    result is fully deterministic.
    """
    import random

    from data.generator import generate_interactions
    from evaluation.evaluation import build_engine
    from settings import load_settings
    from tests import scenarios

    cfg = config if config is not None else load_settings()
    engine = build_engine(cfg)
    traces: list[DecisionTrace] = []

    if include_demo_scenarios:
        demo_interactions = [factory() for factory in scenarios.ALL_SINGLE_TURN.values()]
        demo_interactions += list(scenarios.scenario_g_multi_turn())
        demo_interactions.append(scenarios.scenario_i_policy_counterfactual()[0])
        demo_interactions.append(scenarios.scenario_j_consequence_counterfactual()[0])
        for interaction in demo_interactions:
            traces.append(
                engine.evaluate(
                    interaction,
                    timestamp=interaction.timestamp,
                    record_session=False,
                )
            )

    if traffic_sample > 0:
        rng = random.Random(cfg["seed"])
        for interaction in generate_interactions(cfg, rng)[:traffic_sample]:
            traces.append(
                engine.evaluate(
                    interaction,
                    timestamp=interaction.timestamp,
                    record_session=False,
                )
            )

    return traces


def _print_operational_monitor(report: MonitoringReport) -> None:
    rule = "=" * 60
    print(rule)
    print("CONTROLPLANE OPERATIONAL MONITOR")
    print(rule)
    r = report.risk
    v = report.verification
    d = report.decisions
    lat = report.latency
    print(f"  Interactions   : {report.total_interactions}")
    print(f"  Mean risk      : {_fmt(r.mean_overall_risk)}")
    print(f"  P50 risk       : {_fmt(r.p50_overall_risk)}")
    print(f"  P95 risk       : {_fmt(r.p95_overall_risk)}")
    print(f"  Max risk       : {_fmt(r.max_overall_risk)}")
    print(f"  FAST           : {_pct(v.fast_rate)}   ({v.fast_count})")
    print(f"  DEEP           : {_pct(v.deep_rate)}   ({v.deep_count})")
    print(f"  Human review   : {_pct(d.human_review_rate)}   ({d.human_review_count})")
    print(f"  Block          : {_pct(d.block_rate)}   ({d.block_count})")
    print(f"  Mean latency   : {_fmt(lat.mean_total_latency_ms)} ms")
    print(f"  P95 latency    : {_fmt(lat.p95_total_latency_ms)} ms")

    print(f"\n{'-' * 60}\nTOP REASON CODES\n{'-' * 60}")
    if report.reason_codes:
        for row in report.reason_codes[:10]:
            print(f"  {row.reason_code:32s} {row.count}")
    else:
        print("  (none)")

    print(f"\n{'-' * 60}\nTOP POLICY RULES (fired)\n{'-' * 60}")
    if report.policy_rules:
        for row in report.policy_rules[:10]:
            print(f"  {row.rule:32s} {row.fired_count}")
    else:
        print("  (none)")

    print(f"\n{'-' * 60}\nFAST vs DEEP BY APPLICATION\n{'-' * 60}")
    for row in report.verification_by_application:
        print(
            f"  {row.application:30s} FAST {_pct(row.fast_rate):>6s} / "
            f"DEEP {_pct(row.deep_rate):>6s}"
        )

    print(f"\n{'-' * 60}\nMODEL BREAKDOWN (observed association only)\n{'-' * 60}")
    for row in report.models:
        print(
            f"  {row.model:20s} n={row.interaction_count:<5d} "
            f"mean_risk={_fmt(row.mean_risk)} deep={_pct(row.deep_rate)} "
            f"human_review={_pct(row.human_review_rate)}"
        )

    print(f"\n{'-' * 60}\nOPERATIONAL ATTENTION SIGNALS\n{'-' * 60}")
    if report.attention_signals:
        for sig in report.attention_signals:
            print(
                f"  [{sig.severity}] {sig.code}\n"
                f"      observed {sig.observed_value:.3f} vs threshold {sig.threshold:.3f}\n"
                f"      {sig.explanation}"
            )
    else:
        print("  (none — all monitored aggregates within configured thresholds)")

    dq = report.data_quality
    print(f"\n{'-' * 60}\nDATA QUALITY\n{'-' * 60}")
    print(f"  records seen      : {dq.total_records_seen}")
    print(f"  valid records     : {dq.valid_records}")
    print(f"  invalid (skipped) : {dq.invalid_records_skipped}")
    print(f"  missing fields    : {dq.missing_trace_fields or '(none)'}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(encoding="utf-8")

    print("[monitoring] running the real pipeline over demo scenarios + "
          "synthetic production traffic ...")
    traces = collect_operational_traces(traffic_sample=300)
    print(f"[monitoring] collected {len(traces)} real DecisionTrace records\n")

    report = MonitoringService().report(
        traces, generated_at=datetime(2026, 8, 28, 12, 0, 0)
    )
    _print_operational_monitor(report)

    # Same real traffic, a stricter operator profile — shows the attention
    # signals reacting to configuration, not to fabricated data.
    strict_cfg = load_settings()
    strict_cfg = {**strict_cfg, "monitoring": {**strict_cfg["monitoring"]}}
    strict_cfg["monitoring"]["attention_signals"] = {
        **strict_cfg["monitoring"]["attention_signals"],
        "high_deep_verification_rate": 0.50,
        "high_risk_concentration": 0.12,
        "high_human_review_rate": 0.04,
    }
    strict = MonitoringService(strict_cfg).report(traces)
    print(f"\n{'-' * 60}\nATTENTION SIGNALS — STRICT OPERATOR PROFILE\n{'-' * 60}")
    for sig in strict.attention_signals:
        print(
            f"  [{sig.severity}] {sig.code}: observed {sig.observed_value:.3f} "
            f"vs threshold {sig.threshold:.3f}"
        )

    out_path = "monitoring_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(f"\n[monitoring_report.json written — {out_path}]")


if __name__ == "__main__":
    main()
