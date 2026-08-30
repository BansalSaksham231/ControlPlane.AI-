"""
Enterprise observability / monitoring tests (Phase 5).

Every metric is aggregated from REAL ``DecisionTrace`` records produced by
the production pipeline over the demo scenarios + a sample of synthetic
*production traffic* (never the evaluation dataset). Monitoring never
re-runs a detector/engine and never reads ground truth.
"""

from __future__ import annotations

import copy
import pathlib
from datetime import datetime

import pytest

from data.schemas import InterventionTier
from monitoring.metrics import percentile_or_none, rate_or_none, truncate_timestamp
from monitoring.schemas import MonitoringReport
from monitoring.service import MonitoringService, collect_operational_traces
from settings import load_settings
from tests import scenarios

_MON_DIR = pathlib.Path(__file__).resolve().parent.parent / "monitoring"
_GEN_AT = datetime(2026, 8, 28, 12, 0, 0)


# ------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def traces():
    """Real traces: demo scenarios + 120 synthetic production interactions."""
    return collect_operational_traces(traffic_sample=120)


@pytest.fixture(scope="module")
def report(traces):
    return MonitoringService().report(traces, generated_at=_GEN_AT)


@pytest.fixture(scope="module")
def demo_only_traces():
    return collect_operational_traces(traffic_sample=0)


def _single_trace(factory):
    from evaluation.evaluation import build_engine

    engine = build_engine()
    interaction = factory()
    return engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )


# ------------------------------------------------------------------
# 1. empty collection
# ------------------------------------------------------------------


def test_empty_trace_collection():
    rep = MonitoringService().report([])
    assert isinstance(rep, MonitoringReport)
    assert rep.total_interactions == 0
    assert rep.decisions.allow_count == 0
    assert rep.decisions.human_review_rate is None
    assert rep.risk.mean_overall_risk is None
    assert rep.risk.p95_overall_risk is None
    assert rep.verification.fast_rate is None
    assert rep.attention_signals == []
    assert rep.trend == []
    # detector rows still present, with no fabricated numbers
    assert {d.detector for d in rep.detectors} == {"performance", "responsibility", "cost"}
    assert all(d.error_count is None and d.mean_latency_ms is None for d in rep.detectors)
    assert rep.data_quality.total_records_seen == 0


# ------------------------------------------------------------------
# 2. single clean trace
# ------------------------------------------------------------------


def test_single_clean_trace():
    trace = _single_trace(scenarios.scenario_a_clean)
    rep = MonitoringService().report([trace], generated_at=_GEN_AT)
    assert rep.total_interactions == 1
    assert rep.decisions.allow_count == 1
    assert sum(
        (
            rep.decisions.allow_count,
            rep.decisions.annotate_count,
            rep.decisions.verify_count,
            rep.decisions.human_review_count,
            rep.decisions.block_count,
        )
    ) == 1
    assert rep.risk.mean_overall_risk == pytest.approx(trace.final_decision.overall_risk)
    assert rep.risk.max_overall_risk == rep.risk.p95_overall_risk == rep.risk.mean_overall_risk


# ------------------------------------------------------------------
# 3. mixed decision distribution
# ------------------------------------------------------------------


def test_mixed_decision_distribution(report, traces):
    d = report.decisions
    total = (
        d.allow_count + d.annotate_count + d.verify_count
        + d.human_review_count + d.block_count
    )
    assert total == report.total_interactions == len(traces)
    # the demo scenarios alone guarantee more than one tier is present
    present = [c for c in (d.allow_count, d.verify_count, d.human_review_count, d.block_count) if c]
    assert len(present) >= 2


# ------------------------------------------------------------------
# 4. risk statistics
# ------------------------------------------------------------------


def test_risk_statistics(report):
    r = report.risk
    for value in (r.mean_overall_risk, r.p50_overall_risk, r.p95_overall_risk, r.max_overall_risk):
        assert value is not None and 0.0 <= value <= 1.0
    assert r.p50_overall_risk <= r.p95_overall_risk <= r.max_overall_risk


# ------------------------------------------------------------------
# 5. p50 / p95 calculations
# ------------------------------------------------------------------


def test_percentile_helper_is_deterministic_and_correct():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert percentile_or_none(values, 50) == 0.5
    assert percentile_or_none(values, 95) == 1.0
    assert percentile_or_none([], 50) is None
    assert percentile_or_none([0.42], 95) == 0.42
    # unsorted input -> same answer
    assert percentile_or_none(list(reversed(values)), 95) == 1.0


def test_report_percentiles_match_manual(report, traces):
    risks = sorted(t.final_decision.overall_risk for t in traces)
    assert report.risk.p95_overall_risk == percentile_or_none(risks, 95)
    assert report.risk.max_overall_risk == pytest.approx(risks[-1])


# ------------------------------------------------------------------
# 6. FAST / DEEP rates
# ------------------------------------------------------------------


def test_fast_deep_rates(report, traces):
    v = report.verification
    assert v.fast_count + v.deep_count + v.unknown_count == len(traces)
    assert v.fast_rate + v.deep_rate == pytest.approx(1.0, abs=1e-6)
    # DEEP-triggered reasons only recorded where a VerificationReport exists
    assert isinstance(v.deep_trigger_reason_counts, dict)
    manual_deep = sum(1 for t in traces if (t.verification_path or "DEEP").upper() == "DEEP")
    assert v.deep_count == manual_deep


# ------------------------------------------------------------------
# 7. application breakdown
# ------------------------------------------------------------------


def test_application_breakdown(report):
    apps = {a.application for a in report.applications}
    assert apps <= {"customer_support", "internal_knowledge_assistant", "decision_support"}
    assert apps  # at least one present
    assert sum(a.interactions for a in report.applications) == report.total_interactions
    for a in report.applications:
        assert a.mean_risk is not None and 0.0 <= a.mean_risk <= 1.0
        assert a.deep_rate is not None and 0.0 <= a.deep_rate <= 1.0


# ------------------------------------------------------------------
# 8. model breakdown
# ------------------------------------------------------------------


def test_model_breakdown(report):
    cfg_models = set(load_settings()["models"]) | {"unknown"}
    assert {m.model for m in report.models} <= cfg_models
    assert sum(m.interaction_count for m in report.models) == report.total_interactions
    for m in report.models:
        assert "observed association" in m.interpretation
        assert m.mean_risk is not None


# ------------------------------------------------------------------
# 9. detector latency metrics
# ------------------------------------------------------------------


def test_detector_latency_metrics(report):
    assert len(report.detectors) == 3
    for d in report.detectors:
        assert d.invocation_count == report.total_interactions
        assert d.mean_latency_ms is not None and d.mean_latency_ms >= 0.0
        assert d.p95_latency_ms is not None and d.p95_latency_ms >= 0.0


# ------------------------------------------------------------------
# 10. policy rule frequency
# ------------------------------------------------------------------


def test_policy_rule_frequency(report, traces):
    rules = {r.rule: r.fired_count for r in report.policy_rules}
    assert rules  # some rules fired
    # RISK_BAND fires on every interaction
    assert rules.get("RISK_BAND") == report.total_interactions
    # sorted descending by frequency
    counts = [r.fired_count for r in report.policy_rules]
    assert counts == sorted(counts, reverse=True)
    # only fired entries are counted
    manual = 0
    for t in traces:
        manual += sum(1 for e in t.policy.rule_trace if e.fired and e.rule == "RISK_BAND")
    assert manual == rules["RISK_BAND"]


# ------------------------------------------------------------------
# 11. reason-code frequency
# ------------------------------------------------------------------


def test_reason_code_frequency(report, traces):
    codes = {c.reason_code: c.count for c in report.reason_codes}
    assert codes  # the demo scenarios raise several reason codes
    counts = [c.count for c in report.reason_codes]
    assert counts == sorted(counts, reverse=True)
    assert all(v > 0 for v in codes.values())
    manual_total = sum(len(t.final_decision.reason_codes) for t in traces)
    assert sum(codes.values()) == manual_total


# ------------------------------------------------------------------
# 12. time-window filtering
# ------------------------------------------------------------------


def test_time_window_filtering(traces):
    svc = MonitoringService()
    all_ts = sorted(t.timestamp for t in traces)
    midpoint = all_ts[len(all_ts) // 2]

    full = svc.report(traces)
    windowed = svc.report(traces, start_time=midpoint)

    assert windowed.window.applied is True
    assert windowed.total_interactions < full.total_interactions
    assert windowed.window.traces_excluded_by_window > 0
    assert (
        windowed.total_interactions + windowed.window.traces_excluded_by_window
        == full.total_interactions
    )
    # every trend bucket is inside the window
    assert all(b.bucket_start >= truncate_timestamp(midpoint, "hourly") for b in windowed.trend)
    # deterministic
    again = svc.report(traces, start_time=midpoint)
    assert windowed.model_dump() == again.model_dump()


def test_window_uses_supplied_bounds_only(traces):
    # an end_time before all data -> empty, not "now"
    svc = MonitoringService()
    rep = svc.report(traces, end_time=datetime(2000, 1, 1))
    assert rep.total_interactions == 0
    assert rep.window.traces_excluded_by_window == len(traces)


# ------------------------------------------------------------------
# 13. empty denominator returns None
# ------------------------------------------------------------------


def test_empty_denominator_returns_none(traces):
    deep_only = [t for t in traces if (t.verification_path or "DEEP").upper() == "DEEP"]
    rep = MonitoringService().report(deep_only)
    assert rep.verification.fast_count == 0
    assert rep.verification.fast_rate == 0.0                 # 0 / N is a real 0
    assert rep.verification.mean_total_latency_fast_ms is None   # mean of nothing -> None
    assert rep.verification.mean_verification_latency_fast_ms is None
    assert rate_or_none(0, 0) is None


# ------------------------------------------------------------------
# 14 & 15. operational attention signals + threshold configuration
# ------------------------------------------------------------------


def _config_with_signal_thresholds(**overrides):
    cfg = copy.deepcopy(load_settings())
    cfg["monitoring"] = copy.deepcopy(cfg["monitoring"])
    cfg["monitoring"]["attention_signals"] = {
        **cfg["monitoring"]["attention_signals"],
        **overrides,
    }
    return cfg


def test_attention_signal_generation(traces):
    # thresholds set low enough that signals must fire on this traffic
    cfg = _config_with_signal_thresholds(
        high_deep_verification_rate=0.01,
        high_human_review_rate=0.001,
        low_mean_confidence=0.99,
    )
    rep = MonitoringService(cfg).report(traces)
    codes = {s.code for s in rep.attention_signals}
    assert "HIGH_DEEP_VERIFICATION_RATE" in codes
    assert "LOW_MEAN_CONFIDENCE" in codes
    for s in rep.attention_signals:
        assert s.severity in ("WARNING", "CRITICAL")
        assert s.observed_value is not None
        assert s.threshold is not None
        assert s.explanation
    # not asserted as statistical anomalies
    assert "operational" in " ".join(rep.notes).lower()


def test_threshold_configuration_changes_alerts(traces):
    strict = MonitoringService(
        _config_with_signal_thresholds(
            high_deep_verification_rate=0.01,
            high_block_rate=0.001,
            high_human_review_rate=0.001,
            high_risk_concentration=0.001,
            low_mean_confidence=0.99,
            high_p95_latency_ms=0.001,
        )
    ).report(traces)
    lax = MonitoringService(
        _config_with_signal_thresholds(
            high_deep_verification_rate=0.999,
            high_block_rate=0.999,
            high_human_review_rate=0.999,
            high_risk_concentration=0.999,
            low_mean_confidence=0.0001,
            high_p95_latency_ms=1e9,
        )
    ).report(traces)
    assert len(strict.attention_signals) > len(lax.attention_signals)
    assert lax.attention_signals == []


def test_no_signals_when_empty():
    assert MonitoringService().report([]).attention_signals == []


# ------------------------------------------------------------------
# 16. trend aggregation
# ------------------------------------------------------------------


def test_trend_aggregation(report, traces):
    assert report.trend
    assert sum(b.interaction_count for b in report.trend) == report.total_interactions
    starts = [b.bucket_start for b in report.trend]
    assert starts == sorted(starts)
    for b in report.trend:
        assert b.mean_risk is not None
        assert b.deep_rate is not None

    hourly = MonitoringService().report(traces, trend_granularity="hourly")
    daily = MonitoringService().report(traces, trend_granularity="daily")
    assert len(daily.trend) <= len(hourly.trend)
    assert daily.trend_granularity == "daily"
    assert sum(b.interaction_count for b in daily.trend) == len(traces)


# ------------------------------------------------------------------
# 17. data-quality reporting
# ------------------------------------------------------------------


def test_data_quality_reporting(traces):
    dirty = list(traces[:20]) + ["not a trace", {"interaction_id": "bad"}, 42]
    rep = MonitoringService().report(dirty)
    dq = rep.data_quality
    assert dq.total_records_seen == 23
    assert dq.valid_records == 20
    assert dq.invalid_records_skipped == 3
    assert len(dq.exclusion_reasons) == 3
    assert rep.total_interactions == 20  # good records still aggregated


def test_data_quality_flags_missing_optional_fields(traces):
    t = traces[0].model_copy(update={"model": None, "verification": None})
    rep = MonitoringService().report([t])
    assert rep.data_quality.missing_trace_fields.get("model") == 1
    assert rep.data_quality.missing_trace_fields.get("verification") == 1
    # record is NOT discarded
    assert rep.total_interactions == 1


# ------------------------------------------------------------------
# 18. monitoring does not access ground truth
# ------------------------------------------------------------------

_FORBIDDEN_TOKENS = (
    "ground_truth_hallucination",
    "ground_truth_pii",
    "ground_truth_toxicity",
    "ground_truth_bias",
    "ground_truth_cost_anomaly",
    "ground_truth_performance_risk",
    "ground_truth_responsibility_risk",
    "ground_truth_cost_risk",
    "expected_decision",
    "final_outcome",
)


def test_monitoring_does_not_reference_ground_truth():
    offenders: list[str] = []
    for path in _MON_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], offenders


def test_monitoring_does_not_import_evaluation_or_rerun_engine():
    for name in ("schemas.py", "metrics.py"):
        text = (_MON_DIR / name).read_text(encoding="utf-8")
        assert "evaluation" not in text
        assert "DecisionEngine" not in text
        assert ".detect(" not in text
        assert ".evaluate(" not in text


# ------------------------------------------------------------------
# 19. monitoring does not mutate traces
# ------------------------------------------------------------------


def test_monitoring_does_not_mutate_traces(traces):
    before = [t.model_dump(mode="json") for t in traces]
    MonitoringService().report(traces, generated_at=_GEN_AT)
    after = [t.model_dump(mode="json") for t in traces]
    assert before == after


# ------------------------------------------------------------------
# 20. deterministic repeated execution
# ------------------------------------------------------------------


def test_deterministic_repeated_execution(traces):
    svc = MonitoringService()
    a = svc.report(traces, generated_at=_GEN_AT).model_dump()
    b = svc.report(traces, generated_at=_GEN_AT).model_dump()
    assert a == b  # same trace objects -> byte-identical, latency included


def test_deterministic_across_independent_collections():
    t1 = collect_operational_traces(traffic_sample=40)
    t2 = collect_operational_traces(traffic_sample=40)
    a = MonitoringService().report(t1, generated_at=_GEN_AT).model_dump()
    b = MonitoringService().report(t2, generated_at=_GEN_AT).model_dump()

    def _strip(node):
        if isinstance(node, dict):
            return {
                k: _strip(v)
                for k, v in node.items()
                if not (k.endswith("_ms") or k == "latency")
            }
        if isinstance(node, list):
            return [_strip(x) for x in node]
        return node

    assert _strip(a) == _strip(b)


# ------------------------------------------------------------------
# 21. real demo scenarios -> non-empty report
# ------------------------------------------------------------------


def test_real_demo_scenarios_produce_non_empty_report(demo_only_traces):
    rep = MonitoringService().report(demo_only_traces, generated_at=_GEN_AT)
    assert rep.total_interactions == len(demo_only_traces) > 0
    assert rep.reason_codes
    assert rep.policy_rules
    assert rep.risk.mean_overall_risk is not None
    assert rep.risk_distribution.total == rep.total_interactions
    assert sum(b.count for b in rep.risk_distribution.buckets) == rep.total_interactions
    # risk-distribution percentages sum to ~100
    pct = sum(b.percentage for b in rep.risk_distribution.buckets)
    assert pct == pytest.approx(100.0, abs=1e-3)


# ------------------------------------------------------------------
# 22. no fake error counts
# ------------------------------------------------------------------


def test_no_fake_detector_error_counts(report):
    assert all(d.error_count is None for d in report.detectors)


# ------------------------------------------------------------------
# extra: FAST/DEEP by application + O(n) sanity
# ------------------------------------------------------------------


def test_verification_by_application(report):
    for row in report.verification_by_application:
        assert row.fast_count + row.deep_count <= report.total_interactions
        if row.fast_count + row.deep_count:
            assert row.fast_rate + row.deep_rate == pytest.approx(1.0, abs=1e-6)
