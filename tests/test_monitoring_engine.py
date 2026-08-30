"""
Phase 8 — Step 1: operational risk monitoring + incident intelligence.

Monitoring is an OBSERVATIONAL layer over real DecisionTrace records:
never re-runs the pipeline, never reads ground truth, never leaks PII,
deterministic for a given trace collection.
"""

from __future__ import annotations

import pathlib
import random

import pytest

from data.generator import generate_interactions
from data.schemas import InterventionTier
from evaluation.evaluation import build_engine
from feedback.store import FeedbackStore
from monitoring.engine import OperationalMonitor
from monitoring.schemas import (
    IncidentSeverity,
    MonitoringConfig,
    OperationalMonitoringReport,
)
from settings import load_settings
from tests import scenarios

_MON_DIR = pathlib.Path(__file__).resolve().parent.parent / "monitoring"
_GEN_AT = None


# ------------------------------------------------------------------
# real traces — built ONCE, the pipeline is never touched again
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    return build_engine()


@pytest.fixture(scope="module")
def traces(engine):
    out = []
    for f in scenarios.ALL_SINGLE_TURN.values():
        it = f()
        out.append(engine.evaluate(it, timestamp=it.timestamp, record_session=False))
    for it in scenarios.scenario_g_multi_turn():
        out.append(engine.evaluate(it, timestamp=it.timestamp, record_session=False))
    cfg = load_settings()
    for it in generate_interactions(cfg, random.Random(cfg["seed"]))[:60]:
        out.append(engine.evaluate(it, timestamp=it.timestamp, record_session=False))
    return out


@pytest.fixture(scope="module")
def report(traces):
    return OperationalMonitor().report(traces)


def _scenario_trace(engine, factory):
    it = factory()
    return engine.evaluate(it, timestamp=it.timestamp, record_session=False)


# ------------------------------------------------------------------
# 1. empty input
# ------------------------------------------------------------------


def test_empty_input_returns_valid_report():
    r = OperationalMonitor().report([])
    assert isinstance(r, OperationalMonitoringReport)
    assert r.total_interactions == 0
    assert r.snapshot.total_interactions == 0
    assert r.snapshot.allow_rate == 0.0
    assert r.snapshot.average_risk == 0.0
    assert r.incidents == []
    assert r.incident_digest.total == 0
    assert r.incident_digest.incident_rate == 0.0
    assert r.applications == []
    assert len(r.detectors) == 3
    assert all(d.average_risk == 0.0 for d in r.detectors)
    assert r.risk_distribution.total == 0
    assert all(mt.direction.value == "stable" for mt in r.trend.metrics)
    assert r.feedback.feedback_available is False


# ------------------------------------------------------------------
# 2-5. snapshot counts / rates / bounds
# ------------------------------------------------------------------


def test_snapshot_counts_match_traces(report, traces):
    s = report.snapshot
    assert s.total_interactions == len(traces) == report.total_interactions
    assert (
        s.allow_count + s.annotate_count + s.verify_count
        + s.human_review_count + s.block_count
        == len(traces)
    )
    manual_allow = sum(
        1 for t in traces if t.final_decision.decision is InterventionTier.ALLOW
    )
    assert s.allow_count == manual_allow


def test_decision_rates_sum_to_one(report):
    s = report.snapshot
    total = (
        s.allow_rate + s.annotate_rate + s.verify_rate
        + s.human_review_rate + s.block_rate
    )
    assert total == pytest.approx(1.0, abs=1e-4)
    assert s.fast_path_rate + s.deep_path_rate == pytest.approx(1.0, abs=1e-4)


def test_risk_and_confidence_metrics_are_bounded(report):
    s = report.snapshot
    for v in (
        s.average_risk, s.p95_risk, s.average_confidence, s.low_confidence_rate,
        s.average_cost_risk, s.high_consequence_rate, s.high_criticality_rate,
        s.multi_risk_rate,
    ):
        assert 0.0 <= v <= 1.0
    assert s.average_latency_ms >= 0.0 and s.p95_latency_ms >= 0.0


def test_snapshot_matches_manual_aggregation(report, traces):
    n = len(traces)
    manual_avg_risk = sum(t.final_decision.overall_risk for t in traces) / n
    assert report.snapshot.average_risk == pytest.approx(manual_avg_risk, abs=1e-5)
    manual_multi = sum(1 for t in traces if t.fusion.multi_risk) / n
    assert report.snapshot.multi_risk_rate == pytest.approx(manual_multi, abs=1e-5)


# ------------------------------------------------------------------
# 6-8. application + detector breakdown
# ------------------------------------------------------------------


def test_application_grouping(report, traces):
    apps = {a.application for a in report.applications}
    assert apps <= {
        "customer_support", "internal_knowledge_assistant", "decision_support"
    }
    assert sum(a.interaction_count for a in report.applications) == len(traces)
    # deterministic alphabetical order
    assert [a.application for a in report.applications] == sorted(apps)


def test_application_decision_distribution_is_consistent(report):
    for a in report.applications:
        assert sum(a.decision_distribution.values()) == a.interaction_count
        for v in (a.human_review_rate, a.block_rate, a.fast_path_rate, a.deep_path_rate):
            assert 0.0 <= v <= 1.0


def test_dominant_detector_calculation(report, traces):
    # performance is the most-frequent dominant dimension on this traffic
    perf = next(d for d in report.detectors if d.detector == "performance")
    manual = sum(1 for t in traces if t.fusion.dominant_dimension == "performance")
    assert perf.dominant_dimension_count == manual
    assert sum(d.dominant_dimension_count for d in report.detectors) <= len(traces)


# ------------------------------------------------------------------
# 9. reason-code aggregation
# ------------------------------------------------------------------


def test_reason_code_aggregation(report, traces):
    counts = {rc.reason_code: rc.count for rc in report.reason_codes}
    manual = {}
    for t in traces:
        for code in t.final_decision.reason_codes:
            manual[code] = manual.get(code, 0) + 1
    assert counts == manual
    # sorted by descending frequency
    freqs = [rc.count for rc in report.reason_codes]
    assert freqs == sorted(freqs, reverse=True)
    for rc in report.reason_codes:
        assert 0.0 <= rc.share_of_interventions <= 1.0


# ------------------------------------------------------------------
# 10. FAST/DEEP aggregation
# ------------------------------------------------------------------


def test_fast_deep_aggregation(report, traces):
    v = report.verification
    assert v.fast_count + v.deep_count == len(traces)
    assert v.fast_rate + v.deep_rate == pytest.approx(1.0, abs=1e-4)
    manual_deep = sum(
        1 for t in traces if (t.verification_path or "DEEP").upper() == "DEEP"
    )
    assert v.deep_count == manual_deep
    assert isinstance(v.deep_trigger_reason_counts, dict)


# ------------------------------------------------------------------
# 11-13. incident detection / severity / ordering
# ------------------------------------------------------------------


def test_incident_detection(engine, report):
    incident_ids = {i.interaction_id for i in report.incidents}
    block = _scenario_trace(engine, scenarios.scenario_c_pii)
    allow = _scenario_trace(engine, scenarios.scenario_a_clean)
    r = OperationalMonitor().report([block, allow])
    ids = {i.interaction_id for i in r.incidents}
    assert block.interaction_id in ids           # BLOCK -> incident
    assert allow.interaction_id not in ids       # clean ALLOW -> not an incident
    # a plain VERIFY is not automatically an incident
    verify = _scenario_trace(engine, scenarios.scenario_b_hallucination)
    r2 = OperationalMonitor().report([verify])
    if verify.final_decision.decision is InterventionTier.VERIFY and not verify.fusion.multi_risk:
        assert r2.incident_digest.total == 0


def test_incident_severity_is_transparent(engine):
    block = _scenario_trace(engine, scenarios.scenario_c_pii)
    r = OperationalMonitor().report([block])
    inc = r.incidents[0]
    assert inc.severity is IncidentSeverity.CRITICAL
    assert inc.severity_rationale
    assert "BLOCK" in inc.triggers
    # severity never changes the decision
    assert inc.decision == block.final_decision.decision.value


def test_incident_ordering_is_deterministic(report):
    ranks = [
        (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}[i.severity.value],
            -i.overall_risk,
            i.timestamp,
            i.interaction_id,
        )
        for i in report.incidents
    ]
    assert ranks == sorted(ranks)


# ------------------------------------------------------------------
# 14. trend
# ------------------------------------------------------------------


def test_trend_calculation(engine):
    low = _scenario_trace(engine, scenarios.scenario_a_clean)
    high = _scenario_trace(engine, scenarios.scenario_e_multi_risk)
    # 6 low then 6 high, chronologically ordered
    seq = []
    for i in range(12):
        base = low if i < 6 else high
        seq.append(
            base.model_copy(
                update={
                    "interaction_id": f"INT-TREND-{i:02d}",
                    "timestamp": base.timestamp.replace(microsecond=i),
                }
            )
        )
    r = OperationalMonitor().report(seq)
    risk_trend = next(m for m in r.trend.metrics if m.metric == "average_risk")
    assert risk_trend.first_half_n == 6 and risk_trend.second_half_n == 6
    assert risk_trend.second_half_value > risk_trend.first_half_value
    assert risk_trend.direction.value == "increasing"
    assert r.trend.method  # documented, no significance claim
    assert "significance" in r.trend.method


# ------------------------------------------------------------------
# 15. operational shift
# ------------------------------------------------------------------


def test_operational_shift_calculation(engine):
    low = _scenario_trace(engine, scenarios.scenario_a_clean)
    high = _scenario_trace(engine, scenarios.scenario_e_multi_risk)
    seq = []
    for i in range(20):
        base = low if i < 15 else high  # last 25% are high-risk
        seq.append(
            base.model_copy(
                update={
                    "interaction_id": f"INT-SHIFT-{i:02d}",
                    "timestamp": base.timestamp.replace(microsecond=i),
                }
            )
        )
    r = OperationalMonitor().report(seq)
    shift = {s.metric: s for s in r.operational_shift.shifts}
    assert r.operational_shift.recent_window_size == 5
    assert r.operational_shift.baseline_window_size == 15
    assert shift["average_risk"].recent_value > shift["average_risk"].baseline_value
    assert shift["average_risk"].direction == "up"
    assert "NOT AI/model-drift" in r.operational_shift.disclaimer


# ------------------------------------------------------------------
# 16. feedback aggregation
# ------------------------------------------------------------------


def test_feedback_aggregation(engine):
    block = _scenario_trace(engine, scenarios.scenario_c_pii)
    verify = _scenario_trace(engine, scenarios.scenario_b_hallucination)
    store = FeedbackStore(path=None)
    store.submit(
        interaction_id=block.interaction_id,
        system_decision=block.final_decision.decision,
        outcome="approved",
        reviewer="r1",
    )
    store.submit(
        interaction_id=verify.interaction_id,
        system_decision=verify.final_decision.decision,
        reviewer_decision=InterventionTier.ALLOW,
        outcome="rejected",
        reviewer="r2",
    )
    r = OperationalMonitor().report([block, verify], feedback_store=store)
    fb = r.feedback
    assert fb.feedback_available is True
    assert fb.feedback_count == 2
    assert fb.approved == 1 and fb.rejected == 1
    assert fb.override_count == 1  # the rejection is an override
    assert fb.override_rate == pytest.approx(0.5)
    assert "not ground truth" in fb.note


# ------------------------------------------------------------------
# 17. PII never leaks
# ------------------------------------------------------------------


def test_pii_never_leaks(engine):
    it = scenarios.scenario_c_pii()
    trace = engine.evaluate(it, timestamp=it.timestamp, record_session=False)
    r = OperationalMonitor().report([trace])
    blob = r.model_dump_json()

    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans
    for span in raw_spans:
        assert span not in blob
    assert it.response not in blob
    assert "matched_text" not in blob


# ------------------------------------------------------------------
# 18. ground-truth isolation (source level)
# ------------------------------------------------------------------


def test_monitoring_engine_has_no_ground_truth_or_evaluation():
    for name in ("engine.py", "schemas.py", "__main__.py", "metrics.py"):
        text = (_MON_DIR / name).read_text(encoding="utf-8")
        for token in (
            "ground_truth_", "expected_decision", "final_outcome",
        ):
            assert token not in text, f"{name}: {token}"
    engine_src = (_MON_DIR / "engine.py").read_text(encoding="utf-8")
    assert "import evaluation" not in engine_src
    assert "from evaluation" not in engine_src
    assert "EvaluationCase" not in engine_src


# ------------------------------------------------------------------
# 19. monitoring does not invoke pipeline components
# ------------------------------------------------------------------


def test_monitoring_does_not_rerun_the_pipeline(traces, monkeypatch):
    import detectors.cost.detector as cost_mod
    import detectors.performance.detector as perf_mod
    import detectors.responsibility.detector as resp_mod
    import decision.engine as dec_mod
    import verification.router as router_mod

    def boom(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("monitoring re-ran a pipeline component")

    monkeypatch.setattr(perf_mod.PerformanceDetector, "detect", boom)
    monkeypatch.setattr(resp_mod.ResponsibilityDetector, "detect", boom)
    monkeypatch.setattr(cost_mod.CostDetector, "detect", boom)
    monkeypatch.setattr(dec_mod.DecisionEngine, "evaluate", boom)
    monkeypatch.setattr(router_mod.VerificationRouter, "route", boom)

    r = OperationalMonitor().report(traces)
    assert r.total_interactions == len(traces)
    assert r.incident_digest.total >= 0


# ------------------------------------------------------------------
# 20. determinism
# ------------------------------------------------------------------


def test_same_traces_produce_deterministic_output(traces):
    a = OperationalMonitor().report(traces).model_dump()
    b = OperationalMonitor().report(traces).model_dump()
    assert a == b


def test_shuffled_input_produces_same_report(traces):
    shuffled = list(traces)
    random.Random(1).shuffle(shuffled)
    a = OperationalMonitor().report(traces).model_dump()
    b = OperationalMonitor().report(shuffled).model_dump()
    assert a == b  # internal ordering is by (timestamp, interaction_id)


# ------------------------------------------------------------------
# 21. empty optional fields handled + config
# ------------------------------------------------------------------


def test_missing_verification_and_config_override(engine):
    it = scenarios.scenario_a_clean()
    trace = engine.evaluate(it, timestamp=it.timestamp, record_session=False)
    trace = trace.model_copy(update={"verification": None})
    r = OperationalMonitor(
        MonitoringConfig(elevated_risk_threshold=0.10, trend_stable_band=0.0)
    ).report([trace])
    assert r.total_interactions == 1
    assert r.verification.deep_trigger_reason_counts == {}
    assert r.config.elevated_risk_threshold == 0.10
    with pytest.raises(Exception):
        MonitoringConfig(shift_recent_fraction=1.5)


def test_malformed_records_are_reported_not_crashed(traces):
    r = OperationalMonitor().report(list(traces[:5]) + ["nope", {"bad": 1}])
    assert r.data_quality.total_records_seen == 7
    assert r.data_quality.invalid_records_skipped == 2
    assert r.total_interactions == 5


# ------------------------------------------------------------------
# 22. scenarios A-J unchanged
# ------------------------------------------------------------------


def test_core_scenarios_unchanged(engine):
    got = {
        k: engine.evaluate(f(), timestamp=None, record_session=False).final_decision.decision.value
        for k, f in {
            "A": scenarios.scenario_a_clean,
            "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii,
            "E": scenarios.scenario_e_multi_risk,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK", "E": "BLOCK"}


# ------------------------------------------------------------------
# serialization round-trip
# ------------------------------------------------------------------


def test_report_round_trips_through_json(report):
    restored = OperationalMonitoringReport.model_validate_json(report.model_dump_json())
    assert restored.total_interactions == report.total_interactions
    assert len(restored.incidents) == len(report.incidents)


# ------------------------------------------------------------------
# cascade-router telemetry: semantic bypass + multi-turn critical floor
# ------------------------------------------------------------------


def test_verification_summary_reports_semantic_bypass(report):
    v = report.verification
    # scenarios C and E hit the deterministic hard boundary -> bypassed DEEP
    assert v.semantic_bypass_count >= 1
    assert v.semantic_bypass_count <= v.deep_count
    assert 0.0 <= v.semantic_bypass_rate_of_deep <= 1.0
    assert (
        v.estimated_bypass_compute_saved_ms is None
        or v.estimated_bypass_compute_saved_ms >= 0.0
    )


def test_multi_turn_summary_empty_for_stateless_traces(report):
    # the module fixture records no sessions (record_session=False)
    mt = report.multi_turn
    assert mt.total_sessions == 0
    assert mt.multi_turn_sessions == 0
    assert mt.sessions_hitting_critical_floor == 0
    assert mt.critical_floor_session_rate is None


def test_legacy_trace_without_verification_report_is_handled(engine):
    from monitoring.schemas import MultiTurnSummary

    trace = _scenario_trace(engine, scenarios.scenario_b_hallucination)
    legacy = trace.model_copy(update={"verification": None, "session": None})
    r = OperationalMonitor().report([legacy, trace])
    # no crash; the legacy trace simply doesn't count as bypassed
    assert r.verification.semantic_bypass_count >= 0
    assert isinstance(r.multi_turn, MultiTurnSummary)


def test_multi_turn_summary_detects_critical_floor_sessions():
    from decision.engine import DecisionEngine
    from session.manager import SessionManager

    sm = SessionManager()
    eng = DecisionEngine(session_manager=sm)
    traces = []
    critical_seq = [
        scenarios.scenario_a_clean, scenarios.scenario_c_pii,
        scenarios.scenario_a_clean, scenarios.scenario_a_clean,
    ]
    for i, f in enumerate(critical_seq, 1):
        it = f().model_copy(update={"session_id": "MT-CRIT", "interaction_id": f"MTC-{i}"})
        traces.append(eng.evaluate(it, timestamp=it.timestamp))
    for i in range(1, 3):  # a benign 2-turn session
        it = scenarios.scenario_a_clean().model_copy(
            update={"session_id": "MT-OK", "interaction_id": f"MTO-{i}"}
        )
        traces.append(eng.evaluate(it, timestamp=it.timestamp))

    mt = OperationalMonitor().report(traces).multi_turn
    assert mt.multi_turn_sessions == 2
    assert mt.sessions_hitting_critical_floor == 1
    assert mt.critical_floor_session_rate == pytest.approx(0.5)
    assert mt.critical_floor_events >= 1
