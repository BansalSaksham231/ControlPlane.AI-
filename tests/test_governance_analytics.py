"""
Phase 9 — Governance Intelligence analytics.

Read-only aggregation over stored DecisionTrace records + governance
actions + feedback. Never re-runs the pipeline, never reads ground truth,
never writes config, and never treats reviewer feedback as ground truth.
"""

from __future__ import annotations

import copy
import pathlib
import random

import pytest

from api.service import ControlPlaneService
from data.generator import generate_interactions
from data.schemas import InterventionTier
from governance.analytics import build_overview, compare_applications
from governance.recommendations import build_recommendations
from governance.report import build_governance_report
from governance.schemas import (
    GovernanceConfig,
    GovernanceIntelligenceReport,
    RecommendationDisposition,
)
from governance.signals import collect_signals, summarize_signals
from governance.trends import build_trends
from settings import load_settings
from tests import scenarios

_GOV_DIR = pathlib.Path(__file__).resolve().parent.parent / "governance"
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


# ------------------------------------------------------------------
# shared real data: pipeline traces + governance actions + feedback
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def svc():
    s = ControlPlaneService(fit_cost_baseline=False)
    for factory in scenarios.ALL_SINGLE_TURN.values():
        it = factory()
        s.check(it, timestamp=it.timestamp)
    for turn in scenarios.scenario_g_multi_turn():
        s.check(turn, timestamp=turn.timestamp)
    cfg = load_settings()
    for it in generate_interactions(cfg, random.Random(cfg["seed"]))[:90]:
        s.check(it, timestamp=it.timestamp)

    traces = s.all_traces()
    blocks = [t for t in traces if t.final_decision.decision is InterventionTier.BLOCK][:3]
    hrs = [t for t in traces if t.final_decision.decision is InterventionTier.HUMAN_REVIEW][:4]
    for t in blocks:
        s.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        s.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="route to human review", reviewer_decision="HUMAN_REVIEW",
        )
    for t in hrs[:2]:
        s.record_governance_action(t.interaction_id, action="APPROVE_DECISION", comment="agree")
    for t in hrs[2:]:
        s.record_governance_action(t.interaction_id, action="ESCALATE", comment="senior review")
        s.submit_feedback(
            interaction_id=t.interaction_id, system_decision=None,
            reviewer_decision="VERIFY", outcome="modified", reviewer="r",
        )
    return s


@pytest.fixture(scope="module")
def report(svc) -> GovernanceIntelligenceReport:
    return build_governance_report(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    )


# ------------------------------------------------------------------
# 1-2. empty / single
# ------------------------------------------------------------------


def test_empty_dataset():
    rep = build_governance_report([], [], [])
    assert rep.overview.traffic.total_interactions == 0
    assert rep.overview.decisions.human_review_rate is None
    assert rep.overview.risk.average_risk is None
    assert rep.application_comparison.applications == []
    assert rep.signals.signal_count == 0
    assert rep.trends.signals == []
    assert rep.insights == []
    assert rep.recommendations  # a NO_ACTION recommendation is still returned
    assert rep.recommendations[0].disposition is RecommendationDisposition.NO_ACTION


def test_single_trace():
    s = ControlPlaneService(fit_cost_baseline=False)
    it = scenarios.scenario_a_clean()
    s.check(it, timestamp=it.timestamp)
    rep = build_governance_report(s.all_traces(), [], [])
    assert rep.overview.traffic.total_interactions == 1
    assert rep.overview.decisions.allow == 1
    assert rep.application_comparison.applications[0].volume == 1


# ------------------------------------------------------------------
# 3-9. overview metrics
# ------------------------------------------------------------------


def test_traffic_and_decision_distribution(report, svc):
    traces = svc.all_traces()
    o = report.overview
    assert o.traffic.total_interactions == len(traces)
    assert sum(o.traffic.by_application.values()) == len(traces)
    d = o.decisions
    assert d.allow + d.annotate + d.verify + d.human_review + d.block == len(traces)
    assert d.human_oversight_rate == pytest.approx(
        (d.human_review + d.block) / len(traces), abs=1e-4
    )


def test_risk_and_confidence_metrics_bounded(report):
    r = report.overview.risk
    for v in (r.average_risk, r.p50_risk, r.p95_risk, r.max_risk, r.high_risk_rate):
        assert v is not None and 0.0 <= v <= 1.0
    assert r.p50_risk <= r.p95_risk <= r.max_risk
    c = report.overview.confidence
    assert 0.0 <= c.average_confidence <= 1.0
    assert 0.0 <= c.low_confidence_rate <= 1.0


def test_fast_deep_metrics(report, svc):
    v = report.overview.verification
    assert v.fast_count + v.deep_count == len(svc.all_traces())
    assert v.fast_rate + v.deep_rate == pytest.approx(1.0, abs=1e-4)
    assert v.average_total_latency_ms is not None and v.average_total_latency_ms > 0


def test_detector_contribution(report):
    dc = report.overview.detector_contribution
    assert dc.total_incidents >= 0
    assert dc.performance_driven_incidents + dc.responsibility_driven_incidents \
        + dc.cost_driven_incidents <= dc.total_incidents + dc.multi_risk_incidents
    assert "not a causal claim" in dc.note


def test_policy_rule_counts(report):
    rules = {r.rule: r for r in report.overview.policy.rules}
    assert "RISK_BAND" in rules
    assert rules["RISK_BAND"].fire_count == report.overview.policy.total_interactions
    for r in report.overview.policy.rules:
        assert r.human_review_count <= r.tier_changing_count
        assert r.block_count <= r.tier_changing_count


def test_governance_action_aggregation(report, svc):
    hg = report.overview.human_governance
    assert hg.incidents_investigated >= 1
    assert hg.reviewed >= 1               # MODIFY_DECISION -> REVIEWED
    assert hg.escalated >= 1
    assert sum(hg.action_counts.values()) == len(svc.governance.get_all_actions())
    # each investigated incident lands in exactly one status bucket
    assert hg.open + hg.acknowledged + hg.reviewed + hg.escalated + hg.closed \
        == hg.incidents_investigated


def test_reviewer_override_and_decision_preserved_separately(report):
    rd = report.overview.reviewer_disagreement
    assert rd.reviewed_count >= 3
    assert rd.override_count >= 1
    assert rd.override_rate is not None and 0.0 <= rd.override_rate <= 1.0
    # transitions record BOTH the automated decision and the reviewer decision
    for key in rd.automated_to_reviewer_transitions:
        assert " -> " in key
    assert "NOT ground truth" in rd.note


# ------------------------------------------------------------------
# 12. automated decision is preserved, not overwritten
# ------------------------------------------------------------------


def test_automated_decision_not_mutated_by_analytics(svc):
    before = {
        t.interaction_id: t.final_decision.decision.value for t in svc.all_traces()
    }
    build_governance_report(svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all())
    after = {
        t.interaction_id: t.final_decision.decision.value for t in svc.all_traces()
    }
    assert before == after


# ------------------------------------------------------------------
# 13. feedback signal aggregation
# ------------------------------------------------------------------


def test_feedback_signal_aggregation(report):
    types = report.signals.by_signal_type
    assert "feedback_modified" in types or "feedback_approved" in types
    assert "reviewer_override" in types
    assert report.signals.override_rate is not None
    assert "NOT ground truth" in report.signals.note
    for sig in report.signal_details:
        assert sig.source.value in (
            "reviewer_override", "feedback_modified", "feedback_rejected", "feedback_approved"
        )
        # automated decision and reviewer outcome are distinct fields
        assert hasattr(sig, "automated_decision") and hasattr(sig, "reviewer_outcome")


# ------------------------------------------------------------------
# 14-15. application comparison + insights
# ------------------------------------------------------------------


def test_application_comparison(report):
    apps = report.application_comparison
    names = {a.application for a in apps.applications}
    assert names <= {"customer_support", "internal_knowledge_assistant", "decision_support"}
    assert apps.highest_volume in names
    assert apps.highest_risk in names
    for a in apps.applications:
        for v in (a.average_risk, a.deep_rate, a.human_oversight_rate):
            assert v is None or 0.0 <= v <= 1.0


def test_insight_generation_is_deterministic_and_structured(report):
    for i in report.insights:
        assert i.code and i.title and i.explanation
        assert i.recommended_action.value in (
            "REVIEW_POLICY", "REVIEW_THRESHOLD", "REVIEW_APPLICATION",
            "REVIEW_DETECTOR", "INVESTIGATE_INCIDENTS", "NONE",
        )
        assert i.supporting_metrics
        assert "not a truth claim" in i.note
    # severity-ordered, then code
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    order = [(sev_rank[i.severity.value], i.code) for i in report.insights]
    assert order == sorted(order)


# ------------------------------------------------------------------
# 16-17. recommendations
# ------------------------------------------------------------------


def test_recommendation_generation(report):
    assert report.recommendations
    for r in report.recommendations:
        assert r.disposition in (
            RecommendationDisposition.RECOMMENDED_FOR_EVALUATION,
            RecommendationDisposition.REVIEW_REQUIRED,
            RecommendationDisposition.NO_ACTION,
        )
        assert "NOT APPLIED" in r.disclaimer


def test_recommendation_does_not_modify_config(svc):
    before = copy.deepcopy(load_settings())
    build_recommendations(
        build_overview(svc.all_traces(), svc.governance.get_all_actions()),
        compare_applications(svc.all_traces(), svc.governance.get_all_actions()),
        summarize_signals(collect_signals(svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all())),
    )
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


# ------------------------------------------------------------------
# 18. trends
# ------------------------------------------------------------------


def test_trend_calculation(report, svc):
    t = report.trends
    assert t.first_window_n + t.second_window_n == len(svc.all_traces())
    metrics = {s.metric for s in t.signals}
    assert {"average_risk", "average_confidence", "human_oversight_rate", "deep_rate"} <= metrics
    for s in t.signals:
        assert s.direction.value in ("increasing", "decreasing", "stable")
        assert s.label in ("TREND", "SIGNAL", "POTENTIAL_DRIFT")
    assert "no statistical-significance claim" in " ".join(t.notes)


def test_trend_direction_detection():
    s = ControlPlaneService(fit_cost_baseline=False)
    low = scenarios.scenario_a_clean()
    high = scenarios.scenario_e_multi_risk()
    for i in range(12):
        base = low if i < 6 else high
        it = base.model_copy(update={
            "interaction_id": f"GOV-TREND-{i:02d}",
            "timestamp": base.timestamp.replace(microsecond=i),
        })
        s.check(it, timestamp=it.timestamp)
    trends = build_trends(s.all_traces(), [])
    risk = next(x for x in trends.signals if x.metric == "average_risk")
    assert risk.direction.value == "increasing"
    assert risk.current > risk.baseline


# ------------------------------------------------------------------
# 30. determinism
# ------------------------------------------------------------------


def test_governance_report_is_deterministic(svc):
    a = build_governance_report(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    ).model_dump(mode="json")
    b = build_governance_report(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    ).model_dump(mode="json")

    def _strip(node):
        if isinstance(node, dict):
            return {
                k: _strip(v) for k, v in node.items()
                if not (k.endswith("_ms") or k.endswith("_latency") or k == "generated_at")
            }
        if isinstance(node, list):
            return [_strip(x) for x in node]
        return node

    assert _strip(a) == _strip(b)


# ------------------------------------------------------------------
# CRITICAL SAFETY / ARCHITECTURE GUARDS  (A-J)
# ------------------------------------------------------------------


def test_governance_never_imports_evaluation_or_ground_truth():
    for path in _GOV_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import evaluation" not in text, path.name
        assert "from evaluation" not in text, path.name
        assert "EvaluationCase" not in text, path.name
        for token in ("ground_truth_", "expected_decision", "final_outcome",
                      "actual_correctness"):
            assert token not in text, f"{path.name}: {token}"


def test_governance_does_not_orchestrate_pipeline():
    banned = (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "RiskFusionEngine(", "PolicyEngine(",
        "ConsequenceEngine(", "CriticalityEngine(",
        ".detect(", ".evaluate(", ".fuse_scores(", ".route(", ".assess(",
    )
    for path in _GOV_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name}: {token}"
    # ``.decide(`` may appear only as prose, never as a call — check for "engine.decide("
    for path in _GOV_DIR.glob("*.py"):
        assert "engine.decide(" not in path.read_text(encoding="utf-8")


def test_governance_recommendations_never_write_settings():
    src = (_GOV_DIR / "recommendations.py").read_text(encoding="utf-8")
    assert "settings.yaml" not in src
    for token in ("open(", "write(", "yaml.dump", "yaml.safe_dump", ".write_text"):
        assert token not in src, token


def test_governance_does_not_rerun_pipeline(svc, monkeypatch):
    import decision.engine as dec_mod
    import detectors.performance.detector as perf_mod
    import policy.engine as pol_mod

    def boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("governance analytics re-ran a pipeline component")

    monkeypatch.setattr(perf_mod.PerformanceDetector, "detect", boom)
    monkeypatch.setattr(dec_mod.DecisionEngine, "evaluate", boom)
    monkeypatch.setattr(pol_mod.PolicyEngine, "decide", boom)

    rep = build_governance_report(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    )
    assert rep.overview.traffic.total_interactions == len(svc.all_traces())


def test_reviewer_feedback_not_labelled_ground_truth():
    for path in _GOV_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        # the words "reviewer"/"override"/"governance signal" are used
        assert "governance signal" in text or "reviewer" in text or "override" in text
    # explicit disclaimers present in the schemas
    sch = (_GOV_DIR / "schemas.py").read_text(encoding="utf-8")
    assert "NOT ground truth" in sch
    assert "not correctness" in sch or "not a correctness label" in sch


def test_no_raw_pii_in_governance_report(svc):
    # scenario C (PII) is in the fixture traffic
    blob = build_governance_report(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    ).model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in blob, needle
    assert "matched_text" not in blob


# ------------------------------------------------------------------
# 28. scenarios A-J unchanged
# ------------------------------------------------------------------


def test_core_scenarios_unchanged():
    from evaluation.evaluation import build_engine

    engine = build_engine()
    got = {
        k: engine.evaluate(f(), timestamp=None, record_session=False).final_decision.decision.value
        for k, f in {
            "A": scenarios.scenario_a_clean,
            "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii,
            "D": scenarios.scenario_d_high_consequence,
            "E": scenarios.scenario_e_multi_risk,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK", "D": "VERIFY", "E": "BLOCK"}
