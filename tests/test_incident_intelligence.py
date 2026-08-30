"""
Phase 10 — Incident Intelligence (clustering / patterns / drift / attribution).

Read-only, deterministic, no ground truth, no raw PII, no pipeline re-run.
"""

from __future__ import annotations

import pathlib
import random
from datetime import timedelta

import pytest

from api.service import ControlPlaneService
from data.generator import generate_interactions
from data.schemas import InterventionTier
from incident.clustering import cluster_incidents
from incident.report import build_incident_intelligence
from incident.schemas import IncidentIntelligenceReport, Phase10IncidentConfig
from incident.store import IncidentStore, signature_for
from settings import load_settings
from tests import scenarios

_INC_DIR = pathlib.Path(__file__).resolve().parent.parent / "incident"
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def svc():
    s = ControlPlaneService(fit_cost_baseline=False)
    for f in scenarios.ALL_SINGLE_TURN.values():
        it = f()
        s.check(it, timestamp=it.timestamp)
    for turn in scenarios.scenario_g_multi_turn():
        s.check(turn, timestamp=turn.timestamp)
    cfg = load_settings()
    for it in generate_interactions(cfg, random.Random(cfg["seed"]))[:100]:
        s.check(it, timestamp=it.timestamp)
    # recurring reviewer disagreement (BLOCK -> HUMAN_REVIEW)
    for t in [t for t in s.all_traces() if t.final_decision.decision is InterventionTier.BLOCK][:4]:
        s.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        s.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="hr only", reviewer_decision="HUMAN_REVIEW",
        )
    return s


@pytest.fixture(scope="module")
def report(svc) -> IncidentIntelligenceReport:
    return build_incident_intelligence(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    )


# ------------------------------------------------------------------ incidents


def test_empty_dataset():
    rep = build_incident_intelligence([], [], [])
    assert rep.total_incidents == 0
    assert rep.clusters == [] and rep.patterns == []
    assert rep.drift.signals == []


def test_one_incident():
    s = ControlPlaneService(fit_cost_baseline=False)
    it = scenarios.scenario_c_pii()
    s.check(it, timestamp=it.timestamp)
    rep = build_incident_intelligence(s.all_traces(), [], [])
    assert rep.total_incidents == 1
    assert len(rep.clusters) == 1
    assert rep.clusters[0].incident_count == 1
    assert rep.clusters[0].is_recurring is False


def test_incident_records_are_pii_safe(svc):
    store = IncidentStore()
    records = store.ingest(svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all())
    assert records
    blob = "".join(r.model_dump_json() for r in records)
    for needle in _KNOWN_PII:
        assert needle not in blob
    assert "matched_text" not in blob
    # no ground-truth field names
    for r in records:
        assert not any(k.startswith("ground_truth") for k in r.model_dump())


# ------------------------------------------------------------------ clustering


def test_grouping_is_deterministic(svc):
    store = IncidentStore()
    recs = store.ingest(svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all())
    a = cluster_incidents(recs)
    b = cluster_incidents(list(reversed(recs)))
    assert [c.cluster_id for c in a] == [c.cluster_id for c in b]
    assert [c.incident_count for c in a] == [c.incident_count for c in b]


def test_signature_uses_only_structured_fields():
    fields = {
        "application": "customer_support", "dominant_dimension": "responsibility",
        "decision": "BLOCK", "verification_path": "DEEP",
        "reason_codes": ["CRITICAL_PII", "PII_EXPOSURE"], "tier_changing_rules": ["CRITICAL_PII"],
        "consequence_band": "low", "criticality_band": "low",
    }
    sig1 = signature_for(fields)
    sig2 = signature_for({**fields, "reason_codes": ["PII_EXPOSURE", "CRITICAL_PII"]})
    assert sig1 == sig2  # order-independent
    assert "CRITICAL_PII" in sig1 and "BLOCK" in sig1


def test_representative_incidents_and_stats(report):
    for c in report.clusters:
        assert len(c.representative_incidents) <= 3
        assert c.average_risk is not None and c.max_risk >= c.average_risk
        assert sum(c.decisions.values()) == c.incident_count


# ------------------------------------------------------------------ patterns


def test_recurring_and_reviewer_override_patterns(report):
    types = {p.type.value for p in report.patterns}
    assert types  # at least one pattern on this traffic
    # a reviewer-override pattern exists (we injected BLOCK -> HUMAN_REVIEW x3+)
    assert any(op.transition == "BLOCK -> HUMAN_REVIEW" for op in report.reviewer_override_patterns)
    for p in report.patterns:
        assert 0.0 <= p.detection_confidence <= 1.0
        assert "NOT a statement about whether any AI response" in p.detection_confidence_note


def test_pattern_severity_ordering(report):
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    order = [rank[p.severity.value] for p in report.patterns]
    assert order == sorted(order)


def test_policy_rule_and_detector_patterns(report):
    for p in report.patterns:
        if p.type.value == "POLICY_RULE_DOMINANCE":
            assert p.affected_policy_rule
        if p.type.value == "DETECTOR_DOMINANCE":
            assert p.affected_dimension in ("performance", "responsibility", "cost")


# ------------------------------------------------------------------ drift


def test_drift_stable_on_uniform_traffic():
    s = ControlPlaneService(fit_cost_baseline=False)
    it = scenarios.scenario_a_clean()
    for i in range(8):
        c = it.model_copy(update={
            "interaction_id": f"DR-STABLE-{i}", "timestamp": it.timestamp.replace(microsecond=i)})
        s.check(c, timestamp=c.timestamp)
    # clean ALLOWs are not incidents -> no drift signals from incident_rate spikes
    rep = build_incident_intelligence(s.all_traces(), [], [])
    assert all(sig.signal in ("STABLE", "TREND") for sig in rep.drift.signals)


def test_drift_detects_increasing_risk():
    s = ControlPlaneService(fit_cost_baseline=False)
    low, high = scenarios.scenario_a_clean(), scenarios.scenario_e_multi_risk()
    for i in range(20):
        base = low if i < 10 else high
        c = base.model_copy(update={
            "interaction_id": f"DR-UP-{i:02d}", "timestamp": base.timestamp + timedelta(minutes=i)})
        s.check(c, timestamp=c.timestamp)
    rep = build_incident_intelligence(s.all_traces(), [], [])
    g_risk = next(x for x in rep.drift.signals if x.scope == "global" and x.metric == "average_risk")
    assert g_risk.direction.value == "increasing"
    assert g_risk.recent > g_risk.baseline


def test_drift_insufficient_samples_flagged():
    s = ControlPlaneService(fit_cost_baseline=False)
    for i in range(6):
        it = scenarios.scenario_e_multi_risk().model_copy(update={
            "interaction_id": f"DR-SMALL-{i}", "timestamp": scenarios.scenario_e_multi_risk().timestamp.replace(microsecond=i)})
        s.check(it, timestamp=it.timestamp)
    rep = build_incident_intelligence(s.all_traces(), [], [])
    assert any(not sig.sample_sufficient for sig in rep.drift.signals)
    assert "not proof of model degradation" in rep.drift.disclaimer
    assert "no statistical-significance" in rep.drift.disclaimer.lower()


# ------------------------------------------------------------------ attribution


def test_attribution_has_no_causal_language(report):
    for a in report.attributions:
        text = (a.narrative + " " + a.disclaimer).lower()
        assert "caused by" not in text
        assert "associated with" in text or "observed alongside" in text or "dominant contributor" in text
        assert "not causal proof" in a.disclaimer
        for share in list(a.dimension_shares.values()) + list(a.reason_code_shares.values()):
            assert 0.0 <= share <= 1.0


# ------------------------------------------------------------------ safety guards


def test_incident_source_has_no_ground_truth_or_evaluation():
    for path in _INC_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import evaluation" not in text
        assert "from evaluation" not in text
        assert "EvaluationCase" not in text
        for token in ("ground_truth_", "expected_decision", "final_outcome", "actual_correctness"):
            assert token not in text, f"{path.name}: {token}"


def test_incident_does_not_orchestrate_pipeline():
    banned = (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "RiskFusionEngine(", "PolicyEngine(",
        "ConsequenceEngine(", "CriticalityEngine(",
        ".detect(", ".evaluate(", ".fuse_scores(", ".route(", ".assess(", "engine.decide(",
    )
    for path in _INC_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name}: {token}"


def test_incident_does_not_rerun_pipeline(svc, monkeypatch):
    import decision.engine as dec_mod
    import detectors.performance.detector as perf_mod
    import policy.engine as pol_mod

    def boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("incident intelligence re-ran a pipeline component")

    monkeypatch.setattr(perf_mod.PerformanceDetector, "detect", boom)
    monkeypatch.setattr(dec_mod.DecisionEngine, "evaluate", boom)
    monkeypatch.setattr(pol_mod.PolicyEngine, "decide", boom)
    rep = build_incident_intelligence(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    )
    assert rep.total_incidents > 0


def test_report_is_deterministic(svc):
    a = build_incident_intelligence(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    ).model_dump(mode="json")
    b = build_incident_intelligence(
        svc.all_traces(), svc.governance.get_all_actions(), svc.feedback.all()
    ).model_dump(mode="json")
    assert a == b


def test_no_raw_pii_in_full_report(report):
    blob = report.model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in blob
    assert "matched_text" not in blob and "ground_truth" not in blob


def test_reviewer_feedback_not_ground_truth(report):
    for op in report.reviewer_override_patterns:
        assert "NOT evidence that the automated decision was incorrect" in op.note
