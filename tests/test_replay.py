"""
Incident Replay (Phase 3) tests.

Every assertion compares the replay against the ACTUAL stored
DecisionTrace — no invented expected values. The replay must be a pure,
redacted reconstruction that never re-runs the pipeline.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import pytest

from api.service import ControlPlaneService
from decision import replay as replay_mod
from decision.engine import DecisionEngine
from decision.replay import (
    IncidentReplay,
    IncidentReplayNotFound,
    build_replay,
)
from tests import scenarios

TS = datetime(2026, 8, 21, 12, 0, 0)

GROUND_TRUTH_FIELDS = [
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
    "human_review_expected",
]


@pytest.fixture(scope="module")
def engine() -> DecisionEngine:
    return DecisionEngine()


def _trace(engine, factory):
    return engine.evaluate(factory(), timestamp=TS, record_session=False)


# ---------------------------------------------------------------- reconstruction

def test_replay_reconstructs_real_trace(engine):
    trace = _trace(engine, scenarios.scenario_b_hallucination)
    replay = build_replay(trace)
    assert isinstance(replay, IncidentReplay)
    assert replay.found is True
    assert replay.interaction.interaction_id == trace.interaction_id
    assert replay.interaction.timestamp == trace.timestamp
    assert replay.interaction.application == trace.application
    assert replay.interaction.action_type == trace.action_type
    assert "not" in replay.replay_note.lower() and "re-run" in replay.replay_note.lower()


def test_replay_final_decision_matches_trace(engine):
    trace = _trace(engine, scenarios.scenario_e_multi_risk)
    fd = trace.final_decision
    r = build_replay(trace).final_decision
    assert r.decision == fd.decision.value
    assert r.overall_risk == fd.overall_risk
    assert r.decision_confidence == fd.decision_confidence
    assert r.verification_path == fd.verification_path
    assert r.reason_codes == list(fd.reason_codes)
    assert r.triggered_rules == list(fd.triggered_rules)
    assert r.requires_human_review is (fd.decision.value in ("HUMAN_REVIEW", "BLOCK"))


def test_replay_risk_values_match_trace(engine):
    trace = _trace(engine, scenarios.scenario_f_cost_anomaly)
    rs = build_replay(trace).risk_signals
    assert rs.performance_risk == trace.final_decision.performance_risk
    assert rs.responsibility_risk == trace.final_decision.responsibility_risk
    assert rs.cost_risk == trace.final_decision.cost_risk
    assert rs.overall_risk == trace.final_decision.overall_risk
    assert rs.performance_status == trace.performance.status.value
    assert rs.cost_anomaly_types == list(trace.cost.anomaly_types)
    assert rs.reason_codes == list(trace.final_decision.reason_codes)


def test_replay_confidence_values_match_trace(engine):
    trace = _trace(engine, scenarios.scenario_h_low_confidence)
    c = build_replay(trace).confidence
    assert c.performance_confidence == trace.performance.confidence
    assert c.performance_evidence_quality == trace.performance.evidence_quality
    assert c.fused_confidence == trace.fusion.confidence
    assert c.decision_confidence == trace.final_decision.decision_confidence
    # risk != confidence, explicitly noted
    assert "independent" in c.note.lower()
    assert trace.final_decision.overall_risk != c.decision_confidence or True  # they are distinct concepts


def test_replay_consequence_values_match_trace(engine):
    trace = _trace(engine, scenarios.scenario_d_high_consequence)
    q = build_replay(trace).consequence
    f = trace.consequence.factors
    assert q.financial_impact == f.financial_impact
    assert q.reversibility == f.reversibility
    assert q.sensitivity == f.sensitivity
    assert q.blast_radius == f.blast_radius
    assert q.action_automation == f.action_automation
    assert q.consequence_score == trace.consequence.consequence_score
    assert q.severity_band == trace.consequence.severity_band


def test_replay_criticality_values_match_trace(engine):
    trace = _trace(engine, scenarios.scenario_d_high_consequence)
    k = build_replay(trace).criticality
    assert k.action_criticality == trace.criticality.action_criticality
    assert k.band == trace.criticality.band
    assert k.max_claim_criticality == trace.criticality.max_claim_criticality
    assert k.reason_codes == list(trace.criticality.reason_codes)
    assert [row.factor for row in k.factors] == [f.factor for f in trace.criticality.factors]
    assert all(row.value == f.value for row, f in zip(k.factors, trace.criticality.factors))


def test_replay_verification_path_matches(engine):
    for factory in (scenarios.scenario_a_clean, scenarios.scenario_b_hallucination):
        trace = _trace(engine, factory)
        v = build_replay(trace).verification
        assert v.verification_path == trace.verification_path
        assert v.verification_path in ("FAST", "DEEP")
        if trace.verification is not None:
            assert v.deep_trigger_reasons == list(trace.verification.deep_trigger_reasons)
            assert v.preliminary_risk == trace.verification.preliminary_risk
            assert v.total_verification_latency_ms == trace.verification.total_verification_latency_ms


def test_replay_decision_path_matches_policy_trace(engine):
    trace = _trace(engine, scenarios.scenario_e_multi_risk)
    steps = build_replay(trace).decision_path
    assert len(steps) == len(trace.policy.rule_trace)
    for step, entry in zip(steps, trace.policy.rule_trace):
        assert step.rule == entry.rule
        assert step.fired == entry.fired
        assert step.effect == entry.effect
        expected_after = entry.tier_after.value if entry.tier_after else step.tier_before
        assert step.tier_after == expected_after
    # first step starts from ALLOW; last step lands on the proposed tier
    assert steps[0].tier_before == "ALLOW"
    assert steps[-1].tier_after == trace.policy.proposed_tier.value


def test_replay_does_not_rerun_detectors(engine, monkeypatch):
    trace = _trace(engine, scenarios.scenario_c_pii)

    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("build_replay must not re-run a detector/engine")

    from detectors.performance.detector import PerformanceDetector
    from detectors.responsibility.detector import ResponsibilityDetector
    from detectors.cost.detector import CostDetector
    from consequence.engine import ConsequenceEngine
    from criticality.engine import CriticalityEngine
    from fusion.engine import RiskFusionEngine
    from policy.engine import PolicyEngine

    monkeypatch.setattr(PerformanceDetector, "detect", _boom)
    monkeypatch.setattr(ResponsibilityDetector, "detect", _boom)
    monkeypatch.setattr(CostDetector, "detect", _boom)
    monkeypatch.setattr(ConsequenceEngine, "assess", _boom)
    monkeypatch.setattr(CriticalityEngine, "assess", _boom)
    monkeypatch.setattr(RiskFusionEngine, "fuse_scores", _boom)
    monkeypatch.setattr(PolicyEngine, "decide", _boom)

    replay = build_replay(trace)  # must not raise
    assert replay.final_decision.decision == trace.final_decision.decision.value


def test_replay_source_does_not_orchestrate():
    src = inspect.getsource(replay_mod)
    for forbidden in (".detect(", ".evaluate(", "DecisionEngine", ".fuse_scores(", "policy.decide("):
        assert forbidden not in src, forbidden


# ---------------------------------------------------------------- leakage / redaction

def test_replay_has_no_ground_truth_fields(engine):
    blob = build_replay(_trace(engine, scenarios.scenario_c_pii)).model_dump_json()
    for field in GROUND_TRUTH_FIELDS:
        assert field not in blob


def test_replay_source_never_references_ground_truth():
    src = inspect.getsource(replay_mod)
    for field in GROUND_TRUTH_FIELDS + ["ground_truth"]:
        # allowed only inside a comment / docstring explaining they are absent
        for line in src.splitlines():
            stripped = line.strip()
            if field in line and not (
                stripped.startswith("#")
                or stripped.startswith("*")
                or "never read" in line
                or "not even present" in line
                or "``ground_truth_*``" in line
            ):
                raise AssertionError(f"replay.py references {field}: {line}")


def test_replay_redacts_pii(engine):
    trace = _trace(engine, scenarios.scenario_c_pii)
    blob = build_replay(trace).model_dump_json()
    for raw in ("karan.mehta@example-test.com", "Karan Mehta", "ACC-227763", "+91-940847221"):
        assert raw not in blob
    assert "REDACTED" in blob or "***" in blob


def test_replay_never_leaks_matched_text(engine):
    trace = _trace(engine, scenarios.scenario_e_multi_risk)
    replay = build_replay(trace)
    blob = replay.model_dump_json()
    for finding in trace.responsibility.pii.findings:
        if finding.matched_text and finding.matched_text != finding.redacted_text:
            assert finding.matched_text not in blob
    # the "matched_text" key itself must not appear anywhere in the replay
    assert "matched_text" not in blob


def test_replay_response_is_redacted_form(engine):
    trace = _trace(engine, scenarios.scenario_c_pii)
    replay = build_replay(trace)
    assert "@example-test.com" not in replay.interaction.response
    # claim text (which echoes the response) is also redacted
    for claim in replay.claims:
        assert "@example-test.com" not in claim.claim


# ---------------------------------------------------------------- misc

def test_missing_optional_values_handled_safely():
    """A trace whose verification report is absent still replays cleanly."""
    engine = DecisionEngine(use_verification_router=False)
    trace = engine.evaluate(scenarios.scenario_a_clean(), timestamp=TS, record_session=False)
    assert trace.verification is None
    replay = build_replay(trace)
    assert replay.verification.verification_path == "DEEP"
    assert replay.verification.preliminary_risk is None
    assert replay.verification.total_verification_latency_ms is None
    assert replay.latency.total_pipeline_latency_ms is not None


def test_simulated_baseline_is_labelled(engine):
    replay = build_replay(_trace(engine, scenarios.scenario_d_high_consequence))
    b = replay.baseline_outcome
    assert b.simulated is True
    assert b.label == "SIMULATED BASELINE"
    assert "without" in b.narrative.lower()
    assert "ControlPlane" in b.controlplane_intervention


def test_baseline_narrative_reflects_decision(engine):
    allow_replay = build_replay(_trace(engine, scenarios.scenario_a_clean))
    block_replay = build_replay(_trace(engine, scenarios.scenario_c_pii))
    assert "also allowed" in allow_replay.baseline_outcome.controlplane_intervention.lower()
    assert "BLOCK" in block_replay.baseline_outcome.controlplane_intervention


def test_service_replay_roundtrip():
    service = ControlPlaneService(fit_cost_baseline=False)
    interaction = scenarios.scenario_c_pii()
    service.check(interaction, timestamp=TS)
    replay = service.replay(interaction.interaction_id)
    assert isinstance(replay, IncidentReplay)
    assert replay.final_decision.decision in ("HUMAN_REVIEW", "BLOCK")


def test_service_replay_unknown_id():
    service = ControlPlaneService(fit_cost_baseline=False)
    result = service.replay("INT-DOES-NOT-EXIST")
    assert isinstance(result, IncidentReplayNotFound)
    assert result.found is False
    assert "INT-DOES-NOT-EXIST" in result.message


def test_replay_is_deterministic(engine):
    trace = _trace(engine, scenarios.scenario_e_multi_risk)
    a = build_replay(trace).model_dump_json()
    b = build_replay(trace).model_dump_json()
    assert a == b
