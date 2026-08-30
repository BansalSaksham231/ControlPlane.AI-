"""
Phase 6 — Explainability, Step 2: the view builder.

    build_explanation(trace) -> ExplainabilitySummary

Every test uses a REAL ``DecisionTrace`` from the existing pipeline and
proves the builder only *translates* it — no value is changed, no PII
leaks, no ground truth is touched.
"""

from __future__ import annotations

import pathlib

import pytest

from data.schemas import InterventionTier
from explainability.builder import build_explanation
from explainability.schemas import CounterfactualExplanation, ExplainabilitySummary
from evaluation.evaluation import build_engine
from tests import scenarios

_BUILDER_SRC = (
    pathlib.Path(__file__).resolve().parent.parent / "explainability" / "builder.py"
)

_FORBIDDEN_TOKENS = (
    "matched_text",
    "ground_truth_",
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


@pytest.fixture(scope="module")
def engine():
    return build_engine()


def _trace(engine, factory):
    interaction = factory()
    return engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )


@pytest.fixture(scope="module")
def traces(engine):
    return {
        "A": _trace(engine, scenarios.scenario_a_clean),
        "B": _trace(engine, scenarios.scenario_b_hallucination),
        "C": _trace(engine, scenarios.scenario_c_pii),
        "D": _trace(engine, scenarios.scenario_d_high_consequence),
        "E": _trace(engine, scenarios.scenario_e_multi_risk),
        "F": _trace(engine, scenarios.scenario_f_cost_anomaly),
        "H": _trace(engine, scenarios.scenario_h_low_confidence),
    }


# ------------------------------------------------------------------
# 1-7. scenario shapes
# ------------------------------------------------------------------


def test_clean_scenario_builds_valid_summary(traces):
    s = build_explanation(traces["A"])
    assert isinstance(s, ExplainabilitySummary)
    assert s.decision is InterventionTier.ALLOW
    assert s.human_review_required is False
    assert len(s.risk_dimensions) == 3
    assert {d.dimension for d in s.risk_dimensions} == {
        "performance",
        "responsibility",
        "cost",
    }


def test_hallucination_scenario_is_verify(traces):
    s = build_explanation(traces["B"])
    assert s.decision is InterventionTier.VERIFY
    assert s.decision_path
    assert s.decision_path[-1].to_tier is InterventionTier.VERIFY


def test_pii_scenario_is_block(traces):
    s = build_explanation(traces["C"])
    assert s.decision is InterventionTier.BLOCK
    assert s.human_review_required is True
    assert s.human_review.triggering_conditions


def test_high_consequence_information_preserved(traces):
    trace = traces["D"]
    s = build_explanation(trace)
    assert s.consequence_summary.consequence_score == trace.consequence.consequence_score
    assert s.consequence_summary.severity_band == trace.consequence.severity_band
    assert s.consequence_summary.dominant_factors == list(
        trace.consequence.dominant_factors
    )
    assert len(s.consequence_summary.factors) == len(trace.consequence.contributions)


def test_multi_risk_reason_codes_preserved(traces):
    trace = traces["E"]
    s = build_explanation(trace)
    assert s.primary_reasons == list(trace.final_decision.reason_codes)
    assert len(s.primary_reasons) >= 2


def test_fast_scenario_path(traces):
    s = build_explanation(traces["A"])
    assert s.verification_path.value == "FAST"
    assert s.verification_summary.verification_path.value == "FAST"
    assert s.verification_summary.used_deep is False


def test_deep_scenario_path(traces):
    s = build_explanation(traces["C"])
    assert s.verification_path.value == "DEEP"
    assert s.verification_summary.used_deep is True


# ------------------------------------------------------------------
# 8-13. values must be IDENTICAL to the trace
# ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "E", "F", "H"])
def test_core_values_match_trace_exactly(traces, key):
    trace = traces[key]
    s = build_explanation(trace)
    fd = trace.final_decision

    # final decision
    assert s.decision == fd.decision
    assert s.overall_risk == fd.overall_risk
    assert s.decision_confidence == fd.decision_confidence
    assert s.verification_path.value == trace.verification_path
    assert s.human_review_required == (fd.decision.value in ("HUMAN_REVIEW", "BLOCK"))

    # risk dimensions (copied from fusion.risk_breakdown)
    by_dim = {d.dimension: d for d in s.risk_dimensions}
    for row in trace.fusion.risk_breakdown:
        assert by_dim[row.dimension].risk == row.risk
        assert by_dim[row.dimension].confidence == row.confidence
        assert by_dim[row.dimension].weight == row.weight
        assert by_dim[row.dimension].weighted_contribution == row.weighted_contribution
    assert s.risk_summary.overall_risk == fd.overall_risk
    assert s.risk_summary.dominant_dimension == trace.fusion.dominant_dimension
    assert s.risk_summary.dominant_risk == trace.fusion.dominant_risk
    assert s.risk_summary.multi_risk == bool(trace.fusion.multi_risk)
    assert (
        s.risk_summary.criticality_weighted_performance_risk
        == trace.criticality_weighted_performance_risk
    )

    # confidence
    assert s.confidence_summary.fused_confidence == trace.fusion.confidence
    assert s.confidence_summary.fused_uncertainty == trace.fusion.uncertainty
    assert s.confidence_summary.performance_confidence == trace.performance.confidence
    assert (
        s.confidence_summary.verification_confidence
        == trace.performance.verification_confidence
    )

    # consequence
    assert s.consequence_summary.consequence_score == trace.consequence.consequence_score
    assert s.consequence_summary.severity_band == trace.consequence.severity_band

    # criticality
    assert s.criticality_summary.action_criticality == trace.criticality.action_criticality
    assert s.criticality_summary.band == trace.criticality.band
    assert (
        s.criticality_summary.max_claim_criticality
        == trace.criticality.max_claim_criticality
    )

    # policy rule trace: one presentation entry per real trace entry, in order
    assert [p.rule for p in s.policy_rules] == [
        e.rule for e in trace.policy.rule_trace
    ]
    assert [p.fired for p in s.policy_rules] == [
        e.fired for e in trace.policy.rule_trace
    ]
    assert [p.effect for p in s.policy_rules] == [
        e.effect for e in trace.policy.rule_trace
    ]

    # decision path: same transitions the engine recorded
    assert [(d.rule, d.from_tier, d.to_tier) for d in s.decision_path] == [
        (d.rule, d.from_tier, d.to_tier) for d in trace.decision_path
    ]

    # reasons / drivers copied, not invented
    assert s.primary_reasons == list(fd.reason_codes)
    assert s.decision_drivers == [d.rule for d in trace.decision_drivers]


# ------------------------------------------------------------------
# 14. determinism
# ------------------------------------------------------------------


def test_builder_is_deterministic(traces):
    for trace in traces.values():
        a = build_explanation(trace).model_dump_json()
        b = build_explanation(trace).model_dump_json()
        assert a == b


# ------------------------------------------------------------------
# 15. PII is not leaked
# ------------------------------------------------------------------


def test_pii_not_leaked(engine):
    interaction = scenarios.scenario_c_pii()
    trace = engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )
    blob = build_explanation(trace).model_dump_json()

    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans, "scenario_c_pii must produce PII findings"
    for span in raw_spans:
        assert span not in blob

    # the raw (unredacted) response must not appear either
    assert interaction.response not in blob
    assert "matched_text" not in blob


def test_pii_not_leaked_multi_risk(engine):
    interaction = scenarios.scenario_e_multi_risk()
    trace = engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )
    blob = build_explanation(trace).model_dump_json()
    for f in trace.responsibility.pii.findings:
        if f.matched_text and f.matched_text.strip():
            assert f.matched_text not in blob


# ------------------------------------------------------------------
# 16. forbidden ground-truth fields absent
# ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "E"])
def test_no_forbidden_tokens_in_serialized_output(traces, key):
    blob = build_explanation(traces[key]).model_dump_json()
    for token in _FORBIDDEN_TOKENS:
        assert token not in blob


# ------------------------------------------------------------------
# 17. builder does not instantiate / invoke detectors or engines
# ------------------------------------------------------------------


def test_builder_source_has_no_engine_or_detector_calls():
    text = _BUILDER_SRC.read_text(encoding="utf-8")
    banned = (
        "DecisionEngine",
        "PerformanceDetector",
        "ResponsibilityDetector",
        "CostDetector",
        "RiskFusionEngine",
        "PolicyEngine",
        "ConsequenceEngine",
        "CriticalityEngine",
        "VerificationRouter",
        "SessionManager",
        "simulate_policies",
        "compare_decisions",
        "import detectors",
        "from detectors",
        "from evaluation",
        "from simulation",
        ".detect(",
        ".evaluate(",
        ".fuse_scores(",
        ".decide(",
        ".route(",
        ".assess(",
        "ground_truth_",
        "expected_decision",
        "final_outcome",
    )
    for token in banned:
        assert token not in text, token


def test_builder_does_not_import_forbidden_modules():
    import explainability.builder as mod

    src = _BUILDER_SRC.read_text(encoding="utf-8")
    # the only pipeline dependency permitted is the redacted replay + the trace schema
    assert "from decision.replay import build_replay" in src
    for mod_name in ("detectors", "fusion", "policy.engine", "consequence.engine",
                     "criticality.engine", "verification.router", "simulation",
                     "evaluation"):
        assert f"import {mod_name}" not in src
        assert f"from {mod_name}" not in src
    assert mod.build_explanation is not None


# ------------------------------------------------------------------
# 18. missing optional verification data handled safely
# ------------------------------------------------------------------


def test_missing_verification_data_is_safe(traces):
    trace = traces["A"].model_copy(update={"verification": None})
    s = build_explanation(trace)
    assert s.verification_summary.explanation == ""
    assert s.verification_summary.deep_trigger_reasons == []
    assert s.verification_summary.deep_was_forced is False
    assert s.verification_path.value in ("FAST", "DEEP")
    # still round-trips
    ExplainabilitySummary.model_validate_json(s.model_dump_json())


# ------------------------------------------------------------------
# 19. empty decision path handled safely
# ------------------------------------------------------------------


def test_empty_policy_trace_is_safe(traces):
    trace = traces["A"]
    empty_policy = trace.policy.model_copy(update={"rule_trace": []})
    trace = trace.model_copy(update={"policy": empty_policy})
    s = build_explanation(trace)
    assert s.policy_rules == []
    assert s.decision_path == []
    assert s.human_review.triggering_conditions == []
    assert isinstance(s, ExplainabilitySummary)


# ------------------------------------------------------------------
# 20. JSON serialization / deserialization
# ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "E", "F", "H"])
def test_json_roundtrip(traces, key):
    s = build_explanation(traces[key])
    restored = ExplainabilitySummary.model_validate_json(s.model_dump_json())
    assert restored == s


# ------------------------------------------------------------------
# counterfactual pass-through (result supplied, never computed here)
# ------------------------------------------------------------------


def test_counterfactual_is_passed_through_unchanged(traces):
    cfx = CounterfactualExplanation(
        rule_removed="HIGH_CONSEQUENCE",
        original_decision=InterventionTier.VERIFY,
        counterfactual_decision=InterventionTier.ALLOW,
        decision_changed=True,
        rules_no_longer_firing=["HIGH_CONSEQUENCE"],
        reason_codes_removed=["HIGH_CONSEQUENCE"],
        summary="Without the high-consequence rule this would have been ALLOW.",
    )
    s = build_explanation(traces["D"], counterfactuals=[cfx])
    assert s.counterfactuals == [cfx]
    assert s.counterfactuals[0].simulated is True


def test_counterfactuals_default_empty(traces):
    assert build_explanation(traces["A"]).counterfactuals == []


# ------------------------------------------------------------------
# regression: presentation layer does not change the result
# ------------------------------------------------------------------


@pytest.mark.parametrize("key", ["A", "B", "C", "D", "E"])
def test_regression_builder_does_not_change_outcome(traces, key):
    trace = traces[key]
    e = build_explanation(trace)
    assert e.decision == trace.final_decision.decision
    assert e.overall_risk == trace.final_decision.overall_risk
    assert e.decision_confidence == trace.final_decision.decision_confidence
    assert e.verification_path.value == trace.verification_path
    assert e.risk_summary.dominant_dimension == trace.fusion.dominant_dimension
    assert e.consequence_summary.consequence_score == trace.consequence.consequence_score
    assert e.criticality_summary.action_criticality == trace.criticality.action_criticality
    # trace itself is untouched
    assert trace.final_decision.decision.value == e.decision.value
