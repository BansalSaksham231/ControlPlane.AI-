"""
Phase 6 — Explainability Foundation, Step 1.

Contract tests for ``explainability.schemas`` only. Test objects are built
from REAL ``DecisionTrace`` records (run through the existing pipeline) and
the already-redacted ``IncidentReplay`` — no new detector, no new risk
calculation, no ground truth.
"""

from __future__ import annotations

import pathlib

import pytest
from pydantic import BaseModel, ValidationError

import explainability.schemas as esch
from data.schemas import InterventionTier
from decision.replay import build_replay
from evaluation.evaluation import build_engine
from explainability.schemas import (
    ConfidenceSummary,
    ConsequenceExplanation,
    ConsequenceFactorRow,
    CounterfactualExplanation,
    CriticalityExplanation,
    DecisionPathExplanation,
    EvidenceExplanation,
    ExplainabilitySummary,
    HumanReviewExplanation,
    PolicyRuleExplanation,
    RiskDimensionExplanation,
    RiskSummary,
    VerificationExplanation,
)
from tests import scenarios

_EXPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "explainability"

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
    "matched_text",
)


# ------------------------------------------------------------------
# helpers — build the VIEW from a real trace (Step-1 test scaffold only)
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    return build_engine()


def _trace(engine, factory):
    interaction = factory()
    return engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )


def _explain(trace, counterfactuals=None) -> ExplainabilitySummary:
    """
    Map a stored trace onto the presentation contract, taking every free-text
    field from the already-redacted IncidentReplay. Pure field access — no
    detector call, no new score.
    """
    r = build_replay(trace)
    fd = trace.final_decision
    fusion = trace.fusion
    crit = trace.criticality
    cons = trace.consequence

    text_by_dim = {
        "performance": r.risk_signals.performance_explanation,
        "responsibility": r.risk_signals.responsibility_explanation,
        "cost": r.risk_signals.cost_explanation,
    }
    risk_dimensions = [
        RiskDimensionExplanation(
            dimension=c.dimension,
            risk=c.risk,
            confidence=c.confidence,
            weight=c.weight,
            weighted_contribution=c.weighted_contribution,
            status=(
                trace.performance.status.value if c.dimension == "performance" else None
            ),
            is_dominant=(c.dimension == fusion.dominant_dimension),
            explanation=text_by_dim.get(c.dimension, ""),
        )
        for c in fusion.risk_breakdown
    ]

    evidence = [
        EvidenceExplanation(
            claim=c.claim,
            status=c.status,
            retrieval_similarity=c.retrieval_similarity,
            nli_label=c.nli_label,
            nli_confidence=c.nli_confidence,
            evidence_strength=c.evidence_strength,
            claim_risk=c.claim_risk,
            supporting_evidence=c.top_evidence,
        )
        for c in r.claims
    ]

    policy_rules = [
        PolicyRuleExplanation(
            rule=s.rule,
            fired=s.fired,
            tier_before=s.tier_before,
            tier_after=s.tier_after,
            changed_tier=bool(s.fired and s.tier_before != s.tier_after),
            effect=s.effect,
            detail=s.detail,
        )
        for s in r.decision_path
    ]
    decision_path = [
        DecisionPathExplanation(
            rule=s.rule,
            from_tier=s.tier_before,
            to_tier=s.tier_after,
            reason=s.detail,
        )
        for s in r.decision_path
        if s.rule == "RISK_BAND" or (s.fired and s.tier_before != s.tier_after)
    ]

    human_conditions = [
        s.rule
        for s in r.decision_path
        if s.fired and s.tier_after in ("HUMAN_REVIEW", "BLOCK")
    ]

    return ExplainabilitySummary(
        decision=fd.decision,
        overall_risk=fd.overall_risk,
        decision_confidence=fd.decision_confidence,
        verification_path=trace.verification_path,
        human_review_required=r.final_decision.requires_human_review,
        primary_reasons=list(fd.reason_codes),
        decision_drivers=[d.rule for d in trace.decision_drivers],
        risk_summary=RiskSummary(
            overall_risk=fd.overall_risk,
            dominant_dimension=fusion.dominant_dimension,
            dominant_risk=fusion.dominant_risk,
            multi_risk=fusion.multi_risk,
            weighted_only_risk=fusion.weighted_only_risk,
            criticality_weighted_performance_risk=(
                trace.criticality_weighted_performance_risk
            ),
            severity_rule_applied=fusion.severity_rule_applied,
            severity_floor_applied=fusion.severity_floor_applied,
        ),
        confidence_summary=ConfidenceSummary(
            decision_confidence=fd.decision_confidence,
            fused_confidence=fusion.confidence,
            fused_uncertainty=fusion.uncertainty,
            performance_confidence=trace.performance.confidence,
            verification_confidence=trace.performance.verification_confidence,
        ),
        risk_dimensions=risk_dimensions,
        verification_summary=VerificationExplanation(
            verification_path=trace.verification_path,
            used_deep=trace.verification_path == "DEEP",
            deep_was_forced=r.verification.deep_was_forced,
            deep_trigger_reasons=list(r.verification.deep_trigger_reasons),
            reason_for_deep_verification=r.verification.reason_for_deep_verification,
            preliminary_risk=r.verification.preliminary_risk,
            preliminary_confidence=r.verification.preliminary_confidence,
            final_risk=r.verification.final_risk,
            final_confidence=r.verification.final_confidence,
            disagreement_score=r.verification.disagreement_score,
            evidence_available=r.verification.evidence_available,
            explanation=r.risk_signals.fusion_explanation,
        ),
        evidence=evidence,
        consequence_summary=ConsequenceExplanation(
            consequence_score=cons.consequence_score,
            severity_band=cons.severity_band,
            dominant_factors=list(cons.dominant_factors),
            factors=[
                ConsequenceFactorRow(
                    factor=fc.factor,
                    value=fc.value,
                    weight=fc.weight,
                    weighted_contribution=fc.weighted_contribution,
                )
                for fc in cons.contributions
            ],
            explanation=r.consequence.explanation,
        ),
        criticality_summary=CriticalityExplanation(
            action_criticality=crit.action_criticality,
            band=crit.band,
            dominant_factors=list(crit.dominant_factors),
            max_claim_criticality=crit.max_claim_criticality,
            explanation=r.criticality.explanation,
        ),
        policy_rules=policy_rules,
        decision_path=decision_path,
        human_review=HumanReviewExplanation(
            required=r.final_decision.requires_human_review,
            decision=fd.decision,
            triggering_conditions=human_conditions,
            explanation=r.explanation,
        ),
        counterfactuals=counterfactuals or [],
        explanation=r.explanation,
    )


# ------------------------------------------------------------------
# valid ALLOW / VERIFY / BLOCK explanations
# ------------------------------------------------------------------


def test_valid_allow_explanation(engine):
    summary = _explain(_trace(engine, scenarios.scenario_a_clean))
    assert summary.decision is InterventionTier.ALLOW
    assert summary.human_review_required is False
    assert isinstance(summary.risk_summary, RiskSummary)
    assert summary.risk_dimensions and len(summary.risk_dimensions) == 3


def test_valid_verify_explanation(engine):
    summary = _explain(_trace(engine, scenarios.scenario_b_hallucination))
    assert summary.decision is InterventionTier.VERIFY
    # a VERIFY path should have at least one tier-moving driver or reason
    assert summary.primary_reasons or summary.decision_drivers
    assert summary.decision_path  # RISK_BAND at minimum


def test_valid_block_explanation(engine):
    summary = _explain(_trace(engine, scenarios.scenario_c_pii))
    assert summary.decision is InterventionTier.BLOCK
    assert summary.human_review_required is True
    assert summary.human_review.required is True
    assert summary.human_review.triggering_conditions


# ------------------------------------------------------------------
# bounded numeric fields
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        scenarios.scenario_a_clean,
        scenarios.scenario_b_hallucination,
        scenarios.scenario_c_pii,
        scenarios.scenario_e_multi_risk,
        scenarios.scenario_f_cost_anomaly,
        scenarios.scenario_h_low_confidence,
    ],
)
def test_risk_and_confidence_values_bounded(engine, factory):
    s = _explain(_trace(engine, factory))
    assert 0.0 <= s.overall_risk <= 1.0
    assert 0.0 <= s.decision_confidence <= 1.0
    assert 0.0 <= s.risk_summary.overall_risk <= 1.0
    assert 0.0 <= s.risk_summary.dominant_risk <= 1.0
    for d in s.risk_dimensions:
        assert 0.0 <= d.risk <= 1.0
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.weight <= 1.0
        assert 0.0 <= d.weighted_contribution <= 1.0
    for c in s.confidence_summary.model_dump().values():
        if isinstance(c, (int, float)):
            assert 0.0 <= c <= 1.0
    for e in s.evidence:
        assert 0.0 <= e.retrieval_similarity <= 1.0
        assert 0.0 <= e.evidence_strength <= 1.0
        assert 0.0 <= e.claim_risk <= 1.0
        assert e.nli_confidence is None or 0.0 <= e.nli_confidence <= 1.0
    assert 0.0 <= s.consequence_summary.consequence_score <= 1.0
    assert 0.0 <= s.criticality_summary.action_criticality <= 1.0


def test_out_of_range_values_are_rejected():
    with pytest.raises(ValidationError):
        RiskSummary(
            overall_risk=1.5, dominant_dimension="performance", dominant_risk=0.2
        )
    with pytest.raises(ValidationError):
        EvidenceExplanation(
            claim="x",
            status="NEUTRAL",
            retrieval_similarity=-0.1,
            evidence_strength=0.0,
            claim_risk=0.0,
        )


# ------------------------------------------------------------------
# verification path
# ------------------------------------------------------------------


def test_verification_path_is_valid(engine):
    for factory in (scenarios.scenario_a_clean, scenarios.scenario_c_pii):
        s = _explain(_trace(engine, factory))
        assert s.verification_path.value in ("FAST", "DEEP")
        assert s.verification_summary.verification_path.value in ("FAST", "DEEP")
        assert s.verification_summary.used_deep == (
            s.verification_path.value == "DEEP"
        )


def test_invalid_verification_path_rejected():
    with pytest.raises(ValidationError):
        VerificationExplanation(verification_path="SORT_OF", used_deep=False)


# ------------------------------------------------------------------
# decision path representation
# ------------------------------------------------------------------


def test_decision_path_representation(engine):
    s = _explain(_trace(engine, scenarios.scenario_c_pii))
    assert s.decision_path
    for step in s.decision_path:
        assert isinstance(step, DecisionPathExplanation)
        assert isinstance(step.from_tier, InterventionTier)
        assert isinstance(step.to_tier, InterventionTier)
    # the last transition lands on the final decision tier
    assert s.decision_path[-1].to_tier is s.decision
    for rule in s.policy_rules:
        assert isinstance(rule, PolicyRuleExplanation)
        if rule.changed_tier:
            assert rule.fired is True


# ------------------------------------------------------------------
# counterfactual contract (result comes from the existing simulation engine)
# ------------------------------------------------------------------


def test_counterfactual_contract_accepts_simulation_result(engine):
    from simulation.engine import compare_decisions

    interaction, modified = scenarios.scenario_j_consequence_counterfactual()
    cf = compare_decisions(engine, interaction, modified)
    cfx = CounterfactualExplanation(
        rule_removed="action_amount_inr:480000->100",
        original_decision=cf.original_decision,
        counterfactual_decision=cf.counterfactual_decision,
        decision_changed=cf.original_decision != cf.counterfactual_decision,
        rules_no_longer_firing=list(cf.rules_removed),
        reason_codes_removed=list(cf.reason_codes_removed),
        summary=cf.summary,
    )
    assert cfx.simulated is True
    trace = engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )
    s = _explain(trace, counterfactuals=[cfx])
    assert s.counterfactuals and s.counterfactuals[0].decision_changed in (True, False)


# ------------------------------------------------------------------
# no forbidden ground-truth fields / no matched_text — anywhere
# ------------------------------------------------------------------


def _all_explainability_models():
    return [
        obj
        for obj in vars(esch).values()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


def test_no_forbidden_field_names_in_any_model():
    offenders: list[str] = []
    for model in _all_explainability_models():
        for name in model.model_fields:
            if name in _FORBIDDEN_TOKENS or name.startswith("ground_truth"):
                offenders.append(f"{model.__name__}.{name}")
    assert offenders == [], offenders


def test_models_forbid_extra_fields():
    # a forbidden field cannot be smuggled in
    with pytest.raises(ValidationError):
        RiskSummary(
            overall_risk=0.1,
            dominant_dimension="performance",
            dominant_risk=0.1,
            ground_truth_pii=True,
        )
    with pytest.raises(ValidationError):
        ExplainabilitySummary(
            decision="ALLOW",
            overall_risk=0.1,
            decision_confidence=0.9,
            verification_path="FAST",
            human_review_required=False,
            risk_summary=RiskSummary(
                overall_risk=0.1, dominant_dimension="performance", dominant_risk=0.1
            ),
            confidence_summary=ConfidenceSummary(
                decision_confidence=0.9,
                fused_confidence=0.9,
                fused_uncertainty=0.1,
                performance_confidence=0.9,
                verification_confidence=0.9,
            ),
            verification_summary=VerificationExplanation(
                verification_path="FAST", used_deep=False
            ),
            consequence_summary=ConsequenceExplanation(
                consequence_score=0.0, severity_band="LOW"
            ),
            criticality_summary=CriticalityExplanation(
                action_criticality=0.0, band="low"
            ),
            human_review=HumanReviewExplanation(required=False, decision="ALLOW"),
            explanation="ok",
            expected_decision="BLOCK",
        )


def test_source_files_do_not_reference_ground_truth():
    offenders: list[str] = []
    for path in _EXPL_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("ground_truth_", "expected_decision", "final_outcome"):
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], offenders


def test_schemas_do_not_import_detectors_or_engines():
    text = (_EXPL_DIR / "schemas.py").read_text(encoding="utf-8")
    for banned in (
        "import detectors",
        "from detectors",
        "DecisionEngine",
        "PerformanceDetector",
        "ResponsibilityDetector",
        "CostDetector",
        ".detect(",
        ".evaluate(",
        ".fuse_scores(",
    ):
        assert banned not in text, banned


# ------------------------------------------------------------------
# PII-safe serialization
# ------------------------------------------------------------------


def test_pii_safe_serialization(engine):
    interaction = scenarios.scenario_c_pii()
    trace = engine.evaluate(
        interaction, timestamp=interaction.timestamp, record_session=False
    )
    summary = _explain(trace)
    blob = summary.model_dump_json()

    # raw matched PII spans from the trace must not appear in the view
    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans, "scenario_c_pii should produce PII findings"
    for span in raw_spans:
        assert span not in blob

    assert "matched_text" not in blob
    for token in _FORBIDDEN_TOKENS:
        assert token not in blob


# ------------------------------------------------------------------
# deterministic serialization
# ------------------------------------------------------------------


def test_deterministic_serialization(engine):
    interaction = scenarios.scenario_e_multi_risk()

    def _blob():
        trace = engine.evaluate(
            interaction, timestamp=interaction.timestamp, record_session=False
        )
        return _explain(trace).model_dump_json()

    assert _blob() == _blob()


def test_summary_roundtrips_through_json(engine):
    s = _explain(_trace(engine, scenarios.scenario_e_multi_risk))
    restored = ExplainabilitySummary.model_validate_json(s.model_dump_json())
    assert restored == s


# ------------------------------------------------------------------
# multi-turn session memory (ContextualSnapshot -> ExplainabilitySummary)
# ------------------------------------------------------------------


def _multi_turn_summaries(seq_factories, sid="EXP-SESSION"):
    from decision.engine import DecisionEngine
    from explainability.builder import build_explanation
    from session.manager import SessionManager

    sm = SessionManager()
    engine = DecisionEngine(session_manager=sm)
    out = []
    for i, factory in enumerate(seq_factories, 1):
        interaction = factory().model_copy(
            update={"session_id": sid, "interaction_id": f"{sid}-{i}"}
        )
        trace = engine.evaluate(interaction, timestamp=interaction.timestamp)
        out.append(build_explanation(trace))
    return out


def test_single_turn_trace_has_no_session_memory(engine):
    from explainability.builder import build_explanation

    s = build_explanation(_trace(engine, scenarios.scenario_c_pii))
    assert s.session_memory is None


def test_session_memory_populated_after_a_critical_turn():
    summaries = _multi_turn_summaries(
        [scenarios.scenario_a_clean, scenarios.scenario_c_pii,
         scenarios.scenario_a_clean, scenarios.scenario_a_clean]
    )
    assert summaries[0].session_memory is None          # turn 1: no history
    turn3 = summaries[2].session_memory
    assert turn3 is not None
    assert turn3.turns_recorded == 2
    assert turn3.has_critical_history is True
    assert turn3.critical_floor == pytest.approx(0.75)
    assert turn3.critical_floor_applied is True         # forced this clean turn up
    assert summaries[2].decision.value in ("HUMAN_REVIEW", "BLOCK")

    events = turn3.critical_events
    assert [e.turn_index for e in events] == [2]
    assert events[0].trigger == "CRITICAL_PII"
    assert 0.0 <= events[0].risk_at_event <= 1.0

    assert turn3.peak_responsibility_risk > 0.0
    assert turn3.reason_code_counts.get("CRITICAL_PII", 0) >= 1


def test_session_memory_never_contains_raw_pii():
    summaries = _multi_turn_summaries(
        [scenarios.scenario_c_pii, scenarios.scenario_a_clean, scenarios.scenario_a_clean]
    )
    mem = summaries[-1].session_memory
    assert mem is not None and mem.pii_entity_keys
    blob = summaries[-1].model_dump_json()
    for raw in ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta"):
        assert raw not in blob
    # keys are the redacted "<subtype>:<masked>" form
    assert all(":" in k for k in mem.pii_entity_keys)


def test_session_memory_roundtrips_and_is_deterministic():
    a = _multi_turn_summaries(
        [scenarios.scenario_c_pii, scenarios.scenario_a_clean, scenarios.scenario_a_clean]
    )[-1]
    b = _multi_turn_summaries(
        [scenarios.scenario_c_pii, scenarios.scenario_a_clean, scenarios.scenario_a_clean]
    )[-1]
    assert a.model_dump_json() == b.model_dump_json()
    assert ExplainabilitySummary.model_validate_json(a.model_dump_json()) == a
