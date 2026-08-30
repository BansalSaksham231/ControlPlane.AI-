"""
Progressive Verification (Phase 2) tests — the ControlPlane Adaptive Guard.

Verifies the router spends deep verification effort only where risk,
uncertainty, disagreement or consequence justify it, records the path and
real latency, and never changes an existing decision on the demo
scenarios.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schemas import (
    ActionType,
    Application,
    Interaction,
    InterventionTier,
    ModelName,
    UserType,
)
from decision.engine import DecisionEngine
from detectors.performance.detector import PerformanceDetector
from settings import load_settings
from tests import scenarios
from verification.router import VerificationRouter
from verification.schemas import VerificationPath

TS = datetime(2026, 8, 21, 12, 0, 0)

REFUND_CONTEXT = (
    "Company policy allows customers to request a refund within 30 business days of "
    "purchase, provided the item is unused."
)


def _interaction(**overrides) -> Interaction:
    base = dict(
        interaction_id="INT-VER-1",
        timestamp=TS,
        application=Application.CUSTOMER_SUPPORT,
        user_type=UserType.EXTERNAL_CUSTOMER,
        model=ModelName.GPT_4O_MINI,
        session_id="S1",
        prompt="What is the refund policy?",
        context=REFUND_CONTEXT,
        response="You are eligible for a refund within 30 business days as long as the item is unused.",
        tokens_in=40,
        tokens_out=30,
        latency_ms=300.0,
        tool_calls=0,
        retry_count=0,
        action_type=ActionType.INFORMATION,
        action_amount_inr=0.0,
        affected_entities=1,
    )
    base.update(overrides)
    return Interaction(**base)


@pytest.fixture(scope="module")
def router() -> VerificationRouter:
    return VerificationRouter()


def _route(router, interaction):
    from common.timing import Stopwatch

    perf, resp, cost, report = router.route(interaction, Stopwatch(), {})
    return perf, resp, cost, report


# 1. clean / high-confidence -> FAST
def test_clean_high_confidence_takes_fast_path(router):
    _, _, _, report = _route(router, _interaction())
    assert report.verification_path is VerificationPath.FAST
    assert report.deep_trigger_reasons == []
    assert report.preliminary_confidence >= 0.7
    assert report.deep_path_latency_ms == 0.0


# 2. weak / ambiguous evidence -> DEEP
def test_weak_evidence_takes_deep_path(router):
    _, _, _, report = _route(
        router,
        _interaction(
            response="Your entire account balance has been permanently deleted and cannot be restored.",
            context="The customer asked about opening hours on public holidays.",
        ),
    )
    assert report.verification_path is VerificationPath.DEEP
    assert {"MISSING_EVIDENCE", "DETECTOR_DISAGREEMENT", "LOW_CONFIDENCE"} & set(
        report.deep_trigger_reasons
    )


# 3. high consequence -> DEEP even when the response itself looks clean
def test_high_consequence_forces_deep(router):
    _, _, _, report = _route(
        router,
        _interaction(
            # a well-grounded response (shallow verifies it as SUPPORTED,
            # high confidence) ...
            response="You are eligible for a refund within 30 business days as long as the item is unused.",
            context=REFUND_CONTEXT,
            # ... but the action carries a large financial exposure.
            action_type=ActionType.REFUND,
            action_amount_inr=480_000.0,
        ),
    )
    assert report.verification_path is VerificationPath.DEEP
    assert "HIGH_CONSEQUENCE" in report.deep_trigger_reasons
    assert report.preliminary_risk <= 0.35
    assert report.preliminary_confidence >= 0.70
    # deep entered purely because of consequence, not because the response looked risky
    assert report.deep_was_forced is True


# 4. high criticality -> DEEP
def test_high_criticality_forces_deep(router):
    _, _, _, report = _route(
        router,
        _interaction(
            response="The account has been cancelled and the mailing list export sent to the vendor.",
            context="Routine request.",
            action_type=ActionType.ACCOUNT_CANCELLATION,
            affected_entities=800,
        ),
    )
    assert report.verification_path is VerificationPath.DEEP
    assert "HIGH_CRITICALITY" in report.deep_trigger_reasons or "HIGH_CONSEQUENCE" in report.deep_trigger_reasons


# 5. low confidence -> DEEP
def test_low_confidence_forces_deep(router):
    _, _, _, report = _route(
        router,
        _interaction(
            response="You will receive a full store-credit voucher and a courtesy upgrade within 24 hours.",
            context="Vouchers and upgrades are handled case by case depending on the account tier.",
        ),
    )
    assert report.verification_path is VerificationPath.DEEP
    assert "LOW_CONFIDENCE" in report.deep_trigger_reasons


# 6. missing evidence -> DEEP when configured
def test_missing_evidence_forces_deep(router):
    _, _, _, report = _route(
        router,
        _interaction(response="You can return the item within 45 days.", context=""),
    )
    assert report.verification_path is VerificationPath.DEEP
    assert "MISSING_EVIDENCE" in report.deep_trigger_reasons
    assert report.evidence_available is False


# 7. configurable thresholds actually change routing
def test_thresholds_change_routing():
    cfg = load_settings()
    # A moderately-risky unverified response that normally routes DEEP...
    interaction = _interaction(
        response="You will get a full refund within 24 hours guaranteed.",
        context="Refund timelines vary by product and region.",
    )
    strict = VerificationRouter(cfg)
    _, _, _, r_strict = _route(strict, interaction)
    assert r_strict.verification_path is VerificationPath.DEEP

    relaxed_cfg = dict(cfg)
    relaxed_cfg["verification"] = dict(cfg["verification"])
    relaxed_cfg["verification"].update(
        fast_path_max_risk=0.99,
        fast_path_min_confidence=0.0,
        deep_verification_risk_threshold=0.99,
        deep_verification_consequence_threshold=0.99,
        deep_verification_criticality_threshold=0.99,
        deep_verification_extreme_factor=0.99,
        disagreement_trigger=0.99,
        always_deep_on_missing_evidence=False,
    )
    relaxed = VerificationRouter(relaxed_cfg)
    _, _, _, r_relaxed = _route(relaxed, interaction)
    assert r_relaxed.verification_path is VerificationPath.FAST


# 8 + 9. verification path & deep-trigger reasons recorded on the trace
def test_trace_records_path_and_reasons():
    engine = DecisionEngine()
    trace = engine.evaluate(
        scenarios.scenario_b_hallucination(), timestamp=TS, record_session=False
    )
    assert trace.verification_path == "DEEP"
    assert trace.final_decision.verification_path == "DEEP"
    assert trace.verification is not None
    assert trace.verification.deep_trigger_reasons
    assert trace.decision_path  # tier transitions recorded

    clean = engine.evaluate(scenarios.scenario_a_clean(), timestamp=TS, record_session=False)
    assert clean.verification_path == "FAST"
    assert clean.verification.verification_path is VerificationPath.FAST


# 10. real latency values recorded (measured, not fabricated)
def test_real_latency_recorded(router):
    _, _, _, report = _route(
        router, _interaction(response="You are eligible for a refund within 90 business days.", context=REFUND_CONTEXT)
    )
    assert report.fast_path_latency_ms > 0.0
    assert report.deep_path_latency_ms > 0.0
    assert report.total_verification_latency_ms == pytest.approx(
        report.fast_path_latency_ms + report.deep_path_latency_ms, abs=1e-3
    )
    fast_report = _route(router, _interaction())[3]
    assert fast_report.deep_path_latency_ms == 0.0


# 11. risk and confidence remain separate
def test_risk_and_confidence_are_separate(router):
    _, _, _, report = _route(
        router,
        _interaction(
            response="Your account balance of 999999 has been permanently deleted forever.",
            context="The customer asked about weekend delivery.",
        ),
    )
    # elevated risk, low confidence — not the same number
    assert report.preliminary_risk != report.preliminary_confidence
    assert report.disagreement_breakdown.score >= 0.0


# 12. existing deep behaviour is unchanged
def test_deep_pass_matches_pre_router_behaviour():
    """
    DEEP path == the old always-full pipeline, EXCEPT where a deterministic hard
    boundary lets the router skip the semantic pass. For those interactions the
    DECISION and the primary responsibility reason codes still match the legacy
    pipeline; ``overall_risk`` is intentionally not recomputed.
    """
    router_engine = DecisionEngine(use_verification_router=True)
    legacy_engine = DecisionEngine(use_verification_router=False)
    for factory in scenarios.ALL_SINGLE_TURN.values():
        interaction = factory()
        ra = router_engine.evaluate(interaction, timestamp=TS, record_session=False)
        rb = legacy_engine.evaluate(interaction, timestamp=TS, record_session=False)
        a, b = ra.final_decision, rb.final_decision
        assert a.decision == b.decision
        if ra.verification is not None and ra.verification.semantics_bypassed:
            assert {"CRITICAL_PII", "PII_EXPOSURE"} <= set(a.reason_codes)
        else:
            assert a.overall_risk == pytest.approx(b.overall_risk, abs=1e-6)
            assert a.reason_codes == b.reason_codes


# 13. ground truth is never read
def test_no_ground_truth_in_verification():
    import inspect

    import verification.router as router_mod
    import verification.backend as backend_mod

    for mod in (router_mod, backend_mod):
        src = inspect.getsource(mod)
        assert "ground_truth" not in src
        assert "expected_decision" not in src
    assert "ground_truth_hallucination" not in Interaction.model_fields


# 14. deterministic
def test_deterministic_routing(router):
    interaction = scenarios.scenario_d_high_consequence()
    a = _route(router, interaction)[3]
    b = _route(router, interaction)[3]
    assert a.verification_path == b.verification_path
    assert a.deep_trigger_reasons == b.deep_trigger_reasons
    assert a.preliminary_risk == b.preliminary_risk
    assert a.disagreement_score == b.disagreement_score


def test_all_demo_scenarios_decisions_unchanged():
    """The router must not change any A–H decision."""
    engine = DecisionEngine()
    expected = {
        "A_clean": InterventionTier.ALLOW,
        "B_hallucination": InterventionTier.VERIFY,
        "C_pii": InterventionTier.BLOCK,
        "D_high_consequence": InterventionTier.VERIFY,
        "E_multi_risk": InterventionTier.BLOCK,
        "F_cost_anomaly": InterventionTier.VERIFY,
        "H_low_confidence": InterventionTier.VERIFY,
    }
    for name, factory in scenarios.ALL_SINGLE_TURN.items():
        decision = engine.evaluate(
            factory(), timestamp=TS, record_session=False
        ).final_decision.decision
        assert decision == expected[name], name


def test_shallow_is_subset_of_deep():
    detector = PerformanceDetector()
    for response in (
        "You are eligible for a refund within 30 business days as long as the item is unused.",
        "You are eligible for a refund within 90 business days of your purchase.",
    ):
        shallow = detector.detect(response, REFUND_CONTEXT, depth="shallow")
        deep = detector.detect(response, REFUND_CONTEXT, depth="deep")
        # unambiguous cases agree
        assert shallow.status == deep.status
        assert shallow.performance_risk == pytest.approx(deep.performance_risk, abs=1e-6)
        # shallow did strictly less retrieval work
        for s_claim, d_claim in zip(shallow.claim_results, deep.claim_results):
            assert len(s_claim.top_evidence) <= len(d_claim.top_evidence)


# ------------------------------------------------------------------ #
# 16. deterministic semantic bypass on a critical-PII hard boundary
# ------------------------------------------------------------------ #
def _cfg(**verification_overrides):
    settings = load_settings()
    return {
        **settings,
        "verification": {**settings.get("verification", {}), **verification_overrides},
    }


def test_bypass_is_on_by_default_and_can_be_disabled():
    on = _route(VerificationRouter(), scenarios.scenario_c_pii())[3]
    assert on.semantics_bypassed is True

    off = _route(
        VerificationRouter(_cfg(bypass_semantics_on_hard_boundary=False)),
        scenarios.scenario_c_pii(),
    )[3]
    assert off.semantics_bypassed is False


def test_bypass_skips_semantics_on_critical_pii():
    router = VerificationRouter(_cfg(bypass_semantics_on_hard_boundary=True))
    perf, resp, cost, report = _route(router, scenarios.scenario_c_pii())

    assert report.semantics_bypassed is True
    assert report.bypass_reason.startswith("CRITICAL_PII")
    assert report.verification_path is VerificationPath.DEEP
    assert report.deep_trigger_reasons == ["DETERMINISTIC_HARD_BOUNDARY"]
    # no semantic work ran
    assert perf.method == "deterministic_semantic_bypass"
    assert perf.claim_results == []
    assert perf.latency.nli_ms == 0.0 and perf.latency.retrieval_ms == 0.0
    assert report.deep_path_latency_ms == 0.0
    # responsibility + cost still ran
    assert resp.contains_critical_pii is True


def test_bypass_does_not_change_the_decision():
    """
    Contract: the bypass preserves the DECISION on every scenario (the
    responsibility override blocks regardless of grounding). It does not
    guarantee cross-dimension reason codes such as MULTI_RISK, because
    performance is no longer independently measured — the primary
    responsibility reason codes are always retained.
    """
    from decision.engine import DecisionEngine

    on = DecisionEngine(config=_cfg(bypass_semantics_on_hard_boundary=True))
    off = DecisionEngine()
    for factory in scenarios.ALL_SINGLE_TURN.values():
        interaction = factory()
        a = on.evaluate(interaction, timestamp=TS, record_session=False).final_decision
        b = off.evaluate(interaction, timestamp=TS, record_session=False).final_decision
        assert a.decision == b.decision
        if b.decision.value == "BLOCK":
            assert {"CRITICAL_PII", "PII_EXPOSURE"} <= set(a.reason_codes)


def test_bypass_leaves_non_pii_interactions_untouched():
    router = VerificationRouter(_cfg(bypass_semantics_on_hard_boundary=True))
    for factory in (scenarios.scenario_a_clean, scenarios.scenario_b_hallucination,
                    scenarios.scenario_f_cost_anomaly):
        _, _, _, report = _route(router, factory())
        assert report.semantics_bypassed is False


def test_backend_bypass_returns_valid_result_without_detecting():
    from verification.backend import LexicalDeepVerifier

    class _Boom:
        def detect(self, *a, **k):  # pragma: no cover
            raise AssertionError("detector must not run when bypass_semantics=True")

    backend = LexicalDeepVerifier(detector=_Boom())
    result = backend.verify(
        scenarios.scenario_c_pii(), depth="deep",
        bypass_semantics=True, bypass_reason="CRITICAL_PII:test",
    )
    assert result.status.value == "UNVERIFIED"
    assert result.evidence_available is False
    assert result.latency.total_ms == 0.0


# ------------------------------------------------------------------ #
# 17. cascade router promoted to AUTHORITATIVE (cascade_shadow_mode: false)
# ------------------------------------------------------------------ #
def test_cascade_authoritative_routes_the_demo_scenarios():
    """
    With the cascade promoted (config cascade_shadow_mode: false), the FAST/DEEP
    path is driven by the Tier-1 ML router plus the Tier-0.5 deterministic
    overrides. Scenario A is deterministically clean and the MF router routes it
    FAST; B-H each trip a Tier-0.5 gate (or the hard-boundary bypass) -> DEEP.
    """
    from verification.cascade_router import CascadeRouter
    from verification.routing_models import build_default_embedding_mf_router

    router = VerificationRouter(
        cascade=CascadeRouter(classifier=build_default_embedding_mf_router())
    )
    assert router.cascade_shadow is False

    expected = {
        "A_clean": VerificationPath.FAST,
        "B_hallucination": VerificationPath.DEEP,
        "C_pii": VerificationPath.DEEP,
        "D_high_consequence": VerificationPath.DEEP,
        "E_multi_risk": VerificationPath.DEEP,
        "F_cost_anomaly": VerificationPath.DEEP,
        "H_low_confidence": VerificationPath.DEEP,
    }
    for name, factory in scenarios.ALL_SINGLE_TURN.items():
        report = _route(router, factory())[3]
        assert report.verification_path is expected[name], name
        if expected[name] is VerificationPath.DEEP:
            assert report.deep_trigger_reasons, name


def test_tier_0_5_override_forces_deep_over_a_fast_prediction():
    """
    A high-consequence interaction is forced DEEP by the Tier-0.5 deterministic
    override even though the ML router (which never sees consequence) would not
    escalate on its own.
    """
    router = VerificationRouter()  # default: authoritative, embedding MF router
    _, _, _, report = _route(router, scenarios.scenario_d_high_consequence())

    assert report.verification_path is VerificationPath.DEEP
    assert "HIGH_CONSEQUENCE" in report.deep_trigger_reasons


def test_cascade_authoritative_is_deterministic():
    router = VerificationRouter()
    a = _route(router, scenarios.scenario_b_hallucination())[3]
    b = _route(router, scenarios.scenario_b_hallucination())[3]
    assert a.verification_path == b.verification_path
    assert a.deep_trigger_reasons == b.deep_trigger_reasons
    assert a.predicted_complexity_score == b.predicted_complexity_score
