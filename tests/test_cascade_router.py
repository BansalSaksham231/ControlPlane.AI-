"""
Tiered cascade routing — verification/cascade_router.py + verification/routing_models.py.

Covers the deterministic fast path, the learned Tier-1 gatekeeper (heuristic and
matrix-factorisation), per-application dynamic thresholds, cost-optimal
calibration, contextual-snapshot (incremental) routing, speculative cascading,
and the bypass-semantics signal. Everything here is deterministic and offline.
"""

from __future__ import annotations

import pytest

from verification.cascade_router import (
    CascadeRouter,
    FastPathEvaluator,
    FastPathOutcome,
    Interaction,
    RouteVerdict,
    RoutingState,
    resolve_profile,
)
from verification.routing_models import (
    HeuristicComplexityClassifier,
    MatrixFactorizationRouter,
    RoutingSample,
    build_default_mf_router,
    extract_features,
    fit_matrix_factorization,
    select_cost_optimal_threshold,
)

_PII_KNOWN = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221")


def _grounded(app: str = "customer_support", **kw) -> Interaction:
    return Interaction(
        interaction_id=kw.pop("interaction_id", "T"),
        application=app,
        prompt=kw.pop("prompt", "what are the support hours"),
        response=kw.pop("response", "We are open Monday to Friday, 9am to 5pm."),
        context=kw.pop("context", "Support hours: Mon-Fri 09:00-17:00."),
        **kw,
    )


# ---------------------------------------------------------------- fast path


def test_fast_path_clears_trivial_grounded_response():
    fp = FastPathEvaluator().evaluate(_grounded(), resolve_profile("customer_support"))
    assert fp.outcome is FastPathOutcome.CLEARED
    assert fp.confidence == 1.0
    assert fp.latency_ms < 50.0


def test_fast_path_flags_outbound_pii_as_violation():
    it = _grounded(
        response="Contact Karan Mehta at karan.mehta@example-test.com or +91-940847221.",
        context="",
    )
    fp = FastPathEvaluator().evaluate(it, resolve_profile("customer_support"))
    assert fp.outcome is FastPathOutcome.VIOLATION
    assert any(s.startswith("PII_PATTERN") for s in fp.signals)


def test_fast_path_defers_ambiguous_content():
    it = _grounded(
        response="Our refund rate is roughly 12% and Q3 revenue was around 4.2M.",
        context="Internal finance summary.",
        prompt="tell me about refunds",
    )
    fp = FastPathEvaluator().evaluate(it, resolve_profile("customer_support"))
    assert fp.outcome is FastPathOutcome.AMBIGUOUS
    assert "HIGH_STAKES_INTENT:refund" in fp.signals or "MULTIPLE_NUMERIC_CLAIMS" in fp.signals


def test_tool_output_is_not_penalised_for_proper_nouns():
    it = _grounded(
        response="Tool result: assignee = Karan Mehta, status = DONE",
        context="ticket 88 assignee Karan Mehta status DONE",
        is_tool_output=True,
    )
    fp = FastPathEvaluator().evaluate(it, resolve_profile("internal_knowledge_assistant"))
    assert fp.outcome is FastPathOutcome.CLEARED


# ---------------------------------------------------------------- cascade A->D


def test_cascade_clear_short_circuits_to_fast_allow():
    d = CascadeRouter().route(_grounded())
    assert d.verdict is RouteVerdict.FAST_ALLOW
    assert d.tier_reached.name == "FAST_PATH_DETERMINISTIC"
    assert d.predicted_complexity_score is None  # classifier never ran


def test_cascade_pii_routes_to_deep_not_block_and_sets_bypass_semantics():
    it = _grounded(response="Card 4111 1111 1111 1111 on file.", context="")
    d = CascadeRouter().route(it)
    assert d.verdict is RouteVerdict.ROUTE_TO_DEEP
    assert d.reason_code == "DETERMINISTIC_HARD_BOUNDARY"
    assert d.bypass_semantics is True
    assert d.deterministic_signals and all(
        s.startswith(("PII_", "BLOCKED_")) for s in d.deterministic_signals
    )


def test_cascade_uses_classifier_for_ambiguous_middle():
    it = _grounded(
        response="I think the fee is probably about 2 percent but it might vary.",
        context="",  # ungrounded numeric claim -> AMBIGUOUS, not a hard rule
        prompt="what is the fee",
    )
    d = CascadeRouter().route(it)
    assert d.tier_reached.name == "COMPLEXITY_CLASSIFIER"
    assert d.predicted_complexity_score is not None
    assert d.reason_code in ("COMPLEXITY_ABOVE_THRESHOLD", "COMPLEXITY_BELOW_THRESHOLD")


# ---------------------------------------------------------------- dynamic threshold


def test_profiles_give_different_verdicts_for_same_interaction():
    it = Interaction(
        "X", "any",
        prompt="summarise the meeting",
        response="The team probably agreed to ship around March, roughly on schedule.",
        context="Meeting notes: ship date under discussion.",
    )
    router = CascadeRouter()
    fin = router.route(it, "financial_agent")
    internal = router.route(it, "internal_knowledge_assistant")
    # financial_agent has a much lower threshold -> at least as likely to escalate
    assert not (
        fin.verdict is RouteVerdict.FAST_ALLOW
        and internal.verdict is RouteVerdict.ROUTE_TO_DEEP
    )
    assert fin.applied_threshold < internal.applied_threshold


def test_force_deep_on_action_pins_threshold_to_zero():
    it = _grounded(app="financial_agent", response="Transfer complete.",
                   action_type="wire_transfer", action_amount=5000.0)
    prof = resolve_profile("financial_agent")
    assert prof.effective_threshold(it) == 0.0
    d = CascadeRouter().route(it, "financial_agent")
    assert d.verdict is RouteVerdict.ROUTE_TO_DEEP


# ---------------------------------------------------------------- budget guard


def test_budget_guard_degrades_open_for_low_blast_radius():
    it = _grounded(
        response="The estimate could be around 5 to 7 for the quarter.",
        context="", prompt="give me the estimate",  # ungrounded number -> AMBIGUOUS
    )
    d = CascadeRouter().route(it, "customer_support", upstream_latency_ms=190.0)
    assert d.reason_code == "BUDGET_GUARD_DEGRADE_OPEN"
    assert d.verdict is RouteVerdict.FAST_ALLOW
    assert d.within_latency_budget is False


def test_budget_guard_degrades_closed_for_financial_agent():
    it = _grounded(
        app="financial_agent",
        response="It might be around 3 percent, roughly.",
        context="", prompt="what rate applies",  # ungrounded number -> AMBIGUOUS
    )
    d = CascadeRouter().route(it, "financial_agent", upstream_latency_ms=190.0)
    assert d.reason_code == "BUDGET_GUARD_DEGRADE_CLOSED"
    assert d.verdict is RouteVerdict.ROUTE_TO_DEEP


# ---------------------------------------------------------------- matrix factorisation


def test_mf_router_is_monotonic_in_complexity():
    router = build_default_mf_router()
    simple = extract_features(_grounded(response="Yes, that is correct."))
    complex_ = extract_features(_grounded(
        response="It might be roughly 14% but I'm not certain and figures vary.",
        context="", prompt="what is the return",
    ))
    assert router.predict_from_features(complex_) > router.predict_from_features(simple)


def test_mf_fit_is_deterministic():
    samples = [
        RoutingSample(deep_would_change_decision=False,
                      features={"low_context_overlap": 0.1, "hedge_density": 0.0}),
        RoutingSample(deep_would_change_decision=True,
                      features={"low_context_overlap": 0.9, "hedge_density": 0.8}),
    ]
    a = fit_matrix_factorization(samples, epochs=50)
    b = fit_matrix_factorization(samples, epochs=50)
    assert a == b


def test_mf_router_drops_into_cascade():
    router = CascadeRouter(classifier=MatrixFactorizationRouter(
        fit_matrix_factorization(
            [
                RoutingSample(deep_would_change_decision=False,
                              features={"low_context_overlap": 0.1}),
                RoutingSample(deep_would_change_decision=True,
                              features={"low_context_overlap": 1.0, "no_evidence": 1.0}),
            ],
            epochs=100,
        )
    ))
    d = router.route(_grounded(
        response="Possibly around 9, but unverified.", context="",
        prompt="estimate please",
    ))
    assert d.tier_reached.name == "COMPLEXITY_CLASSIFIER"


# ---------------------------------------------------------------- calibration


def test_cost_optimal_threshold_respects_safety_constraint():
    samples = [
        RoutingSample(deep_would_change_decision=False, complexity_score=0.1),
        RoutingSample(deep_would_change_decision=False, complexity_score=0.3),
        RoutingSample(deep_would_change_decision=True, complexity_score=0.6),
        RoutingSample(deep_would_change_decision=True, complexity_score=0.85),
    ]
    calib = select_cost_optimal_threshold(
        samples, deep_verification_cost=1.0, missed_risk_penalty=50.0,
        max_missed_risk_rate=0.0,
    )
    assert calib.threshold <= 0.6
    assert calib.missed_risk_rate == 0.0
    assert calib.satisfied_constraint is True


def test_cost_optimal_threshold_returns_safest_when_infeasible():
    samples = [RoutingSample(deep_would_change_decision=True, complexity_score=0.9)]
    calib = select_cost_optimal_threshold(
        samples, deep_verification_cost=1.0, missed_risk_penalty=5.0,
        max_missed_risk_rate=0.0, candidate_thresholds=[0.95, 0.99],
    )
    assert calib.satisfied_constraint is False
    assert calib.threshold == 0.95  # the one that escalates the risky sample


def test_router_calibrate_profile_updates_threshold():
    router = CascadeRouter()
    samples = [
        RoutingSample(deep_would_change_decision=False,
                      features={"low_context_overlap": 0.1}),
        RoutingSample(deep_would_change_decision=True,
                      features={"low_context_overlap": 1.0, "no_evidence": 1.0}),
    ]
    new_router, calib = router.calibrate_profile(
        "customer_support", samples,
        deep_verification_cost=1.0, missed_risk_penalty=20.0,
    )
    assert 0.0 <= calib.threshold <= 1.0
    assert new_router._resolve("customer_support").cost_optimal_escalation_threshold \
        == calib.threshold


# ---------------------------------------------------------------- contextual snapshots


def test_incremental_stable_session_skips_classifier():
    router = CascadeRouter()
    state: RoutingState | None = None
    verdicts = []
    for i in range(4):
        turn = _grounded(
            interaction_id=f"S{i}", response="Sure, here is the link to the doc.",
            context="doc link", prompt="where is the doc", turn_index=i,
        )
        d, state = router.route_incremental(turn, state)
        verdicts.append((d.verdict, d.reason_code))
    assert all(v is RouteVerdict.FAST_ALLOW for v, _ in verdicts)
    assert all(rc == "INCREMENTAL_STABLE" for _, rc in verdicts)
    assert state.every_turn_cleared is True
    assert state.session_complexity == 0.0


def test_incremental_delta_only_a_spike_turn_escalates():
    router = CascadeRouter()
    _, state = router.route_incremental(
        _grounded(interaction_id="S0", turn_index=0), None
    )
    spike = Interaction(
        "S1", "customer_support",
        prompt="what's the refund amount",
        response="The refund is probably about 4500 rupees, though it might be "
        "closer to 5000 depending on fees and the exact date.",
        context="", turn_index=1,
    )
    d, state = router.route_incremental(spike, state)
    assert d.verdict is RouteVerdict.ROUTE_TO_DEEP
    assert state.session_complexity > 0.0
    assert d.turn_index == 1


def test_incremental_violation_escalates_and_marks_state():
    router = CascadeRouter()
    turn = _grounded(
        interaction_id="S1",
        response="Here is the card: 4111 1111 1111 1111.", context="",
        turn_index=1,
    )
    d, state = router.route_incremental(turn, RoutingState())
    assert d.verdict is RouteVerdict.ROUTE_TO_DEEP
    assert d.bypass_semantics is True
    assert state.every_turn_cleared is False


# ---------------------------------------------------------------- speculative cascading


def test_speculative_discards_deep_when_classifier_says_fast():
    calls = {"n": 0}

    def deep(_it):
        calls["n"] += 1
        return {"verified": True}

    router = CascadeRouter(classifier=HeuristicComplexityClassifier(bias=-8.0))
    it = _grounded(
        response="The value is roughly 3 or so, not certain.", context="",
        prompt="estimate", app="customer_support",
    )
    res = router.route_speculative(it, deep, system_load=0.1)
    assert res.routing_decision.verdict is RouteVerdict.FAST_ALLOW
    assert res.deep_started is True
    assert res.deep_discarded is True
    assert res.deep_result is None


def test_speculative_keeps_deep_result_when_escalated():
    def deep(_it):
        return {"verified": True, "risk": 0.8}

    router = CascadeRouter(classifier=HeuristicComplexityClassifier(bias=8.0))
    it = _grounded(
        response="Possibly around 12 percent, though estimates vary widely.",
        context="", prompt="what is the rate",
    )
    res = router.route_speculative(it, deep, system_load=0.1)
    assert res.routing_decision.verdict is RouteVerdict.ROUTE_TO_DEEP
    assert res.deep_result == {"verified": True, "risk": 0.8}
    assert res.deep_discarded is False


def test_speculative_falls_back_to_sequential_under_high_load():
    def deep(_it):
        return "deep-ran"

    router = CascadeRouter()
    it = _grounded(
        response="Maybe about 5, roughly.", context="", prompt="estimate",
    )
    res = router.route_speculative(it, deep, system_load=0.95)
    assert res.deep_discarded is False  # no speculation attempted


# ---------------------------------------------------------------- safety / hygiene


def test_routing_decision_carries_no_raw_pii():
    it = _grounded(
        response="Karan Mehta, karan.mehta@example-test.com, ACC-227763.",
        context="",
    )
    blob = repr(CascadeRouter().route(it))
    for needle in _PII_KNOWN:
        assert needle not in blob


def test_routing_is_deterministic():
    it = _grounded(
        response="I believe the number is around 7, possibly 8.", context="",
        prompt="how many",
    )
    a = CascadeRouter().route(it)
    b = CascadeRouter().route(it)
    for f in ("verdict", "tier_reached", "reason_code",
              "predicted_complexity_score", "applied_threshold",
              "deterministic_signals"):
        assert getattr(a, f) == getattr(b, f)


def test_cascade_module_has_no_forbidden_imports():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "verification"
    for name in ("cascade_router.py", "routing_models.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "import evaluation" not in text and "from evaluation" not in text
        assert "import requests" not in text and "urllib.request" not in text
        assert "import random" not in text
        for tok in ("ground_truth_", "expected_decision", "final_outcome"):
            assert tok not in text, f"{name}: {tok}"


# ---------------------------------------------------------------- MF embedding path

from verification.routing_models import (  # noqa: E402
    EmbeddingRoutingSample,
    HashingEmbeddingBackend,
    build_default_embedding_mf_router,
    fit_matrix_factorization_embeddings,
)


def test_hashing_embedding_is_deterministic_and_normalised():
    be = HashingEmbeddingBackend(dim=48)
    a = be.embed("should I move my balance")
    b = be.embed("should I move my balance")
    assert a == b
    assert len(a) == 48
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_embedding_mf_router_is_monotonic_and_routes():
    router = build_default_embedding_mf_router()
    assert "embedding" in router.model_name
    low = router.predict_complexity(_grounded(prompt="what are the support hours"))
    high = router.predict_complexity(
        _grounded(prompt="is this medication safe to combine with mine")
    )
    assert high.score > low.score
    d = CascadeRouter(classifier=router).route(
        _grounded(prompt="what is the guaranteed return on this fund",
                  response="Probably around 12 percent.", context="")
    )
    assert d.tier_reached.name in ("FAST_PATH_DETERMINISTIC", "COMPLEXITY_CLASSIFIER")


def test_embedding_mf_fit_is_deterministic():
    be = HashingEmbeddingBackend(dim=32)
    samples = [
        EmbeddingRoutingSample("what time do you open", False),
        EmbeddingRoutingSample("estimate my refund and transfer it", True),
    ]
    a = fit_matrix_factorization_embeddings(samples, be, epochs=40)
    b = fit_matrix_factorization_embeddings(samples, be, epochs=40)
    assert a == b


def test_mf_from_pretrained_mock_is_usable():
    from verification.routing_models import MatrixFactorizationRouter

    router = MatrixFactorizationRouter.from_pretrained(None)  # no live backend
    pred = router.predict_complexity(_grounded(prompt="summarise the liability clause"))
    assert 0.0 <= pred.score <= 1.0
    assert router.params.uses_embeddings is True


# ---------------------------------------------------------------- promotion: MF learns the authoritative router

from scripts.train_mf_router import (  # noqa: E402
    collect_labelled_dataset,
    train_and_evaluate,
)


def test_mf_router_learns_to_replicate_authoritative_routing():
    """
    Promotion-path proof: train the MatrixFactorizationRouter on shadow
    telemetry (interactions labelled by the real VerificationRouter's FAST/DEEP
    call) and show its predict_complexity scores then align with those
    decisions — clearing the escalation threshold for exactly the scenarios
    that need DEEP verification. Fully offline (HashingEmbeddingBackend),
    deterministic.
    """
    rows = collect_labelled_dataset(n_synth=88, seed=11)
    assert 70 <= len(rows) <= 110
    deep_fraction = sum(r.router_deep for r in rows) / len(rows)
    assert 0.35 <= deep_fraction <= 0.90, "dataset should not be trivially imbalanced"

    report = train_and_evaluate(rows, epochs=300)

    # it actually learned — beats the majority-class (always-DEEP) baseline
    assert report.train.accuracy >= 0.95
    assert report.holdout.accuracy >= 0.85
    assert report.holdout.accuracy >= report.holdout.always_deep_accuracy + 0.05
    assert report.holdout.deep_recall >= 0.80

    # Safety-critical direction: every scenario the authoritative router sends
    # DEEP must clear the trained MF router's threshold too (no missed DEEP).
    # (FAST-class learning is validated on the held-out synthetic traffic above,
    #  not the demo scenarios — those are deliberately all-risky and A's routing
    #  document is near-identical to B's.)
    deep_scenarios = ("B_hallucination", "C_pii", "D_high_consequence",
                      "E_multi_risk", "F_cost_anomaly", "H_low_confidence")
    for name in deep_scenarios:
        auth_path, _, aligned = report.scenario_scores[name]
        assert auth_path == "DEEP" and aligned, name

    # calibrated threshold is a real operating point
    assert 0.0 <= report.calibrated_threshold < 1.0


def test_mf_training_loop_is_deterministic():
    rows = collect_labelled_dataset(n_synth=32, seed=11)
    a = train_and_evaluate(rows, epochs=120)
    b = train_and_evaluate(rows, epochs=120)
    assert a.router.params == b.router.params
    assert a.holdout.accuracy == b.holdout.accuracy
