"""
Calibration Advisor (Phase 4) tests — OFFLINE / EVALUATION subsystem.

The advisor simulates alternative threshold configurations against the
synthetic evaluation dataset. It MAY read ground-truth labels. It must
NEVER be reachable from the production decision path and must NEVER
mutate production configuration.
"""

from __future__ import annotations

import copy
import pathlib
from datetime import datetime

import pytest

from calibration.advisor import (
    CalibrationAdvisor,
    CalibrationCache,
    _deep_merge,
    _frange,
)
from calibration.schemas import (
    CalibrationMetrics,
    CalibrationReport,
    CounterfactualAnalysis,
)
from data.schemas import (
    ActionType,
    Application,
    Interaction,
    ModelName,
    UserType,
)
from settings import load_settings


# ---------------------------------------------------------------- shared advisor

@pytest.fixture(scope="module")
def advisor() -> CalibrationAdvisor:
    """One advisor -> detectors precomputed once, reused across the module."""
    return CalibrationAdvisor()


# ---------------------------------------------------------------- fixtures for
# tests 9 & 10 (consequence / criticality) which need a controlled dataset
# where those specific triggers decide the FAST/DEEP outcome.

def _grounded_recommendation(entities: int) -> Interaction:
    ctx = (
        "Warehouse B is running a 20 percent higher order backlog than Warehouse A "
        "this month, so extra staff should be moved to Warehouse B."
    )
    return Interaction(
        interaction_id=f"INT-FIX-{entities}",
        timestamp=datetime(2026, 8, 21, 12, 0, 0),
        application=Application.DECISION_SUPPORT,
        user_type=UserType.OPERATIONS_AGENT,
        model=ModelName.GPT_4O_MINI,
        session_id="S",
        prompt="Where should we move staff?",
        context=ctx,
        response=(
            "Since Warehouse B has a 20 percent higher backlog than Warehouse A, "
            "move extra staff to Warehouse B."
        ),
        tokens_in=40,
        tokens_out=40,
        latency_ms=300.0,
        tool_calls=0,
        retry_count=0,
        action_type=ActionType.RECOMMENDATION,
        action_amount_inr=0.0,
        affected_entities=entities,
    )


def _pinned_config(**pins: float) -> dict:
    cfg = copy.deepcopy(load_settings())
    cfg["verification"].update(pins)
    return cfg


@pytest.fixture(scope="module")
def fixture_cache() -> CalibrationCache:
    dataset = [(_grounded_recommendation(350), {}), (_grounded_recommendation(1), {})]
    return CalibrationCache(load_settings(), dataset=dataset)


# ---------------------------------------------------------------- 1-5 basics

def test_calibration_loads_evaluation_cases(advisor):
    assert len(advisor._cache.records) == 150
    assert any(r.gt_any for r in advisor._cache.records)
    assert any(r.gt_clean for r in advisor._cache.records)


def test_frange_deterministic():
    assert _frange([0.1, 0.3, 0.1]) == [0.1, 0.2, 0.3]
    assert _frange([0.1, 0.3, 0.1]) == _frange([0.1, 0.3, 0.1])


def test_threshold_sweep_is_deterministic(advisor):
    a = advisor.sweep("deep_verification_risk").model_dump()
    b = advisor.sweep("deep_verification_risk").model_dump()
    assert a == b


def test_sweep_does_not_mutate_settings(advisor):
    before = copy.deepcopy(load_settings())
    advisor.sweep("deep_verification_risk")
    advisor.evaluate_config({"verification": {"deep_verification_risk_threshold": 0.99}})
    assert load_settings() == before
    # advisor's own config also untouched
    assert advisor._config["verification"]["deep_verification_risk_threshold"] == \
        before["verification"]["deep_verification_risk_threshold"]


def test_deep_merge_is_pure():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = _deep_merge(base, {"a": {"y": 9}})
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}  # unchanged


def test_metrics_are_mathematically_consistent(advisor):
    m = advisor.current_operating_point()
    assert m.fast_verification_rate == pytest.approx(1 - m.deep_verification_rate, abs=1e-4)
    if m.intervention_recall is not None:
        assert m.false_negative_rate == pytest.approx(1 - m.intervention_recall, abs=1e-4)
    assert 0.0 <= m.human_review_rate <= 1.0
    assert m.mean_total_pipeline_latency_ms >= m.mean_verification_latency_ms


def test_zero_denominator_metrics_are_none():
    # a dataset with NO risky cases -> recall / precision cannot be computed
    dataset = [(_grounded_recommendation(1), {})]
    cache = CalibrationCache(load_settings(), dataset=dataset)
    adv = CalibrationAdvisor(load_settings(), cache=cache)
    m = adv.evaluate_config({})
    assert m.n_risky == 0
    assert m.intervention_recall is None
    assert m.false_negative_rate is None
    # this clean case is not intervened on -> no interventions -> precision None
    assert m.intervention_precision is None


# ---------------------------------------------------------------- 6-10 sweeps move

def test_risk_threshold_sweep_produces_different_operating_points(advisor):
    sweep = advisor.sweep("deep_verification_risk")
    deep_rates = {p.metrics.deep_verification_rate for p in sweep.points}
    assert len(deep_rates) > 1
    # monotone-ish: lowest threshold -> most DEEP
    assert sweep.points[0].metrics.deep_verification_rate >= sweep.points[-1].metrics.deep_verification_rate


def test_verification_threshold_changes_fast_deep_distribution(advisor):
    sweep = advisor.sweep("fast_path_min_confidence")
    deep_rates = [p.metrics.deep_verification_rate for p in sweep.points]
    assert max(deep_rates) - min(deep_rates) > 0.05


def test_confidence_threshold_affects_routing(advisor):
    low = advisor.evaluate_config({"verification": {"fast_path_min_confidence": 0.30}})
    high = advisor.evaluate_config({"verification": {"fast_path_min_confidence": 0.95}})
    assert high.deep_verification_rate > low.deep_verification_rate


def test_consequence_threshold_affects_routing(fixture_cache):
    # pin criticality / disagreement / confidence so ONLY the consequence
    # threshold can decide FAST vs DEEP for the fixture interaction. Runs with
    # the cascade in shadow so this exercises the deterministic Tier-0.5 knob
    # in isolation (with the cascade authoritative it would also escalate).
    cfg = _pinned_config(
        deep_verification_criticality_threshold=0.99,
        disagreement_trigger=0.99,
        fast_path_min_confidence=0.0,
        deep_verification_extreme_factor=0.99,
        fast_path_max_risk=0.99,
        deep_verification_risk_threshold=0.99,
        cascade_shadow_mode=True,
    )
    adv = CalibrationAdvisor(cfg, cache=fixture_cache)
    strict = adv.evaluate_config({"verification": {"deep_verification_consequence_threshold": 0.30}})
    lax = adv.evaluate_config({"verification": {"deep_verification_consequence_threshold": 0.60}})
    assert strict.deep_verification_rate > lax.deep_verification_rate


def test_criticality_threshold_affects_routing(fixture_cache):
    # cascade in shadow — isolates the deterministic Tier-0.5 criticality knob.
    cfg = _pinned_config(
        deep_verification_consequence_threshold=0.99,
        disagreement_trigger=0.99,
        fast_path_min_confidence=0.0,
        deep_verification_extreme_factor=0.99,
        fast_path_max_risk=0.99,
        deep_verification_risk_threshold=0.99,
        cascade_shadow_mode=True,
    )
    adv = CalibrationAdvisor(cfg, cache=fixture_cache)
    strict = adv.evaluate_config({"verification": {"deep_verification_criticality_threshold": 0.30}})
    lax = adv.evaluate_config({"verification": {"deep_verification_criticality_threshold": 0.60}})
    assert strict.deep_verification_rate > lax.deep_verification_rate


# ---------------------------------------------------------------- 11-14 recommendation

def test_recommendation_respects_constraints(advisor):
    rec = advisor.recommend("deep_verification_risk")
    assert rec.status == "RECOMMENDATION"
    m = rec.recommended_metrics
    c = advisor._constraints()
    assert m.intervention_recall >= c["target_recall"]
    if m.false_positive_rate is not None:
        assert m.false_positive_rate <= c["max_false_positive_rate"]
    assert m.human_review_rate <= c["max_human_review_rate"]
    assert "recall" in rec.explanation and "safety constraint" in rec.explanation.lower()


def test_recommendation_is_not_just_highest_f1(advisor):
    """Objective is min_deep_workload — the recommended point need not maximise F1."""
    rec = advisor.recommend("deep_verification_risk")
    sweep = advisor.sweep("deep_verification_risk")
    safe = [p for p in sweep.points if p.satisfies_safety_constraints]
    best_f1 = max(safe, key=lambda p: p.metrics.intervention_f1 or 0.0)
    chosen = next(p for p in safe if p.threshold_value == rec.recommended_value)
    assert chosen.metrics.deep_verification_rate <= best_f1.metrics.deep_verification_rate


def test_no_safe_operating_point_is_explicit(advisor):
    # impossible constraint -> must NOT silently relax
    cfg = copy.deepcopy(load_settings())
    cfg["calibration"]["target_recall"] = 1.01  # unachievable
    adv = CalibrationAdvisor(cfg, cache=advisor._cache)
    rec = adv.recommend("deep_verification_risk")
    assert rec.status == "NO_SAFE_OPERATING_POINT"
    assert rec.recommended_value is None
    assert "NOT relaxed" in rec.explanation


def test_recommendation_respects_max_human_review(advisor):
    cfg = copy.deepcopy(load_settings())
    cfg["calibration"]["max_human_review_rate"] = 0.01  # very strict
    adv = CalibrationAdvisor(cfg, cache=advisor._cache)
    rec = adv.recommend("deep_verification_risk")
    # every real operating point has ~0.39 human-review rate -> no safe point
    assert rec.status == "NO_SAFE_OPERATING_POINT"
    assert "human-review" in rec.explanation


# ---------------------------------------------------------------- 15 counterfactual

def test_counterfactual_produces_baseline_vs_alternative(advisor):
    cf = advisor.counterfactual("deep_verification_risk", 0.35, 0.85)
    assert isinstance(cf, CounterfactualAnalysis)
    assert cf.baseline_value == 0.35 and cf.counterfactual_value == 0.85
    assert isinstance(cf.baseline_metrics, CalibrationMetrics)
    assert isinstance(cf.counterfactual_metrics, CalibrationMetrics)
    # raising the threshold reduces DEEP verification here
    assert cf.changes.deep_verification_delta is not None
    assert cf.counterfactual_metrics.deep_verification_rate <= cf.baseline_metrics.deep_verification_rate
    assert "0.35" in cf.summary and "0.85" in cf.summary


def test_counterfactual_is_pure_simulation(advisor):
    before = copy.deepcopy(load_settings())
    advisor.counterfactual("deep_verification_risk", 0.20, 0.90)
    assert load_settings() == before


# ---------------------------------------------------------------- 16-18 boundary

def test_production_modules_do_not_import_calibration():
    root = pathlib.Path(__file__).resolve().parent.parent
    production_dirs = ["decision", "verification", "detectors", "fusion", "policy",
                       "consequence", "criticality"]
    for d in production_dirs:
        for py in (root / d).rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "import calibration" not in text, f"{py} imports calibration"
            assert "from calibration" not in text, f"{py} imports calibration"


def test_calibration_may_read_ground_truth():
    """The advisor is an evaluation subsystem — it is ALLOWED to use labels."""
    import inspect

    import calibration.advisor as mod

    src = inspect.getsource(mod)
    assert "ground_truth" in src  # it legitimately reads _RISKY_GT_KEYS
    # and it does so from the evaluation dataset, not from an Interaction
    assert "ground_truth" not in Interaction.model_fields.keys()


def test_calibration_does_not_mutate_production_config(advisor):
    before = copy.deepcopy(load_settings())
    advisor.report()
    assert load_settings() == before


# ---------------------------------------------------------------- 19-20 report

def test_full_report_is_serializable_and_deterministic(advisor):
    r1 = advisor.report()
    r2 = advisor.report()
    assert isinstance(r1, CalibrationReport)

    def _strip_latency(obj):
        if isinstance(obj, dict):
            for k in list(obj):
                if k.endswith("_ms") or k.endswith("_delta_ms") or "latency" in k:
                    obj.pop(k, None)
                else:
                    _strip_latency(obj[k])
        elif isinstance(obj, list):
            for it in obj:
                _strip_latency(it)

    d1, d2 = r1.model_dump(mode="json"), r2.model_dump(mode="json")
    d1.pop("generated_at", None)
    d2.pop("generated_at", None)
    _strip_latency(d1)
    _strip_latency(d2)
    assert d1 == d2
    assert r1.model_dump_json()  # serializes


def test_report_current_point_matches_real_pipeline(advisor):
    """The cached-detector engine must reproduce the real decisions."""
    from evaluation.evaluation import evaluate

    real = evaluate().model_dump()
    cur = advisor.current_operating_point()
    # human-review rate and abstention are computed identically in both places
    assert cur.human_review_rate == pytest.approx(real["human_review_rate"], abs=1e-3)
    assert cur.abstention_rate == pytest.approx(real["abstention_rate"], abs=1e-3)
