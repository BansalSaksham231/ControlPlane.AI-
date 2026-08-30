"""
Phase 11 — Enterprise Command Center views (read-only presentation layer).

No new risk formula, no pipeline re-run, no ground truth, no raw PII,
deterministic.
"""

from __future__ import annotations

import copy
import pathlib

import pytest

from api.service import ControlPlaneService
from data.schemas import InterventionTier
from enterprise.schemas import CommandCenterView
from enterprise.service import EnterpriseService
from settings import load_settings
from tests import scenarios

_ENT_DIR = pathlib.Path(__file__).resolve().parent.parent / "enterprise"
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def svc():
    s = ControlPlaneService(fit_cost_baseline=True)
    s.populate_operational_demo(200)
    for t in [t for t in s.all_traces() if t.final_decision.decision is InterventionTier.BLOCK][:3]:
        s.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        s.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="hr only", reviewer_decision="HUMAN_REVIEW",
        )
    return s


@pytest.fixture(scope="module")
def view(svc) -> CommandCenterView:
    return svc.command_center_view()


# ------------------------------------------------------------------ empty state


def test_empty_state():
    s = ControlPlaneService(fit_cost_baseline=False)
    cc = s.command_center_view()
    assert cc.kpi.has_data is False
    assert cc.executive_summary.has_data is False
    assert cc.application_posture == []
    assert cc.recent_decisions == []


# ------------------------------------------------------------------ KPI consistency


def test_kpi_matches_monitoring_snapshot(svc, view):
    s = svc.get_operational_monitoring().snapshot
    k = view.kpi
    assert k.total_interactions == s.total_interactions
    assert k.allow_rate == s.allow_rate
    assert k.block_rate == s.block_rate
    assert k.fast_rate == s.fast_path_rate
    assert k.deep_rate == s.deep_path_rate
    # rates bounded
    for r in (k.allow_rate, k.verify_rate, k.human_review_rate, k.block_rate,
              k.incident_rate, k.override_rate, k.average_risk, k.average_confidence):
        assert 0.0 <= r <= 1.0


def test_no_fabricated_metrics(svc, view):
    # every KPI must be reconstructable from stored traces
    traces = svc.all_traces()
    manual_avg = sum(t.final_decision.overall_risk for t in traces) / len(traces)
    assert view.kpi.average_risk == pytest.approx(manual_avg, abs=1e-4)
    manual_block = sum(
        1 for t in traces if t.final_decision.decision is InterventionTier.BLOCK
    ) / len(traces)
    assert view.kpi.block_rate == pytest.approx(manual_block, abs=1e-4)


# ------------------------------------------------------------------ risk posture / heatmap


def test_risk_posture_from_traces(svc, view):
    traces = svc.all_traces()
    rp = view.risk_posture
    manual_perf = sum(t.final_decision.performance_risk for t in traces) / len(traces)
    assert rp.performance_average == pytest.approx(manual_perf, abs=1e-4)
    assert rp.dominant_dimension in ("performance", "responsibility", "cost")
    assert rp.overall_high_risk_count >= 0


def test_heatmap_shape_and_values(view):
    hm = view.heatmap
    assert hm.dimensions == ["performance", "responsibility", "cost", "consequence"]
    assert len(hm.cells) == len(hm.applications) * 4
    for c in hm.cells:
        assert c.value is None or 0.0 <= c.value <= 1.0
    assert "No new risk formula" in hm.note


# ------------------------------------------------------------------ application matrix


def test_application_posture(svc, view):
    apps = {r.application for r in view.application_posture}
    assert apps <= {"customer_support", "internal_knowledge_assistant", "decision_support"}
    total = sum(r.interactions for r in view.application_posture)
    assert total == len(svc.all_traces())
    for r in view.application_posture:
        assert r.posture in ("LOW", "MODERATE", "HIGH")
        assert r.posture_rationale
        for v in (r.average_risk, r.high_risk_rate, r.human_review_rate, r.deep_rate):
            assert 0.0 <= v <= 1.0
        assert r.override_rate is None or 0.0 <= r.override_rate <= 1.0


def test_override_rate_comes_from_governance(svc, view):
    gov = svc.governance_report().application_comparison.applications
    gov_by_app = {a.application: a.reviewer_override_rate for a in gov}
    seen = 0
    for r in view.application_posture:
        assert r.override_rate == gov_by_app.get(r.application)
        seen += r.override_rate is not None
    # the fixture records reviewer MODIFY_DECISION actions, so at least one app
    # has a measured disagreement rate
    assert seen >= 1


def test_recommended_posture_uses_adaptive(svc, view):
    adaptive = svc.adaptive_report()
    rec_apps = {r.application for r in adaptive.recommendations if r.application}
    for row in view.application_posture:
        if row.recommended_posture is not None:
            assert row.application in rec_apps
            assert row.recommended_posture in (
                "REVIEW_POLICY", "REVIEW_VERIFICATION_THRESHOLD", "REVIEW_APPLICATION",
                "REVIEW_DETECTOR", "INVESTIGATE_DRIFT",
            )


# ------------------------------------------------------------------ executive summary


def test_executive_summary(view):
    es = view.executive_summary
    assert es.has_data is True
    assert es.interactions_evaluated == view.kpi.total_interactions
    assert es.ai_systems_monitored == len(view.application_posture)
    assert {b.posture for b in es.application_posture} <= {"LOW", "MODERATE", "HIGH"}
    assert "No production changes" in es.safety_status


# ------------------------------------------------------------------ live decision feed


def test_recent_decisions_are_stored_traces_newest_first(view):
    rows = view.recent_decisions
    assert rows
    ts = [r.timestamp for r in rows]
    assert ts == sorted(ts, reverse=True)
    for r in rows:
        assert r.source == "STORED_TRACE"


# ------------------------------------------------------------------ governance timeline


def test_governance_timeline(svc):
    tl = svc.governance_timeline()
    assert tl.events
    types = [e.event_type for e in tl.events]
    assert "DECISION" in types
    assert any(t in types for t in ("REVIEWER_FEEDBACK", "PATTERN", "RECOMMENDATION"))
    orders = [e.order for e in tl.events]
    assert orders == sorted(orders)
    for e in tl.events:
        if e.timestamp is None:
            assert "causal workflow order" in e.timestamp_note


# ------------------------------------------------------------------ determinism


def test_command_center_is_deterministic(svc):
    a = svc.command_center_view().model_dump(mode="json")
    b = svc.command_center_view().model_dump(mode="json")

    def _strip(n):
        if isinstance(n, dict):
            return {k: _strip(v) for k, v in n.items()
                    if not (k.endswith("_ms") or k == "generated_at" or k == "timestamp")}
        if isinstance(n, list):
            return [_strip(x) for x in n]
        return n

    assert _strip(a) == _strip(b)


# ------------------------------------------------------------------ safety / architecture guards


def test_enterprise_never_imports_evaluation_or_ground_truth():
    for path in _ENT_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import evaluation" not in text
        assert "from evaluation" not in text
        assert "EvaluationCase" not in text
        for token in ("ground_truth_", "expected_decision", "final_outcome",
                      "actual_correctness"):
            assert token not in text, f"{path.name}: {token}"


def test_enterprise_never_writes_config():
    for path in _ENT_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("yaml.dump", "yaml.safe_dump", ".write_text(", "'w')", '"w")',
                      "settings.yaml"):
            assert token not in text, f"{path.name}: {token}"


def test_enterprise_does_not_orchestrate_pipeline():
    banned = (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "RiskFusionEngine(", "PolicyEngine(",
        ".detect(", ".evaluate(", ".fuse_scores(", ".route(", ".assess(",
    )
    for path in _ENT_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name}: {token}"


def test_command_center_does_not_rerun_pipeline(svc, monkeypatch):
    import decision.engine as dec_mod
    import detectors.performance.detector as perf_mod
    import policy.engine as pol_mod

    def boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("enterprise view re-ran a pipeline component")

    monkeypatch.setattr(perf_mod.PerformanceDetector, "detect", boom)
    monkeypatch.setattr(dec_mod.DecisionEngine, "evaluate", boom)
    monkeypatch.setattr(pol_mod.PolicyEngine, "decide", boom)
    cc = EnterpriseService(svc).command_center()
    assert cc.kpi.total_interactions == len(svc.all_traces())
    tl = EnterpriseService(svc).governance_timeline()
    assert tl.events


def test_no_raw_pii_in_command_center(view):
    blob = view.model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in blob
    assert "matched_text" not in blob and "ground_truth" not in blob


def test_command_center_does_not_change_config(svc):
    before = copy.deepcopy(load_settings())
    svc.command_center_view()
    svc.governance_timeline()
    svc.application_posture()
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


# ------------------------------------------------------------------ scenarios A-J


def test_core_scenarios_unchanged():
    from evaluation.evaluation import build_engine

    engine = build_engine()
    got = {
        k: engine.evaluate(f(), timestamp=None, record_session=False).final_decision.decision.value
        for k, f in {
            "A": scenarios.scenario_a_clean, "B": scenarios.scenario_b_hallucination,
            "C": scenarios.scenario_c_pii, "D": scenarios.scenario_d_high_consequence,
            "E": scenarios.scenario_e_multi_risk,
        }.items()
    }
    assert got == {"A": "ALLOW", "B": "VERIFY", "C": "BLOCK", "D": "VERIFY", "E": "BLOCK"}
