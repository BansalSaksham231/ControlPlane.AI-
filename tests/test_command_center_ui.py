"""
Phase 8 — Step 2: Enterprise AI Risk Command Center (Streamlit).

Every displayed value is read from the OperationalMonitoringReport
produced by the monitoring backend (stashed in
``session_state["cc_report"]``). The UI recomputes nothing and re-runs no
pipeline component.
"""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from evaluation.evaluation import build_engine  # noqa: E402  (test-only import)
from monitoring.schemas import OperationalMonitoringReport  # noqa: E402
from tests import scenarios  # noqa: E402

TIMEOUT = 300
_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
APP = str(_APP_SRC)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    # populate operational demo traffic
    for b in at.button:
        if "Populate" in b.label:
            b.click()
            break
    at.run()
    assert not at.exception
    return at


@pytest.fixture(scope="module")
def report(app) -> OperationalMonitoringReport:
    r = app.session_state["cc_report"]
    assert isinstance(r, OperationalMonitoringReport)
    return r


def _metric(at, label: str):
    vals = [m.value for m in at.metric if m.label == label]
    return vals[0] if vals else None


def _all_text(at) -> str:
    parts = []
    for coll in (at.markdown, at.info, at.success, at.warning, at.error, at.caption):
        for el in coll:
            parts.append(str(el.value))
    for m in at.metric:
        parts.append(f"{m.label} = {m.value}")
    for df in at.dataframe:
        try:
            parts.append(df.value.to_csv(index=False))
        except Exception:  # pragma: no cover
            parts.append(str(df.value))
    return "\n".join(parts)


# ------------------------------------------------------------------
# 1-3. renders / empty state / population
# ------------------------------------------------------------------


def test_command_center_renders(app):
    assert not app.exception
    text = _all_text(app)
    assert "Enterprise AI Risk Command Center" in text
    assert "Incident command" in text


def test_demo_population_records_traffic(report):
    assert report.total_interactions >= 150


def test_empty_state_message_exists_in_source():
    src = _APP_SRC.read_text(encoding="utf-8")
    assert "No operational traffic yet." in src


# ------------------------------------------------------------------
# 4. executive metrics match the report
# ------------------------------------------------------------------


def test_executive_metrics_match_report(app, report):
    s = report.snapshot
    assert _metric(app, "Interactions") == str(s.total_interactions)
    assert _metric(app, "Average risk") == f"{s.average_risk:.3f}"
    assert _metric(app, "P95 risk") == f"{s.p95_risk:.3f}"
    assert _metric(app, "Human review") == f"{s.human_review_rate * 100:.1f}%"
    assert _metric(app, "Block") == f"{s.block_rate * 100:.1f}%"
    assert _metric(app, "FAST-path") == f"{s.fast_path_rate * 100:.1f}%"
    assert _metric(app, "Incident rate") == f"{report.incident_digest.incident_rate * 100:.1f}%"


def test_risk_confidence_distinction_is_shown(app):
    assert "Risk ≠ confidence" in _all_text(app)


# ------------------------------------------------------------------
# 5. application table matches report
# ------------------------------------------------------------------


def test_application_table_matches_report(app, report):
    text = _all_text(app)
    assert report.applications, "expected application summaries"
    for a in report.applications:
        assert a.application in text
    # a per-application inspect metric is rendered from the first app's data
    assert "Observed risk" in {m.label for m in app.metric}


# ------------------------------------------------------------------
# 6-7. incidents
# ------------------------------------------------------------------


def test_incident_counts_match_report(app, report):
    d = report.incident_digest
    assert _metric(app, "Incidents") == str(d.total)
    assert _metric(app, "Critical") == str(d.by_severity.get("CRITICAL", 0))
    assert _metric(app, "High") == str(d.by_severity.get("HIGH", 0))
    assert _metric(app, "Medium") == str(d.by_severity.get("MEDIUM", 0))


def test_incident_severity_and_ids_appear(app, report):
    if not report.incidents:
        pytest.skip("no incidents in demo traffic")
    text = _all_text(app)
    top = report.incidents[0]
    assert top.interaction_id in text
    assert top.severity.value in text
    # deterministic ordering from the backend is preserved
    ranks = [
        {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}[i.severity.value]
        for i in report.incidents
    ]
    assert ranks == sorted(ranks)


def test_incident_drilldown_reaches_explainability(app, report):
    # Phase 8 Step 3: drill-down is now "select incident -> Investigate Incident",
    # which opens the investigation workspace (it embeds the explainability view).
    if not report.incidents:
        pytest.skip("no incidents in demo traffic")
    target = report.incidents[0].interaction_id
    sel = None
    for s in app.selectbox:
        if s.label.startswith("Select an incident"):
            sel = s
            break
    assert sel is not None
    sel.select(target)
    app.run()
    for b in app.button:
        if "Investigate Incident" in b.label:
            b.click()
            break
    app.run()
    assert not app.exception
    text = _all_text(app)
    assert "Incident Investigation" in text
    assert "Why did ControlPlane decide this?" in text     # reuses existing explainability
    assert "Decision path" in text


# ------------------------------------------------------------------
# 8. reason-code frequencies match report
# ------------------------------------------------------------------


def test_reason_code_frequencies_match_report(app, report):
    text = _all_text(app)
    for rc in report.reason_codes[:5]:
        assert rc.reason_code in text
        assert str(rc.count) in text


# ------------------------------------------------------------------
# 9. FAST/DEEP values match report
# ------------------------------------------------------------------


def test_fast_deep_values_match_report(app, report):
    v = report.verification
    assert _metric(app, "FAST") == f"{v.fast_rate * 100:.1f}%"
    assert _metric(app, "DEEP") == f"{v.deep_rate * 100:.1f}%"
    text = _all_text(app)
    for trig in list(v.deep_trigger_reason_counts)[:3]:
        assert trig in text


# ------------------------------------------------------------------
# 10-11. trends + operational shifts
# ------------------------------------------------------------------


def test_trend_values_match_report(app, report):
    text = _all_text(app)
    for mt in report.trend.metrics:
        assert mt.metric in text
    assert report.trend.method in text
    # arrow indicators present
    assert ("↑" in text) or ("↓" in text) or ("→" in text)


def test_operational_shift_values_match_report(app, report):
    text = _all_text(app)
    assert "Operational shifts" in text
    assert "not AI/model-drift" in text.lower() or "NOT AI/model-drift" in text
    for sh in report.operational_shift.shifts:
        assert sh.metric in text
    assert "drift" in report.operational_shift.disclaimer.lower()


# ------------------------------------------------------------------
# 12. feedback values match report
# ------------------------------------------------------------------


def test_feedback_values_match_report(app, report):
    fb = report.feedback
    assert _metric(app, "Feedback count") == str(fb.feedback_count)
    assert _metric(app, "Approved") == str(fb.approved)
    assert _metric(app, "Rejected") == str(fb.rejected)
    assert "not ground truth" in _all_text(app)


# ------------------------------------------------------------------
# 13. PII never appears
# ------------------------------------------------------------------


def test_pii_does_not_appear(app, report):
    engine = build_engine()
    it = scenarios.scenario_c_pii()
    trace = engine.evaluate(it, timestamp=it.timestamp, record_session=False)
    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans

    rendered = _all_text(app)
    blob = report.model_dump_json()
    for span in raw_spans:
        assert span not in rendered
        assert span not in blob
    assert it.response not in rendered
    assert "matched_text" not in blob


# ------------------------------------------------------------------
# 14-15. ground-truth isolation + no pipeline engines in the UI
# ------------------------------------------------------------------


def test_ui_source_has_no_ground_truth_or_evaluation():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome",
                  "import evaluation", "from evaluation"):
        assert token not in src, token


def test_ui_does_not_instantiate_pipeline_engines():
    src = _APP_SRC.read_text(encoding="utf-8")
    # the UI obtains the report via the service seam, it does not build one
    assert "OperationalMonitor(" not in src
    for banned in (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "RiskFusionEngine(", "PolicyEngine(",
    ):
        assert banned not in src, banned


def test_monitoring_backend_is_the_source(app):
    # the report the UI rendered is exactly the backend report object
    r = app.session_state["cc_report"]
    assert isinstance(r, OperationalMonitoringReport)
    assert "Observational view" in _all_text(app)


# ------------------------------------------------------------------
# 16-20. existing functionality still works
# ------------------------------------------------------------------


def _load_and_check(at, letter):
    for b in at.button:
        if b.label.startswith(f"{letter} "):
            b.click()
            break
    at.run()
    for b in at.button:
        if "CHECK" in b.label:
            b.click()
            break
    at.run()
    assert not at.exception


def test_existing_check_tab_still_works(app):
    _load_and_check(app, "A")
    assert app.session_state["result_decision"] == "ALLOW"


def test_existing_explainability_still_works(app):
    _load_and_check(app, "C")
    assert app.session_state["result_decision"] == "BLOCK"
    assert "Why did ControlPlane decide this?" in _all_text(app)


def test_existing_simulation_and_counterfactual_still_work(app):
    # use a low-risk scenario so this shared-fixture check never nudges a
    # borderline scenario's multi-turn session toward escalation.
    _load_and_check(app, "A")
    labels = [b.label for b in app.button]
    assert "Simulate" in labels and "Compare" in labels
    for b in app.button:
        if b.label == "Simulate":
            b.click()
            break
    app.run()
    assert not app.exception
    for b in app.button:
        if b.label == "Compare":
            b.click()
            break
    app.run()
    assert not app.exception


def test_existing_audit_tab_still_works(app):
    _load_and_check(app, "D")
    iid = app.session_state["result_trace_id"]
    for ti in app.text_input:
        if ti.label == "Interaction ID":
            ti.set_value(iid)
            break
    app.run()
    assert not app.exception
    assert iid in _all_text(app)


def test_command_center_still_populates_after_other_tabs(app):
    for b in app.button:
        if "Populate" in b.label:
            b.click()
            break
    app.run()
    assert not app.exception
    interactions = [m.value for m in app.metric if m.label == "Interactions"]
    assert interactions and int(interactions[0]) >= 150


# ------------------------------------------------------------------
# routing & compute efficiency + multi-turn critical floor
# ------------------------------------------------------------------


def test_routing_and_compute_efficiency_row_renders(app):
    # read the report fresh from session_state — earlier tests may re-populate
    v = app.session_state["cc_report"].verification
    text = _all_text(app)
    assert "Routing & compute efficiency" in text
    assert _metric(app, "FAST path") == f"{v.fast_rate * 100:.1f}%"
    assert _metric(app, "Semantic bypasses") == str(v.semantic_bypass_count)
    assert any(m.label == "Compute cycles saved by bypass" for m in app.metric)


def test_semantic_bypass_count_matches_report(app):
    v = app.session_state["cc_report"].verification
    if v.semantic_bypass_count:
        assert str(v.semantic_bypass_count) in _all_text(app)
        assert "semantic bypass" in _all_text(app).lower()


def test_multi_turn_critical_floor_section_is_wired():
    src = _APP_SRC.read_text(encoding="utf-8")
    assert "Sessions hitting critical floor" in src
    assert "r.multi_turn" in src
    assert "non-decaying critical floor" in src.lower()
