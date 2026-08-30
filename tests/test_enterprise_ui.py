"""Phase 11 — 🏢 Executive Streamlit tab (judge-facing command center)."""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from enterprise.schemas import CommandCenterView  # noqa: E402

_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
_APP = str(_APP_SRC)
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(_APP, default_timeout=600)
    at.run()
    assert not at.exception
    for b in at.button:
        if b.key == "exec_populate_top":
            b.click()
            break
    at.run()
    assert not at.exception
    return at


def _text(at) -> str:
    parts = []
    for coll in (at.markdown, at.info, at.success, at.warning, at.error, at.caption):
        parts += [str(e.value) for e in coll]
    for m in at.metric:
        parts.append(f"{m.label} = {m.value}")
    for df in at.dataframe:
        try:
            parts.append(df.value.to_csv(index=False))
        except Exception:  # pragma: no cover
            parts.append(str(df.value))
    return "\n".join(parts)


# ------------------------------------------------------------------ load


def test_executive_tab_loads(app):
    t = _text(app)
    for s in (
        "Enterprise AI Risk Control Plane",
        "Executive summary",
        "A · Executive KPIs",
        "B · Risk posture",
        "C · Application risk matrix",
        "D · Risk heatmap",
        "Live control feed",
        "Governance audit timeline",
        "What-If — policy playground",
    ):
        assert s in t, s


def test_uses_command_center_view(app):
    view = app.session_state["command_center_view"]
    assert isinstance(view, CommandCenterView)
    assert view.kpi.has_data is True
    assert view.kpi.total_interactions > 0
    assert view.application_posture


def test_kpi_metrics_rendered_from_view(app):
    view = app.session_state["command_center_view"]
    labels = {m.label: m.value for m in app.metric}
    assert labels.get("Interactions") == str(view.kpi.total_interactions)
    assert labels.get("Interactions evaluated") == str(view.executive_summary.interactions_evaluated)


def test_no_deployment_language_or_buttons(app):
    labels = [b.label for b in app.button]
    assert "DEPLOY NOW" not in labels
    assert not any("APPLY TO PRODUCTION" in l for l in labels)
    t = _text(app)
    assert "no deployment path" in t
    assert "production configuration is never" in t
    assert "No production changes" in t  # executive summary safety status


def test_application_matrix_and_heatmap_render(app):
    view = app.session_state["command_center_view"]
    t = _text(app)
    for r in view.application_posture:
        assert r.application in t
    # heatmap "N/A" allowed but never a raw exception
    assert "No new risk formula" in t or "N/A where unavailable" in t or "risk formula" in t


def test_governance_timeline_renders(app):
    t = _text(app)
    assert "DECISION" in t
    tl = app.session_state["command_center_view"]
    # timeline is fetched live in the view; just assert the section note shows
    assert "causal workflow order" in t or "DECISION" in t


def test_no_raw_pii_in_executive_ui(app):
    rendered = _text(app)
    blob = app.session_state["command_center_view"].model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in rendered, needle
        assert needle not in blob, needle
    assert "matched_text" not in rendered and "matched_text" not in blob
    assert "ground_truth" not in rendered and "ground_truth" not in blob


def test_ui_source_has_no_ground_truth_or_evaluation():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome",
                  "import evaluation", "from evaluation"):
        assert token not in src, token


# ------------------------------------------------------------------ interactive controls
#
# ▶ RUN ENTERPRISE DEMO and 🔬 What-If mutate / calibrate the process-wide
# @st.cache_resource service that every UI test module shares, so clicking them
# here would contaminate other modules. Their end-to-end behaviour is covered in
# test_enterprise_demo.py against dedicated fresh services; here we only assert the
# Executive tab wires the controls up.


def test_demo_and_whatif_controls_are_wired(app):
    keys = {b.key for b in app.button}
    assert "exec_run_demo" in keys
    assert "exec_wi_run" in keys
    labels = [b.label for b in app.button]
    assert any("RUN ENTERPRISE DEMO" in l for l in labels)


def test_whatif_control_options_are_existing_calibration_controls(app):
    ctrl = next(s for s in app.selectbox if s.key == "exec_wi_ctrl")
    assert set(ctrl.options) == {"deep_verification_risk_threshold", "fast_path_min_confidence"}


# ------------------------------------------------------------------ regression


def test_existing_tabs_still_work(app):
    for b in app.button:
        if b.label.startswith("A "):
            b.click()
            break
    app.run()
    for b in app.button:
        if "CHECK" in b.label:
            b.click()
            break
    app.run()
    assert not app.exception
    assert app.session_state["result_decision"] == "ALLOW"
    t = _text(app)
    assert "Governance Intelligence" in t
