"""Phase 10 — 🧠 Adaptive Intelligence Streamlit tab."""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from adaptive.schemas import AdaptiveGovernanceReport  # noqa: E402

_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
_APP = str(_APP_SRC)
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(_APP, default_timeout=300)
    at.run()
    assert not at.exception
    for b in at.button:
        if b.key == "ad_populate":
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
    for t in getattr(at, "text", []):
        parts.append(str(t.value))
    return "\n".join(parts)


def test_tab_loads(app):
    t = _text(app)
    for s in ("Adaptive Intelligence", "Adaptive executive summary", "Incident patterns",
              "Drift", "Root-cause / attribution", "Recommendations", "Approval"):
        assert s in t, s


def test_uses_the_adaptive_report(app):
    rep = app.session_state["adaptive_report"]
    assert isinstance(rep, AdaptiveGovernanceReport)
    assert rep.production_configuration_status == "UNCHANGED"


def test_patterns_and_drift_render(app):
    t = _text(app)
    rep = app.session_state["adaptive_report"]
    if rep.top_patterns:
        assert rep.top_patterns[0]["type"] in t
    assert "not proof of model degradation" in t


def test_recommendation_and_no_deploy_language(app):
    errors = " ".join(str(e.value) for e in app.error)
    assert "RECOMMENDATION ONLY" in errors
    assert "no auto-deployment path" in errors
    assert "config/settings.yaml` is never modified" in errors
    labels = [b.label for b in app.button]
    assert "DEPLOY NOW" not in labels
    assert any("APPROVE FOR EVALUATION" in l for l in labels)


def test_approval_works_and_shows_not_applied(app):
    for b in app.button:
        if "APPROVE FOR EVALUATION" in b.label:
            b.click()
            break
    app.run()
    assert not app.exception
    successes = " ".join(str(s.value) for s in app.success)
    assert "Approved for evaluation only" in successes
    assert "Production configuration remains unchanged" in successes


def test_no_raw_pii_in_adaptive_ui(app):
    rendered = _text(app)
    blob = app.session_state["adaptive_report"].model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in rendered, needle
        assert needle not in blob, needle
    assert "matched_text" not in rendered and "matched_text" not in blob


def test_ui_source_has_no_ground_truth_or_evaluation():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome",
                  "import evaluation", "from evaluation"):
        assert token not in src, token


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
    assert "Enterprise AI Risk Command Center" in t
    assert "Governance Intelligence" in t
