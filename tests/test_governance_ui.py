"""Phase 9 — Governance Intelligence Streamlit section (§8, §25-27)."""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from governance.schemas import GovernanceIntelligenceReport  # noqa: E402

_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
_APP = str(_APP_SRC)
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(_APP, default_timeout=300)
    at.run()
    assert not at.exception
    for b in at.button:
        if b.label == "Populate demo operational traffic" and b.key == "gov_populate":
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


def test_governance_section_renders(app):
    t = _text(app)
    for section in (
        "Governance Intelligence",
        "Executive governance overview",
        "Application comparison",
        "Policy intelligence",
        "Reviewer signals",
        "Governance insights",
        "Trend monitoring",
        "Calibration recommendations",
    ):
        assert section in t, section


def test_ui_uses_the_governance_report(app):
    rep = app.session_state["gov_report"]
    assert isinstance(rep, GovernanceIntelligenceReport)
    assert rep.overview.traffic.total_interactions > 0
    # a metric rendered from the report
    interactions = [m.value for m in app.metric if m.label == "Interactions"]
    assert interactions and int(interactions[0]) == rep.overview.traffic.total_interactions


def test_automated_vs_reviewer_vs_ground_truth_is_visually_explicit(app):
    warnings = " ".join(str(w.value) for w in app.warning).lower()
    assert "automated decision" in warnings and "ground truth" in warnings


def test_recommendation_only_label_is_prominent(app):
    errors = " ".join(str(e.value) for e in app.error)
    assert "RECOMMENDATION ONLY — NOT APPLIED TO PRODUCTION." in errors


def test_ui_governance_has_no_pii():
    at = AppTest.from_file(_APP, default_timeout=300)
    at.run()
    for b in at.button:
        if b.key == "gov_populate":
            b.click()
            break
    at.run()
    assert not at.exception
    rendered = _text(at)
    blob = at.session_state["gov_report"].model_dump_json()
    for needle in _KNOWN_PII:
        assert needle not in rendered, needle
        assert needle not in blob, needle
    assert "matched_text" not in rendered
    assert "matched_text" not in blob


def test_ui_source_has_no_ground_truth_or_evaluation():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome",
                  "import evaluation", "from evaluation"):
        assert token not in src, token


def test_existing_tabs_still_work(app):
    # Check tab
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
    # Command Center still present
    assert "Enterprise AI Risk Command Center" in _text(app)
