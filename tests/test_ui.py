"""
Streamlit UI smoke tests via streamlit.testing AppTest.

Verifies the UI renders and that clicking a scenario + CHECK runs the
real pipeline and shows a decision. Skipped automatically if the
streamlit testing harness is unavailable.
"""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py")
TIMEOUT = 240


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    return at


def _click(at, predicate):
    for button in at.button:
        if predicate(button.label):
            button.click()
            return
    raise AssertionError("button not found")


def test_app_loads(app):
    assert not app.exception
    labels = [b.label for b in app.button]
    assert any(l.startswith("A ") for l in labels)
    assert any("CHECK" in l for l in labels)


def test_clean_scenario_allows(app):
    _click(app, lambda l: l.startswith("A "))
    app.run()
    _click(app, lambda l: "CHECK" in l)
    app.run()
    assert not app.exception
    assert app.session_state["result_decision"] == "ALLOW"


def test_pii_scenario_blocks(app):
    _click(app, lambda l: l.startswith("C "))
    app.run()
    _click(app, lambda l: "CHECK" in l)
    app.run()
    assert not app.exception
    assert app.session_state["result_decision"] == "BLOCK"
    human_review_metrics = [m.value for m in app.metric if m.label == "Human review"]
    assert "YES" in human_review_metrics


def test_monitoring_populate(app):
    _click(app, lambda l: "Populate" in l)
    app.run()
    assert not app.exception
    interactions_metric = [m.value for m in app.metric if m.label == "Interactions"]
    assert interactions_metric and int(interactions_metric[0]) >= 150


def test_policy_simulation_and_counterfactual_render(app):
    # load a scenario + check so the sim/counterfactual tabs have an interaction
    _click(app, lambda l: l.startswith("B "))
    app.run()
    _click(app, lambda l: "CHECK" in l)
    app.run()
    assert not app.exception
    labels = [b.label for b in app.button]
    assert any("Simulate" == l for l in labels)
    assert any("Compare" == l for l in labels)

    _click(app, lambda l: l == "Simulate")
    app.run()
    assert not app.exception
    _click(app, lambda l: l == "Compare")
    app.run()
    assert not app.exception
