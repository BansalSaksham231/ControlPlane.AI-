"""
Phase 6 — Step 3: Explainability UI integration.

Drives the existing Streamlit app via ``streamlit.testing`` AppTest and
checks that the new "Why did ControlPlane decide this?" section is
rendered from the ``ExplainabilitySummary`` produced by
``build_explanation(trace)`` — same pipeline, better explanation.
"""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from explainability.schemas import ExplainabilitySummary  # noqa: E402

TIMEOUT = 240
_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
APP = str(_APP_SRC)


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


def _check_scenario(at, letter: str):
    _click(at, lambda l: l.startswith(f"{letter} "))
    at.run()
    _click(at, lambda l: "CHECK" in l)
    at.run()
    assert not at.exception


def _all_text(at) -> str:
    parts: list[str] = []
    for coll in (at.markdown, at.info, at.success, at.warning, at.error, at.caption):
        for el in coll:
            parts.append(str(el.value))
    for m in at.metric:
        parts.append(f"{m.label} = {m.value}")
    for df in at.dataframe:
        try:
            parts.append(df.value.to_csv(index=False))
        except Exception:  # pragma: no cover - defensive
            parts.append(str(df.value))
    return "\n".join(parts)


# ------------------------------------------------------------------
# 1-3. scenario decisions
# ------------------------------------------------------------------


def test_clean_scenario_renders_allow(app):
    _check_scenario(app, "A")
    assert app.session_state["result_decision"] == "ALLOW"
    assert app.session_state["result_explanation"].decision.value == "ALLOW"


def test_hallucination_scenario_renders_verify(app):
    _check_scenario(app, "B")
    assert app.session_state["result_decision"] == "VERIFY"
    assert app.session_state["result_explanation"].decision.value == "VERIFY"


def test_pii_scenario_renders_block(app):
    _check_scenario(app, "C")
    assert app.session_state["result_decision"] == "BLOCK"
    assert app.session_state["result_explanation"].decision.value == "BLOCK"


# ------------------------------------------------------------------
# 4. risk + confidence displayed
# ------------------------------------------------------------------


def test_risk_and_confidence_are_displayed(app):
    _check_scenario(app, "B")
    labels = {m.label for m in app.metric}
    assert {"Overall risk", "Confidence", "Verification", "Decision"} <= labels
    text = _all_text(app)
    assert "Overall risk = " in text and "%" in text


# ------------------------------------------------------------------
# 5-6. FAST / DEEP path
# ------------------------------------------------------------------


def test_fast_path_is_displayed(app):
    _check_scenario(app, "A")
    assert app.session_state["result_explanation"].verification_path.value == "FAST"
    assert "FAST verification" in _all_text(app)


def test_deep_path_is_displayed(app):
    _check_scenario(app, "C")
    summary = app.session_state["result_explanation"]
    assert summary.verification_path.value == "DEEP"
    text = _all_text(app)
    assert "DEEP verification" in text
    assert "Preliminary risk" in {m.label for m in app.metric} or "Preliminary risk" in text
    for reason in summary.verification_summary.deep_trigger_reasons:
        assert reason in text


# ------------------------------------------------------------------
# 7. reason codes appear
# ------------------------------------------------------------------


def test_reason_codes_appear(app):
    _check_scenario(app, "C")
    summary = app.session_state["result_explanation"]
    assert summary.primary_reasons
    text = _all_text(app)
    for code in summary.primary_reasons:
        assert code in text


# ------------------------------------------------------------------
# 8. evidence section
# ------------------------------------------------------------------


def test_evidence_section_appears(app):
    _check_scenario(app, "B")
    expander_labels = [e.label for e in app.expander]
    assert any(lbl.startswith("Evidence & Claims") for lbl in expander_labels)


# ------------------------------------------------------------------
# 9-10. consequence + criticality
# ------------------------------------------------------------------


def test_consequence_and_criticality_appear(app):
    _check_scenario(app, "D")
    summary = app.session_state["result_explanation"]
    text = _all_text(app)
    assert "CONSEQUENCE" in text and "CRITICALITY" in text
    assert "Consequence if wrong" in text  # the probability != consequence framing
    assert f"{summary.consequence_summary.consequence_score:.2f}" in text
    assert "Max claim criticality" in {m.label for m in app.metric}


# ------------------------------------------------------------------
# 11. decision path
# ------------------------------------------------------------------


def test_decision_path_appears(app):
    _check_scenario(app, "C")
    summary = app.session_state["result_explanation"]
    text = _all_text(app)
    assert "Decision path" in text
    chain = [summary.decision_path[0].from_tier.value] + [
        s.to_tier.value for s in summary.decision_path
    ]
    assert "  →  ".join(chain) in text


# ------------------------------------------------------------------
# 12. human review prominence
# ------------------------------------------------------------------


def test_human_review_is_prominent_when_required(app):
    _check_scenario(app, "C")
    errors = " ".join(str(e.value) for e in app.error)
    assert "HUMAN REVIEW REQUIRED" in errors


def test_no_human_review_message_when_not_required(app):
    _check_scenario(app, "A")
    successes = " ".join(str(s.value) for s in app.success)
    assert "No human review required." in successes


# ------------------------------------------------------------------
# 13. PII is not leaked
# ------------------------------------------------------------------


def test_pii_not_leaked_in_ui(app):
    _check_scenario(app, "C")
    trace = app.session_state["result_trace"]
    summary: ExplainabilitySummary = app.session_state["result_explanation"]

    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans, "scenario C must produce PII findings"

    blob = summary.model_dump_json()
    rendered = _all_text(app)
    for span in raw_spans:
        assert span not in blob, f"{span!r} leaked into the ExplainabilitySummary"
        assert span not in rendered, f"{span!r} leaked into the rendered result screen"

    assert "matched_text" not in blob
    assert "matched_text" not in rendered


# ------------------------------------------------------------------
# 14. ExplainabilitySummary is the UI source
# ------------------------------------------------------------------


def test_ui_uses_explainability_summary(app):
    _check_scenario(app, "D")
    trace = app.session_state["result_trace"]
    summary = app.session_state["result_explanation"]
    assert isinstance(summary, ExplainabilitySummary)
    assert summary.decision == trace.final_decision.decision
    assert summary.overall_risk == trace.final_decision.overall_risk
    assert summary.decision_confidence == trace.final_decision.decision_confidence
    assert summary.verification_path.value == trace.verification_path


def test_app_source_calls_build_explanation():
    src = _APP_SRC.read_text(encoding="utf-8")
    assert "from explainability.builder import build_explanation" in src
    assert "build_explanation(trace)" in src
    # the UI must not re-run detectors / the engine to build the explanation
    assert "PerformanceDetector" not in src
    assert "DecisionEngine(" not in src


# ------------------------------------------------------------------
# 15. existing functionality still works
# ------------------------------------------------------------------


def test_existing_tabs_still_work(app):
    # load + check keeps the old detector deep-dive and downstream tabs alive
    _check_scenario(app, "B")
    text = _all_text(app)
    assert "Detector deep-dive" in text
    # simulate + counterfactual buttons still present and runnable
    labels = [b.label for b in app.button]
    assert "Simulate" in labels and "Compare" in labels
    _click(app, lambda l: l == "Simulate")
    app.run()
    assert not app.exception
    _click(app, lambda l: l == "Compare")
    app.run()
    assert not app.exception


def test_monitoring_tab_still_populates(app):
    _click(app, lambda l: "Populate" in l)
    app.run()
    assert not app.exception
    interactions = [m.value for m in app.metric if m.label == "Interactions"]
    assert interactions and int(interactions[0]) >= 150


# ------------------------------------------------------------------
# ground-truth isolation
# ------------------------------------------------------------------


def test_ui_source_has_no_ground_truth_access():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome"):
        assert token not in src


# ------------------------------------------------------------------
# multi-turn session memory section
# ------------------------------------------------------------------


def test_session_memory_section_renders_for_a_non_critical_multi_turn_session(app):
    # scenario B (hallucination -> VERIFY) is never a critical violation *on
    # a fresh session*. This module-scoped ``app`` fixture is shared with
    # earlier tests in this file that also check scenario B on its default
    # "SESSION-SCEN-B" id (see test_hallucination_scenario_renders_verify /
    # test_risk_and_confidence_are_displayed / test_evidence_section_appears /
    # test_existing_tabs_still_work) — by this point in the file that shared
    # session has already accumulated enough high-risk turns to have
    # legitimately escalated to BLOCK (multi-turn compounding risk is a
    # real, intended feature: see session/manager.py). So this test uses its
    # own private session id to observe a genuinely fresh, non-critical
    # multi-turn session in isolation.
    #
    # session_memory only populates once a turn's *prior* history is itself
    # multi-turn (turns_recorded > 1 prior turns) when there is no critical
    # history to short-circuit that gate (see explainability/builder.py
    # ``_session_memory`` and the equivalent non-UI contract in
    # tests/test_explainability.py::test_session_memory_populated_after_a_critical_turn),
    # so this needs a third turn on the same session, not two.
    _click(app, lambda l: l.startswith("B "))
    app.run()
    for ti in app.text_input:
        if ti.key == "f_session_id":
            ti.set_value("SESSION-TEST-NONCRITICAL-B")
    app.run()
    for _ in range(3):
        _click(app, lambda l: "CHECK" in l)
        app.run()
        assert not app.exception
    mem = app.session_state["result_explanation"].session_memory
    assert mem is not None and mem.turns_recorded > 1
    assert mem.has_critical_history is False          # no BLOCK / critical-PII turn
    assert mem.critical_floor == 0.0
    assert mem.critical_floor_applied is False        # -> no critical callout for this turn
    text = _all_text(app)
    assert "Multi-Turn Session Memory" in text        # rendered: turns_recorded > 1
    assert "Turns in memory = " in text


def test_session_memory_section_appears_after_a_critical_turn(app):
    # scenario C carries session_id SESSION-SCEN-C — check it twice so the
    # second decision inherits the first turn's critical-PII history.
    _check_scenario(app, "C")
    _check_scenario(app, "C")

    summary = app.session_state["result_explanation"]
    mem = summary.session_memory
    assert mem is not None
    assert mem.has_critical_history is True
    assert any(e.trigger == "CRITICAL_PII" for e in mem.critical_events)

    text = _all_text(app)
    assert "Multi-Turn Session Memory" in text
    assert "non-decaying" in text.lower()                 # floor surfaced (applied or history)
    assert "Turns in memory = " in text
    assert "Critical events timeline" in text
    # redacted PII entity chips are shown, raw PII is not
    assert mem.pii_entity_keys
    for raw in ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta"):
        assert raw not in text


def test_session_memory_render_is_wired_into_explanation():
    src = _APP_SRC.read_text(encoding="utf-8")
    assert "def render_session_memory" in src
    assert "render_session_memory(summary.session_memory)" in src
