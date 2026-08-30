"""
Phase 8 — Step 3: incident investigation workspace in the Streamlit
Command Center (§35).

Command Center -> incident -> Investigate -> workspace (decision / risk /
confidence / verification / decision path) -> counterfactual -> reviewer
action -> governance history updates. Existing tabs keep working.
"""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

_APP_SRC = pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"
_APP = str(_APP_SRC)
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(_APP, default_timeout=300)
    at.run()
    assert not at.exception
    for b in at.button:
        if "Populate" in b.label:
            b.click()
            break
    at.run()
    assert not at.exception
    # select the first incident and open the investigation workspace
    for s in at.selectbox:
        if s.label.startswith("Select an incident"):
            s.select(s.options[1])
            break
    at.run()
    for b in at.button:
        if "Investigate Incident" in b.label:
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


def _metric(at, label):
    v = [m.value for m in at.metric if m.label == label]
    return v[0] if v else None


# ------------------------------------------------------------------
# workspace renders
# ------------------------------------------------------------------


def test_investigation_workspace_appears(app):
    t = _text(app)
    assert "🔎 Incident Investigation" in t
    assert "Executive assessment" in t
    assert "immutable" in t.lower()


def test_decision_risk_confidence_verification_appear(app):
    labels = {m.label for m in app.metric}
    assert "ControlPlane decision" in labels
    assert "Risk" in labels
    assert "Confidence" in labels
    assert "Verification" in labels
    assert "Human review" in labels


def test_why_and_decision_path_appear(app):
    t = _text(app)
    assert "Why did ControlPlane decide this?" in t     # embedded explainability
    assert "Decision path" in t
    assert "Probability of error" in t                  # consequence vs criticality framing


def test_governance_section_and_history_appear(app):
    t = _text(app)
    assert "Human Governance & Override" in t
    assert "Governance history" in t
    assert "Governance status" in t


def test_original_vs_effective_governed_decision_shown(app):
    labels = {m.label for m in app.metric}
    assert "Original ControlPlane decision" in labels
    assert "Effective governed decision" in labels
    tiers = ("ALLOW", "ANNOTATE", "VERIFY", "HUMAN_REVIEW", "BLOCK")
    assert _metric(app, "Original ControlPlane decision") in tiers
    assert _metric(app, "Effective governed decision") in tiers
    # with no override recorded yet, the two agree
    assert (
        _metric(app, "Original ControlPlane decision")
        == _metric(app, "Effective governed decision")
    )
    assert "immutable" in _text(app).lower()


# ------------------------------------------------------------------
# counterfactual works
# ------------------------------------------------------------------


def test_counterfactual_runs_and_is_labelled_simulation(app):
    for b in app.button:
        if b.label == "Run counterfactual":
            b.click()
            break
    app.run()
    assert not app.exception
    labels = {m.label for m in app.metric}
    assert "Current decision" in labels and "Counterfactual decision" in labels
    warnings = " ".join(str(w.value) for w in app.warning)
    assert "SIMULATION" in warnings


# ------------------------------------------------------------------
# reviewer action works + history updates + decision immutable
# ------------------------------------------------------------------


def test_reviewer_action_updates_governance_history(app):
    # pick ACKNOWLEDGE (no comment/reviewer-decision required) and submit
    for s in app.selectbox:
        if s.label == "Governance action":
            s.select("ACKNOWLEDGE")
            break
    app.run()
    for b in app.button:
        if b.label == "Submit governance action":
            b.click()
            break
    app.run()
    assert not app.exception
    t = _text(app)
    assert "ACKNOWLEDGE" in t
    assert "OPEN → ACKNOWLEDGED" in t or "ACKNOWLEDGED" in t
    # decision metric still shows the automated tier (immutable)
    decision = _metric(app, "ControlPlane decision")
    assert decision in ("ALLOW", "ANNOTATE", "VERIFY", "HUMAN_REVIEW", "BLOCK")


def test_effective_decision_and_override_warning_wired_in_source():
    # the MODIFY-override divergence + immutability is proven end-to-end in
    # tests/test_investigation.py and tests/test_investigation_api.py; here we
    # only assert the UI reads the fields (not fragile multi-step form driving).
    src = _APP_SRC.read_text(encoding="utf-8")
    assert '"Original ControlPlane decision"' in src
    assert '"Effective governed decision"' in src
    assert "inv.effective_governed_decision" in src
    assert "inv.is_overridden" in src
    assert "Human Governance & Override" in src


# ------------------------------------------------------------------
# PII never appears in the rendered workspace
# ------------------------------------------------------------------


def test_no_raw_pii_in_investigation_ui():
    # deterministically investigate the PII incident
    at = AppTest.from_file(_APP, default_timeout=300)
    at.run()
    for b in at.button:
        if "Populate" in b.label:
            b.click()
            break
    at.run()
    pii_id = None
    for s in at.selectbox:
        if s.label.startswith("Select an incident"):
            pii_id = next((o for o in s.options if "SCEN-C" in o or o.endswith("SCEN-C")), None)
            pii_id = pii_id or next((o for o in s.options if "C" in o and o != "—"), s.options[1])
            s.select(pii_id)
            break
    at.run()
    for b in at.button:
        if "Investigate Incident" in b.label:
            b.click()
            break
    at.run()
    assert not at.exception
    rendered = _text(at)
    # find the trace to get its raw PII spans
    from streamlit.testing.v1 import AppTest as _AT  # noqa
    for needle in _KNOWN_PII:
        assert needle not in rendered, needle
    assert "matched_text" not in rendered


# ------------------------------------------------------------------
# ground-truth isolation (UI source)
# ------------------------------------------------------------------


def test_ui_source_has_no_ground_truth_or_evaluation():
    src = _APP_SRC.read_text(encoding="utf-8")
    for token in ("ground_truth_", "expected_decision", "final_outcome",
                  "import evaluation", "from evaluation"):
        assert token not in src, token


# ------------------------------------------------------------------
# existing tabs still work
# ------------------------------------------------------------------


def test_existing_tabs_still_work(app):
    # check tab
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
    # audit tab
    iid = app.session_state["result_trace_id"]
    for ti in app.text_input:
        if ti.label == "Interaction ID":
            ti.set_value(iid)
            break
    app.run()
    assert not app.exception
    # command center still there
    assert "Enterprise AI Risk Command Center" in _text(app)
