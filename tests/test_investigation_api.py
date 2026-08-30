"""Phase 8 — Step 3: investigation API endpoints (§34)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app, set_service
from api.service import ControlPlaneService
from tests import scenarios

_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def client():
    svc = ControlPlaneService(fit_cost_baseline=False)
    ids = {}
    for name in ("scenario_a_clean", "scenario_b_hallucination", "scenario_c_pii",
                 "scenario_d_high_consequence"):
        it = getattr(scenarios, name)()
        svc.check(it, timestamp=it.timestamp)
        ids[name] = it.interaction_id
    set_service(svc)
    with TestClient(app) as c:
        c.scenario_ids = ids
        yield c


# ------------------------------------------------------------------
# GET /investigation/{id}
# ------------------------------------------------------------------


def test_get_investigation_valid(client):
    iid = client.scenario_ids["scenario_c_pii"]
    resp = client.get(f"/investigation/{iid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["original_decision"] == "BLOCK"
    assert body["incident"]["severity"] == "CRITICAL"
    assert body["explanation"]["decision"] == "BLOCK"
    assert body["investigation_status"] == "OPEN"
    assert "ACKNOWLEDGE" in body["available_actions"]


def test_get_investigation_clean_scenario(client):
    iid = client.scenario_ids["scenario_a_clean"]
    body = client.get(f"/investigation/{iid}").json()
    assert body["found"] is True
    assert body["original_decision"] == "ALLOW"
    assert body["incident"] is None


def test_get_investigation_unknown_is_404(client):
    resp = client.get("/investigation/NO-SUCH-INCIDENT")
    assert resp.status_code == 404


def test_get_investigation_pii_scenario_has_no_raw_pii(client):
    iid = client.scenario_ids["scenario_c_pii"]
    text = client.get(f"/investigation/{iid}").text
    for needle in _KNOWN_PII:
        assert needle not in text
    assert "matched_text" not in text


# ------------------------------------------------------------------
# POST /investigation/{id}/action
# ------------------------------------------------------------------


def _action(client, iid, **payload):
    return client.post(f"/investigation/{iid}/action", json=payload)


def test_action_acknowledge_then_approve(client):
    iid = client.scenario_ids["scenario_b_hallucination"]
    r1 = _action(client, iid, action="ACKNOWLEDGE", comment="seen")
    assert r1.status_code == 200
    assert r1.json()["investigation_status"] == "ACKNOWLEDGED"
    r2 = _action(client, iid, action="APPROVE_DECISION", comment="agree with VERIFY")
    assert r2.status_code == 200
    assert r2.json()["investigation_status"] == "REVIEWED"


def test_action_modify_records_both_decisions(client):
    iid = client.scenario_ids["scenario_d_high_consequence"]
    r = _action(
        client, iid,
        action="MODIFY_DECISION",
        comment="would have allowed with annotation",
        reviewer_decision="ANNOTATE",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["original_decision"] == "VERIFY"          # unchanged automated decision
    assert body["latest_reviewer_decision"] == "ANNOTATE"
    last = body["governance_history"][-1]
    assert last["original_decision"] == "VERIFY"
    assert last["reviewer_decision"] == "ANNOTATE"
    # the stored decision (via /audit) is still VERIFY
    assert client.get(f"/audit/{iid}").json()["decision"] == "VERIFY"


def test_action_reject_and_escalate_and_close(client):
    it = scenarios.scenario_e_multi_risk().model_copy(update={"interaction_id": "API-INV-E"})
    from api.app import get_service
    get_service().check(it, timestamp=it.timestamp)

    assert _action(client, "API-INV-E", action="REJECT_DECISION", comment="wrong call").status_code == 200
    assert _action(client, "API-INV-E", action="ESCALATE", comment="senior please").status_code == 200
    r = _action(client, "API-INV-E", action="CLOSE")
    assert r.status_code == 200
    assert r.json()["investigation_status"] == "CLOSED"
    assert r.json()["available_actions"] == []


def test_action_invalid_value_is_422(client):
    iid = client.scenario_ids["scenario_a_clean"]
    r = _action(client, iid, action="DEFINITELY_NOT_AN_ACTION")
    assert r.status_code == 422


def test_action_modify_without_reviewer_decision_is_422(client):
    it = scenarios.scenario_b_hallucination().model_copy(update={"interaction_id": "API-INV-B2"})
    from api.app import get_service
    get_service().check(it, timestamp=it.timestamp)
    r = _action(client, "API-INV-B2", action="MODIFY_DECISION", comment="x")
    assert r.status_code == 422


def test_action_escalate_without_comment_is_422(client):
    it = scenarios.scenario_b_hallucination().model_copy(update={"interaction_id": "API-INV-B3"})
    from api.app import get_service
    get_service().check(it, timestamp=it.timestamp)
    r = _action(client, "API-INV-B3", action="ESCALATE")
    assert r.status_code == 422


def test_action_unknown_incident_is_404(client):
    r = _action(client, "NO-SUCH-INCIDENT", action="ACKNOWLEDGE")
    assert r.status_code == 404


def test_action_rejects_unknown_fields(client):
    iid = client.scenario_ids["scenario_a_clean"]
    r = client.post(f"/investigation/{iid}/action", json={"action": "ACKNOWLEDGE", "junk": 1})
    assert r.status_code == 422


# ------------------------------------------------------------------
# history + counterfactual
# ------------------------------------------------------------------


def test_get_history(client):
    iid = client.scenario_ids["scenario_b_hallucination"]
    body = client.get(f"/investigation/{iid}/history").json()
    assert body["interaction_id"] == iid
    assert len(body["governance_history"]) >= 1
    assert client.get("/investigation/NOPE/history").status_code == 404


def test_post_counterfactual(client):
    iid = client.scenario_ids["scenario_d_high_consequence"]
    r = client.post(
        f"/investigation/{iid}/counterfactual", json={"modified_fields": {"action_amount_inr": 1.0}}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["simulated"] is True
    assert body["current_decision"] == "VERIFY"
    assert "SIMULATION" in body["note"]
    # stored decision unchanged
    assert client.get(f"/audit/{iid}").json()["decision"] == "VERIFY"


def test_counterfactual_unknown_incident_is_404(client):
    r = client.post("/investigation/NOPE/counterfactual", json={"modified_fields": {}})
    assert r.status_code == 404


# ------------------------------------------------------------------
# existing endpoints still work
# ------------------------------------------------------------------


def test_existing_endpoints_intact(client):
    assert client.get("/health").status_code == 200
    iid = client.scenario_ids["scenario_a_clean"]
    assert client.get(f"/audit/{iid}").status_code == 200
    assert client.get("/feedback/summary").status_code == 200
    r = client.post("/check", json={
        "application": "customer_support",
        "response": "You can get a refund within 30 days.",
        "context": "Refunds are allowed within 30 days.",
    })
    assert r.status_code == 200


# ------------------------------------------------------------------
# reviewer-facing governance override aliases
# ------------------------------------------------------------------


def test_post_governance_action_records_override_and_preserves_trace(client):
    iid = client.scenario_ids["scenario_c_pii"]
    before = client.get(f"/investigation/{iid}").json()["original_decision"]
    assert before == "BLOCK"

    resp = client.post("/governance/action", json={
        "interaction_id": iid, "action_type": "MODIFY", "new_tier": "HUMAN_REVIEW",
        "justification": "route to a human, not a hard block", "reviewer_id": "alice",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_decision"] == "BLOCK"                 # immutable
    assert body["effective_governed_decision"] == "HUMAN_REVIEW"
    assert body["is_overridden"] is True

    # the stored trace / investigation still reports the automated BLOCK
    inv = client.get(f"/investigation/{iid}").json()
    assert inv["original_decision"] == "BLOCK"
    assert inv["effective_governed_decision"] == "HUMAN_REVIEW"


def test_get_governance_actions_returns_history(client):
    from api.app import get_service

    it = scenarios.scenario_d_high_consequence().model_copy(update={"interaction_id": "API-GOV-HIST"})
    get_service().check(it, timestamp=it.timestamp)

    client.post("/governance/action", json={
        "interaction_id": "API-GOV-HIST", "action_type": "APPROVED",
        "justification": "agree with the call",
    })
    resp = client.get("/governance/actions/API-GOV-HIST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_decision"] == body["effective_governed_decision"]   # APPROVE != override
    assert body["is_overridden"] is False
    assert body["history"] and body["history"][-1]["action"] == "APPROVE_DECISION"
    assert body["history"][-1]["original_decision"] == body["original_decision"]


def test_governance_action_endpoints_404_for_unknown_interaction(client):
    assert client.get("/governance/actions/NO-SUCH-ID").status_code == 404
    r = client.post("/governance/action", json={
        "interaction_id": "NO-SUCH-ID", "action_type": "APPROVE",
    })
    assert r.status_code == 404


def test_governance_analytics_routes_not_shadowed_by_the_alias(client):
    # the 8 pre-existing /governance/* analytics endpoints still resolve
    for path in ("/governance/overview", "/governance/recommendations",
                 "/governance/insights", "/governance/signals"):
        assert client.get(path).status_code == 200


def test_governance_override_json_has_no_raw_pii(client):
    from api.app import get_service

    it = scenarios.scenario_c_pii().model_copy(update={"interaction_id": "API-GOV-PII"})
    get_service().check(it, timestamp=it.timestamp)
    r = client.post("/governance/action", json={
        "interaction_id": "API-GOV-PII", "action_type": "REJECT",
        "justification": "the automated call was wrong",
    })
    assert r.status_code == 200
    text = client.get("/governance/actions/API-GOV-PII").text
    for needle in _KNOWN_PII:
        assert needle not in text
