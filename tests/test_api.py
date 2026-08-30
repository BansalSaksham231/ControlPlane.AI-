"""API tests — health, /check, validation, session, audit, feedback."""

from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

from api.app import app, set_service
from api.service import ControlPlaneService

# Each _check_payload() call gets a fresh session_id unless one is passed
# explicitly, so single-turn cases never share multi-turn session state
# (a critical violation in one test would otherwise raise the session floor
# for every later test on the same id).
_api_session_ids = itertools.count(1)


@pytest.fixture(scope="module")
def client():
    # fast_start: skip the cost-baseline fit (static baseline is fine for tests)
    set_service(ControlPlaneService(fit_cost_baseline=False))
    with TestClient(app) as test_client:
        yield test_client


def _check_payload(**overrides) -> dict:
    payload = dict(
        application="customer_support",
        session_id=f"SESSION-API-{next(_api_session_ids)}",
        prompt="What is the refund policy?",
        context=(
            "Company policy allows customers to request a refund within 30 business "
            "days of purchase, provided the item is unused."
        ),
        response=(
            "You are eligible for a refund within 30 business days of your purchase, "
            "as long as the item is unused."
        ),
        tokens_in=40,
        tokens_out=30,
        latency_ms=300.0,
    )
    payload.update(overrides)
    return payload


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "performance" in body["detectors"]


def test_check_clean_allow(client):
    resp = client.post("/check", json=_check_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"]["decision"] == "ALLOW"
    assert 0.0 <= body["decision"]["overall_risk"] <= 1.0
    assert body["interaction_id"].startswith("INT-")
    assert body["trace"] is None


def test_check_pii_escalates(client):
    resp = client.post(
        "/check",
        json=_check_payload(
            response=(
                "The contact details on file for account ACC-227763 are: Karan Mehta, "
                "email karan.mehta@example-test.com, phone +91-940847221."
            ),
            include_trace=True,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"]["decision"] in ("HUMAN_REVIEW", "BLOCK")
    assert "CRITICAL_PII" in body["decision"]["triggered_rules"]
    # trace present, and redacted
    assert body["trace"] is not None
    assert "karan.mehta@example-test.com" not in str(body["trace"]["responsibility"]["findings"])


def test_check_validation_error(client):
    resp = client.post("/check", json={"application": "not_a_real_app", "response": "x"})
    assert resp.status_code == 422


def test_check_missing_response(client):
    resp = client.post("/check", json={"application": "customer_support"})
    assert resp.status_code == 422


def test_session_endpoint_tracks_turns(client):
    sid = "SESSION-API-MULTI"
    for _ in range(3):
        client.post(
            "/check",
            json=_check_payload(
                session_id=sid,
                response=(
                    "You are eligible for a refund within 90 business days of your "
                    "purchase, and the item condition does not matter."
                ),
            ),
        )
    resp = client.get(f"/session/{sid}")
    assert resp.status_code == 200
    state = resp.json()
    assert state["interaction_count"] == 3
    assert len(state["recent_decisions"]) == 3


def test_session_reset(client):
    sid = "SESSION-API-RESET"
    client.post("/check", json=_check_payload(session_id=sid))
    assert client.get(f"/session/{sid}").json()["interaction_count"] == 1
    assert client.delete(f"/session/{sid}").status_code == 200
    assert client.get(f"/session/{sid}").json()["interaction_count"] == 0


def test_audit_endpoint(client):
    resp = client.post("/check", json=_check_payload(interaction_id="INT-API-AUDIT-1"))
    assert resp.status_code == 200
    audit = client.get("/audit/INT-API-AUDIT-1")
    assert audit.status_code == 200
    body = audit.json()
    assert body["interaction_id"] == "INT-API-AUDIT-1"
    assert body["decision"] == "ALLOW"
    assert "explanation" in body


def test_audit_not_found(client):
    assert client.get("/audit/INT-DOES-NOT-EXIST").status_code == 404


def test_feedback_flow(client):
    check = client.post("/check", json=_check_payload(interaction_id="INT-API-FB-1"))
    system_tier = check.json()["decision"]["decision"]

    resp = client.post(
        "/feedback",
        json={
            "interaction_id": "INT-API-FB-1",
            "reviewer_decision": "VERIFY",
            "reason": "wanted a second look",
            "reviewer": "qa-1",
        },
    )
    assert resp.status_code == 200
    record = resp.json()
    assert record["system_decision"] == system_tier
    assert record["reviewer_decision"] == "VERIFY"
    assert record["human_override"] is (system_tier != "VERIFY")

    summary = client.get("/feedback/summary").json()
    assert summary["total"] >= 1


def test_feedback_unknown_interaction(client):
    resp = client.post("/feedback", json={"interaction_id": "INT-NOPE", "reviewer_decision": "ALLOW"})
    assert resp.status_code == 404


def test_no_ground_truth_fields_accepted(client):
    """Ground-truth keys in the payload must be ignored, never consumed."""
    resp = client.post(
        "/check",
        json=_check_payload(ground_truth_hallucination=True, expected_decision="BLOCK"),
    )
    assert resp.status_code == 200
    # extra keys ignored; decision still driven purely by content
    assert resp.json()["decision"]["decision"] == "ALLOW"


def test_check_exposes_reason_codes(client):
    resp = client.post(
        "/check",
        json=_check_payload(
            response=(
                "The contact details on file for account ACC-1 are: Karan Mehta, "
                "email karan.mehta@example-test.com, phone +91-940847221."
            )
        ),
    )
    body = resp.json()["decision"]
    assert "reason_codes" in body
    assert "CRITICAL_PII" in body["reason_codes"]


def test_simulate_policy_endpoint(client):
    resp = client.post(
        "/simulate-policy",
        json={
            "interaction": _check_payload(
                response=(
                    "You are eligible for a refund within 90 business days of your "
                    "purchase, and the item condition does not matter."
                )
            ),
            "profiles": ["customer_support", "decision_support"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {o["profile"] for o in body["outcomes"]} == {"customer_support", "decision_support"}
    assert isinstance(body["differs"], bool)


def test_counterfactual_endpoint(client):
    resp = client.post(
        "/counterfactual",
        json={
            "interaction": _check_payload(
                response="The approved refund has been processed and will settle in 7 days.",
                context="Finance approved a goodwill refund. Refunds are processed within 7 business days.",
                action_type="refund",
                action_amount_inr=480000,
            ),
            "modified_fields": {"action_amount_inr": 100},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_decision"] in {"VERIFY", "HUMAN_REVIEW", "ANNOTATE"}
    assert "changed_fields" in body
    assert body["rejected_fields"] == []


def test_counterfactual_rejects_ground_truth(client):
    resp = client.post(
        "/counterfactual",
        json={
            "interaction": _check_payload(),
            "modified_fields": {"ground_truth_pii": True, "action_amount_inr": 500},
        },
    )
    assert resp.status_code == 200
    assert "ground_truth_pii" in resp.json()["rejected_fields"]
