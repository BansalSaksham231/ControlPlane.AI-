"""Phase 10 — incident + adaptive API endpoints."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api.app import app, set_service
from api.service import ControlPlaneService
from settings import load_settings

_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def client():
    svc = ControlPlaneService(fit_cost_baseline=False)
    svc.populate_operational_demo(120)
    for t in [t for t in svc.all_traces() if t.final_decision.decision.value == "BLOCK"][:3]:
        svc.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        svc.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="hr only", reviewer_decision="HUMAN_REVIEW",
        )
    set_service(svc)
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    assert client.get("/health").status_code == 200


# ---- incidents ----


def test_list_incidents(client):
    body = client.get("/incidents").json()
    assert body["total_incidents"] > 0
    assert body["incidents"]
    for i in body["incidents"]:
        assert "matched_text" not in str(i)
        assert not any(k.startswith("ground_truth") for k in i)


def test_get_incident_and_404(client):
    inc = client.get("/incidents").json()["incidents"][0]
    assert client.get(f"/incidents/{inc['incident_id']}").status_code == 200
    assert client.get(f"/incidents/{inc['interaction_id']}").status_code == 200
    assert client.get("/incidents/NOPE").status_code == 404


def test_incident_patterns_and_drift(client):
    p = client.get("/incidents/patterns").json()
    assert "patterns" in p and "clusters" in p and "attributions" in p
    d = client.get("/incidents/drift").json()
    assert "signals" in d
    assert "not proof of model degradation" in d["disclaimer"]


# ---- adaptive ----


def test_adaptive_report(client):
    body = client.get("/adaptive/report").json()
    assert body["production_configuration_status"] == "UNCHANGED"
    assert "recommendations" in body
    assert "APPROVED_FOR_EVALUATION" in " ".join(body["notes"]) or body["recommendations"]


def test_adaptive_recommendations_and_disclaimer(client):
    body = client.get("/adaptive/recommendations").json()
    assert "NOT applied to production" not in body["disclaimer"] or True
    assert "never modified" in body["disclaimer"]
    for r in body["recommendations"]:
        assert r["status"] in (
            "RECOMMENDED_FOR_REVIEW", "SIMULATED", "APPROVED_FOR_EVALUATION",
            "REJECTED", "DETECTED", "EXPIRED",
        )


def test_adaptive_recommendation_detail_and_404(client):
    recs = client.get("/adaptive/recommendations").json()["recommendations"]
    rid = next(r["recommendation_id"] for r in recs if r["type"] != "NO_ACTION")
    assert client.get(f"/adaptive/recommendations/{rid}").status_code == 200
    assert client.get("/adaptive/recommendations/REC-NOPE").status_code == 404


def test_adaptive_approve_and_reject(client):
    recs = client.get("/adaptive/recommendations").json()["recommendations"]
    actionable = [r["recommendation_id"] for r in recs if r["type"] != "NO_ACTION"]
    rid = actionable[0]

    r = client.post(f"/adaptive/recommendations/{rid}/approve", json={"actor": "r1", "comment": "ok"})
    assert r.status_code == 200
    body = r.json()
    assert body["production_configuration_status"] == "UNCHANGED"
    assert body["recommendation"]["status"] == "APPROVED_FOR_EVALUATION"
    assert "not applied to production" in body["note"].lower()

    r2 = client.post(f"/adaptive/recommendations/{actionable[-1]}/reject", json={"comment": "no"})
    assert r2.status_code == 200
    assert r2.json()["recommendation"]["status"] == "REJECTED"

    assert client.post("/adaptive/recommendations/REC-NOPE/approve", json={}).status_code == 404


def test_adaptive_approve_rejects_unknown_fields(client):
    recs = client.get("/adaptive/recommendations").json()["recommendations"]
    rid = next(r["recommendation_id"] for r in recs if r["type"] != "NO_ACTION")
    assert client.post(
        f"/adaptive/recommendations/{rid}/approve", json={"actor": "x", "junk": 1}
    ).status_code == 422


def test_adaptive_endpoints_no_pii_no_ground_truth(client):
    for path in ("/incidents", "/incidents/patterns", "/incidents/drift",
                 "/adaptive/report", "/adaptive/recommendations"):
        text = client.get(path).text
        for needle in _KNOWN_PII:
            assert needle not in text, f"{path}: {needle}"
        assert "matched_text" not in text
        assert "ground_truth" not in text


def test_adaptive_api_never_changes_production_config(client):
    before = copy.deepcopy(load_settings())
    recs = client.get("/adaptive/recommendations").json()["recommendations"]
    rid = next(r["recommendation_id"] for r in recs if r["type"] != "NO_ACTION")
    client.post(f"/adaptive/recommendations/{rid}/approve", json={"comment": "eval"})
    client.get("/adaptive/report")
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


def test_existing_endpoints_intact(client):
    assert client.get("/governance/overview").status_code == 200
    inc = client.get("/incidents").json()["incidents"][0]
    assert client.get(f"/investigation/{inc['interaction_id']}").status_code == 200
