"""Phase 9 — Governance Intelligence API endpoints."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api.app import app, set_service
from api.service import ControlPlaneService
from settings import load_settings
from tests import scenarios

_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def client():
    svc = ControlPlaneService(fit_cost_baseline=False)
    svc.populate_operational_demo(120)
    traces = svc.all_traces()
    blocks = [t for t in traces if t.final_decision.decision.value == "BLOCK"][:2]
    for t in blocks:
        svc.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        svc.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="human review only", reviewer_decision="HUMAN_REVIEW",
        )
    set_service(svc)
    with TestClient(app) as c:
        yield c


def test_health_still_ok(client):
    assert client.get("/health").status_code == 200


def test_governance_overview(client):
    r = client.get("/governance/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["traffic"]["total_interactions"] > 0
    assert set(body["decisions"]) >= {"allow", "block", "human_oversight_rate"}
    assert "NOT ground truth" in body["reviewer_disagreement"]["note"]


def test_governance_applications(client):
    body = client.get("/governance/applications").json()
    assert body["applications"]
    assert body["highest_volume"] in {a["application"] for a in body["applications"]}


def test_governance_application_detail_and_404(client):
    apps = client.get("/governance/applications").json()["applications"]
    name = apps[0]["application"]
    assert client.get(f"/governance/applications/{name}").status_code == 200
    assert client.get("/governance/applications/no_such_app").status_code == 404


def test_governance_insights(client):
    body = client.get("/governance/insights").json()
    assert "insights" in body
    for i in body["insights"]:
        assert i["code"] and i["recommended_action"]
        assert "not a truth claim" in i["note"]


def test_governance_signals(client):
    body = client.get("/governance/signals").json()
    assert "summary" in body and "signals" in body
    assert body["summary"]["signal_count"] >= 2      # the 2 reviewer overrides
    assert "reviewer_override" in body["summary"]["by_signal_type"]
    assert "NOT ground truth" in body["summary"]["note"]


def test_governance_trends(client):
    body = client.get("/governance/trends").json()
    assert "signals" in body
    assert "no statistical-significance claim" in " ".join(body["notes"])


def test_governance_recommendations(client):
    body = client.get("/governance/recommendations").json()
    assert body["disclaimer"] == "RECOMMENDATION ONLY — NOT APPLIED TO PRODUCTION."
    assert body["recommendations"]
    for r in body["recommendations"]:
        assert r["disposition"] in (
            "RECOMMENDED_FOR_EVALUATION", "REVIEW_REQUIRED", "NO_ACTION"
        )
        assert "NOT APPLIED" in r["disclaimer"]


def test_governance_endpoints_never_leak_pii(client):
    for path in ("/governance/overview", "/governance/applications",
                 "/governance/insights", "/governance/signals", "/governance/trends",
                 "/governance/recommendations"):
        text = client.get(path).text
        for needle in _KNOWN_PII:
            assert needle not in text, f"{path}: {needle}"
        assert "matched_text" not in text
        assert "ground_truth" not in text


def test_governance_api_does_not_change_production_config(client):
    before = copy.deepcopy(load_settings())
    for path in ("/governance/overview", "/governance/insights",
                 "/governance/recommendations"):
        client.get(path)
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


def test_existing_endpoints_intact(client):
    it = scenarios.scenario_a_clean().model_copy(update={"interaction_id": "GOV-API-A"})
    from api.app import get_service
    get_service().check(it, timestamp=it.timestamp)
    assert client.get("/audit/GOV-API-A").status_code == 200
    assert client.get("/investigation/GOV-API-A").status_code == 200
    assert client.get("/feedback/summary").status_code == 200
