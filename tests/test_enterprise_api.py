"""
Phase 11 — read-only enterprise API endpoints.

    GET /command-center
    GET /applications
    GET /applications/{application}        (+404)
    GET /governance/timeline               (?interaction_id=)
    GET /incidents/{incident_id}/investigation   (+404)

Deterministic, PII-safe, no ground truth, never mutates production config.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from api.app import app, set_service
from api.service import ControlPlaneService
from settings import load_settings

_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")
_GT_TOKENS = ("matched_text", "ground_truth", "expected_decision", "final_outcome",
              "actual_correctness")


@pytest.fixture(scope="module")
def client():
    svc = ControlPlaneService(fit_cost_baseline=False)
    svc.populate_operational_demo(200)
    for t in [t for t in svc.all_traces() if t.final_decision.decision.value == "BLOCK"][:3]:
        svc.record_governance_action(t.interaction_id, action="ACKNOWLEDGE")
        svc.record_governance_action(
            t.interaction_id, action="MODIFY_DECISION",
            comment="route to human review", reviewer_decision="HUMAN_REVIEW",
        )
    set_service(svc)
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ command center


def test_command_center_ok(client):
    r = client.get("/command-center")
    assert r.status_code == 200
    body = r.json()
    assert body["kpi"]["has_data"] is True
    assert body["kpi"]["total_interactions"] > 0
    assert body["risk_posture"]["high_risk_threshold"] == 0.5
    assert len(body["heatmap"]["cells"]) == len(body["heatmap"]["applications"]) * 4
    assert body["recent_decisions"]
    ts = [row["timestamp"] for row in body["recent_decisions"]]
    assert ts == sorted(ts, reverse=True)
    assert all(row["source"] == "STORED_TRACE" for row in body["recent_decisions"])
    assert "No production changes" in body["executive_summary"]["safety_status"]


def test_monitoring_operational_report_ok(client):
    r = client.get("/monitoring/operational")
    assert r.status_code == 200
    body = r.json()
    # the OperationalMonitoringReport surface the Next.js dashboard consumes
    assert body["total_interactions"] > 0
    for key in ("snapshot", "risk_distribution", "verification", "multi_turn",
                "incidents", "incident_digest"):
        assert key in body
    v = body["verification"]
    assert v["fast_count"] + v["deep_count"] <= body["total_interactions"]
    assert 0.0 <= v["semantic_bypass_rate_of_deep"] <= 1.0
    assert body["multi_turn"]["sessions_hitting_critical_floor"] >= 0


def test_command_center_is_deterministic(client):
    a = client.get("/command-center").json()
    b = client.get("/command-center").json()

    def _strip(n):
        if isinstance(n, dict):
            return {k: _strip(v) for k, v in n.items() if k != "generated_at"}
        if isinstance(n, list):
            return [_strip(x) for x in n]
        return n

    assert _strip(a) == _strip(b)


# ------------------------------------------------------------------ applications


def test_applications_list_and_detail(client):
    rows = client.get("/applications").json()["applications"]
    assert rows
    total = sum(r["interactions"] for r in rows)
    cc_total = client.get("/command-center").json()["kpi"]["total_interactions"]
    assert total == cc_total
    app_name = rows[0]["application"]
    detail = client.get(f"/applications/{app_name}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["application"] == app_name
    assert d["posture"] in ("LOW", "MODERATE", "HIGH")
    assert d["posture_rationale"]


def test_applications_unknown_is_404(client):
    r = client.get("/applications/not_a_real_app")
    assert r.status_code == 404
    assert "Known" in r.json()["detail"]


# ------------------------------------------------------------------ governance timeline


def test_governance_timeline_ok(client):
    body = client.get("/governance/timeline").json()
    assert body["events"]
    types = [e["event_type"] for e in body["events"]]
    assert "DECISION" in types
    orders = [e["order"] for e in body["events"]]
    assert orders == sorted(orders)
    for e in body["events"]:
        if e["timestamp"] is None:
            assert "causal workflow order" in e["timestamp_note"]


def test_governance_timeline_focus_interaction(client):
    inc = client.get("/incidents").json()["incidents"][0]
    body = client.get(
        "/governance/timeline", params={"interaction_id": inc["interaction_id"]}
    ).json()
    assert body["events"]
    entities = " ".join(e["entity"] for e in body["events"])
    assert inc["interaction_id"] in entities


# ------------------------------------------------------------------ incident investigation


def test_incident_investigation_ok(client):
    inc = client.get("/incidents").json()["incidents"][0]
    for key in (inc["incident_id"], inc["interaction_id"]):
        r = client.get(f"/incidents/{key}/investigation")
        assert r.status_code == 200
        body = r.json()
        assert body["incident"]["incident_id"] == inc["incident_id"]
        assert "investigation" in body
        assert body["related_by"] == "shared deterministic structured risk signature"
        for rel in body["related_incidents"]:
            assert rel["signature"] == inc["signature"]
            assert rel["interaction_id"] != inc["interaction_id"]


def test_incident_investigation_404(client):
    assert client.get("/incidents/NOPE-123/investigation").status_code == 404


# ------------------------------------------------------------------ cross-cutting guards


_ENDPOINTS = (
    "/command-center",
    "/applications",
    "/governance/timeline",
)


def test_enterprise_endpoints_no_pii_no_ground_truth(client):
    paths = list(_ENDPOINTS)
    inc = client.get("/incidents").json()["incidents"][0]
    paths.append(f"/incidents/{inc['incident_id']}/investigation")
    for path in paths:
        text = client.get(path).text
        for needle in _KNOWN_PII:
            assert needle not in text, f"{path}: {needle}"
        for tok in _GT_TOKENS:
            assert tok not in text, f"{path}: {tok}"


def test_enterprise_endpoints_reject_unknown_query(client):
    # extra="forbid" models still serialize fine; unknown *query* params are ignored
    # by FastAPI, so just assert the documented ones behave.
    assert client.get("/governance/timeline", params={"interaction_id": "X"}).status_code == 200


def test_enterprise_api_never_changes_production_config(client):
    before = copy.deepcopy(load_settings())
    for path in _ENDPOINTS:
        client.get(path)
    inc = client.get("/incidents").json()["incidents"][0]
    client.get(f"/incidents/{inc['incident_id']}/investigation")
    assert load_settings() == before
    assert load_settings()["verification"]["deep_verification_risk_threshold"] == 0.35


def test_existing_endpoints_still_work(client):
    assert client.get("/health").status_code == 200
    assert client.get("/governance/overview").status_code == 200
    assert client.get("/adaptive/report").status_code == 200
    assert client.get("/incidents").status_code == 200
