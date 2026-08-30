"""
ControlPlane.ai FastAPI application.

Thin HTTP layer over :class:`ControlPlaneService`. Routes validate input,
call one service method, and shape the response — no business logic here.

Run locally:

    uvicorn api.app:app --reload
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import get_db
from api.security import verify_api_key
from api.schemas import (
    AdaptiveApprovalRequest,
    CheckRequest,
    CheckResponse,
    CounterfactualRequest,
    FeedbackRequest,
    GovernanceActionRequest,
    GovernanceOverrideRequest,
    HealthResponse,
    InvestigationCounterfactualRequest,
    SimulatePolicyRequest,
)
from api.service import VERSION, ControlPlaneService
from data.schemas import Interaction

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ControlPlane.ai",
    version=VERSION,
    description=(
        "Real-time AI risk control plane (Round 2 prototype). Heuristic, "
        "deterministic detectors — not a production safety guarantee."
    ),
    # Global API-key gate. A no-op until CONTROLPLANE_API_KEY is set, so the
    # default / test / local-dev experience is unchanged.
    dependencies=[Depends(verify_api_key)],
)

# CORS — origins are configurable; default is permissive for local dev. Set
# CONTROLPLANE_CORS_ORIGINS to a comma-separated allowlist in production.
_cors_origins = [
    o.strip()
    for o in os.environ.get("CONTROLPLANE_CORS_ORIGINS", "*").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global safety net: never leak a stack trace to a client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error_type": type(exc).__name__},
    )


def _wire_persistence() -> None:
    """When persistence is enabled, make ``get_db`` yield real DB sessions."""
    if os.environ.get("CONTROLPLANE_PERSISTENCE") != "1":
        return
    try:
        from database.session import get_db as _real_get_db, init_engine

        init_engine()
        app.dependency_overrides[get_db] = _real_get_db
        logger.info("Persistence enabled: get_db bound to database.session")
    except Exception:  # pragma: no cover - defensive
        logger.exception("CONTROLPLANE_PERSISTENCE=1 but database layer failed to load")


_wire_persistence()

_service: ControlPlaneService | None = None


def get_service() -> ControlPlaneService:
    global _service
    if _service is None:
        _service = ControlPlaneService(
            feedback_path=os.environ.get("CONTROLPLANE_FEEDBACK_PATH"),
            fit_cost_baseline=os.environ.get("CONTROLPLANE_FAST_START") != "1",
        )
    return _service


def set_service(service: ControlPlaneService) -> None:
    """Test hook: inject a pre-built service."""
    global _service
    _service = service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    stats = service.stats()
    return HealthResponse(
        status="ok",
        service="controlplane.ai",
        version=VERSION,
        detectors=["performance", "responsibility", "cost"],
        checks_served=stats["checks_served"],
        active_sessions=stats["active_sessions"],
    )


def _interaction_from_request(request: CheckRequest) -> Interaction:
    interaction_id = request.interaction_id or _auto_id()
    try:
        return Interaction(
            interaction_id=interaction_id,
            timestamp=request.timestamp or datetime.now(timezone.utc),
            application=request.application,
            user_type=request.user_type,
            model=request.model,
            session_id=request.session_id,
            prompt=request.prompt,
            context=request.context,
            response=request.response,
            tokens_in=request.tokens_in,
            tokens_out=request.tokens_out,
            latency_ms=request.latency_ms,
            tool_calls=request.tool_calls,
            retry_count=request.retry_count,
            action_type=request.action_type,
            action_amount_inr=request.action_amount_inr,
            affected_entities=request.affected_entities,
        )
    except Exception as exc:  # pragma: no cover - pydantic already validated most
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/check", response_model=CheckResponse)
def check(request: CheckRequest, db=Depends(get_db)) -> CheckResponse:
    # ``db`` is the request-scoped DB session when persistence is wired
    # (else ``None``). The service owns the durable audit write; the session
    # is injected here so per-request transactional handoff to the
    # DecisionEngine can be added without touching the route contract.
    service = get_service()
    interaction = _interaction_from_request(request)
    trace = service.check(interaction, timestamp=interaction.timestamp)
    return CheckResponse(
        interaction_id=interaction.interaction_id,
        decision=trace.final_decision,
        trace=trace.redacted_dump() if request.include_trace else None,
    )


@app.post("/simulate-policy")
def simulate_policy(request: SimulatePolicyRequest) -> dict:
    service = get_service()
    interaction = _interaction_from_request(request.interaction)
    return service.simulate_policy(interaction, request.profiles)


@app.post("/counterfactual")
def counterfactual(request: CounterfactualRequest) -> dict:
    service = get_service()
    interaction = _interaction_from_request(request.interaction)
    return service.counterfactual(interaction, request.modified_fields)


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    return get_service().get_session(session_id)


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict:
    get_service().reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/audit/{interaction_id}")
def get_audit(interaction_id: str) -> dict:
    summary = get_service().get_audit(interaction_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No audit record for {interaction_id}")
    return summary


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict:
    service = get_service()
    try:
        record = service.submit_feedback(
            interaction_id=request.interaction_id,
            system_decision=None,
            reviewer_decision=request.reviewer_decision,
            outcome=request.outcome,
            actual_outcome=request.actual_outcome,
            reason=request.reason,
            reviewer=request.reviewer,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@app.get("/feedback/summary")
def feedback_summary() -> dict:
    return get_service().feedback_summary()


# ------------------------------------------------------------------ investigation


@app.get("/investigation/{interaction_id}")
def get_investigation(interaction_id: str) -> dict:
    result = get_service().investigate_incident(interaction_id)
    if getattr(result, "found", True) is False:
        raise HTTPException(status_code=404, detail=result.message)
    return result.model_dump(mode="json")


@app.get("/investigation/{interaction_id}/history")
def get_investigation_history(interaction_id: str) -> dict:
    service = get_service()
    if service.get_audit_trace(interaction_id) is None:
        raise HTTPException(
            status_code=404, detail=f"No decision trace for {interaction_id}"
        )
    return {
        "interaction_id": interaction_id,
        "governance_history": [
            a.model_dump(mode="json")
            for a in service.governance_history(interaction_id)
        ],
    }


@app.post("/investigation/{interaction_id}/action")
def post_investigation_action(
    interaction_id: str, request: GovernanceActionRequest
) -> dict:
    from investigation.service import GovernanceError

    service = get_service()
    try:
        result = service.record_governance_action(
            interaction_id,
            action=request.action,
            actor=request.actor,
            comment=request.comment,
            reviewer_decision=(
                request.reviewer_decision.value
                if request.reviewer_decision is not None
                else None
            ),
        )
    except GovernanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if getattr(result, "found", True) is False:
        raise HTTPException(status_code=404, detail=result.message)
    return result.model_dump(mode="json")


@app.post("/investigation/{interaction_id}/counterfactual")
def post_investigation_counterfactual(
    interaction_id: str, request: InvestigationCounterfactualRequest
) -> dict:
    result = get_service().investigation_counterfactual(
        interaction_id, request.modified_fields
    )
    if getattr(result, "found", True) is False:
        raise HTTPException(status_code=404, detail=result.message)
    return result.model_dump(mode="json")


# ------------------------------------------------------------------ human governance override
# Reviewer-facing aliases over the incident-investigation governance track.
# (`POST /investigation/{id}/action` + `GET /investigation/{id}/history` are the
#  workflow-native equivalents.) The DecisionTrace is never mutated.


@app.post("/governance/action")
def post_governance_action(request: GovernanceOverrideRequest) -> dict:
    from investigation.service import GovernanceError

    service = get_service()
    try:
        result = service.record_governance_override(
            request.interaction_id,
            action_type=request.action_type,
            new_tier=(request.new_tier.value if request.new_tier is not None else None),
            justification=request.justification,
            reviewer_id=request.reviewer_id,
        )
    except GovernanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if getattr(result, "found", True) is False:
        raise HTTPException(status_code=404, detail=result.message)
    inv = result.model_dump(mode="json")
    return {
        "interaction_id": inv["interaction_id"],
        "original_decision": inv["original_decision"],          # immutable
        "effective_governed_decision": inv["effective_governed_decision"],
        "is_overridden": inv["is_overridden"],
        "governance_history": inv["governance_history"],
    }


@app.get("/governance/actions/{interaction_id}")
def get_governance_actions(interaction_id: str) -> dict:
    service = get_service()
    trace = service.get_audit_trace(interaction_id)
    if trace is None:
        raise HTTPException(
            status_code=404, detail=f"No decision trace for {interaction_id}"
        )
    original = trace.final_decision.decision.value
    history = service.governance_history(interaction_id)
    return {
        "interaction_id": interaction_id,
        "original_decision": original,                          # immutable
        "effective_governed_decision": service.governance.effective_decision(
            interaction_id, original
        ),
        "is_overridden": service.governance.effective_decision(interaction_id, original)
        != original,
        "history": [a.model_dump(mode="json") for a in history],
    }


# ------------------------------------------------------------------ governance intelligence


@app.get("/governance/overview")
def governance_overview() -> dict:
    return get_service().governance_report().overview.model_dump(mode="json")


@app.get("/governance/applications")
def governance_applications() -> dict:
    return get_service().governance_report().application_comparison.model_dump(mode="json")


@app.get("/governance/applications/{application}")
def governance_application(application: str) -> dict:
    comparison = get_service().governance_report().application_comparison
    match = next(
        (a for a in comparison.applications if a.application == application), None
    )
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No governed traffic for application '{application}'. Known: "
                f"{[a.application for a in comparison.applications]}"
            ),
        )
    return match.model_dump(mode="json")


@app.get("/governance/insights")
def governance_insights() -> dict:
    report = get_service().governance_report()
    return {"insights": [i.model_dump(mode="json") for i in report.insights]}


@app.get("/governance/signals")
def governance_signals() -> dict:
    report = get_service().governance_report()
    return {
        "summary": report.signals.model_dump(mode="json"),
        "signals": [s.model_dump(mode="json") for s in report.signal_details],
    }


@app.get("/governance/trends")
def governance_trends() -> dict:
    return get_service().governance_report().trends.model_dump(mode="json")


@app.get("/governance/recommendations")
def governance_recommendations(calibration: bool = False) -> dict:
    report = get_service().governance_report(with_calibration=calibration)
    return {
        "recommendations": [r.model_dump(mode="json") for r in report.recommendations],
        "disclaimer": "RECOMMENDATION ONLY — NOT APPLIED TO PRODUCTION.",
    }


# ------------------------------------------------------------------ incident intelligence


@app.get("/incidents")
def list_incidents() -> dict:
    report = get_service().incident_intelligence()
    return {
        "total_incidents": report.total_incidents,
        "incidents": [i.model_dump(mode="json") for i in report.incidents],
    }


@app.get("/incidents/patterns")
def incident_patterns() -> dict:
    report = get_service().incident_intelligence()
    return {
        "patterns": [p.model_dump(mode="json") for p in report.patterns],
        "clusters": [c.model_dump(mode="json") for c in report.clusters],
        "attributions": [a.model_dump(mode="json") for a in report.attributions],
        "reviewer_override_patterns": [
            op.model_dump(mode="json") for op in report.reviewer_override_patterns
        ],
    }


@app.get("/incidents/drift")
def incident_drift() -> dict:
    return get_service().incident_intelligence().drift.model_dump(mode="json")


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict:
    report = get_service().incident_intelligence()
    match = next(
        (
            i
            for i in report.incidents
            if i.incident_id == incident_id or i.interaction_id == incident_id
        ),
        None,
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"No incident '{incident_id}'.")
    return match.model_dump(mode="json")


# ------------------------------------------------------------------ adaptive guardrails


@app.get("/adaptive/report")
def adaptive_report(counterfactual: bool = False) -> dict:
    return (
        get_service()
        .adaptive_report(with_counterfactual=counterfactual)
        .model_dump(mode="json")
    )


@app.get("/adaptive/recommendations")
def adaptive_recommendations(counterfactual: bool = False) -> dict:
    recs = get_service().adaptive_recommendations(with_counterfactual=counterfactual)
    return {
        "recommendations": [r.model_dump(mode="json") for r in recs],
        "disclaimer": (
            "RECOMMENDATION ONLY. Approval means APPROVED_FOR_EVALUATION — never "
            "applied to production. config/settings.yaml is never modified."
        ),
    }


@app.get("/adaptive/recommendations/{recommendation_id}")
def adaptive_recommendation(recommendation_id: str) -> dict:
    from adaptive.service import RecommendationNotFound

    try:
        rec = get_service().adaptive_get_recommendation(recommendation_id)
    except RecommendationNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"No recommendation '{recommendation_id}'."
        ) from exc
    return rec.model_dump(mode="json")


@app.post("/adaptive/recommendations/{recommendation_id}/approve")
def adaptive_approve(recommendation_id: str, request: AdaptiveApprovalRequest) -> dict:
    from adaptive.service import RecommendationNotFound

    try:
        rec = get_service().adaptive_approve(
            recommendation_id, actor=request.actor, comment=request.comment
        )
    except RecommendationNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"No recommendation '{recommendation_id}'."
        ) from exc
    return {
        "recommendation": rec.model_dump(mode="json"),
        "production_configuration_status": "UNCHANGED",
        "note": "APPROVED_FOR_EVALUATION — not applied to production.",
    }


@app.post("/adaptive/recommendations/{recommendation_id}/reject")
def adaptive_reject(recommendation_id: str, request: AdaptiveApprovalRequest) -> dict:
    from adaptive.service import RecommendationNotFound

    try:
        rec = get_service().adaptive_reject(
            recommendation_id, actor=request.actor, comment=request.comment
        )
    except RecommendationNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"No recommendation '{recommendation_id}'."
        ) from exc
    return {"recommendation": rec.model_dump(mode="json")}


# ------------------------------------------------------------------ enterprise command center


@app.get("/command-center")
def command_center() -> dict:
    """Read-only enterprise command-center view (no calibration / counterfactual)."""
    return get_service().command_center_view().model_dump(mode="json")


@app.get("/monitoring/operational")
def monitoring_operational() -> dict:
    """
    Full ``OperationalMonitoringReport`` over the stored audit log — FAST/DEEP
    split, semantic-bypass savings, multi-turn critical-floor accumulation,
    risk distribution and incident digest. Observes stored traces only; never
    re-runs a detector or the decision engine.
    """
    return get_service().get_operational_monitoring().model_dump(mode="json")


@app.get("/applications")
def applications() -> dict:
    rows = get_service().application_posture()
    return {"applications": [r.model_dump(mode="json") for r in rows]}


@app.get("/applications/{application}")
def application_detail(application: str) -> dict:
    rows = get_service().application_posture()
    match = next((r for r in rows if r.application == application), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"No governed traffic for '{application}'. Known: "
            f"{[r.application for r in rows]}",
        )
    return match.model_dump(mode="json")


@app.get("/governance/timeline")
def governance_timeline(interaction_id: str | None = None) -> dict:
    return (
        get_service()
        .governance_timeline(focus_interaction_id=interaction_id)
        .model_dump(mode="json")
    )


@app.get("/incidents/{incident_id}/investigation")
def incident_investigation(incident_id: str) -> dict:
    """Full investigation for an incident (reuses Phase-8 investigation)."""
    service = get_service()
    report = service.incident_intelligence()
    match = next(
        (
            i
            for i in report.incidents
            if i.incident_id == incident_id or i.interaction_id == incident_id
        ),
        None,
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"No incident '{incident_id}'.")
    result = service.investigate_incident(match.interaction_id)
    if getattr(result, "found", True) is False:
        raise HTTPException(status_code=404, detail=result.message)
    related = [
        i.model_dump(mode="json")
        for i in report.incidents
        if i.signature == match.signature and i.interaction_id != match.interaction_id
    ]
    return {
        "incident": match.model_dump(mode="json"),
        "investigation": result.model_dump(mode="json"),
        "related_incidents": related,
        "related_by": "shared deterministic structured risk signature",
    }


def _auto_id() -> str:
    from uuid import uuid4

    return f"INT-API-{uuid4().hex[:12]}"
