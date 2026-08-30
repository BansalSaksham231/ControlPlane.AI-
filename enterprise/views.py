"""
Pure view builders for the Enterprise Command Center.

Every function takes the reports the earlier phases already produced (plus
stored traces) and reshapes them for a judge-facing screen. No risk
formula, no pipeline call, no ground truth, no raw PII.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from decision.schemas import DecisionTrace
from enterprise.schemas import (
    ApplicationPostureBadge,
    ApplicationPostureRow,
    ArchitectureStage,
    ExecutiveKpiStrip,
    ExecutiveSummary,
    HeatmapCell,
    LiveDecisionRow,
    RiskHeatmap,
    RiskPosture,
    TechnicalArchitecture,
    WhatIfMetricRow,
    WhatIfResult,
)
from monitoring.incidents import classify_incident
from monitoring.schemas import MonitoringConfig

__all__ = [
    "build_kpi_strip",
    "build_risk_posture",
    "build_application_posture",
    "build_heatmap",
    "recent_decisions",
    "build_executive_summary",
    "build_technical_architecture",
    "whatif_from_selection",
]

_HIGH_RISK = 0.50
_DIMS = ("performance", "responsibility", "cost", "consequence")


def _ts_key(ts):
    return ts.astimezone(ts.tzinfo).replace(tzinfo=None) if ts.tzinfo else ts


# ----------------------------------------------------------------------
# A. KPI strip
# ----------------------------------------------------------------------


def build_kpi_strip(monitoring, governance, incident_intel, adaptive) -> ExecutiveKpiStrip:
    s = monitoring.snapshot
    if s.total_interactions == 0:
        return ExecutiveKpiStrip(has_data=False)
    override = governance.overview.reviewer_disagreement.override_rate or 0.0
    active = sum(
        1 for r in adaptive.recommendations if r.type.value != "NO_ACTION"
        and r.status.value in ("RECOMMENDED_FOR_REVIEW", "SIMULATED", "DETECTED")
    )
    return ExecutiveKpiStrip(
        has_data=True,
        total_interactions=s.total_interactions,
        allow_rate=s.allow_rate,
        annotate_rate=s.annotate_rate,
        verify_rate=s.verify_rate,
        human_review_rate=s.human_review_rate,
        block_rate=s.block_rate,
        average_risk=s.average_risk,
        average_confidence=s.average_confidence,
        fast_rate=s.fast_path_rate,
        deep_rate=s.deep_path_rate,
        incident_rate=monitoring.incident_digest.incident_rate,
        override_rate=round(override, 4),
        potential_drift_count=incident_intel.drift.potential_drift_count,
        active_recommendations=active,
    )


# ----------------------------------------------------------------------
# B. risk posture
# ----------------------------------------------------------------------


def build_risk_posture(traces: list[DecisionTrace], monitoring) -> RiskPosture:
    n = len(traces)
    if n == 0:
        return RiskPosture(high_risk_threshold=_HIGH_RISK)

    def _mean(get) -> float:
        return round(sum(get(t) for t in traces) / n, 6)

    def _high(get) -> int:
        return sum(1 for t in traces if get(t) >= _HIGH_RISK)

    perf = lambda t: t.final_decision.performance_risk
    resp = lambda t: t.final_decision.responsibility_risk
    cost = lambda t: t.final_decision.cost_risk
    overall = lambda t: t.final_decision.overall_risk

    means = {"performance": _mean(perf), "responsibility": _mean(resp), "cost": _mean(cost)}
    dominant = max(means, key=means.get) if any(means.values()) else None

    risk_trend = None
    for mt in monitoring.trend.metrics:
        if mt.metric == "average_risk":
            risk_trend = mt.direction.value
            break

    return RiskPosture(
        performance_average=means["performance"],
        responsibility_average=means["responsibility"],
        cost_average=means["cost"],
        overall_average=_mean(overall),
        performance_high_risk_count=_high(perf),
        responsibility_high_risk_count=_high(resp),
        cost_high_risk_count=_high(cost),
        overall_high_risk_count=_high(overall),
        high_risk_threshold=_HIGH_RISK,
        dominant_dimension=dominant,
        risk_trend=risk_trend,
    )


# ----------------------------------------------------------------------
# posture banding  (a LABEL, not a score)
# ----------------------------------------------------------------------


def _posture_band(average_risk: float, oversight_rate: float, incident_rate: float) -> tuple[str, str]:
    if average_risk >= 0.45 or oversight_rate >= 0.35 or incident_rate >= 0.30:
        return "HIGH", (
            f"average risk {average_risk:.2f}, human oversight {oversight_rate:.0%}, "
            f"incident rate {incident_rate:.0%}"
        )
    if average_risk < 0.22 and oversight_rate < 0.12 and incident_rate < 0.10:
        return "LOW", (
            f"average risk {average_risk:.2f}, human oversight {oversight_rate:.0%}, "
            f"incident rate {incident_rate:.0%}"
        )
    return "MODERATE", (
        f"average risk {average_risk:.2f}, human oversight {oversight_rate:.0%}, "
        f"incident rate {incident_rate:.0%}"
    )


# ----------------------------------------------------------------------
# C. application posture
# ----------------------------------------------------------------------


def build_application_posture(
    traces: list[DecisionTrace], monitoring, incident_intel, adaptive, governance=None
) -> list[ApplicationPostureRow]:
    app_summ = {a.application: a for a in monitoring.applications}
    incidents_by_app: Counter[str] = Counter(i.application for i in incident_intel.incidents)
    # reviewer-override (disagreement) rate per app — reuse the Phase-9
    # application comparison; a governance signal, never proof of error.
    override_by_app: dict[str, float] = {}
    if governance is not None:
        for a in governance.application_comparison.applications:
            if a.reviewer_override_rate is not None:
                override_by_app[a.application] = a.reviewer_override_rate

    # adaptive recommendations by app, preserving the adaptive engine's own
    # priority order (recommendations list is already priority-sorted)
    rec_by_app: dict[str, list[str]] = {}
    for rec in adaptive.recommendations:
        if rec.application and rec.type.value != "NO_ACTION":
            rec_by_app.setdefault(rec.application, [])
            if rec.type.value not in rec_by_app[rec.application]:
                rec_by_app[rec.application].append(rec.type.value)

    rows: list[ApplicationPostureRow] = []
    for app in sorted(app_summ):
        a = app_summ[app]
        n = a.interaction_count
        group = [t for t in traces if t.application == app]
        high = sum(1 for t in group if t.final_decision.overall_risk >= _HIGH_RISK)
        oversight = a.human_review_rate + a.block_rate
        inc_rate = round(incidents_by_app.get(app, 0) / n, 4) if n else 0.0
        band, why = _posture_band(a.average_risk, oversight, inc_rate)
        recs = rec_by_app.get(app, [])
        rows.append(
            ApplicationPostureRow(
                application=app,
                interactions=n,
                average_risk=a.average_risk,
                high_risk_rate=round(high / n, 4) if n else 0.0,
                human_review_rate=a.human_review_rate,
                block_rate=a.block_rate,
                fast_rate=a.fast_path_rate,
                deep_rate=a.deep_path_rate,
                dominant_risk_dimension=a.dominant_risk_dimension,
                incident_count=incidents_by_app.get(app, 0),
                override_rate=override_by_app.get(app),
                posture=band,
                posture_rationale=why,
                recommended_posture=(recs[0] if recs else None),
                open_recommendations=recs,
            )
        )
    return rows


# ----------------------------------------------------------------------
# D. heatmap
# ----------------------------------------------------------------------


def build_heatmap(traces: list[DecisionTrace]) -> RiskHeatmap:
    by_app: dict[str, list[DecisionTrace]] = {}
    for t in traces:
        by_app.setdefault(t.application, []).append(t)
    apps = sorted(by_app)
    getters = {
        "performance": lambda t: t.final_decision.performance_risk,
        "responsibility": lambda t: t.final_decision.responsibility_risk,
        "cost": lambda t: t.final_decision.cost_risk,
        "consequence": lambda t: t.consequence.consequence_score,
    }
    cells: list[HeatmapCell] = []
    for app in apps:
        group = by_app[app]
        for dim in _DIMS:
            get = getters[dim]
            try:
                value = round(sum(get(t) for t in group) / len(group), 4) if group else None
            except Exception:  # pragma: no cover
                value = None
            cells.append(HeatmapCell(application=app, dimension=dim, value=value))
    return RiskHeatmap(applications=apps, dimensions=list(_DIMS), cells=cells)


# ----------------------------------------------------------------------
# live decision feed
# ----------------------------------------------------------------------


def recent_decisions(traces: list[DecisionTrace], limit: int = 25) -> list[LiveDecisionRow]:
    ordered = sorted(traces, key=lambda t: (_ts_key(t.timestamp), t.interaction_id), reverse=True)
    out: list[LiveDecisionRow] = []
    for t in ordered[:limit]:
        fd = t.final_decision
        out.append(
            LiveDecisionRow(
                interaction_id=t.interaction_id,
                application=t.application,
                decision=fd.decision.value,
                overall_risk=fd.overall_risk,
                confidence=fd.decision_confidence,
                verification_path=(t.verification_path or "DEEP").upper(),
                dominant_dimension=getattr(t.fusion, "dominant_dimension", None),
                reason_codes=list(fd.reason_codes),
                human_review_required=fd.decision.value in ("HUMAN_REVIEW", "BLOCK"),
                timestamp=t.timestamp,
                source="STORED_TRACE",
            )
        )
    return out


# ----------------------------------------------------------------------
# executive summary / story mode
# ----------------------------------------------------------------------


def build_executive_summary(
    monitoring, governance, incident_intel, adaptive, application_posture
) -> ExecutiveSummary:
    s = monitoring.snapshot
    if s.total_interactions == 0:
        return ExecutiveSummary(has_data=False)

    high_risk = sum(
        b.count
        for b in monitoring.risk_distribution.buckets
        if b.bucket_name in ("HIGH", "CRITICAL")
    )
    oversight = s.human_review_count + s.block_count

    top_dim = None
    detectors = sorted(
        governance.overview.detector_contribution.model_dump().items(),
        key=lambda kv: -(kv[1] if isinstance(kv[1], int) else 0),
    )
    for name, cnt in detectors:
        if name.endswith("_driven_incidents") and isinstance(cnt, int) and cnt > 0:
            top_dim = name.replace("_driven_incidents", "")
            break

    top_issue = None
    if governance.insights:
        gi = governance.insights[0]
        top_issue = f"{gi.title}"
    elif incident_intel.reviewer_override_patterns:
        op = incident_intel.reviewer_override_patterns[0]
        top_issue = f"Repeated reviewer override {op.transition} in {op.application}"

    action = None
    actionable = [r for r in adaptive.recommendations if r.type.value != "NO_ACTION"]
    if actionable:
        r0 = actionable[0]
        action = r0.type.value + (f" — {r0.application}" if r0.application else "")

    return ExecutiveSummary(
        has_data=True,
        ai_systems_monitored=len({r.application for r in application_posture}),
        interactions_evaluated=s.total_interactions,
        high_risk_interactions=high_risk,
        human_oversight_count=oversight,
        potential_drift_signals=incident_intel.drift.potential_drift_count,
        open_governance_recommendations=len(actionable),
        application_posture=[
            ApplicationPostureBadge(application=r.application, posture=r.posture)
            for r in application_posture
        ],
        top_risk_dimension=top_dim,
        top_governance_issue=top_issue,
        recommended_action=action,
    )


# ----------------------------------------------------------------------
# technical architecture (static map — nothing is executed)
# ----------------------------------------------------------------------


def build_technical_architecture() -> TechnicalArchitecture:
    return TechnicalArchitecture(
        stages=[
            ArchitectureStage(stage="INPUT", module="data/schemas.py (Interaction)",
                              description="Production-visible interaction only — never ground truth."),
            ArchitectureStage(stage="DETECT", module="detectors/{performance,responsibility,cost}",
                              description="Independent risk dimensions (lexical, deterministic)."),
            ArchitectureStage(stage="VERIFY", module="verification/router.py",
                              description="FAST / DEEP progressive verification."),
            ArchitectureStage(stage="ASSESS", module="consequence/engine.py · criticality/engine.py",
                              description="How bad if wrong; how much it matters."),
            ArchitectureStage(stage="FUSE", module="fusion/engine.py",
                              description="Weighted risk + severity pull + hard floors."),
            ArchitectureStage(stage="POLICY", module="policy/engine.py",
                              description="Per-application risk bands + rules."),
            ArchitectureStage(stage="DECIDE", module="decision/engine.py",
                              description="ALLOW / ANNOTATE / VERIFY / HUMAN_REVIEW / BLOCK + DecisionTrace."),
            ArchitectureStage(stage="EXPLAIN", module="explainability/builder.py · decision/replay.py",
                              description="Redacted reconstruction — no detector re-run."),
            ArchitectureStage(stage="MONITOR", module="monitoring/engine.py",
                              description="Operational aggregation over stored traces."),
            ArchitectureStage(stage="GOVERN", module="governance/ · investigation/",
                              description="Insights, reviewer signals, incident investigation."),
            ArchitectureStage(stage="INCIDENT INTELLIGENCE", module="incident/",
                              description="Deterministic clustering, patterns, drift, attribution."),
            ArchitectureStage(stage="ADAPT", module="adaptive/",
                              description="Recommendation + counterfactual (calibration.sweep/select), safety-first."),
            ArchitectureStage(stage="HUMAN APPROVAL", module="adaptive/approval.py",
                              description="APPROVED_FOR_EVALUATION only — no deployment path."),
        ]
    )


# ----------------------------------------------------------------------
# what-if / counterfactual view  (maps a ConfigurationSelection)
# ----------------------------------------------------------------------


def _direction(cur: float | None, cand: float | None, lower_better: bool) -> str:
    if cur is None or cand is None or abs(cand - cur) < 1e-6:
        return "flat"
    up = cand > cur
    return "up" if up else "down"


def whatif_from_selection(
    selection: Any, *, application: str | None, control: str
) -> WhatIfResult:
    base = selection.baseline_result
    sel = selection.selected_result
    passed = selection.status == "SELECTED" and sel is not None

    def row(metric, cur, cand, lower_better):
        return WhatIfMetricRow(
            metric=metric,
            current=None if cur is None else round(cur, 4),
            candidate=None if cand is None else round(cand, 4),
            direction=_direction(cur, cand, lower_better),
        )

    metrics = [
        row("recall", base.safety.recall, sel.safety.recall if sel else None, False),
        row("precision", base.safety.precision, sel.safety.precision if sel else None, False),
        row("false_positive_rate", base.safety.false_positive_rate,
            sel.safety.false_positive_rate if sel else None, True),
        row("missed_risk_rate", base.safety.missed_risk_rate,
            sel.safety.missed_risk_rate if sel else None, True),
        row("fast_rate", base.efficiency.fast_path_rate,
            sel.efficiency.fast_path_rate if sel else None, False),
        row("deep_rate", base.efficiency.deep_path_rate,
            sel.efficiency.deep_path_rate if sel else None, True),
        row("human_review_rate", base.efficiency.human_review_rate,
            sel.efficiency.human_review_rate if sel else None, True),
        row("average_latency_ms", base.efficiency.average_latency_ms,
            sel.efficiency.average_latency_ms if sel else None, True),
    ]

    if not passed:
        safety_status = "NO_CANDIDATE"
        interpretation = (
            "No configuration among the tested candidates satisfies the configured "
            "safety constraints. No safe configuration recommended — constraints were "
            "NOT relaxed."
        )
    else:
        d_recall = sel.safety.recall - base.safety.recall
        d_missed = sel.safety.missed_risk_rate - base.safety.missed_risk_rate
        d_fast = sel.efficiency.fast_path_rate - base.efficiency.fast_path_rate
        safety_status = "PASS"
        if d_recall < -1e-6 or d_missed > 1e-6:
            interpretation = (
                f"Candidate WORSENS safety: recall {d_recall:+.2f}, missed-risk "
                f"{d_missed:+.2f}. Not recommended despite efficiency gains."
            )
        elif abs(d_recall) < 1e-6 and abs(d_missed) < 1e-6 and abs(d_fast) < 0.02:
            interpretation = (
                "No measured safety improvement. Candidate differs primarily in "
                "efficiency / routing."
            )
        elif abs(d_recall) < 1e-6 and abs(d_missed) < 1e-6:
            interpretation = (
                f"Candidate improves FAST-path coverage ({base.efficiency.fast_path_rate:.0%}"
                f" -> {sel.efficiency.fast_path_rate:.0%}) without reducing measured recall "
                f"({base.safety.recall:.2f}) on the evaluation set."
            )
        else:
            interpretation = (
                f"Candidate changes recall {d_recall:+.2f}, missed-risk {d_missed:+.2f}, "
                f"FAST-path {d_fast:+.2f}."
            )

    return WhatIfResult(
        application=application,
        control=control,
        current_configuration={k: round(v, 4) for k, v in base.resolved_thresholds.items()},
        candidate_configuration=(
            {k: round(v, 4) for k, v in sel.resolved_thresholds.items()} if sel else None
        ),
        current_value=base.resolved_thresholds.get(control),
        candidate_value=(sel.resolved_thresholds.get(control) if sel else None),
        metrics=metrics,
        current_decision_distribution=dict(base.decision_counts),
        candidate_decision_distribution=dict(sel.decision_counts) if sel else {},
        safety_status=safety_status,
        safety_constraints=selection.safety_constraints.as_dict(),
        safety_violations=list(selection.baseline_violations) if not passed else [],
        interpretation=interpretation,
    )
