"""
python -m monitoring

Runs the REAL decision pipeline over the demo scenarios plus a sample of
synthetic *production* traffic (never the evaluation dataset / ground
truth), feeds the resulting DecisionTrace collection into the Phase-8
OperationalMonitor, and prints the operational report.

Every number printed comes from an actual trace.
"""

from __future__ import annotations

import contextlib
import random
import sys
from datetime import datetime

from api.service import ControlPlaneService
from data.generator import generate_interactions
from monitoring.engine import OperationalMonitor
from monitoring.schemas import IncidentSeverity
from tests import scenarios

_SMOKE_TS = datetime(2026, 8, 28, 12, 0, 0)


def _collect_traces(service: ControlPlaneService, traffic_sample: int = 250) -> None:
    """Run demo scenarios + synthetic production traffic through the service."""
    demo = [f() for f in scenarios.ALL_SINGLE_TURN.values()]
    demo += list(scenarios.scenario_g_multi_turn())
    demo.append(scenarios.scenario_i_policy_counterfactual()[0])
    demo.append(scenarios.scenario_j_consequence_counterfactual()[0])
    for interaction in demo:
        service.check(interaction, timestamp=interaction.timestamp)

    cfg = service._config
    rng = random.Random(cfg["seed"])
    for interaction in generate_interactions(cfg, rng)[:traffic_sample]:
        service.check(interaction, timestamp=interaction.timestamp)


def _seed_demo_feedback(service: ControlPlaneService) -> None:
    """A few illustrative reviewer feedback entries so the section renders."""
    traces = service.all_traces()
    blocked = [t for t in traces if t.final_decision.decision.value == "BLOCK"][:2]
    human = [t for t in traces if t.final_decision.decision.value == "HUMAN_REVIEW"][:2]
    for t in blocked:
        with contextlib.suppress(Exception):
            service.submit_feedback(
                interaction_id=t.interaction_id,
                system_decision=None,
                outcome="approved",
                reviewer="demo-reviewer",
            )
    for t in human:
        with contextlib.suppress(Exception):
            service.submit_feedback(
                interaction_id=t.interaction_id,
                system_decision=None,
                reviewer_decision="VERIFY",
                outcome="modified",
                reviewer="demo-reviewer",
            )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")

    print("[monitoring] running the real pipeline over demo + synthetic traffic ...")
    service = ControlPlaneService(fit_cost_baseline=True)
    _collect_traces(service)
    _seed_demo_feedback(service)
    traces = service.all_traces()
    print(f"[monitoring] collected {len(traces)} real DecisionTrace records\n")

    report = OperationalMonitor().report(
        traces, feedback_store=service.feedback, generated_at=_SMOKE_TS
    )
    s = report.snapshot
    v = report.verification
    d = report.incident_digest

    rule = "=" * 60
    print(rule)
    print("CONTROLPLANE OPERATIONAL MONITOR")
    print(rule)
    print(f"\nInteractions        {s.total_interactions}")
    print("\nDecisions")
    print(f"  ALLOW             {_pct(s.allow_rate)}")
    print(f"  ANNOTATE          {_pct(s.annotate_rate)}")
    print(f"  VERIFY            {_pct(s.verify_rate)}")
    print(f"  HUMAN_REVIEW      {_pct(s.human_review_rate)}")
    print(f"  BLOCK             {_pct(s.block_rate)}")
    print("\nRisk")
    print(f"  Average           {_num(s.average_risk)}")
    print(f"  P95               {_num(s.p95_risk)}")
    print(f"  Avg confidence    {_num(s.average_confidence)}")
    print(f"  Multi-risk rate   {_pct(s.multi_risk_rate)}")
    print(f"  High-consequence  {_pct(s.high_consequence_rate)}")
    print(f"  High-criticality  {_pct(s.high_criticality_rate)}")
    print("\nVerification")
    print(f"  FAST              {_pct(v.fast_rate)}")
    print(f"  DEEP              {_pct(v.deep_rate)}")
    print(f"  Avg total verif.  {_num(v.average_total_verification_latency_ms)} ms")
    print("\nIncidents")
    print(f"  Total             {d.total}  (rate {_pct(d.incident_rate)})")
    print(f"  Critical          {d.by_severity.get(IncidentSeverity.CRITICAL.value, 0)}")
    print(f"  High              {d.by_severity.get(IncidentSeverity.HIGH.value, 0)}")
    print(f"  Medium            {d.by_severity.get(IncidentSeverity.MEDIUM.value, 0)}")
    print(f"  By trigger        {d.by_trigger}")

    print("\nTop reason codes")
    for rc in report.reason_codes[:6]:
        print(f"  {rc.reason_code:28s} {rc.count}")

    print("\nApplications")
    for app in report.applications:
        print(
            f"  {app.application:30s} n={app.interaction_count:<4d} "
            f"risk={_num(app.average_risk)} HR={_pct(app.human_review_rate)} "
            f"BLOCK={_pct(app.block_rate)} DEEP={_pct(app.deep_path_rate)} "
            f"dominant={app.dominant_risk_dimension}"
        )

    print("\nDetectors")
    for det in report.detectors:
        print(
            f"  {det.detector:16s} avg_risk={_num(det.average_risk)} "
            f"high_risk_rate={_pct(det.high_risk_rate)} "
            f"dominant_count={det.dominant_dimension_count}"
        )

    print("\nTrend (first half -> second half)")
    for mt in report.trend.metrics:
        print(
            f"  {mt.metric:20s} {_num(mt.first_half_value)} -> {_num(mt.second_half_value)}  "
            f"({mt.direction.value})"
        )

    print("\nOperational shifts (historical -> recent)")
    for sh in report.operational_shift.shifts:
        print(
            f"  {sh.metric:20s} {_num(sh.baseline_value)} -> {_num(sh.recent_value)}  "
            f"({sh.direction})"
        )

    fb = report.feedback
    print("\nFeedback (governance signal, NOT ground truth)")
    print(
        f"  count={fb.feedback_count}  approved={fb.approved}  modified={fb.modified}  "
        f"rejected={fb.rejected}  override_rate={_pct(fb.override_rate)}"
    )

    top_incidents = report.incidents[:5]
    if top_incidents:
        print("\nTop incidents")
        for inc in top_incidents:
            print(
                f"  [{inc.severity.value:8s}] {inc.interaction_id}  {inc.application}  "
                f"{inc.decision}  risk={inc.overall_risk:.2f}  triggers={inc.triggers}"
            )

    print("\n[operational monitoring report generated — no pipeline was re-run]")


if __name__ == "__main__":
    main()
