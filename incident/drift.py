"""
Operational drift detection.

Builds on the Phase-9 governance trend system (reused, not duplicated) and
adds per-detector, per-policy-rule and per-application scopes. Transparent
historical-window vs recent-window comparison. Nothing here claims
statistical significance or model degradation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from governance.schemas import GovernanceConfig
from governance.trends import build_trends
from incident.schemas import DriftDirection, DriftReport, DriftSignal, Phase10IncidentConfig
from incident.store import ts_key

__all__ = ["build_drift_report"]

_DIR = {
    "increasing": DriftDirection.INCREASING,
    "decreasing": DriftDirection.DECREASING,
    "stable": DriftDirection.STABLE,
}


def _classify(
    baseline: float | None,
    recent: float | None,
    n1: int,
    n2: int,
    config: Phase10IncidentConfig,
) -> tuple[DriftDirection, str, bool]:
    if baseline is None or recent is None:
        return DriftDirection.STABLE, "STABLE", False
    delta = recent - baseline
    magnitude = abs(delta)
    sufficient = n1 >= config.drift_min_window_samples and n2 >= config.drift_min_window_samples
    if magnitude <= config.drift_stable_band:
        return (DriftDirection.STABLE, "STABLE", sufficient)
    direction = DriftDirection.INCREASING if delta > 0 else DriftDirection.DECREASING
    if magnitude >= config.drift_potential_magnitude and sufficient:
        return direction, "POTENTIAL_DRIFT", sufficient
    return direction, "TREND", sufficient


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def build_drift_report(
    traces: list[DecisionTrace],
    governance_actions: list[Any] | None = None,
    *,
    config: Phase10IncidentConfig | None = None,
) -> DriftReport:
    config = config or Phase10IncidentConfig()
    governance_actions = governance_actions or []
    ordered = sorted(traces, key=lambda t: (ts_key(t.timestamp), t.interaction_id))
    n = len(ordered)
    timestamped = len({ts_key(t.timestamp) for t in ordered}) > 1
    basis = (
        "timestamped (historical half vs recent half of time-ordered traces)"
        if timestamped
        else "sequence-based (historical half vs recent half of ordered traces)"
    )

    if n < 4:
        return DriftReport(
            basis=basis, historical_window_n=n, recent_window_n=0, signals=[]
        )

    mid = n // 2
    first, second = ordered[:mid], ordered[mid:]
    signals: list[DriftSignal] = []

    # --- global metrics: reuse the governance trend engine ---
    gov_trends = build_trends(
        traces,
        governance_actions,
        config=GovernanceConfig(
            trend_stable_band=config.drift_stable_band,
            potential_drift_magnitude=config.drift_potential_magnitude,
            potential_drift_min_samples=config.drift_min_window_samples,
            high_risk_threshold=config.high_risk_threshold,
            low_confidence_threshold=config.low_confidence_threshold,
        ),
    )
    for ts in gov_trends.signals:
        direction, label, sufficient = _classify(
            ts.baseline, ts.current, len(first), len(second), config
        )
        delta = (
            None
            if (ts.baseline is None or ts.current is None)
            else round(ts.current - ts.baseline, 6)
        )
        signals.append(
            DriftSignal(
                metric=ts.metric,
                scope="global",
                baseline=ts.baseline,
                recent=ts.current,
                delta=delta,
                direction=direction,
                signal=label,
                sample_sufficient=sufficient,
                explanation=ts.explanation,
            )
        )

    # --- FAST -> DEEP shift (already covered by deep_rate, add explicit label) ---
    # --- per-detector mean-risk shift ---
    detectors = {
        "performance": lambda t: t.final_decision.performance_risk,
        "responsibility": lambda t: t.final_decision.responsibility_risk,
        "cost": lambda t: t.final_decision.cost_risk,
    }
    for name, get in detectors.items():
        b, r = _mean([get(t) for t in first]), _mean([get(t) for t in second])
        direction, label, sufficient = _classify(b, r, len(first), len(second), config)
        signals.append(
            DriftSignal(
                metric="mean_risk",
                scope=f"detector:{name}",
                baseline=b,
                recent=r,
                delta=None if (b is None or r is None) else round(r - b, 6),
                direction=direction,
                signal=label,
                sample_sufficient=sufficient,
                explanation=(
                    f"{name} detector mean risk moved from {b:.3f} to {r:.3f} "
                    f"(historical n={len(first)}, recent n={len(second)})."
                    if b is not None and r is not None
                    else f"{name} detector mean risk not computable."
                ),
            )
        )

    # --- per-application overall-risk shift ---
    apps = sorted({t.application for t in ordered})
    for app in apps:
        fa = [t for t in first if t.application == app]
        sa = [t for t in second if t.application == app]
        b = _mean([t.final_decision.overall_risk for t in fa])
        r = _mean([t.final_decision.overall_risk for t in sa])
        direction, label, sufficient = _classify(b, r, len(fa), len(sa), config)
        if b is None or r is None:
            continue
        signals.append(
            DriftSignal(
                metric="average_risk",
                scope=f"application:{app}",
                baseline=b,
                recent=r,
                delta=round(r - b, 6),
                direction=direction,
                signal=label,
                sample_sufficient=sufficient,
                explanation=(
                    f"'{app}' average risk moved from {b:.3f} (n={len(fa)}) to "
                    f"{r:.3f} (n={len(sa)})."
                ),
            )
        )

    # --- policy-rule firing-rate shift (top rules only) ---
    def _fire_rate(group: list[DecisionTrace], rule: str) -> float | None:
        k = len(group)
        if not k:
            return None
        return round(
            sum(1 for t in group for e in t.policy.rule_trace if e.fired and e.rule == rule) / k,
            6,
        )

    fired_all: Counter[str] = Counter()
    for t in ordered:
        fired_all.update({e.rule for e in t.policy.rule_trace if e.fired})
    for rule, _ in sorted(fired_all.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
        if rule == "RISK_BAND":
            continue
        b, r = _fire_rate(first, rule), _fire_rate(second, rule)
        direction, label, sufficient = _classify(b, r, len(first), len(second), config)
        if b is None or r is None:
            continue
        signals.append(
            DriftSignal(
                metric="fire_rate",
                scope=f"policy_rule:{rule}",
                baseline=b,
                recent=r,
                delta=round(r - b, 6),
                direction=direction,
                signal=label,
                sample_sufficient=sufficient,
                explanation=(
                    f"policy rule {rule} fired on {b:.1%} of interactions historically "
                    f"vs {r:.1%} recently."
                ),
            )
        )

    potential = sum(1 for s in signals if s.signal == "POTENTIAL_DRIFT")
    return DriftReport(
        basis=basis,
        historical_window_n=len(first),
        recent_window_n=len(second),
        potential_drift_count=potential,
        signals=signals,
    )
