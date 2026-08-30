"""
Phase 8 — Operational Risk Monitoring + Incident Intelligence.

``OperationalMonitor.report(traces)`` turns a collection of real
:class:`decision.schemas.DecisionTrace` records into an
:class:`OperationalMonitoringReport`: a flat snapshot, per-application and
per-detector risk summaries, reason-code and verification breakdowns,
**incident intelligence** (a conservative, documented incident rule + a
transparent CRITICAL / HIGH / MEDIUM severity), a first-half/second-half
**trend**, a recent-vs-historical **operational-shift** signal, and a
governance **feedback** summary.

Boundaries (enforced by tests)
------------------------------
* It NEVER re-runs the pipeline — no detector, engine, fusion, policy or
  verification call. It only reads fields already on the trace.
* It NEVER reads ground truth / evaluation labels and never imports
  ``evaluation``. Operational monitoring and offline evaluation are
  separate.
* It exposes only structured, already-safe fields — no claim text, no
  response text, no ``matched_text`` / raw PII.
* It is deterministic for a given trace collection: stable sorting, no
  randomness, no wall-clock reads.

Incident definition (conservative, config-driven — ``MonitoringConfig``)
----------------------------------------------------------------------
An interaction is an INCIDENT when **any** of these hold:
  * decision == ``BLOCK``
  * decision == ``HUMAN_REVIEW``
  * ``fusion.multi_risk`` is true (two+ risk dimensions elevated at once)
  * reason codes contain ``HIGH_CONSEQUENCE`` **and**
    ``overall_risk >= elevated_risk_threshold``
A plain ``VERIFY`` or a moderate single-dimension risk is **not** an
incident.

Incident severity (transparent — uses only existing trace fields, never
changes the decision)
  * ``CRITICAL`` — decision == ``BLOCK``, or ``overall_risk >=
    critical_risk_threshold``, or ``CRITICAL_PII`` in the reason codes.
  * ``HIGH``     — decision == ``HUMAN_REVIEW``, or
    (``consequence_score >= high_consequence_threshold`` and
     ``overall_risk >= elevated_risk_threshold``), or
    (``multi_risk`` and ``overall_risk >= elevated_risk_threshold``).
  * ``MEDIUM``   — an incident that matches none of the above.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from monitoring.incidents import (
    INCIDENT_DEFINITION as _INCIDENT_DEFINITION,
    SEVERITY_RANK as _SEVERITY_RANK,
    classify_incident,
)
from monitoring.metrics import (
    bucket_index,
    count_by,
    mean_or_none,
    percentile_or_none,
    rate_or_none,
)
from monitoring.schemas import (
    ApplicationRiskSummary,
    DataQualityReport,
    DetectorRiskSummary,
    IncidentDigest,
    IncidentSeverity,
    IncidentSummary,
    MetricTrend,
    MonitoredFeedbackSummary,
    MonitoringConfig,
    MonitoringSnapshot,
    MultiTurnSummary,
    OperationalMonitoringReport,
    OperationalShift,
    OperationalShiftReport,
    ReasonCodeSummary,
    RiskBucket,
    RiskDistribution,
    TrendAnalysis,
    TrendDirection,
    VerificationSummary,
)
from monitoring.service import MonitoringService

__all__ = ["OperationalMonitor"]

_TIERS = ("ALLOW", "ANNOTATE", "VERIFY", "HUMAN_REVIEW", "BLOCK")
_DETECTORS = ("performance", "responsibility", "cost")


def _ts_key(ts: datetime) -> datetime:
    """
    A naive-UTC datetime usable as a sort key even when the trace
    collection mixes timezone-aware and timezone-naive timestamps
    (e.g. live ``datetime.now(utc)`` checks alongside fixed demo
    timestamps).
    """
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _rate0(numerator: int, denominator: int) -> float:
    """Rate that is a real 0.0 (not None) when the denominator is zero."""
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean0(values: Iterable[float]) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def _p95_0(values: Iterable[float]) -> float:
    v = percentile_or_none(list(values), 95)
    return 0.0 if v is None else v


def _path(trace: DecisionTrace) -> str:
    return (trace.verification_path or "DEEP").upper()


def _dominant(traces: list[DecisionTrace]) -> str | None:
    dims = [
        t.fusion.dominant_dimension
        for t in traces
        if getattr(t.fusion, "dominant_dimension", None)
    ]
    if not dims:
        return None
    counts = Counter(dims)
    top = max(counts.values())
    # deterministic tie-break: alphabetical
    return sorted(d for d, c in counts.items() if c == top)[0]


class OperationalMonitor:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self._bucket_bounds = [
            (lo, hi) for _name, lo, hi in self.config.risk_buckets
        ]

    # ------------------------------------------------------------------

    def report(
        self,
        traces: Iterable[Any],
        *,
        feedback_store: Any | None = None,
        generated_at: datetime | None = None,
    ) -> OperationalMonitoringReport:
        # Reuse the Phase-5 validation / data-quality pass (no pipeline run).
        valid, data_quality = MonitoringService()._validate(traces)
        # deterministic ordering for trend / shift (tz-safe)
        ordered = sorted(valid, key=lambda t: (_ts_key(t.timestamp), t.interaction_id))
        n = len(ordered)

        incidents = self._incidents(ordered)
        incident_ids = {inc.interaction_id for inc in incidents}

        return OperationalMonitoringReport(
            generated_at=generated_at,
            total_interactions=n,
            config=self.config,
            snapshot=self._snapshot(ordered),
            risk_distribution=self._risk_distribution(ordered),
            applications=self._applications(ordered),
            detectors=self._detectors(ordered),
            reason_codes=self._reason_codes(ordered),
            verification=self._verification(ordered),
            multi_turn=self._multi_turn(ordered),
            incidents=incidents,
            incident_digest=self._digest(incidents, n),
            trend=self._trend(ordered, incident_ids),
            operational_shift=self._shift(ordered, incident_ids),
            feedback=self._feedback(ordered, feedback_store),
            data_quality=data_quality,
            notes=[
                "Operational monitoring only — aggregated from real DecisionTrace "
                "records. No detector / engine / verification pass was re-run.",
                "No ground truth is read; monitoring observes decisions, it does "
                "not judge whether they were correct.",
                "Incidents expose only structured trace fields (no claim text, no "
                "response text, no raw PII spans).",
                f"Incident rule: {_INCIDENT_DEFINITION}.",
            ],
        )

    # ------------------------------------------------------------------
    # snapshot

    def _snapshot(self, traces: list[DecisionTrace]) -> MonitoringSnapshot:
        n = len(traces)
        if n == 0:
            return MonitoringSnapshot()
        cfg = self.config
        tiers = Counter(t.final_decision.decision.value for t in traces)
        risks = [t.final_decision.overall_risk for t in traces]
        confs = [t.final_decision.decision_confidence for t in traces]
        lats = [t.latency_ms for t in traces]
        deep = sum(1 for t in traces if _path(t) == "DEEP")
        low_conf = sum(1 for c in confs if c < cfg.low_confidence_threshold)
        return MonitoringSnapshot(
            total_interactions=n,
            allow_count=tiers.get("ALLOW", 0),
            annotate_count=tiers.get("ANNOTATE", 0),
            verify_count=tiers.get("VERIFY", 0),
            human_review_count=tiers.get("HUMAN_REVIEW", 0),
            block_count=tiers.get("BLOCK", 0),
            allow_rate=_rate0(tiers.get("ALLOW", 0), n),
            annotate_rate=_rate0(tiers.get("ANNOTATE", 0), n),
            verify_rate=_rate0(tiers.get("VERIFY", 0), n),
            human_review_rate=_rate0(tiers.get("HUMAN_REVIEW", 0), n),
            block_rate=_rate0(tiers.get("BLOCK", 0), n),
            average_risk=_mean0(risks),
            p95_risk=_p95_0(risks),
            average_confidence=_mean0(confs),
            low_confidence_rate=_rate0(low_conf, n),
            fast_path_rate=_rate0(n - deep, n),
            deep_path_rate=_rate0(deep, n),
            average_latency_ms=_mean0(lats),
            p95_latency_ms=_p95_0(lats),
            average_cost_risk=_mean0(t.cost.cost_risk for t in traces),
            high_consequence_rate=_rate0(
                sum(
                    1
                    for t in traces
                    if t.consequence.consequence_score >= cfg.high_consequence_threshold
                ),
                n,
            ),
            high_criticality_rate=_rate0(
                sum(
                    1
                    for t in traces
                    if t.criticality.action_criticality >= cfg.high_criticality_threshold
                ),
                n,
            ),
            multi_risk_rate=_rate0(
                sum(1 for t in traces if bool(getattr(t.fusion, "multi_risk", False))), n
            ),
        )

    # ------------------------------------------------------------------
    # risk distribution (reuses monitoring.metrics.bucket_index)

    def _risk_distribution(self, traces: list[DecisionTrace]) -> RiskDistribution:
        n = len(traces)
        counts = [0] * len(self.config.risk_buckets)
        for t in traces:
            counts[bucket_index(t.final_decision.overall_risk, self._bucket_bounds)] += 1
        buckets = [
            RiskBucket(
                bucket_name=name,
                min_risk=lo,
                max_risk=hi,
                count=c,
                percentage=(None if n == 0 else round(100.0 * c / n, 4)),
            )
            for (name, lo, hi), c in zip(self.config.risk_buckets, counts)
        ]
        return RiskDistribution(buckets=buckets, total=sum(counts))

    # ------------------------------------------------------------------
    # application breakdown

    def _applications(self, traces: list[DecisionTrace]) -> list[ApplicationRiskSummary]:
        cfg = self.config
        by_app: dict[str, list[DecisionTrace]] = {}
        for t in traces:
            by_app.setdefault(t.application, []).append(t)

        out: list[ApplicationRiskSummary] = []
        for app in sorted(by_app):
            rows = by_app[app]
            k = len(rows)
            out.append(
                ApplicationRiskSummary(
                    application=app,
                    interaction_count=k,
                    average_risk=_mean0(r.final_decision.overall_risk for r in rows),
                    p95_risk=_p95_0([r.final_decision.overall_risk for r in rows]),
                    average_confidence=_mean0(
                        r.final_decision.decision_confidence for r in rows
                    ),
                    decision_distribution={
                        tier: sum(
                            1 for r in rows if r.final_decision.decision.value == tier
                        )
                        for tier in _TIERS
                    },
                    human_review_rate=_rate0(
                        sum(
                            1
                            for r in rows
                            if r.final_decision.decision is InterventionTier.HUMAN_REVIEW
                        ),
                        k,
                    ),
                    block_rate=_rate0(
                        sum(
                            1
                            for r in rows
                            if r.final_decision.decision is InterventionTier.BLOCK
                        ),
                        k,
                    ),
                    fast_path_rate=_rate0(
                        sum(1 for r in rows if _path(r) == "FAST"), k
                    ),
                    deep_path_rate=_rate0(
                        sum(1 for r in rows if _path(r) == "DEEP"), k
                    ),
                    high_consequence_rate=_rate0(
                        sum(
                            1
                            for r in rows
                            if r.consequence.consequence_score
                            >= cfg.high_consequence_threshold
                        ),
                        k,
                    ),
                    high_criticality_rate=_rate0(
                        sum(
                            1
                            for r in rows
                            if r.criticality.action_criticality
                            >= cfg.high_criticality_threshold
                        ),
                        k,
                    ),
                    dominant_risk_dimension=_dominant(rows),
                )
            )
        return out

    # ------------------------------------------------------------------
    # detector breakdown  (observational only — no new decision)

    def _detectors(self, traces: list[DecisionTrace]) -> list[DetectorRiskSummary]:
        n = len(traces)
        thr = self.config.detector_high_risk_threshold
        risk_of = {
            "performance": lambda t: t.performance.performance_risk,
            "responsibility": lambda t: t.responsibility.overall_responsibility_risk,
            "cost": lambda t: t.cost.cost_risk,
        }
        out: list[DetectorRiskSummary] = []
        for name in _DETECTORS:
            get = risk_of[name]
            risks = [get(t) for t in traces]
            contribs = [
                row.weighted_contribution
                for t in traces
                for row in t.fusion.risk_breakdown
                if row.dimension == name
            ]
            out.append(
                DetectorRiskSummary(
                    detector=name,
                    interaction_coverage=n,
                    average_risk=_mean0(risks),
                    high_risk_rate=_rate0(sum(1 for r in risks if r >= thr), n),
                    mean_weighted_contribution=mean_or_none(contribs),
                    dominant_dimension_count=sum(
                        1
                        for t in traces
                        if getattr(t.fusion, "dominant_dimension", None) == name
                    ),
                )
            )
        return out

    # ------------------------------------------------------------------
    # reason codes

    @staticmethod
    def _reason_codes(traces: list[DecisionTrace]) -> list[ReasonCodeSummary]:
        interventions = sum(
            1 for t in traces if t.final_decision.decision is not InterventionTier.ALLOW
        )
        counts = count_by(
            code for t in traces for code in t.final_decision.reason_codes
        )
        return [
            ReasonCodeSummary(
                reason_code=code,
                count=c,
                # ``count`` is total occurrences across all traces; a single
                # intervention can cite several codes (and a few codes also
                # appear on ALLOW traces), so the raw ratio can exceed 1 —
                # clamp it, since the field is a share.
                share_of_interventions=min(1.0, _rate0(c, interventions)),
            )
            for code, c in counts.items()
        ]

    # ------------------------------------------------------------------
    # verification

    def _verification(self, traces: list[DecisionTrace]) -> VerificationSummary:
        n = len(traces)
        fast = [t for t in traces if _path(t) == "FAST"]
        deep = [t for t in traces if _path(t) == "DEEP"]
        trig = count_by(
            reason
            for t in traces
            if t.verification is not None
            for reason in t.verification.deep_trigger_reasons
        )
        vlat = [
            t.verification.total_verification_latency_ms
            for t in traces
            if t.verification is not None
        ]

        # Deterministic semantic bypass — a subset of DEEP. Legacy traces
        # (no VerificationReport, or an older schema) report False.
        bypassed = [
            t for t in deep
            if bool(getattr(t.verification, "semantics_bypassed", False))
        ]
        # cost of a *full* DEEP pass, measured on the non-bypassed DEEP traces
        full_deep_latency = mean_or_none(
            [
                t.verification.deep_path_latency_ms
                for t in deep
                if t.verification is not None
                and not getattr(t.verification, "semantics_bypassed", False)
                and t.verification.deep_path_latency_ms > 0.0
            ]
        )
        saved = (
            round(len(bypassed) * full_deep_latency, 3)
            if bypassed and full_deep_latency is not None
            else None
        )

        return VerificationSummary(
            fast_count=len(fast),
            deep_count=len(deep),
            fast_rate=_rate0(len(fast), n),
            deep_rate=_rate0(len(deep), n),
            deep_trigger_reason_counts=trig,
            average_fast_latency_ms=mean_or_none([t.latency_ms for t in fast]),
            average_deep_latency_ms=mean_or_none([t.latency_ms for t in deep]),
            average_total_verification_latency_ms=mean_or_none(vlat),
            p95_total_verification_latency_ms=percentile_or_none(vlat, 95),
            semantic_bypass_count=len(bypassed),
            semantic_bypass_rate_of_deep=_rate0(len(bypassed), len(deep)),
            estimated_bypass_compute_saved_ms=saved,
        )

    # ------------------------------------------------------------------
    # multi-turn session accumulation

    @staticmethod
    def _turn_touched_critical_floor(trace: DecisionTrace) -> bool:
        """
        True if this turn either (a) *set* the non-decaying critical floor —
        it is itself a BLOCK / critical-PII / severe-toxicity turn — or (b)
        *inherited* it (the session block already reports the floor applied).
        """
        session = getattr(trace, "session", None)
        if isinstance(session, dict):
            if session.get("critical_floor_applied") or session.get("has_critical_history"):
                return True
            snap = session.get("snapshot")
            if isinstance(snap, dict) and snap.get("critical_events"):
                return True
        fd = getattr(trace, "final_decision", None)
        if fd is not None:
            if getattr(fd.decision, "value", str(fd.decision)) == "BLOCK":
                return True
            if "CRITICAL_PII" in (getattr(fd, "reason_codes", None) or []):
                return True
        resp = getattr(trace, "responsibility", None)
        return bool(getattr(resp, "contains_critical_pii", False))

    def _multi_turn(self, traces: list[DecisionTrace]) -> MultiTurnSummary:
        by_session: dict[str, list[DecisionTrace]] = {}
        for t in traces:
            session = getattr(t, "session", None)
            if not isinstance(session, dict):
                continue                      # legacy / stateless trace
            sid = str(session.get("session_id") or t.interaction_id)
            by_session.setdefault(sid, []).append(t)

        if not by_session:
            return MultiTurnSummary()

        multi_turn = {sid: g for sid, g in by_session.items() if len(g) >= 2}
        hit = [
            sid for sid, g in multi_turn.items()
            if any(self._turn_touched_critical_floor(t) for t in g)
        ]
        events = 0
        for g in by_session.values():
            last = max(g, key=lambda t: (t.session or {}).get("interaction_count", 0))
            snap = (last.session or {}).get("snapshot") or {}
            events += len(snap.get("critical_events") or [])

        return MultiTurnSummary(
            total_sessions=len(by_session),
            multi_turn_sessions=len(multi_turn),
            sessions_hitting_critical_floor=len(hit),
            critical_floor_session_rate=(
                round(len(hit) / len(multi_turn), 4) if multi_turn else None
            ),
            critical_floor_events=events,
        )

    # ------------------------------------------------------------------
    # incidents

    def _incidents(self, traces: list[DecisionTrace]) -> list[IncidentSummary]:
        out = [
            inc
            for t in traces
            if (inc := classify_incident(t, self.config)) is not None
        ]
        # deterministic: severity, then risk desc, then time, then id
        out.sort(
            key=lambda i: (
                _SEVERITY_RANK[i.severity],
                -i.overall_risk,
                _ts_key(i.timestamp),
                i.interaction_id,
            )
        )
        return out

    @staticmethod
    def _digest(incidents: list[IncidentSummary], total_interactions: int) -> IncidentDigest:
        return IncidentDigest(
            total=len(incidents),
            incident_rate=_rate0(len(incidents), total_interactions),
            by_severity={
                sev.value: sum(1 for i in incidents if i.severity is sev)
                for sev in IncidentSeverity
            },
            by_application=count_by(i.application for i in incidents),
            by_trigger=count_by(tr for i in incidents for tr in i.triggers),
            incident_definition=_INCIDENT_DEFINITION,
        )

    # ------------------------------------------------------------------
    # trend  (first half vs second half of chronologically-ordered traces)

    def _trend(
        self, ordered: list[DecisionTrace], incident_ids: set[str]
    ) -> TrendAnalysis:
        band = self.config.trend_stable_band
        n = len(ordered)
        mid = n // 2
        first, second = ordered[:mid], ordered[mid:]

        def _metric(name: str, fn) -> MetricTrend:
            a = fn(first) if first else None
            b = fn(second) if second else None
            delta = None if (a is None or b is None) else round(b - a, 6)
            if delta is None:
                direction = TrendDirection.STABLE
            elif abs(delta) <= band:
                direction = TrendDirection.STABLE
            elif delta > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING
            return MetricTrend(
                metric=name,
                first_half_value=a,
                second_half_value=b,
                delta=delta,
                direction=direction,
                first_half_n=len(first),
                second_half_n=len(second),
            )

        metrics = [
            _metric(
                "average_risk",
                lambda ts: _mean0(t.final_decision.overall_risk for t in ts),
            ),
            _metric(
                "incident_rate",
                lambda ts: _rate0(
                    sum(1 for t in ts if t.interaction_id in incident_ids), len(ts)
                ),
            ),
            _metric(
                "human_review_rate",
                lambda ts: _rate0(
                    sum(
                        1
                        for t in ts
                        if t.final_decision.decision is InterventionTier.HUMAN_REVIEW
                    ),
                    len(ts),
                ),
            ),
            _metric(
                "block_rate",
                lambda ts: _rate0(
                    sum(
                        1
                        for t in ts
                        if t.final_decision.decision is InterventionTier.BLOCK
                    ),
                    len(ts),
                ),
            ),
        ]
        return TrendAnalysis(
            ordered_by_timestamp=True, stable_band=band, metrics=metrics
        )

    # ------------------------------------------------------------------
    # operational shift  (recent window vs historical window)

    def _shift(
        self, ordered: list[DecisionTrace], incident_ids: set[str]
    ) -> OperationalShiftReport:
        cfg = self.config
        n = len(ordered)
        k = max(1, round(n * cfg.shift_recent_fraction)) if n else 0
        recent = ordered[n - k:] if k else []
        baseline = ordered[: n - k] if k else []

        def _agg(ts: list[DecisionTrace], fn) -> float | None:
            return fn(ts) if ts else None

        specs = {
            "average_risk": lambda ts: _mean0(
                t.final_decision.overall_risk for t in ts
            ),
            "block_rate": lambda ts: _rate0(
                sum(
                    1 for t in ts if t.final_decision.decision is InterventionTier.BLOCK
                ),
                len(ts),
            ),
            "human_review_rate": lambda ts: _rate0(
                sum(
                    1
                    for t in ts
                    if t.final_decision.decision is InterventionTier.HUMAN_REVIEW
                ),
                len(ts),
            ),
            "incident_rate": lambda ts: _rate0(
                sum(1 for t in ts if t.interaction_id in incident_ids), len(ts)
            ),
            "performance_risk": lambda ts: _mean0(
                t.performance.performance_risk for t in ts
            ),
            "responsibility_risk": lambda ts: _mean0(
                t.responsibility.overall_responsibility_risk for t in ts
            ),
            "cost_risk": lambda ts: _mean0(t.cost.cost_risk for t in ts),
        }

        shifts: list[OperationalShift] = []
        for metric, fn in specs.items():
            base_v = _agg(baseline, fn)
            recent_v = _agg(recent, fn)
            delta = (
                None if (base_v is None or recent_v is None) else round(recent_v - base_v, 6)
            )
            if delta is None:
                direction = "flat"
            elif abs(delta) <= cfg.shift_flat_band:
                direction = "flat"
            else:
                direction = "up" if delta > 0 else "down"
            shifts.append(
                OperationalShift(
                    metric=metric,
                    baseline_value=base_v,
                    recent_value=recent_v,
                    delta=delta,
                    direction=direction,
                    baseline_n=len(baseline),
                    recent_n=len(recent),
                )
            )
        return OperationalShiftReport(
            recent_fraction=cfg.shift_recent_fraction,
            recent_window_size=len(recent),
            baseline_window_size=len(baseline),
            flat_band=cfg.shift_flat_band,
            shifts=shifts,
        )

    # ------------------------------------------------------------------
    # feedback  (governance signal — NOT ground truth)

    @staticmethod
    def _feedback(
        traces: list[DecisionTrace], feedback_store: Any | None
    ) -> MonitoredFeedbackSummary:
        if feedback_store is None:
            return MonitoredFeedbackSummary(feedback_available=False)
        try:
            records = list(feedback_store.all())
        except Exception:  # noqa: BLE001 — a missing/odd store must not break monitoring
            return MonitoredFeedbackSummary(feedback_available=False)

        ids = {t.interaction_id for t in traces}
        scoped = [r for r in records if r.interaction_id in ids]
        by_outcome = Counter(r.outcome.value for r in scoped)
        overrides = sum(1 for r in scoped if getattr(r, "human_override", False))
        total = len(scoped)
        return MonitoredFeedbackSummary(
            feedback_available=True,
            feedback_count=total,
            interactions_with_feedback=len({r.interaction_id for r in scoped}),
            approved=by_outcome.get("approved", 0),
            modified=by_outcome.get("modified", 0),
            rejected=by_outcome.get("rejected", 0),
            override_count=overrides,
            override_rate=rate_or_none(overrides, total),
            approval_rate=rate_or_none(by_outcome.get("approved", 0), total),
        )
