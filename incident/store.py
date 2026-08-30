"""
Incident store — turns stored ``DecisionTrace`` records into PII-safe
:class:`IncidentRecord` objects, enriched with governance / feedback
signals. Read-only; re-runs nothing.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable

from data.schemas import InterventionTier
from decision.schemas import DecisionTrace
from incident.schemas import IncidentRecord, Phase10IncidentConfig
from monitoring.incidents import classify_incident
from monitoring.schemas import MonitoringConfig

__all__ = ["IncidentStore", "ts_key", "signature_for"]

_HUMAN = ("HUMAN_REVIEW", "BLOCK")


def ts_key(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _short_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10].upper()}"


def _tier_changing_rules(trace: DecisionTrace) -> list[str]:
    return sorted(
        {s.rule for s in trace.decision_path if s.from_tier != s.to_tier}
    )


def signature_for(record_fields: dict[str, Any]) -> str:
    """Deterministic grouping key from structured fields only."""
    reasons = ",".join(sorted(record_fields["reason_codes"])) or "-"
    rules = ",".join(sorted(record_fields["tier_changing_rules"])) or "-"
    return "|".join(
        [
            f"app:{record_fields['application']}",
            f"dim:{record_fields['dominant_dimension'] or 'none'}",
            f"tier:{record_fields['decision']}",
            f"verify:{record_fields['verification_path']}",
            f"reasons:{reasons}",
            f"rules:{rules}",
            f"cons:{record_fields['consequence_band']}",
            f"crit:{record_fields['criticality_band']}",
        ]
    )


class IncidentStore:
    """
    Builds and holds :class:`IncidentRecord` objects. In-memory, like the
    project's other prototype stores; a DB could replace it behind this
    same interface.
    """

    def __init__(self, config: Phase10IncidentConfig | None = None) -> None:
        self.config = config or Phase10IncidentConfig()
        self._mon = MonitoringConfig(
            low_confidence_threshold=self.config.low_confidence_threshold,
            elevated_risk_threshold=self.config.high_risk_threshold,
        )
        self._records: dict[str, IncidentRecord] = {}

    # ------------------------------------------------------------------

    def ingest(
        self,
        traces: Iterable[DecisionTrace],
        governance_actions: Iterable[Any] = (),
        feedback_records: Iterable[Any] = (),
    ) -> list[IncidentRecord]:
        traces = list(traces)
        gov = list(governance_actions)
        fb = list(feedback_records)

        reviewer_signal: dict[str, str] = {}
        for act in sorted(gov, key=lambda a: a.action_id):
            if act.action.value in ("MODIFY_DECISION", "REJECT_DECISION"):
                if act.action.value == "REJECT_DECISION":
                    reviewer_signal[act.interaction_id] = (
                        f"{act.original_decision} -> REJECTED"
                    )
                elif act.reviewer_decision:
                    reviewer_signal[act.interaction_id] = (
                        f"{act.original_decision} -> {act.reviewer_decision}"
                    )

        feedback_signal: dict[str, str] = {}
        for rec in sorted(fb, key=lambda r: r.feedback_id):
            outcome = rec.outcome.value if hasattr(rec.outcome, "value") else str(rec.outcome)
            feedback_signal.setdefault(rec.interaction_id, outcome)

        out: list[IncidentRecord] = []
        for trace in traces:
            inc = classify_incident(trace, self._mon)
            if inc is None:
                continue
            fd = trace.final_decision
            fields = {
                "application": trace.application,
                "dominant_dimension": getattr(trace.fusion, "dominant_dimension", None),
                "decision": fd.decision.value,
                "verification_path": (trace.verification_path or "DEEP").upper(),
                "reason_codes": list(fd.reason_codes),
                "tier_changing_rules": _tier_changing_rules(trace),
                "consequence_band": trace.consequence.severity_band,
                "criticality_band": trace.criticality.band,
            }
            sig = signature_for(fields)
            record = IncidentRecord(
                incident_id=_short_id("INC", trace.interaction_id),
                interaction_id=trace.interaction_id,
                timestamp=trace.timestamp,
                application=trace.application,
                action_type=trace.action_type,
                decision=fd.decision.value,
                overall_risk=fd.overall_risk,
                decision_confidence=fd.decision_confidence,
                verification_path=fields["verification_path"],
                dominant_dimension=fields["dominant_dimension"],
                reason_codes=fields["reason_codes"],
                triggered_rules=list(fd.triggered_rules),
                tier_changing_rules=fields["tier_changing_rules"],
                human_review_required=fd.decision.value in _HUMAN,
                performance_risk=fd.performance_risk,
                responsibility_risk=fd.responsibility_risk,
                cost_risk=fd.cost_risk,
                multi_risk=bool(getattr(trace.fusion, "multi_risk", False)),
                consequence_band=fields["consequence_band"],
                criticality_band=fields["criticality_band"],
                incident_severity=inc.severity.value,
                incident_triggers=list(inc.triggers),
                reviewer_signal=reviewer_signal.get(trace.interaction_id),
                feedback_signal=feedback_signal.get(trace.interaction_id),
                signature=sig,
            )
            self._records[record.interaction_id] = record
            out.append(record)

        return sorted(out, key=lambda r: (ts_key(r.timestamp), r.interaction_id))

    # ------------------------------------------------------------------

    def all(self) -> list[IncidentRecord]:
        return sorted(
            self._records.values(), key=lambda r: (ts_key(r.timestamp), r.interaction_id)
        )

    def get(self, incident_id: str) -> IncidentRecord | None:
        for rec in self._records.values():
            if rec.incident_id == incident_id:
                return rec
        return self._records.get(incident_id)  # allow lookup by interaction_id too
