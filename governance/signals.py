"""
Governance signals — a unified view of reviewer overrides + reviewer
feedback, with source provenance preserved.

A ``GovernanceSignal`` is an input for human governance analysis. It is
NEVER ground truth and is NEVER consumed by any production decision.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from decision.schemas import DecisionTrace
from governance.schemas import (
    GovernanceSignal,
    GovernanceSignalSummary,
    GovernanceSignalType,
)
from monitoring.metrics import rate_or_none

__all__ = ["collect_signals", "summarize_signals"]

_DISAGREEMENT_ACTIONS = ("MODIFY_DECISION", "REJECT_DECISION")
_FEEDBACK_TYPE = {
    "modified": GovernanceSignalType.FEEDBACK_MODIFIED,
    "rejected": GovernanceSignalType.FEEDBACK_REJECTED,
    "approved": GovernanceSignalType.FEEDBACK_APPROVED,
}


def _safe_comment(text: str | None) -> str:
    """Reviewer-authored text, length-capped. Not a raw PII span."""
    return (text or "").strip()[:200]


def collect_signals(
    traces: list[DecisionTrace],
    governance_actions: list[Any],
    feedback_records: list[Any],
) -> list[GovernanceSignal]:
    """
    Build the unified signal list. Deterministic order: reviewer overrides
    (by action_id), then feedback (by feedback_id).
    """
    trace_by_id = {t.interaction_id: t for t in traces}
    signals: list[GovernanceSignal] = []

    # --- reviewer overrides (from investigation GovernanceAction) ---
    for act in sorted(governance_actions, key=lambda a: a.action_id):
        if act.action.value not in _DISAGREEMENT_ACTIONS:
            continue
        trace = trace_by_id.get(act.interaction_id)
        if act.action.value == "REJECT_DECISION":
            reviewer_outcome = "rejected"
            disagreement = True
        else:  # MODIFY_DECISION
            reviewer_outcome = act.reviewer_decision or "modified"
            disagreement = (
                act.reviewer_decision is not None
                and act.reviewer_decision != act.original_decision
            )
        signals.append(
            GovernanceSignal(
                source=GovernanceSignalType.REVIEWER_OVERRIDE,
                interaction_id=act.interaction_id,
                application=trace.application if trace else None,
                automated_decision=act.original_decision,
                reviewer_outcome=reviewer_outcome,
                is_disagreement=disagreement,
                timestamp=act.timestamp,
                comment=_safe_comment(act.comment),
            )
        )

    # --- reviewer feedback (from FeedbackStore) ---
    for rec in sorted(feedback_records, key=lambda r: r.feedback_id):
        outcome = rec.outcome.value if hasattr(rec.outcome, "value") else str(rec.outcome)
        sig_type = _FEEDBACK_TYPE.get(outcome)
        if sig_type is None:
            continue
        trace = trace_by_id.get(rec.interaction_id)
        signals.append(
            GovernanceSignal(
                source=sig_type,
                interaction_id=rec.interaction_id,
                application=trace.application if trace else None,
                automated_decision=(
                    rec.system_decision.value
                    if hasattr(rec.system_decision, "value")
                    else str(rec.system_decision)
                ),
                reviewer_outcome=(
                    rec.reviewer_decision.value
                    if getattr(rec, "reviewer_decision", None) is not None
                    else outcome
                ),
                is_disagreement=outcome in ("modified", "rejected"),
                timestamp=getattr(rec, "timestamp", None),
                comment=_safe_comment(getattr(rec, "reason", "")),
            )
        )
    return signals


def summarize_signals(signals: list[GovernanceSignal]) -> GovernanceSignalSummary:
    n = len(signals)
    disagreements = sum(1 for s in signals if s.is_disagreement)
    modified = sum(
        1
        for s in signals
        if s.source
        in (
            GovernanceSignalType.FEEDBACK_MODIFIED,
            GovernanceSignalType.REVIEWER_OVERRIDE,
        )
        and s.is_disagreement
        and s.reviewer_outcome != "rejected"
    )
    rejected = sum(
        1
        for s in signals
        if s.reviewer_outcome == "rejected"
        or s.source is GovernanceSignalType.FEEDBACK_REJECTED
    )
    return GovernanceSignalSummary(
        signal_count=n,
        by_application=dict(
            Counter(s.application or "unknown" for s in signals)
        ),
        by_decision=dict(Counter(s.automated_decision for s in signals)),
        by_signal_type=dict(Counter(s.source.value for s in signals)),
        override_rate=rate_or_none(disagreements, n),
        modification_rate=rate_or_none(modified, n),
        rejection_rate=rate_or_none(rejected, n),
    )
