"""
Incident investigation & governance service.

``InvestigationService`` reconstructs a complete, auditable investigation
from a stored ``DecisionTrace`` and records human governance actions. It
does NOT re-run the pipeline during investigation (see the module
docstring in ``investigation/__init__.py`` and the source-guard tests).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from decision.replay import build_replay
from explainability.builder import build_explanation
from investigation.schemas import (
    ACTION_TARGET_STATUS,
    ACTIONS_REQUIRING_COMMENT,
    ACTIONS_REQUIRING_REVIEWER_DECISION,
    GovernanceAction,
    GovernanceActionType,
    IncidentInvestigation,
    InvestigationCounterfactual,
    InvestigationNotFound,
    InvestigationStatus,
    available_actions_for,
)
from monitoring.incidents import classify_incident
from monitoring.schemas import MonitoringConfig

__all__ = ["GovernanceStore", "InvestigationService", "GovernanceError"]

# Production-visible, non-free-text fields a reviewer counterfactual may touch.
_CF_ALLOWED_FIELDS = ("action_amount_inr", "affected_entities", "action_type")
_VALID_TIERS = ("ALLOW", "ANNOTATE", "VERIFY", "HUMAN_REVIEW", "BLOCK")


class GovernanceError(ValueError):
    """A governance action was malformed (bad action, missing comment, …)."""


class GovernanceStore:
    """
    Append-only, in-memory governance-action log.

    Kept deliberately small and behind a stable interface
    (``record_action`` / ``get_actions`` / ``get_all_actions`` /
    ``current_status``) so a database-backed store can replace it without
    touching callers. Consistent with the project's documented
    process-local session / audit / feedback state.
    """

    def __init__(self) -> None:
        self._by_interaction: dict[str, list[GovernanceAction]] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def record_action(
        self,
        *,
        interaction_id: str,
        action: GovernanceActionType,
        actor: str,
        comment: str,
        previous_status: InvestigationStatus,
        new_status: InvestigationStatus,
        original_decision: str,
        reviewer_decision: str | None = None,
        timestamp: datetime | None = None,
    ) -> GovernanceAction:
        with self._lock:
            self._counter += 1
            record = GovernanceAction(
                action_id=f"GOV-{self._counter:06d}",
                interaction_id=interaction_id,
                timestamp=timestamp or datetime.now(timezone.utc),
                actor=actor or "reviewer",
                action=action,
                comment=comment,
                previous_status=previous_status,
                new_status=new_status,
                original_decision=original_decision,
                reviewer_decision=reviewer_decision,
            )
            self._by_interaction.setdefault(interaction_id, []).append(record)
        return record

    def get_actions(self, interaction_id: str) -> list[GovernanceAction]:
        with self._lock:
            return list(self._by_interaction.get(interaction_id, []))

    def get_all_actions(self) -> list[GovernanceAction]:
        with self._lock:
            return [a for acts in self._by_interaction.values() for a in acts]

    def current_status(self, interaction_id: str) -> InvestigationStatus:
        actions = self.get_actions(interaction_id)
        return actions[-1].new_status if actions else InvestigationStatus.OPEN

    # -- reviewer-facing convenience (views over the append-only log) -------- #

    def get_history(self, interaction_id: str) -> list[GovernanceAction]:
        """Alias for :meth:`get_actions` — the chronological human-action log."""
        return self.get_actions(interaction_id)

    def effective_decision(self, interaction_id: str, original_decision: str) -> str:
        """
        The tier in effect after human governance: the latest MODIFY_DECISION
        override if one exists, otherwise the immutable automated decision.
        A pure view — nothing on the trace or the log is mutated.
        """
        for action in reversed(self.get_actions(interaction_id)):
            if (
                action.action is GovernanceActionType.MODIFY_DECISION
                and action.reviewer_decision
            ):
                return action.reviewer_decision
        return original_decision


# Short reviewer-facing aliases -> canonical GovernanceActionType.
GOVERNANCE_ACTION_ALIASES: dict[str, GovernanceActionType] = {
    "APPROVE": GovernanceActionType.APPROVE_DECISION,
    "APPROVED": GovernanceActionType.APPROVE_DECISION,
    "MODIFY": GovernanceActionType.MODIFY_DECISION,
    "MODIFIED": GovernanceActionType.MODIFY_DECISION,
    "OVERRIDE": GovernanceActionType.MODIFY_DECISION,
    "REJECT": GovernanceActionType.REJECT_DECISION,
    "REJECTED": GovernanceActionType.REJECT_DECISION,
}


def canonical_governance_action(action: GovernanceActionType | str) -> GovernanceActionType:
    """Accept the short reviewer forms (APPROVE/MODIFY/REJECT, past tense too)
    as well as the canonical ``*_DECISION`` names or an enum value."""
    if isinstance(action, GovernanceActionType):
        return action
    key = str(action).strip().upper()
    if key in GOVERNANCE_ACTION_ALIASES:
        return GOVERNANCE_ACTION_ALIASES[key]
    return GovernanceActionType(key)


class InvestigationService:
    """
    Investigate incidents and record governance actions.

    ``control_plane`` is duck-typed — it must expose ``get_audit_trace``
    and (for counterfactuals) ``get_stored_interaction`` + ``engine``.
    """

    def __init__(
        self,
        control_plane: Any,
        governance_store: GovernanceStore | None = None,
        *,
        monitoring_config: MonitoringConfig | None = None,
    ) -> None:
        self._cp = control_plane
        self._governance = governance_store or GovernanceStore()
        self._mon_config = monitoring_config or MonitoringConfig()

    # ------------------------------------------------------------------
    # read-only reconstruction

    def get_incident(self, interaction_id: str):
        trace = self._cp.get_audit_trace(interaction_id)
        if trace is None:
            return None
        return classify_incident(trace, self._mon_config)

    def investigate(
        self, interaction_id: str
    ) -> IncidentInvestigation | InvestigationNotFound:
        trace = self._cp.get_audit_trace(interaction_id)
        if trace is None:
            return InvestigationNotFound(
                interaction_id=interaction_id,
                message=(
                    f"No decision trace is stored for interaction "
                    f"'{interaction_id}'. Run a check for it first."
                ),
            )

        replay = build_replay(trace)                 # reconstruction, no re-run
        explanation = build_explanation(trace)       # presentation, no re-run
        incident = classify_incident(trace, self._mon_config)

        history = self._governance.get_actions(interaction_id)
        status = history[-1].new_status if history else InvestigationStatus.OPEN
        latest_modify = next(
            (
                a.reviewer_decision
                for a in reversed(history)
                if a.action is GovernanceActionType.MODIFY_DECISION
            ),
            None,
        )

        original_decision = trace.final_decision.decision.value  # immutable
        return IncidentInvestigation(
            interaction_id=interaction_id,
            incident=incident,
            replay=replay,
            explanation=explanation,
            original_decision=original_decision,
            requires_human_review=original_decision in ("HUMAN_REVIEW", "BLOCK"),
            investigation_status=status,
            available_actions=available_actions_for(status),
            governance_history=history,
            latest_reviewer_decision=latest_modify,
            effective_governed_decision=latest_modify or original_decision,
            is_overridden=latest_modify is not None,
        )

    def get_governance_history(self, interaction_id: str) -> list[GovernanceAction]:
        return self._governance.get_actions(interaction_id)

    # ------------------------------------------------------------------
    # governance action

    def record_governance_action(
        self,
        interaction_id: str,
        *,
        action: GovernanceActionType | str,
        actor: str = "reviewer",
        comment: str = "",
        reviewer_decision: str | None = None,
        timestamp: datetime | None = None,
    ) -> IncidentInvestigation | InvestigationNotFound:
        trace = self._cp.get_audit_trace(interaction_id)
        if trace is None:
            return InvestigationNotFound(
                interaction_id=interaction_id,
                message=f"No decision trace is stored for interaction '{interaction_id}'.",
            )

        try:
            action = canonical_governance_action(action)
        except ValueError as exc:
            raise GovernanceError(f"Unknown governance action: {action!r}") from exc

        previous_status = self._governance.current_status(interaction_id)
        if action not in available_actions_for(previous_status):
            raise GovernanceError(
                f"Action {action.value} is not available from status "
                f"{previous_status.value}."
            )

        comment = (comment or "").strip()
        if action in ACTIONS_REQUIRING_COMMENT and not comment:
            raise GovernanceError(f"Action {action.value} requires a comment.")

        if action in ACTIONS_REQUIRING_REVIEWER_DECISION:
            if reviewer_decision is None:
                raise GovernanceError(
                    f"Action {action.value} requires a reviewer_decision."
                )
            reviewer_decision = str(reviewer_decision).upper()
            if reviewer_decision not in _VALID_TIERS:
                raise GovernanceError(
                    f"reviewer_decision must be one of {_VALID_TIERS}."
                )
        else:
            reviewer_decision = None

        original_decision = trace.final_decision.decision.value  # immutable
        new_status = ACTION_TARGET_STATUS[action]

        self._governance.record_action(
            interaction_id=interaction_id,
            action=action,
            actor=actor,
            comment=comment,
            previous_status=previous_status,
            new_status=new_status,
            original_decision=original_decision,
            reviewer_decision=reviewer_decision,
            timestamp=timestamp,
        )
        return self.investigate(interaction_id)

    # ------------------------------------------------------------------
    # counterfactual  (explicitly a SIMULATION — runs the existing engine)

    def counterfactual(
        self, interaction_id: str, modified_fields: dict[str, Any]
    ) -> InvestigationCounterfactual | InvestigationNotFound:
        interaction = self._cp.get_stored_interaction(interaction_id)
        trace = self._cp.get_audit_trace(interaction_id)
        if interaction is None or trace is None:
            return InvestigationNotFound(
                interaction_id=interaction_id,
                message=(
                    f"No stored interaction/trace for '{interaction_id}' — a "
                    "counterfactual needs the original interaction."
                ),
            )

        safe: dict[str, Any] = {}
        rejected: list[str] = []
        for field, value in (modified_fields or {}).items():
            if field in _CF_ALLOWED_FIELDS:
                safe[field] = value
            else:
                rejected.append(field)

        from simulation.engine import compare_decisions

        result = compare_decisions(self._cp.engine, interaction, safe)

        changed = dict(result.changed_fields)
        current = trace.final_decision.decision.value
        cf = result.counterfactual_decision
        risk0 = trace.final_decision.overall_risk
        risk1 = result.counterfactual_overall_risk
        pieces = ", ".join(f"{k}={v}" for k, v in changed.items()) or "no change"
        summary = (
            f"Simulation: with {pieces}, ControlPlane would decide {cf} "
            f"instead of {current} (risk {risk0:.2f} -> {risk1:.2f}). "
            "The stored decision is unchanged."
        )

        return InvestigationCounterfactual(
            interaction_id=interaction_id,
            changed_fields=changed,
            rejected_fields=sorted(set(rejected) | set(result.rejected_fields)),
            current_decision=current,
            counterfactual_decision=cf,
            decision_changed=current != cf,
            current_overall_risk=risk0,
            counterfactual_overall_risk=risk1,
            rules_removed=list(result.rules_removed),
            rules_added=list(result.rules_added),
            reason_codes_removed=list(result.reason_codes_removed),
            reason_codes_added=list(result.reason_codes_added),
            summary=summary,
        )
