"""
Session Manager — multi-turn risk accumulation with contextual snapshots.

Multi-turn conversations compound risk: a model that hedges once is noise; a
model that hedges (or gets contradicted, or leaks PII) turn after turn is a
pattern. The heavy detectors already run **only on the newest turn** — each
``Interaction`` is one turn and ``DecisionEngine.evaluate`` invokes claim
extraction / NLI / PII regex / cost exactly once. The manager takes the
resulting scalar risk plus a handful of structured signals and folds them into
a per-session :class:`ContextualSnapshot`, so later turns inherit what earlier
turns established without re-processing history.

Accumulation is deliberately bounded and transparent:

    on record:   cumulative <- decay * cumulative + (1 - decay) * turn_risk
                 snapshot   <- merge(prior signals, this turn's signals)
                 if this turn is a CRITICAL violation (BLOCK / critical PII /
                    severe toxicity):
                     snapshot.critical_floor <- max(critical_floor, floor_value)   # NEVER decays

    on decide:   session_risk  = history_weight * cumulative
                               + current_weight * current_turn_risk
                 if <escalation_event_count> high-risk turns in the window:
                     session_risk += escalation_risk_bump   (escalated)
                 adjusted_risk = max(current_turn_risk, session_risk)
                 if escalated:       adjusted_risk += escalation_risk_bump
                 adjusted_risk = max(adjusted_risk, snapshot.critical_floor)   # critical history
                 adjusted_risk = min(adjusted_risk, max_session_risk)

The session can only ever *raise* a turn's risk, never lower it. Ordinary risk
decays; a critical violation does not — once turn 2 blocks on PII, every later
turn in the session inherits at least ``critical_floor_value`` scrutiny.

Storage is in-process (a dict). Production would back this with a shared store
keyed by session_id with a TTL.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import Any, Protocol
from datetime import datetime, timezone

from common.timing import clamp01
from session.schemas import (
    ContextualSnapshot,
    CriticalEvent,
    SessionRiskContribution,
    SessionState,
)
from settings import load_settings

MANAGER_NAME = "session"


class SessionStore(Protocol):
    """
    Storage seam for :class:`SessionManager`. The default implementation is an
    in-process dict (:class:`InMemorySessionStore`); a DB-backed implementation
    with the same four methods (``database.repositories.DbSessionStore``) can be
    injected via ``SessionManager(store=...)`` with no behaviour change on the
    in-memory path.
    """

    def get(self, session_id: str) -> SessionState | None: ...
    def put(self, state: SessionState) -> None: ...
    def pop(self, session_id: str) -> None: ...
    def ids(self) -> list[str]: ...


class InMemorySessionStore:
    """Process-local session storage — the historical default."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def put(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state

    def pop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def ids(self) -> list[str]:
        return list(self._sessions)


class SessionManager:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        store: SessionStore | None = None,
    ) -> None:
        settings = config if config is not None else load_settings()
        scfg = settings["session"]
        self._window = int(scfg["history_window"])
        self._decay = float(scfg["decay"])
        self._history_weight = float(scfg["history_weight"])
        self._current_weight = float(scfg["current_weight"])
        self._high_risk_threshold = float(scfg["high_risk_event_threshold"])
        self._escalation_count = int(scfg["escalation_event_count"])
        self._escalation_bump = float(scfg["escalation_risk_bump"])
        self._max_session_risk = float(scfg["max_session_risk"])
        # Non-decaying floor set by a critical violation.
        self._critical_floor_value = float(scfg.get("critical_floor_value", 0.75))

        self._store: SessionStore = store or InMemorySessionStore()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get_state(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._store.get(session_id)
            return state.model_copy(deep=True) if state else SessionState(session_id=session_id)

    def snapshot(self, session_id: str) -> ContextualSnapshot:
        """The structured contextual snapshot for a session (a copy)."""
        return self.get_state(session_id).snapshot

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id)

    def active_sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._store.ids())

    # ------------------------------------------------------------------
    # risk contribution (call BEFORE record for the current turn)
    # ------------------------------------------------------------------

    def contribution(
        self, session_id: str, current_overall_risk: float
    ) -> dict[str, Any]:
        current = clamp01(current_overall_risk)
        with self._lock:
            state = self._store.get(session_id)
            history = state.cumulative_risk if state else 0.0
            prior_high = state.high_risk_event_count if state else 0
            count = state.interaction_count if state else 0
            critical_floor = state.snapshot.critical_floor if state else 0.0
            has_critical = state.snapshot.has_critical_history if state else False
            snapshot = state.snapshot.model_copy(deep=True) if state else None

        projected_high = prior_high + (1 if current >= self._high_risk_threshold else 0)
        escalated = projected_high >= self._escalation_count

        session_risk = (
            self._history_weight * history + self._current_weight * current
        )
        if escalated:
            session_risk += self._escalation_bump
        session_risk = clamp01(min(session_risk, self._max_session_risk))

        adjusted = max(current, session_risk)
        if escalated:
            adjusted = clamp01(min(adjusted + self._escalation_bump, self._max_session_risk))

        # Non-decaying critical floor — a critical violation earlier in the
        # session forces elevated scrutiny on every later turn.
        critical_floor_applied = critical_floor > adjusted + 1e-9
        adjusted = clamp01(min(max(adjusted, critical_floor), self._max_session_risk))

        contribution = SessionRiskContribution(
            session_id=session_id,
            interaction_count=count,
            current_overall_risk=round(current, 4),
            history_component=round(clamp01(history), 4),
            session_risk=round(session_risk, 4),
            adjusted_overall_risk=round(adjusted, 4),
            high_risk_events=projected_high,
            escalated=escalated,
            critical_floor=round(clamp01(critical_floor), 4),
            critical_floor_applied=critical_floor_applied,
            has_critical_history=has_critical,
            snapshot=snapshot,
            explanation=self._explain(
                count, history, projected_high, escalated, current, adjusted,
                critical_floor, critical_floor_applied,
            ),
        )
        return contribution.model_dump()

    # ------------------------------------------------------------------
    # recording (call AFTER the decision for the current turn)
    # ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        *,
        overall_risk: float,
        decision: str,
        interaction_id: str,
        timestamp: datetime | None = None,
        dimension_risks: Sequence[float] | None = None,   # (performance, responsibility, cost)
        reason_codes: Iterable[str] | None = None,
        tier_changing_rules: Iterable[str] | None = None,
        pii_entity_keys: Iterable[str] | None = None,      # "<subtype>:<redacted>" — never raw
        critical: bool = False,
        critical_trigger: str = "",
    ) -> SessionState:
        risk = clamp01(overall_risk)
        with self._lock:
            state = self._store.get(session_id) or SessionState(session_id=session_id)
            state.interaction_count += 1
            state.recent_risks = (state.recent_risks + [round(risk, 4)])[-self._window :]
            state.recent_decisions = (state.recent_decisions + [decision])[-self._window :]
            state.recent_interaction_ids = (
                state.recent_interaction_ids + [interaction_id]
            )[-self._window :]

            # ordinary risk still decays
            state.cumulative_risk = clamp01(
                self._decay * state.cumulative_risk + (1 - self._decay) * risk
            )
            state.high_risk_event_count = sum(
                1 for r in state.recent_risks if r >= self._high_risk_threshold
            )
            state.escalated = state.high_risk_event_count >= self._escalation_count

            # merge this turn's structured signals into the contextual snapshot
            self._merge_snapshot(
                state.snapshot,
                turn_index=state.interaction_count,
                interaction_id=interaction_id,
                decision=decision,
                risk=risk,
                dimension_risks=dimension_risks,
                reason_codes=reason_codes,
                tier_changing_rules=tier_changing_rules,
                pii_entity_keys=pii_entity_keys,
                critical=critical,
                critical_trigger=critical_trigger or (decision if critical else ""),
            )

            state.last_updated = timestamp or datetime.now(timezone.utc)
            self._store.put(state)
            return state.model_copy(deep=True)

    # ------------------------------------------------------------------

    def _merge_snapshot(
        self,
        snap: ContextualSnapshot,
        *,
        turn_index: int,
        interaction_id: str,
        decision: str,
        risk: float,
        dimension_risks: Sequence[float] | None,
        reason_codes: Iterable[str] | None,
        tier_changing_rules: Iterable[str] | None,
        pii_entity_keys: Iterable[str] | None,
        critical: bool,
        critical_trigger: str,
    ) -> None:
        snap.turns_recorded += 1

        if pii_entity_keys:
            snap.pii_entity_keys = sorted(
                set(snap.pii_entity_keys) | {k for k in pii_entity_keys if k}
            )
        for code in reason_codes or ():
            snap.reason_code_counts[code] = snap.reason_code_counts.get(code, 0) + 1
        if tier_changing_rules:
            snap.tier_changing_rules = sorted(
                set(snap.tier_changing_rules) | {r for r in tier_changing_rules if r}
            )

        if dimension_risks and len(dimension_risks) == 3:
            perf, resp, cost = (clamp01(x) for x in dimension_risks)
            snap.peak_performance_risk = max(snap.peak_performance_risk, perf)
            snap.peak_responsibility_risk = max(snap.peak_responsibility_risk, resp)
            snap.peak_cost_risk = max(snap.peak_cost_risk, cost)

        if critical:
            snap.critical_events.append(
                CriticalEvent(
                    turn_index=turn_index,
                    interaction_id=interaction_id,
                    decision=decision,
                    trigger=critical_trigger or "CRITICAL",
                    risk_at_event=round(risk, 4),
                )
            )
            snap.critical_floor = clamp01(
                max(snap.critical_floor, self._critical_floor_value)
            )

    def _explain(
        self,
        count: int,
        history: float,
        high_events: int,
        escalated: bool,
        current: float,
        adjusted: float,
        critical_floor: float,
        critical_floor_applied: bool,
    ) -> str:
        if count == 0:
            return "First turn in this session; no historical risk contribution."
        base = (
            f"Session has {count} prior turn(s); decayed historical risk "
            f"{history:.2f}, {high_events} high-risk turn(s) in the window."
        )
        if critical_floor_applied:
            return (
                base
                + f" A prior critical violation holds a non-decaying floor of "
                f"{critical_floor:.2f}, raising the effective risk from "
                f"{current:.2f} to {adjusted:.2f}."
            )
        if escalated:
            return (
                base
                + " Repeated high-risk turns crossed the escalation threshold, "
                f"raising the effective risk from {current:.2f} to {adjusted:.2f}."
            )
        if adjusted > current + 1e-6:
            return base + f" History raises the effective risk to {adjusted:.2f}."
        return base + " History does not raise the current turn's risk."
