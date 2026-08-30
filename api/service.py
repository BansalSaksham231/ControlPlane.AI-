"""
ControlPlane service layer.

Owns the long-lived objects (decision engine, session manager, feedback
store, in-memory audit log) and exposes the handful of operations the
API needs. All real logic lives in the engines/detectors; this class
only wires them together and keeps the HTTP layer thin.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from data.schemas import Interaction
from decision.engine import DecisionEngine
from decision.schemas import DecisionTrace
from feedback.schemas import FeedbackRecord
from feedback.store import FeedbackStore
from session.manager import SessionManager
from settings import load_settings

VERSION = "0.2.0"


class ControlPlaneService:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        feedback_path: str | None = None,
        fit_cost_baseline: bool = True,
        use_database: bool | None = None,
    ) -> None:
        self._config = config if config is not None else load_settings()

        # Persistence is opt-in: explicit ``use_database=True`` or the
        # ``CONTROLPLANE_PERSISTENCE=1`` env var. Off by default -> the
        # in-memory managers are used and the import of ``database`` (and
        # SQLAlchemy) never happens.
        if use_database is None:
            use_database = os.environ.get("CONTROLPLANE_PERSISTENCE") == "1"

        session_store = None
        governance_store = None
        self._trace_store = None
        if use_database:
            from database import (  # noqa: PLC0415 - deliberately lazy
                DbGovernanceStore,
                DbSessionStore,
                DbTraceStore,
                init_engine,
            )

            init_engine()
            session_store = DbSessionStore()
            governance_store = DbGovernanceStore()
            self._trace_store = DbTraceStore()
            logger.info("ControlPlaneService: persistence enabled (%s)", type(session_store).__name__)

        self.session_manager = SessionManager(self._config, store=session_store)

        cost_baseline = None
        if fit_cost_baseline:
            cost_baseline = self._fit_cost_baseline()

        self.engine = DecisionEngine(
            self._config,
            cost_baseline=cost_baseline,
            session_manager=self.session_manager,
        )
        self.feedback = FeedbackStore(path=feedback_path)

        self._audit: dict[str, DecisionTrace] = {}
        self._interactions: dict[str, Interaction] = {}
        self._audit_order: list[str] = []
        self._checks_served = 0
        self._lock = threading.Lock()

        # Phase 8 Step 3 — incident investigation & human governance.
        from investigation.service import GovernanceStore, InvestigationService

        self.governance = governance_store or GovernanceStore()
        self.investigation = InvestigationService(self, self.governance)

        # Phase 10 — closed-loop adaptive guardrails & incident intelligence.
        from adaptive.service import AdaptiveGovernanceService

        self.adaptive = AdaptiveGovernanceService(self)

        # Phase 11 — enterprise command center (read-only presentation layer).
        from enterprise.service import EnterpriseService

        self.enterprise = EnterpriseService(self)

    # ------------------------------------------------------------------

    def _fit_cost_baseline(self):
        import random

        from data.generator import generate_interactions
        from detectors.cost.baseline import CostBaseline
        from detectors.cost.detector import CostDetector

        rng = random.Random(self._config["seed"])
        interactions = generate_interactions(self._config, rng)
        return CostBaseline.fit(
            interactions, self._config, estimate_cost=CostDetector(self._config).estimate_cost
        )

    # ------------------------------------------------------------------

    def check(
        self, interaction: Interaction, *, timestamp: datetime | None = None
    ) -> DecisionTrace:
        trace = self.engine.evaluate(
            interaction, timestamp=timestamp or datetime.now(timezone.utc)
        )
        with self._lock:
            self._checks_served += 1
            self._audit[interaction.interaction_id] = trace
            self._interactions[interaction.interaction_id] = interaction
            self._audit_order.append(interaction.interaction_id)
            # keep the in-memory audit log bounded
            if len(self._audit_order) > 5000:
                oldest = self._audit_order.pop(0)
                self._audit.pop(oldest, None)
                self._interactions.pop(oldest, None)

        # Durable, cross-process audit copy (append-only). A storage failure
        # must never block or alter a decision that has already been made.
        if self._trace_store is not None:
            try:
                self._trace_store.put(interaction, trace)
            except Exception:  # pragma: no cover - persistence is best-effort
                logger.exception("Failed to persist decision trace for %s", interaction.interaction_id)
        return trace

    def get_stored_interaction(self, interaction_id: str) -> Interaction | None:
        """
        The original :class:`Interaction` for a stored trace (kept
        server-side, alongside the unredacted trace, for authorised
        investigation / counterfactual). Never serialised to a client.
        """
        with self._lock:
            return self._interactions.get(interaction_id)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.session_manager.get_state(session_id).model_dump(mode="json")

    def reset_session(self, session_id: str) -> None:
        self.session_manager.reset(session_id)

    def get_audit(self, interaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._audit.get(interaction_id)
        return trace.audit_summary() if trace else None

    def get_audit_trace(self, interaction_id: str) -> DecisionTrace | None:
        with self._lock:
            return self._audit.get(interaction_id)

    def replay(self, interaction_id: str):
        """
        Reconstruct an :class:`IncidentReplay` from the stored decision
        trace. Does NOT re-run the pipeline. Returns an
        ``IncidentReplayNotFound`` if the interaction is unknown.
        """
        from decision.replay import build_replay, IncidentReplayNotFound

        trace = self.get_audit_trace(interaction_id)
        if trace is None:
            return IncidentReplayNotFound(
                interaction_id=interaction_id,
                message=(
                    f"No decision trace is stored for interaction "
                    f"'{interaction_id}'. Run /check for it first."
                ),
            )
        return build_replay(trace)

    def all_traces(self) -> list[DecisionTrace]:
        """Every decision trace currently held in the in-memory audit log (deduped)."""
        with self._lock:
            seen: set[str] = set()
            traces: list[DecisionTrace] = []
            for interaction_id in self._audit_order:
                if interaction_id in seen or interaction_id not in self._audit:
                    continue
                seen.add(interaction_id)
                traces.append(self._audit[interaction_id])
            return traces

    def populate_demo(self, limit: int = 150) -> int:
        """
        Run the synthetic evaluation dataset through this service so the
        audit log / monitoring view has real traffic to show. Returns the
        number of interactions processed.
        """
        from evaluation.evaluation import load_evaluation_dataset

        processed = 0
        for interaction, _ in load_evaluation_dataset(self._config)[:limit]:
            self.check(interaction, timestamp=datetime(2026, 8, 21, 12, 0, 0))
            processed += 1
        return processed

    def populate_operational_demo(self, traffic_sample: int = 250) -> int:
        """
        Populate the audit log with **operational** demo traffic — the demo
        scenarios (A-J) plus a sample of the synthetic *production-traffic*
        dataset. The evaluation dataset (which carries ground-truth labels)
        is deliberately NOT used here. Timestamps come from the interactions
        themselves, so the result is deterministic. Returns the count.
        """
        import random

        from data.generator import generate_interactions
        from tests import scenarios

        demo = [factory() for factory in scenarios.ALL_SINGLE_TURN.values()]
        demo += list(scenarios.scenario_g_multi_turn())
        demo.append(scenarios.scenario_i_policy_counterfactual()[0])
        demo.append(scenarios.scenario_j_consequence_counterfactual()[0])

        rng = random.Random(self._config["seed"])
        pool = list(demo) + list(generate_interactions(self._config, rng)[:traffic_sample])

        processed = 0
        for interaction in pool:
            # Namespace operational-demo traffic so it never collides with a
            # user's interactive checks (audit log or session accumulation).
            interaction = interaction.model_copy(
                update={
                    "interaction_id": f"OPS-{interaction.interaction_id}",
                    "session_id": f"OPS-{interaction.session_id}",
                }
            )
            self.check(interaction, timestamp=interaction.timestamp)
            processed += 1

        self._seed_operational_feedback()
        return processed

    def _seed_operational_feedback(self) -> None:
        """A few illustrative reviewer feedback entries on demo incidents."""
        import contextlib

        traces = self.all_traces()
        blocked = [t for t in traces if t.final_decision.decision.value == "BLOCK"][:2]
        human = [t for t in traces if t.final_decision.decision.value == "HUMAN_REVIEW"][:2]
        for trace in blocked:
            if self.feedback.for_interaction(trace.interaction_id):
                continue
            with contextlib.suppress(Exception):
                self.submit_feedback(
                    interaction_id=trace.interaction_id,
                    system_decision=None,
                    outcome="approved",
                    reviewer="demo-reviewer",
                )
        for trace in human:
            if self.feedback.for_interaction(trace.interaction_id):
                continue
            with contextlib.suppress(Exception):
                self.submit_feedback(
                    interaction_id=trace.interaction_id,
                    system_decision=None,
                    reviewer_decision="VERIFY",
                    outcome="modified",
                    reviewer="demo-reviewer",
                )

    def governance_report(self, *, with_calibration: bool = False):
        """
        Assemble the Phase-9 :class:`GovernanceIntelligenceReport` from this
        service's stored traces + governance actions + feedback. Read-only:
        it re-runs nothing and never writes config. ``with_calibration``
        additionally executes ``calibration.sweep`` + ``calibration.select``
        to quantify a threshold recommendation (slower).
        """
        from governance.report import build_governance_report

        selection = None
        if with_calibration:
            from governance.recommendations import run_calibration_bridge

            selection = run_calibration_bridge(self._config)
        return build_governance_report(
            self.all_traces(),
            self.governance.get_all_actions(),
            self.feedback.all(),
            calibration_selection=selection,
        )

    # ------------------------------------------------------------------
    # Phase 10 — incident intelligence & adaptive guardrails

    def incident_intelligence(self):
        """Read-only :class:`IncidentIntelligenceReport` over stored state."""
        return self.adaptive.incident_intelligence()

    def adaptive_report(self, *, with_counterfactual: bool = False):
        """The Phase-10 :class:`AdaptiveGovernanceReport`. Never writes config."""
        return self.adaptive.report(with_counterfactual=with_counterfactual)

    def adaptive_recommendations(self, *, with_counterfactual: bool = False):
        return self.adaptive.recommendations(with_counterfactual=with_counterfactual)

    def adaptive_get_recommendation(self, recommendation_id: str):
        return self.adaptive.get_recommendation(recommendation_id)

    def adaptive_approve(self, recommendation_id: str, **kwargs: Any):
        """Record APPROVED_FOR_EVALUATION — never applies the candidate."""
        return self.adaptive.approve(recommendation_id, **kwargs)

    def adaptive_reject(self, recommendation_id: str, **kwargs: Any):
        return self.adaptive.reject(recommendation_id, **kwargs)

    # ------------------------------------------------------------------
    # Phase 11 — enterprise command center (read-only)

    def command_center_view(self):
        return self.enterprise.command_center()

    def application_posture(self):
        return self.enterprise.application_posture()

    def governance_timeline(self, *, focus_interaction_id: str | None = None):
        return self.enterprise.governance_timeline(focus_interaction_id=focus_interaction_id)

    def enterprise_whatif(self, **kwargs: Any):
        """User-triggered What-If simulation (calibration.sweep + calibration.select)."""
        return self.enterprise.whatif(**kwargs)

    def enterprise_demo(self, *, with_counterfactual: bool = True):
        return self.enterprise.run_demo(with_counterfactual=with_counterfactual)

    def get_operational_monitoring(self, config: Any | None = None):
        """
        Thin seam for the Command Center UI: build the Phase-8
        ``OperationalMonitoringReport`` from the traces already recorded in
        this service's audit log. Observes stored traces only — it never
        re-runs a detector, the decision engine or verification.
        """
        from monitoring.engine import OperationalMonitor
        from monitoring.schemas import MonitoringConfig

        monitor = OperationalMonitor(config or MonitoringConfig())
        return monitor.report(self.all_traces(), feedback_store=self.feedback)

    # ------------------------------------------------------------------
    # incident investigation & human governance (Phase 8 Step 3)

    def investigate_incident(self, interaction_id: str):
        """Reconstruct a full :class:`IncidentInvestigation` (no pipeline re-run)."""
        return self.investigation.investigate(interaction_id)

    def record_governance_action(self, interaction_id: str, **kwargs: Any):
        """Record a human governance action; the automated decision is immutable."""
        return self.investigation.record_governance_action(interaction_id, **kwargs)

    def record_governance_override(
        self,
        interaction_id: str,
        *,
        action_type: str,
        new_tier: str | None = None,
        justification: str = "",
        reviewer_id: str = "admin",
    ):
        """
        Reviewer-facing wrapper: APPROVE / MODIFY / REJECT an interaction's
        decision on the append-only governance track. The DecisionTrace and the
        automated ``original_decision`` are never mutated.
        """
        return self.investigation.record_governance_action(
            interaction_id,
            action=action_type,
            actor=reviewer_id,
            comment=justification,
            reviewer_decision=new_tier,
        )

    def governance_history(self, interaction_id: str):
        return self.investigation.get_governance_history(interaction_id)

    def investigation_counterfactual(
        self, interaction_id: str, modified_fields: dict[str, Any]
    ):
        """An explicitly-labelled 'what if?' simulation for an investigation."""
        return self.investigation.counterfactual(interaction_id, modified_fields)

    def submit_feedback(self, **kwargs: Any) -> FeedbackRecord:
        interaction_id = kwargs["interaction_id"]
        if kwargs.get("system_decision") is None:
            trace = self.get_audit_trace(interaction_id)
            if trace is not None:
                kwargs["system_decision"] = trace.final_decision.decision
        if kwargs.get("system_decision") is None:
            raise KeyError(
                f"No prior decision recorded for interaction '{interaction_id}'. "
                "Call /check first or pass system_decision explicitly."
            )
        return self.feedback.submit(**kwargs)

    def feedback_summary(self) -> dict[str, Any]:
        return self.feedback.aggregate().model_dump()

    # ------------------------------------------------------------------
    # policy simulation & counterfactual analysis

    def simulate_policy(
        self, interaction: Interaction, profiles: list[str]
    ) -> dict[str, Any]:
        from simulation.engine import simulate_policies

        return simulate_policies(self.engine, interaction, profiles).model_dump()

    def counterfactual(
        self, interaction: Interaction, modified_fields: dict[str, Any]
    ) -> dict[str, Any]:
        from simulation.engine import compare_decisions

        return compare_decisions(self.engine, interaction, modified_fields).model_dump()

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            served = self._checks_served
        return {
            "checks_served": served,
            "active_sessions": len(self.session_manager.active_sessions()),
            "feedback_records": len(self.feedback),
        }
