"""
Phase 8 — Step 3: incident investigation & governance workflow.

Investigation is read-only reconstruction (replay + explainability) plus a
human governance-action recorder. It never re-runs the pipeline, never
reads ground truth, never leaks PII, and never mutates the automated
ControlPlane decision.
"""

from __future__ import annotations

import copy
import pathlib

import pytest

from api.service import ControlPlaneService
from data.schemas import InterventionTier
from investigation.schemas import (
    GovernanceActionType,
    IncidentInvestigation,
    InvestigationNotFound,
    InvestigationStatus,
)
from investigation.service import GovernanceError, GovernanceStore, InvestigationService
from settings import load_settings
from tests import scenarios

_INV_DIR = pathlib.Path(__file__).resolve().parent.parent / "investigation"

# raw synthetic PII strings that must NEVER appear anywhere in an investigation
_KNOWN_PII = ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta")


@pytest.fixture(scope="module")
def service():
    svc = ControlPlaneService(fit_cost_baseline=False)
    ids = {}
    for name, factory in scenarios.ALL_SINGLE_TURN.items():
        it = factory()
        svc.check(it, timestamp=it.timestamp)
        ids[name] = it.interaction_id
    # scenario G — multi-turn, last turn escalates
    g_ids = []
    for turn in scenarios.scenario_g_multi_turn():
        svc.check(turn, timestamp=turn.timestamp)
        g_ids.append(turn.interaction_id)
    ids["G_last"] = g_ids[-1]
    svc._scenario_ids = ids
    return svc


def _inv(service, name) -> IncidentInvestigation:
    result = service.investigate_incident(service._scenario_ids[name])
    assert isinstance(result, IncidentInvestigation)
    return result


def _fresh(service, factory, suffix: str) -> str:
    """Check a scenario under a unique id so the test starts from status OPEN."""
    it = factory().model_copy(update={"interaction_id": f"INV-{suffix}"})
    service.check(it, timestamp=it.timestamp)
    return it.interaction_id


# ------------------------------------------------------------------
# scenario coverage (§27)
# ------------------------------------------------------------------


def test_scenario_a_clean_investigation_works(service):
    inv = _inv(service, "A_clean")
    assert inv.original_decision == "ALLOW"
    assert inv.incident is None                 # clean ALLOW is not an incident
    assert inv.explanation.decision is InterventionTier.ALLOW
    assert inv.investigation_status is InvestigationStatus.OPEN


def test_scenario_b_hallucination_shows_reason_codes(service):
    inv = _inv(service, "B_hallucination")
    assert inv.original_decision == "VERIFY"
    assert inv.explanation.primary_reasons          # real reason codes, not invented
    assert any("PERFORMANCE" in c or "CONTRADICT" in c for c in inv.explanation.primary_reasons)


def test_scenario_c_pii_is_block_and_human_review(service):
    inv = _inv(service, "C_pii")
    assert inv.original_decision == "BLOCK"
    assert inv.requires_human_review is True
    assert inv.incident is not None and inv.incident.severity.value == "CRITICAL"
    assert "CRITICAL_PII" in inv.explanation.primary_reasons


def test_scenario_d_high_consequence_reasons(service):
    inv = _inv(service, "D_high_consequence")
    reasons = set(inv.explanation.primary_reasons) | {
        d for d in inv.explanation.decision_drivers
    }
    assert "HIGH_CONSEQUENCE" in reasons
    assert any("FINANCIAL" in r or "CRITICALITY" in r for r in reasons)


def test_scenario_e_multi_risk_has_multiple_drivers(service):
    inv = _inv(service, "E_multi_risk")
    assert len(inv.explanation.primary_reasons) >= 2
    assert inv.explanation.decision_drivers


def test_scenario_g_final_turn_escalates_and_governance_works(service):
    inv = _inv(service, "G_last")
    assert inv.original_decision in ("HUMAN_REVIEW", "BLOCK")
    updated = service.record_governance_action(
        service._scenario_ids["G_last"], action="ACKNOWLEDGE", actor="ops"
    )
    assert updated.investigation_status is InvestigationStatus.ACKNOWLEDGED


# ------------------------------------------------------------------
# governance workflow
# ------------------------------------------------------------------


def test_governance_status_transitions(service):
    iid = service._scenario_ids["B_hallucination"]
    inv0 = service.investigate_incident(iid)
    assert inv0.investigation_status is InvestigationStatus.OPEN

    inv1 = service.record_governance_action(iid, action="ACKNOWLEDGE")
    assert inv1.investigation_status is InvestigationStatus.ACKNOWLEDGED
    assert len(inv1.governance_history) == 1

    inv2 = service.record_governance_action(
        iid, action="ESCALATE", comment="needs senior review"
    )
    assert inv2.investigation_status is InvestigationStatus.ESCALATED


def test_governance_immutability_modify_decision(service):
    """§29 — MODIFY_DECISION records the reviewer's tier but never changes the trace."""
    iid = service._scenario_ids["C_pii"]
    original_before = service.get_audit_trace(iid).final_decision.decision.value
    assert original_before == "BLOCK"

    inv = service.record_governance_action(
        iid,
        action="MODIFY_DECISION",
        comment="I would have routed to human review only",
        reviewer_decision="VERIFY",
    )
    assert inv.original_decision == "BLOCK"
    assert inv.latest_reviewer_decision == "VERIFY"
    action = inv.governance_history[-1]
    assert action.original_decision == "BLOCK"
    assert action.reviewer_decision == "VERIFY"

    # the stored decision is untouched
    assert service.get_audit_trace(iid).final_decision.decision.value == "BLOCK"
    assert service.investigate_incident(iid).explanation.decision is InterventionTier.BLOCK


def test_governance_validation_errors(service):
    iid = service._scenario_ids["D_high_consequence"]
    with pytest.raises(GovernanceError):
        service.record_governance_action(iid, action="NOT_A_REAL_ACTION")
    with pytest.raises(GovernanceError):
        service.record_governance_action(iid, action="MODIFY_DECISION", comment="x")  # no reviewer_decision
    with pytest.raises(GovernanceError):
        service.record_governance_action(iid, action="ESCALATE")  # comment required


def test_governance_action_not_available_from_status(service):
    iid = service._scenario_ids["F_cost_anomaly"]
    service.record_governance_action(iid, action="CLOSE", comment="")  # CLOSE ok from OPEN
    with pytest.raises(GovernanceError):
        service.record_governance_action(iid, action="ACKNOWLEDGE")  # nothing available after CLOSE


def test_unknown_interaction_returns_not_found(service):
    result = service.investigate_incident("NO-SUCH-ID")
    assert isinstance(result, InvestigationNotFound)
    assert result.found is False


# ------------------------------------------------------------------
# counterfactual (§28) — explicitly a simulation
# ------------------------------------------------------------------


def test_counterfactual_is_a_simulation_only(service):
    iid = service._scenario_ids["D_high_consequence"]
    trace_before = copy.deepcopy(service.get_audit_trace(iid).model_dump(mode="json"))
    settings_before = copy.deepcopy(load_settings())

    cf = service.investigation_counterfactual(iid, {"action_amount_inr": 1.0})
    assert cf.simulated is True
    assert cf.current_decision == "VERIFY"
    assert cf.current_decision != cf.counterfactual_decision or not cf.decision_changed
    assert "SIMULATION" in cf.note

    # original trace + production config unchanged
    assert service.get_audit_trace(iid).model_dump(mode="json") == trace_before
    assert load_settings() == settings_before
    assert service.get_audit_trace(iid).final_decision.decision.value == "VERIFY"


def test_counterfactual_rejects_free_text_fields(service):
    iid = service._scenario_ids["B_hallucination"]
    cf = service.investigation_counterfactual(
        iid, {"response": "totally different text", "action_amount_inr": 5.0}
    )
    assert "response" in cf.rejected_fields
    assert "action_amount_inr" not in cf.rejected_fields


# ------------------------------------------------------------------
# §30 — investigation never re-runs the pipeline
# ------------------------------------------------------------------


def test_investigation_does_not_rerun_pipeline(service, monkeypatch):
    # record the trace BEFORE the pipeline is sabotaged
    fresh_id = _fresh(service, scenarios.scenario_c_pii, "norerun")

    import consequence.engine as cons_mod
    import criticality.engine as crit_mod
    import decision.engine as dec_mod
    import detectors.cost.detector as cost_mod
    import detectors.performance.detector as perf_mod
    import detectors.responsibility.detector as resp_mod
    import fusion.engine as fus_mod
    import policy.engine as pol_mod
    import verification.router as router_mod

    def boom(*_a, **_k):  # pragma: no cover
        raise AssertionError("investigation re-ran a pipeline component")

    monkeypatch.setattr(perf_mod.PerformanceDetector, "detect", boom)
    monkeypatch.setattr(resp_mod.ResponsibilityDetector, "detect", boom)
    monkeypatch.setattr(cost_mod.CostDetector, "detect", boom)
    monkeypatch.setattr(cons_mod.ConsequenceEngine, "assess", boom)
    monkeypatch.setattr(crit_mod.CriticalityEngine, "assess", boom)
    monkeypatch.setattr(fus_mod.RiskFusionEngine, "fuse_scores", boom)
    monkeypatch.setattr(pol_mod.PolicyEngine, "decide", boom)
    monkeypatch.setattr(dec_mod.DecisionEngine, "evaluate", boom)
    monkeypatch.setattr(router_mod.VerificationRouter, "route", boom)

    inv = service.investigate_incident(fresh_id)
    assert isinstance(inv, IncidentInvestigation)
    assert inv.original_decision == "BLOCK"
    # recording a governance action also must not touch the pipeline
    inv2 = service.record_governance_action(fresh_id, action="ACKNOWLEDGE")
    assert inv2.investigation_status is InvestigationStatus.ACKNOWLEDGED


# ------------------------------------------------------------------
# §31 — determinism
# ------------------------------------------------------------------


def test_investigation_is_deterministic(service):
    iid = service._scenario_ids["E_multi_risk"]
    a = service.investigate_incident(iid).model_dump(mode="json")
    b = service.investigate_incident(iid).model_dump(mode="json")

    def _strip(node):
        if isinstance(node, dict):
            return {
                k: _strip(v)
                for k, v in node.items()
                if not (k.endswith("_ms") or k == "latency" or k == "generated_at")
            }
        if isinstance(node, list):
            return [_strip(x) for x in node]
        return node

    assert _strip(a) == _strip(b)


# ------------------------------------------------------------------
# §33 — no raw PII anywhere
# ------------------------------------------------------------------


def test_no_raw_pii_in_investigation(service):
    iid = _fresh(service, scenarios.scenario_c_pii, "pii-safety")
    trace = service.get_audit_trace(iid)
    raw_spans = [
        f.matched_text
        for f in trace.responsibility.pii.findings
        if f.matched_text and f.matched_text.strip()
    ]
    assert raw_spans

    # investigate + do a full governance workflow + a counterfactual
    service.record_governance_action(iid, action="ACKNOWLEDGE")
    service.record_governance_action(
        iid, action="MODIFY_DECISION", comment="reviewed", reviewer_decision="HUMAN_REVIEW"
    )
    service.investigation_counterfactual(iid, {"action_amount_inr": 0.0})
    inv = service.investigate_incident(iid)

    blob = inv.model_dump_json()
    interaction = service.get_stored_interaction(iid)
    for needle in list(raw_spans) + list(_KNOWN_PII) + [interaction.response]:
        assert needle not in blob, needle
    assert "matched_text" not in blob


# ------------------------------------------------------------------
# §11 / §32 — source guards
# ------------------------------------------------------------------


def test_investigation_source_does_not_orchestrate_pipeline():
    banned = (
        "DecisionEngine(", "PerformanceDetector(", "ResponsibilityDetector(",
        "CostDetector(", "VerificationRouter(", "OperationalMonitor(",
        "RiskFusionEngine(", "PolicyEngine(", "ConsequenceEngine(", "CriticalityEngine(",
        ".detect(", ".fuse_scores(", ".decide(", ".route(", ".assess(",
    )
    for path in _INV_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name}: {token}"
        # ``.evaluate(`` is allowed only transitively via simulation.engine, never
        # written in the investigation source:
        assert ".evaluate(" not in text, f"{path.name}: .evaluate("


def test_investigation_does_not_import_evaluation():
    for path in _INV_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import evaluation" not in text
        assert "from evaluation" not in text
        assert "EvaluationCase" not in text
        for token in ("ground_truth_", "expected_decision", "final_outcome"):
            assert token not in text, f"{path.name}: {token}"


# ------------------------------------------------------------------
# GovernanceStore unit + swap-in wiring
# ------------------------------------------------------------------


def test_governance_store_is_append_only_and_clean():
    store = GovernanceStore()
    assert store.current_status("x") is InvestigationStatus.OPEN
    store.record_action(
        interaction_id="x",
        action=GovernanceActionType.ACKNOWLEDGE,
        actor="r",
        comment="",
        previous_status=InvestigationStatus.OPEN,
        new_status=InvestigationStatus.ACKNOWLEDGED,
        original_decision="BLOCK",
    )
    assert store.current_status("x") is InvestigationStatus.ACKNOWLEDGED
    assert len(store.get_actions("x")) == 1
    assert len(store.get_all_actions()) == 1


def test_investigation_service_accepts_injected_store(service):
    store = GovernanceStore()
    inv_service = InvestigationService(service, store)
    iid = service._scenario_ids["A_clean"]
    inv_service.record_governance_action(iid, action="ACKNOWLEDGE")
    assert store.current_status(iid) is InvestigationStatus.ACKNOWLEDGED


# ------------------------------------------------------------------
# effective governed decision + reviewer-facing convenience API
# ------------------------------------------------------------------


def test_effective_governed_decision_defaults_to_original(service):
    iid = _fresh(service, scenarios.scenario_c_pii, "EFF-DEFAULT")
    inv = service.investigate_incident(iid)
    assert inv.original_decision == "BLOCK"
    assert inv.effective_governed_decision == "BLOCK"     # no override yet
    assert inv.is_overridden is False


def test_effective_governed_decision_reflects_latest_override(service):
    iid = _fresh(service, scenarios.scenario_c_pii, "EFF-OVERRIDE")
    trace_before = service.get_audit_trace(iid).final_decision.decision.value

    service.record_governance_override(
        iid, action_type="MODIFY", new_tier="HUMAN_REVIEW",
        justification="route to a human rather than a hard block", reviewer_id="alice",
    )
    inv = service.investigate_incident(iid)
    assert inv.original_decision == "BLOCK"
    assert inv.effective_governed_decision == "HUMAN_REVIEW"
    assert inv.is_overridden is True
    # a second override wins
    service.record_governance_override(
        iid, action_type="MODIFY", new_tier="VERIFY", justification="on reflection, verify",
    )
    inv2 = service.investigate_incident(iid)
    assert inv2.effective_governed_decision == "VERIFY"

    # the DecisionTrace is byte-for-byte unchanged
    assert service.get_audit_trace(iid).final_decision.decision.value == trace_before == "BLOCK"


def test_governance_store_effective_decision_and_history_alias(service):
    iid = _fresh(service, scenarios.scenario_b_hallucination, "EFF-STORE")
    original = service.get_audit_trace(iid).final_decision.decision.value
    assert service.governance.effective_decision(iid, original) == original     # none recorded
    service.record_governance_override(iid, action_type="APPROVED", justification="agree")
    assert service.governance.get_history(iid) == service.governance.get_actions(iid)
    # APPROVE does not set an effective override
    assert service.governance.effective_decision(iid, original) == original


def test_canonical_governance_action_accepts_short_and_past_tense_forms():
    from investigation.schemas import GovernanceActionType
    from investigation.service import canonical_governance_action

    assert canonical_governance_action("MODIFY") is GovernanceActionType.MODIFY_DECISION
    assert canonical_governance_action("modified") is GovernanceActionType.MODIFY_DECISION
    assert canonical_governance_action("APPROVED") is GovernanceActionType.APPROVE_DECISION
    assert canonical_governance_action("REJECT") is GovernanceActionType.REJECT_DECISION
    assert (
        canonical_governance_action(GovernanceActionType.CLOSE)
        is GovernanceActionType.CLOSE
    )
    with pytest.raises(ValueError):
        canonical_governance_action("NONSENSE")
