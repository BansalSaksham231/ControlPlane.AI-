"""Session Manager tests — accumulation, bounds/decay, escalation, reset."""

from __future__ import annotations

from datetime import datetime

import pytest

from session.manager import SessionManager
from session.schemas import SessionState

TS = datetime(2026, 8, 21, 12, 0, 0)


@pytest.fixture()
def manager() -> SessionManager:
    return SessionManager()


def _record(manager, sid, risk, decision="VERIFY", n=1):
    for i in range(n):
        manager.record(sid, overall_risk=risk, decision=decision, interaction_id=f"{sid}-{i}", timestamp=TS)


def test_new_session_has_no_history(manager):
    contribution = manager.contribution("S-new", 0.5)
    assert contribution["interaction_count"] == 0
    assert contribution["history_component"] == 0.0
    assert contribution["adjusted_overall_risk"] == 0.5
    assert contribution["escalated"] is False


def test_history_raises_but_never_lowers_current_risk(manager):
    _record(manager, "S1", 0.8, n=3)
    contribution = manager.contribution("S1", 0.1)
    # low current turn, but risky history -> adjusted is >= current
    assert contribution["adjusted_overall_risk"] >= 0.1
    contribution_high = manager.contribution("S1", 0.95)
    assert contribution_high["adjusted_overall_risk"] >= 0.95


def test_accumulation_is_bounded(manager):
    _record(manager, "S2", 1.0, n=50)
    state = manager.get_state("S2")
    assert state.cumulative_risk <= 1.0
    contribution = manager.contribution("S2", 1.0)
    assert contribution["session_risk"] <= 0.98 + 1e-9


def test_decay_reduces_old_risk(manager):
    _record(manager, "S3", 0.9, n=1)
    after_one = manager.get_state("S3").cumulative_risk
    _record(manager, "S3", 0.0, n=5)
    after_calm = manager.get_state("S3").cumulative_risk
    assert after_calm < after_one


def test_escalation_after_repeated_high_risk(manager):
    _record(manager, "S4", 0.7, n=1)
    before = manager.contribution("S4", 0.7)
    assert before["escalated"] is False
    _record(manager, "S4", 0.7, n=1)
    after = manager.contribution("S4", 0.7)  # 2 prior + this turn = 3 high-risk turns
    assert after["escalated"] is True
    assert after["adjusted_overall_risk"] > 0.7


def test_reset_clears_session(manager):
    _record(manager, "S5", 0.9, n=4)
    assert manager.get_state("S5").interaction_count == 4
    manager.reset("S5")
    assert manager.get_state("S5").interaction_count == 0
    assert isinstance(manager.get_state("S5"), SessionState)


def test_window_bounds_recent_history(manager):
    _record(manager, "S6", 0.5, n=40)
    state = manager.get_state("S6")
    assert len(state.recent_risks) <= 20
    assert len(state.recent_decisions) <= 20


def test_explanation_present(manager):
    _record(manager, "S7", 0.8, n=3)
    contribution = manager.contribution("S7", 0.8)
    assert len(contribution["explanation"]) > 20


def test_multi_turn_escalation_end_to_end():
    from decision.engine import DecisionEngine
    from tests import scenarios

    sm = SessionManager()
    engine = DecisionEngine(session_manager=sm)
    decisions = []
    for interaction in scenarios.scenario_g_multi_turn():
        trace = engine.evaluate(interaction, timestamp=TS)
        decisions.append(trace.final_decision.decision.value)

    from policy.schemas import TIER_RANK
    from data.schemas import InterventionTier

    # risk should be non-decreasing in tier severity across the run, ending
    # stronger than it started.
    assert TIER_RANK[InterventionTier(decisions[-1])] > TIER_RANK[InterventionTier(decisions[0])]
    assert "HUMAN_REVIEW" in decisions or "BLOCK" in decisions


def test_get_state_returns_copy(manager):
    _record(manager, "S8", 0.5, n=1)
    state = manager.get_state("S8")
    state.cumulative_risk = 999
    assert manager.get_state("S8").cumulative_risk != 999


# ------------------------------------------------------------------ #
# Contextual snapshots + non-decaying critical floor
# ------------------------------------------------------------------ #
def test_record_is_backward_compatible_without_structured_signals(manager):
    state = manager.record(
        "BC", overall_risk=0.4, decision="VERIFY", interaction_id="BC-1", timestamp=TS
    )
    assert state.snapshot.turns_recorded == 1
    assert state.snapshot.critical_floor == 0.0
    assert manager.contribution("BC", 0.4)["critical_floor_applied"] is False


def test_contextual_snapshot_merges_prior_turn_signals(manager):
    manager.record(
        "SNAP", overall_risk=0.5, decision="VERIFY", interaction_id="SNAP-1", timestamp=TS,
        dimension_risks=(0.6, 0.2, 0.1),
        reason_codes=["CONTRADICTED_EVIDENCE", "HIGH_PERFORMANCE_RISK"],
        tier_changing_rules=["MIN_TIER_CONTRADICTION"],
        pii_entity_keys=["email:k***@e***"],
    )
    manager.record(
        "SNAP", overall_risk=0.3, decision="ANNOTATE", interaction_id="SNAP-2", timestamp=TS,
        dimension_risks=(0.1, 0.4, 0.0),
        reason_codes=["CONTRADICTED_EVIDENCE"],
        pii_entity_keys=["phone:*******21"],
    )
    snap = manager.snapshot("SNAP")
    assert snap.turns_recorded == 2
    assert snap.reason_code_counts["CONTRADICTED_EVIDENCE"] == 2
    assert snap.reason_code_counts["HIGH_PERFORMANCE_RISK"] == 1
    assert snap.pii_entity_keys == ["email:k***@e***", "phone:*******21"]
    assert snap.tier_changing_rules == ["MIN_TIER_CONTRADICTION"]
    assert snap.peak_performance_risk == 0.6          # max, not last
    assert snap.peak_responsibility_risk == 0.4
    assert not snap.has_critical_history


def test_critical_floor_is_set_and_never_decays(manager):
    manager.record(
        "CF", overall_risk=0.9, decision="BLOCK", interaction_id="CF-1", timestamp=TS,
        critical=True, critical_trigger="CRITICAL_PII",
    )
    floor = manager.snapshot("CF").critical_floor
    assert floor == pytest.approx(0.75)

    # many calm turns must NOT decay the critical floor
    for i in range(10):
        manager.record("CF", overall_risk=0.0, decision="ALLOW",
                       interaction_id=f"CF-calm-{i}", timestamp=TS)
    assert manager.snapshot("CF").critical_floor == pytest.approx(0.75)
    assert manager.get_state("CF").cumulative_risk < 0.1   # ordinary risk DID decay

    contribution = manager.contribution("CF", 0.05)         # a benign next turn
    assert contribution["critical_floor_applied"] is True
    assert contribution["adjusted_overall_risk"] == pytest.approx(0.75)
    assert contribution["has_critical_history"] is True


def test_critical_events_are_recorded_with_turn_index(manager):
    manager.record("CE", overall_risk=0.2, decision="ANNOTATE", interaction_id="CE-1", timestamp=TS)
    manager.record("CE", overall_risk=0.9, decision="BLOCK", interaction_id="CE-2", timestamp=TS,
                   critical=True, critical_trigger="BLOCK")
    events = manager.snapshot("CE").critical_events
    assert len(events) == 1
    assert events[0].turn_index == 2
    assert events[0].trigger == "BLOCK"
    assert events[0].interaction_id == "CE-2"


def test_snapshot_stores_no_raw_pii():
    from decision.engine import DecisionEngine
    from tests import scenarios

    sm = SessionManager()
    engine = DecisionEngine(session_manager=sm)
    pii = scenarios.scenario_c_pii().model_copy(update={"session_id": "PII-SNAP"})
    engine.evaluate(pii, timestamp=TS)

    snap = sm.snapshot("PII-SNAP")
    blob = snap.model_dump_json()
    for raw in ("ACC-227763", "karan.mehta@example-test.com", "+91-940847221", "Karan Mehta"):
        assert raw not in blob
    assert snap.has_critical_history          # PII BLOCK -> critical floor engaged


def test_critical_floor_forces_scrutiny_on_later_clean_turns():
    from decision.engine import DecisionEngine
    from tests import scenarios

    sm = SessionManager()
    engine = DecisionEngine(session_manager=sm)
    sid = "INHERIT"
    seq = [
        scenarios.scenario_a_clean(),      # ALLOW
        scenarios.scenario_c_pii(),        # BLOCK -> sets critical floor
        scenarios.scenario_a_clean(),      # would be ALLOW on its own...
        scenarios.scenario_a_clean(),
    ]
    decisions = []
    for i, it in enumerate(seq, 1):
        trace = engine.evaluate(
            it.model_copy(update={"session_id": sid, "interaction_id": f"{sid}-{i}"}),
            timestamp=TS,
        )
        decisions.append(trace.final_decision.decision.value)

    assert decisions[0] == "ALLOW"
    assert decisions[1] == "BLOCK"
    # ...but the non-decaying critical floor keeps later turns under human scrutiny
    assert decisions[2] in ("HUMAN_REVIEW", "BLOCK")
    assert decisions[3] in ("HUMAN_REVIEW", "BLOCK")


def test_session_manager_is_deterministic():
    a, b = SessionManager(), SessionManager()
    for m in (a, b):
        m.record("D", overall_risk=0.7, decision="VERIFY", interaction_id="D-1", timestamp=TS,
                 dimension_risks=(0.5, 0.6, 0.1), reason_codes=["X"], critical=True)
        m.record("D", overall_risk=0.2, decision="ALLOW", interaction_id="D-2", timestamp=TS)
    assert a.get_state("D").model_dump() == b.get_state("D").model_dump()
    assert a.contribution("D", 0.1) == b.contribution("D", 0.1)
