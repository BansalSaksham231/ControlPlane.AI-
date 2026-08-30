"""
Tiered Cascade Routing for progressive verification.
=====================================================

Decides **how much verification compute** an AI interaction deserves before
ControlPlane's decision pipeline adjudicates it, in the spirit of FrugalGPT's
LLM cascades and RouteLLM's learned query routers, under the latency envelope of
the "200-millisecond challenge" for inline AI guardrails.

    Tier 0  FastPathEvaluator      deterministic, < 50 ms, resolves the bulk of
                                   standard traffic with zero semantic reasoning
    Tier 1  ComplexityClassifier   a lightweight learned router (heuristic
                                   fallback or a fitted matrix-factorisation
                                   model) scoring how much a DEEP pass matters
    Tier 2  DEEP verification      the heavy TF-IDF retrieval + NLI pass, run
                                   only when Tier 1 clears the per-application
                                   cost-optimal escalation threshold

Scope boundary
--------------
The cascade routes *verification depth*, not *final decisions*:

    FAST_ALLOW     trust the cheap shallow verification verdict; skip the DEEP
                   re-run. The interaction still flows through responsibility /
                   cost / consequence / policy / decision unchanged.
    ROUTE_TO_DEEP  run the full verification pass before the pipeline continues.

``FAST_ALLOW`` is not a terminal ``ALLOW``. Final ALLOW/…/BLOCK authority stays
with the PolicyEngine + DecisionEngine, so a missed Tier-0 pattern degrades to
"cheaper verification", never "no governance". A deterministic hard-boundary hit
(outbound PII pattern, blocked keyword) therefore escalates to DEEP with a
flagged signal — and sets ``bypass_semantics`` so the DEEP backend can skip the
lexical/NLI work on an interaction the policy layer will almost certainly block.

Three routing entry points
--------------------------
    route(...)              single-shot sequential cascade
    route_incremental(...)  multi-turn agentic routing on "contextual snapshots"
                            — evaluates only the newest turn/tool output and
                            carries a bounded RoutingState forward, avoiding the
                            "stochastic tax" of re-scanning the whole history
    route_speculative(...)  runs Tier 1 and a caller-supplied DEEP verifier in
                            parallel and keeps the winner (latency-critical apps)

Determinism: every verdict is a pure function of (interaction, profile, model).
``time.perf_counter`` is measurement only. No randomness, no network.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final

from verification.routing_models import (
    ComplexityClassifier,
    ComplexityPrediction,
    HeuristicComplexityClassifier,
    RoutingSample,
    ThresholdCalibration,
    estimate_tokens,
    extract_features,
    select_cost_optimal_threshold,
)

__all__ = [
    "RouteVerdict",
    "CascadeTier",
    "FastPathOutcome",
    "Interaction",
    "ApplicationProfile",
    "DEFAULT_PROFILES",
    "resolve_profile",
    "FastPathResult",
    "RoutingDecision",
    "RoutingState",
    "SpeculativeResult",
    "FastPathEvaluator",
    "CascadeRouter",
    "RoutingSample",
    "ThresholdCalibration",
    "select_cost_optimal_threshold",
    "FAST_PATH_BUDGET_MS",
    "CLASSIFIER_BUDGET_MS",
    "TOTAL_GUARDRAIL_BUDGET_MS",
]

# --------------------------------------------------------------------------- #
# Latency budgets (milliseconds) — design targets, measured and reported.
# --------------------------------------------------------------------------- #
FAST_PATH_BUDGET_MS: Final[float] = 50.0
CLASSIFIER_BUDGET_MS: Final[float] = 100.0
TOTAL_GUARDRAIL_BUDGET_MS: Final[float] = 200.0
FAST_PATH_COVERAGE_TARGET: Final[float] = 0.85

# Bounded session-complexity accumulation for incremental routing. Mirrors the
# decay/current-weight shape of ControlPlane's session/ risk accumulator.
_SESSION_DECAY: Final[float] = 0.55
_SESSION_CURRENT_WEIGHT: Final[float] = 0.70
_REPEAT_AMBIGUITY_BUMP: Final[float] = 0.08


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class RouteVerdict(str, Enum):
    FAST_ALLOW = "FAST_ALLOW"        # trust the shallow verdict; skip DEEP
    ROUTE_TO_DEEP = "ROUTE_TO_DEEP"  # run the full TF-IDF + NLI pass


class CascadeTier(int, Enum):
    FAST_PATH_DETERMINISTIC = 0
    COMPLEXITY_CLASSIFIER = 1
    DEEP_VERIFICATION = 2


class FastPathOutcome(str, Enum):
    CLEARED = "CLEARED"
    AMBIGUOUS = "AMBIGUOUS"
    VIOLATION = "VIOLATION"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
_NON_ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"", "none", "information", "informational", "info", "answer", "chat", "advice"}
)


@dataclass(frozen=True, slots=True)
class Interaction:
    """
    Production-shaped view of one AI interaction (or, in incremental routing,
    one turn / tool output). Carries no ground-truth / evaluation fields.
    """

    interaction_id: str
    application: str
    prompt: str
    response: str
    context: str = ""
    action_type: str = "information"
    action_amount: float = 0.0
    affected_entities: int = 0
    tool_calls: int = 0
    retry_count: int = 0
    turn_index: int = 0
    is_tool_output: bool = False

    @property
    def estimated_response_tokens(self) -> int:
        return estimate_tokens(self.response)

    @property
    def estimated_prompt_tokens(self) -> int:
        return estimate_tokens(self.prompt)

    @property
    def has_evidence(self) -> bool:
        return bool(self.context.strip())

    @property
    def is_action(self) -> bool:
        return (
            self.action_type.strip().lower() not in _NON_ACTION_TYPES
            or self.action_amount > 0.0
            or self.affected_entities > 0
        )


@dataclass(frozen=True, slots=True)
class ApplicationProfile:
    """
    Per-application routing policy. ``cost_optimal_escalation_threshold`` is the
    dynamic knob fitted offline per application (see
    :func:`verification.routing_models.select_cost_optimal_threshold`).
    """

    name: str
    cost_optimal_escalation_threshold: float
    description: str = ""
    fast_path_max_response_tokens: int = 120
    blocked_keywords: tuple[str, ...] = ()
    high_stakes_intent_markers: tuple[str, ...] = ()
    force_deep_on_action: bool = False
    fast_path_latency_budget_ms: float = FAST_PATH_BUDGET_MS
    #   "fast" -> degrade open (FAST_ALLOW + async re-check flag)
    #   "deep" -> degrade closed (ROUTE_TO_DEEP, accept the latency hit)
    budget_degradation: str = "fast"

    def __post_init__(self) -> None:
        if not 0.0 <= self.cost_optimal_escalation_threshold <= 1.0:
            raise ValueError(f"{self.name}: threshold must be in [0, 1]")
        if self.budget_degradation not in ("fast", "deep"):
            raise ValueError(f"{self.name}: budget_degradation must be 'fast'|'deep'")

    def effective_threshold(
        self, interaction: Interaction, session_complexity: float = 0.0
    ) -> float:
        """
        Resolve the escalation threshold for this interaction. Never a single
        hardcoded number: an actioned interaction under ``force_deep_on_action``
        pins it to 0.0; a long response or an already-elevated session lowers it.
        """
        if self.force_deep_on_action and interaction.is_action:
            return 0.0
        threshold = self.cost_optimal_escalation_threshold
        if interaction.estimated_response_tokens > 2 * self.fast_path_max_response_tokens:
            threshold = max(0.0, threshold - 0.10)
        if session_complexity >= 0.60:
            threshold = max(0.0, threshold - 0.10)
        return threshold


@dataclass(frozen=True, slots=True)
class FastPathResult:
    outcome: FastPathOutcome
    signals: tuple[str, ...]
    confidence: float
    latency_ms: float
    explanation: str


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Fully-audited cascade result. Every field is safe to log (no raw text, no PII)."""

    interaction_id: str
    application: str
    verdict: RouteVerdict
    tier_reached: CascadeTier
    reason_code: str
    explanation: str
    predicted_complexity_score: float | None
    applied_threshold: float
    deterministic_signals: tuple[str, ...]
    fast_path_latency_ms: float
    classifier_latency_ms: float
    total_latency_ms: float
    within_latency_budget: bool
    turn_index: int = 0

    @property
    def routes_to_deep(self) -> bool:
        return self.verdict is RouteVerdict.ROUTE_TO_DEEP

    @property
    def bypass_semantics(self) -> bool:
        """
        True when the DEEP backend may skip lexical chunking / retrieval / NLI
        and go straight to fusion + policy: the escalation is driven by a
        deterministic hard boundary the policy layer will adjudicate anyway, so
        the semantic grounding score cannot change the outcome.
        """
        return (
            self.verdict is RouteVerdict.ROUTE_TO_DEEP
            and self.reason_code == "DETERMINISTIC_HARD_BOUNDARY"
        )


@dataclass(frozen=True, slots=True)
class RoutingState:
    """
    The "contextual snapshot" carried between turns of an agentic session so the
    router never re-scans the accumulated history string. Bounded — its size
    does not grow with conversation length.
    """

    turn_count: int = 0
    every_turn_cleared: bool = True
    peak_complexity: float = 0.0
    session_complexity: float = 0.0        # bounded, decayed accumulation
    consecutive_ambiguous: int = 0
    high_stakes_seen: bool = False
    carried_signals: tuple[str, ...] = ()

    def advanced(
        self,
        *,
        cleared: bool,
        turn_complexity: float,
        ambiguous: bool,
        high_stakes: bool,
        new_signals: Iterable[str],
    ) -> "RoutingState":
        session = _clamp01(
            _SESSION_DECAY * self.session_complexity
            + _SESSION_CURRENT_WEIGHT * turn_complexity
        )
        consec = self.consecutive_ambiguous + 1 if ambiguous else 0
        if consec >= 2:
            session = _clamp01(session + _REPEAT_AMBIGUITY_BUMP)
        merged = tuple(dict.fromkeys((*self.carried_signals, *new_signals)))[-12:]
        return RoutingState(
            turn_count=self.turn_count + 1,
            every_turn_cleared=self.every_turn_cleared and cleared,
            peak_complexity=max(self.peak_complexity, turn_complexity),
            session_complexity=session,
            consecutive_ambiguous=consec,
            high_stakes_seen=self.high_stakes_seen or high_stakes,
            carried_signals=merged,
        )


@dataclass(frozen=True, slots=True)
class SpeculativeResult:
    """Outcome of :meth:`CascadeRouter.route_speculative`."""

    routing_decision: RoutingDecision
    deep_result: object | None          # populated iff the DEEP pass was kept
    deep_started: bool
    deep_discarded: bool                 # started speculatively, then not needed
    wall_latency_ms: float


# --------------------------------------------------------------------------- #
# Built-in application profiles
# --------------------------------------------------------------------------- #
_DEFAULT_HIGH_STAKES_MARKERS: Final[tuple[str, ...]] = (
    "refund", "cancel", "close account", "delete", "terminate", "transfer",
    "wire", "payment", "password", "credential", "legal", "lawsuit", "medical",
    "diagnos", "prescri", "guarantee",
)

DEFAULT_PROFILES: Final[Mapping[str, ApplicationProfile]] = {
    "customer_support": ApplicationProfile(
        name="customer_support",
        description="External chatbot. Balanced cost/safety; actions always deep.",
        cost_optimal_escalation_threshold=0.55,
        fast_path_max_response_tokens=110,
        blocked_keywords=("guaranteed refund", "we never share data"),
        high_stakes_intent_markers=_DEFAULT_HIGH_STAKES_MARKERS,
        force_deep_on_action=True,
        budget_degradation="fast",
    ),
    "internal_knowledge_assistant": ApplicationProfile(
        name="internal_knowledge_assistant",
        description="Internal RAG bot. Low blast radius; strongly favours FAST.",
        cost_optimal_escalation_threshold=0.74,
        fast_path_max_response_tokens=160,
        high_stakes_intent_markers=("password", "credential", "secret", "offboard"),
        force_deep_on_action=False,
        budget_degradation="fast",
    ),
    "financial_agent": ApplicationProfile(
        name="financial_agent",
        description="Agent that can move money. A missed error is very costly.",
        cost_optimal_escalation_threshold=0.28,
        fast_path_max_response_tokens=90,
        blocked_keywords=("guaranteed return", "risk-free", "insider"),
        high_stakes_intent_markers=_DEFAULT_HIGH_STAKES_MARKERS
        + ("trade", "portfolio", "balance", "invest", "loan", "credit limit"),
        force_deep_on_action=True,
        budget_degradation="deep",
    ),
    "realtime_voice_agent": ApplicationProfile(
        name="realtime_voice_agent",
        description="Sub-500 ms SLA. Uses speculative cascading; degrades closed.",
        cost_optimal_escalation_threshold=0.40,
        fast_path_max_response_tokens=70,
        high_stakes_intent_markers=_DEFAULT_HIGH_STAKES_MARKERS,
        force_deep_on_action=True,
        budget_degradation="deep",
    ),
    "default": ApplicationProfile(
        name="default",
        description="Fallback profile for unregistered applications.",
        cost_optimal_escalation_threshold=0.50,
        high_stakes_intent_markers=_DEFAULT_HIGH_STAKES_MARKERS,
        force_deep_on_action=True,
        budget_degradation="deep",
    ),
}


def resolve_profile(application: str | ApplicationProfile) -> ApplicationProfile:
    if isinstance(application, ApplicationProfile):
        return application
    return DEFAULT_PROFILES.get(application, DEFAULT_PROFILES["default"])


# --------------------------------------------------------------------------- #
# Tier 0 — deterministic fast-path evaluator
# --------------------------------------------------------------------------- #
_PII_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d[\s\-]?){9,14}\d(?!\d)")),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ \-]?){13,16}(?!\d)")),
    ("GOV_ID", re.compile(r"\b[A-Z]{2}\d{6,10}\b")),
    ("ACCOUNT_ID", re.compile(r"\b(?:ACC|ACCT|CUST|EMP)[\-_]?\d{4,}\b", re.I)),
)
_NUMBER_TOKEN = re.compile(r"(?<!\w)(?:[₹$€£]\s?)?\d[\d,]*(?:\.\d+)?%?")
_URL_OR_HANDLE = re.compile(r"https?://|www\.|(?<!\w)@[A-Za-z0-9_]{2,}")
_CAP_RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class FastPathEvaluator:
    """
    Tier 0: ultra-fast, fully deterministic hard-boundary checks over the
    *outbound response* (and prompt intent). Returns ``CLEARED`` (short-circuit
    to FAST_ALLOW), ``VIOLATION`` (short-circuit to ROUTE_TO_DEEP + flag), or
    ``AMBIGUOUS`` (defer to Tier 1). Cost: a handful of pre-compiled regex scans.

    In incremental routing the caller passes an :class:`Interaction` holding
    only the newest turn / tool output, so the regex work is O(delta), not
    O(session).
    """

    def evaluate(
        self, interaction: Interaction, profile: ApplicationProfile
    ) -> FastPathResult:
        start = time.perf_counter()
        response = interaction.response
        lowered = response.lower()
        signals: list[str] = []

        for label, pattern in _PII_PATTERNS:
            if pattern.search(response):
                signals.append(f"PII_PATTERN:{label}")
        for kw in profile.blocked_keywords:
            if kw.lower() in lowered:
                signals.append(f"BLOCKED_KEYWORD:{kw}")
        if signals:
            return self._result(
                FastPathOutcome.VIOLATION, signals, 1.0, start,
                "Deterministic hard boundary crossed in the outbound response; "
                "escalating to DEEP so the policy layer can adjudicate.",
            )

        blockers: list[str] = []
        if interaction.estimated_response_tokens > profile.fast_path_max_response_tokens:
            blockers.append("RESPONSE_TOO_LONG")
        if interaction.is_action:
            blockers.append("SIDE_EFFECTING_ACTION")
        if _URL_OR_HANDLE.search(response):
            blockers.append("CONTAINS_URL_OR_HANDLE")
        # tool outputs legitimately carry proper nouns; only flag entities in
        # free-form model text.
        if not interaction.is_tool_output and _CAP_RUN.search(response):
            blockers.append("CONTAINS_NAMED_ENTITY")
        numeric = len(_NUMBER_TOKEN.findall(response))
        if numeric and not interaction.has_evidence:
            blockers.append("UNGROUNDED_NUMERIC_CLAIM")
        elif numeric > 2:
            blockers.append("MULTIPLE_NUMERIC_CLAIMS")
        prompt_lc = interaction.prompt.lower()
        markers = sorted(
            m for m in profile.high_stakes_intent_markers
            if m in prompt_lc or m in lowered
        )
        if markers:
            blockers.append("HIGH_STAKES_INTENT:" + ",".join(markers))

        if not blockers:
            return self._result(
                FastPathOutcome.CLEARED, ("DETERMINISTIC_CLEAR",), 1.0, start,
                "Short, action-free response with no entities, ungrounded "
                "numbers, links or high-stakes intent; no semantic check adds "
                "information.",
            )
        return self._result(
            FastPathOutcome.AMBIGUOUS, tuple(blockers), 0.0, start,
            "No hard rule fired but the response is non-trivial "
            f"({', '.join(blockers)}); deferring to the complexity classifier.",
        )

    @staticmethod
    def _result(outcome, signals, confidence, start, explanation) -> FastPathResult:
        return FastPathResult(
            outcome=outcome,
            signals=tuple(signals),
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            explanation=explanation,
        )


# --------------------------------------------------------------------------- #
# The cascade orchestrator
# --------------------------------------------------------------------------- #
class CascadeRouter:
    """
    Orchestrates the three-tier cascade. Stateless and thread-safe — build one
    and share it. The Tier-1 model is injected, so a fitted
    :class:`~verification.routing_models.MatrixFactorizationRouter` drops in
    without touching this class.
    """

    def __init__(
        self,
        *,
        classifier: ComplexityClassifier | None = None,
        fast_evaluator: FastPathEvaluator | None = None,
        profiles: Mapping[str, ApplicationProfile] | None = None,
        total_latency_budget_ms: float = TOTAL_GUARDRAIL_BUDGET_MS,
    ) -> None:
        self._classifier = classifier or HeuristicComplexityClassifier()
        self._fast = fast_evaluator or FastPathEvaluator()
        self._profiles = dict(profiles) if profiles is not None else dict(DEFAULT_PROFILES)
        self._budget = total_latency_budget_ms

    @property
    def classifier(self) -> ComplexityClassifier:
        return self._classifier

    # ------------------------------------------------------------------ #
    # A -> D single-shot cascade
    # ------------------------------------------------------------------ #
    def route(
        self,
        interaction: Interaction,
        application: str | ApplicationProfile | None = None,
        *,
        upstream_latency_ms: float = 0.0,
        session_complexity: float = 0.0,
    ) -> RoutingDecision:
        profile = self._resolve(application or interaction.application)
        wall_start = time.perf_counter()

        # Step A: deterministic fast path
        fast = self._fast.evaluate(interaction, profile)
        threshold = profile.effective_threshold(interaction, session_complexity)

        if fast.outcome is FastPathOutcome.CLEARED:
            return self._finish(
                interaction, profile, RouteVerdict.FAST_ALLOW,
                CascadeTier.FAST_PATH_DETERMINISTIC, "DETERMINISTIC_CLEAR",
                fast, None, threshold, wall_start, upstream_latency_ms,
                extra=f"Tier 0 cleared in {fast.latency_ms:.2f} ms. {fast.explanation}",
            )
        if fast.outcome is FastPathOutcome.VIOLATION:
            return self._finish(
                interaction, profile, RouteVerdict.ROUTE_TO_DEEP,
                CascadeTier.FAST_PATH_DETERMINISTIC, "DETERMINISTIC_HARD_BOUNDARY",
                fast, None, threshold, wall_start, upstream_latency_ms,
                extra=fast.explanation,
            )

        # budget guard: never let the guardrail blow the SLA
        spent = upstream_latency_ms + fast.latency_ms
        if spent > self._budget - CLASSIFIER_BUDGET_MS:
            return self._budget_degrade(
                interaction, profile, fast, threshold, wall_start,
                upstream_latency_ms, spent,
            )

        # Step B: learned complexity classifier
        prediction = self._classify(interaction, wall_start)

        # Step C + D: cost-optimal escalation against the dynamic threshold
        if prediction.score >= threshold:
            verdict, reason = RouteVerdict.ROUTE_TO_DEEP, "COMPLEXITY_ABOVE_THRESHOLD"
            extra = (
                f"complexity {prediction.score:.3f} >= {profile.name} threshold "
                f"{threshold:.3f} -> DEEP verification is cost-justified."
            )
        else:
            verdict, reason = RouteVerdict.FAST_ALLOW, "COMPLEXITY_BELOW_THRESHOLD"
            extra = (
                f"complexity {prediction.score:.3f} < {profile.name} threshold "
                f"{threshold:.3f} -> the shallow verification verdict is trusted."
            )
        return self._finish(
            interaction, profile, verdict, CascadeTier.COMPLEXITY_CLASSIFIER,
            reason, fast, prediction, threshold, wall_start, upstream_latency_ms,
            extra=extra,
        )

    # ------------------------------------------------------------------ #
    # Contextual-snapshot routing for multi-turn agentic sessions
    # ------------------------------------------------------------------ #
    def route_incremental(
        self,
        turn: Interaction,
        prior_state: RoutingState | None = None,
        application: str | ApplicationProfile | None = None,
    ) -> tuple[RoutingDecision, RoutingState]:
        """
        Route the newest turn / tool output only, carrying a bounded
        :class:`RoutingState` forward. Avoids the "stochastic tax" of re-running
        regex + complexity features over the whole accumulated conversation on
        every agent step.

        Fast-stable shortcut: if the prior session was entirely Tier-0 clear,
        this turn is Tier-0 clear, and the carried session complexity is still
        below threshold, return FAST_ALLOW without invoking the classifier.
        """
        profile = self._resolve(application or turn.application)
        state = prior_state or RoutingState()
        wall_start = time.perf_counter()

        fast = self._fast.evaluate(turn, profile)
        threshold = profile.effective_threshold(turn, state.session_complexity)

        if fast.outcome is FastPathOutcome.VIOLATION:
            decision = self._finish(
                turn, profile, RouteVerdict.ROUTE_TO_DEEP,
                CascadeTier.FAST_PATH_DETERMINISTIC, "DETERMINISTIC_HARD_BOUNDARY",
                fast, None, threshold, wall_start, 0.0, extra=fast.explanation,
            )
            new_state = state.advanced(
                cleared=False, turn_complexity=1.0, ambiguous=False,
                high_stakes=True, new_signals=fast.signals,
            )
            return decision, new_state

        if (
            fast.outcome is FastPathOutcome.CLEARED
            and state.every_turn_cleared
            and state.session_complexity < threshold
        ):
            decision = self._finish(
                turn, profile, RouteVerdict.FAST_ALLOW,
                CascadeTier.FAST_PATH_DETERMINISTIC, "INCREMENTAL_STABLE",
                fast, None, threshold, wall_start, 0.0,
                extra=(
                    f"turn {turn.turn_index}: conversation has been Tier-0 clear "
                    f"for {state.turn_count} turns and session complexity "
                    f"{state.session_complexity:.2f} < {threshold:.2f} — no "
                    "classifier call needed."
                ),
            )
            new_state = state.advanced(
                cleared=True, turn_complexity=0.0, ambiguous=False,
                high_stakes=False, new_signals=(),
            )
            return decision, new_state

        # ambiguous, or clear-but-session-elevated: score the delta and merge
        prediction = self._classify(turn, wall_start)
        merged_score = max(
            prediction.score,
            _clamp01(
                _SESSION_DECAY * state.session_complexity
                + _SESSION_CURRENT_WEIGHT * prediction.score
            ),
        )
        escalate = merged_score >= threshold
        verdict = RouteVerdict.ROUTE_TO_DEEP if escalate else RouteVerdict.FAST_ALLOW
        reason = "INCREMENTAL_ABOVE_THRESHOLD" if escalate else "INCREMENTAL_BELOW_THRESHOLD"
        decision = self._finish(
            turn, profile, verdict, CascadeTier.COMPLEXITY_CLASSIFIER, reason,
            fast, replace(prediction, score=round(merged_score, 6)),
            threshold, wall_start, 0.0,
            extra=(
                f"turn {turn.turn_index}: delta complexity {prediction.score:.3f}, "
                f"session-merged {merged_score:.3f} vs threshold {threshold:.3f}."
            ),
        )
        new_state = state.advanced(
            cleared=fast.outcome is FastPathOutcome.CLEARED,
            turn_complexity=prediction.score,
            ambiguous=fast.outcome is FastPathOutcome.AMBIGUOUS,
            high_stakes=any(s.startswith("HIGH_STAKES") for s in fast.signals),
            new_signals=fast.signals,
        )
        return decision, new_state

    # ------------------------------------------------------------------ #
    # Speculative cascading for latency-critical apps
    # ------------------------------------------------------------------ #
    def route_speculative(
        self,
        interaction: Interaction,
        deep_verifier: Callable[[Interaction], object],
        application: str | ApplicationProfile | None = None,
        *,
        system_load: float = 0.0,
        load_ceiling: float = 0.75,
        max_workers: int = 2,
    ) -> SpeculativeResult:
        """
        Run Tier 1 and the caller-supplied DEEP verifier *in parallel*, then keep
        the winner:

        * classifier says FAST_ALLOW  -> cancel/ignore the DEEP future (it was
          speculative), return FAST_ALLOW, ``deep_discarded=True``;
        * classifier says ROUTE_TO_DEEP -> the DEEP pass is already running, so
          its result is (mostly) latency-free — return it attached.

        Speculation is skipped when ``system_load >= load_ceiling`` (can't afford
        to burn a DEEP slot on a maybe) — the router falls back to sequential
        :meth:`route`. Real wall-clock benefit requires ``deep_verifier`` to
        release the GIL (I/O or a C-extension model); a pure-Python verifier will
        serialise.
        """
        profile = self._resolve(application or interaction.application)
        wall_start = time.perf_counter()

        fast = self._fast.evaluate(interaction, profile)
        threshold = profile.effective_threshold(interaction)

        # Tier 0 is decisive either way -> no speculation needed.
        if fast.outcome is not FastPathOutcome.AMBIGUOUS or system_load >= load_ceiling:
            decision = self.route(interaction, profile)
            deep_res = (
                deep_verifier(interaction) if decision.routes_to_deep else None
            )
            return SpeculativeResult(
                routing_decision=decision,
                deep_result=deep_res,
                deep_started=decision.routes_to_deep,
                deep_discarded=False,
                wall_latency_ms=(time.perf_counter() - wall_start) * 1000.0,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            deep_future = pool.submit(deep_verifier, interaction)
            pred_future = pool.submit(self._classifier.predict_complexity, interaction)
            prediction = pred_future.result()

            escalate = prediction.score >= threshold
            if escalate:
                deep_res, discarded = deep_future.result(), False
                verdict, reason = RouteVerdict.ROUTE_TO_DEEP, "SPECULATIVE_KEPT_DEEP"
            else:
                deep_future.cancel()
                deep_res, discarded = None, True
                verdict, reason = RouteVerdict.FAST_ALLOW, "SPECULATIVE_DISCARDED_DEEP"

        decision = self._finish(
            interaction, profile, verdict, CascadeTier.COMPLEXITY_CLASSIFIER,
            reason, fast, prediction, threshold, wall_start, 0.0,
            extra=(
                f"speculative: complexity {prediction.score:.3f} vs threshold "
                f"{threshold:.3f}; DEEP {'kept' if escalate else 'discarded'}."
            ),
        )
        return SpeculativeResult(
            routing_decision=decision,
            deep_result=deep_res,
            deep_started=True,
            deep_discarded=discarded,
            wall_latency_ms=(time.perf_counter() - wall_start) * 1000.0,
        )

    # ------------------------------------------------------------------ #
    # Offline calibration helper
    # ------------------------------------------------------------------ #
    def calibrate_profile(
        self,
        application: str,
        samples: Sequence[RoutingSample],
        *,
        deep_verification_cost: float,
        missed_risk_penalty: float,
        max_missed_risk_rate: float = 0.02,
    ) -> tuple["CascadeRouter", ThresholdCalibration]:
        """
        Fit ``application``'s escalation threshold on labelled routing samples
        and return a new router using it. Samples whose ``complexity_score`` is
        unset are scored with the current classifier first.
        """
        scored = [
            s
            if s.complexity_score
            else replace(
                s,
                complexity_score=self._classifier.predict_from_features(s.features),
            )
            for s in samples
        ]
        calib = select_cost_optimal_threshold(
            scored,
            deep_verification_cost=deep_verification_cost,
            missed_risk_penalty=missed_risk_penalty,
            max_missed_risk_rate=max_missed_risk_rate,
        )
        base = self._resolve(application)
        profiles = dict(self._profiles)
        profiles[application] = replace(
            base, cost_optimal_escalation_threshold=calib.threshold
        )
        router = CascadeRouter(
            classifier=self._classifier,
            fast_evaluator=self._fast,
            profiles=profiles,
            total_latency_budget_ms=self._budget,
        )
        return router, calib

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _resolve(self, application: str | ApplicationProfile) -> ApplicationProfile:
        if isinstance(application, ApplicationProfile):
            return application
        return self._profiles.get(
            application, self._profiles.get("default", DEFAULT_PROFILES["default"])
        )

    def _classify(
        self, interaction: Interaction, wall_start: float
    ) -> ComplexityPrediction:
        return self._classifier.predict_complexity(interaction)

    def _budget_degrade(
        self, interaction, profile, fast, threshold, wall_start,
        upstream_latency_ms, spent,
    ) -> RoutingDecision:
        if profile.budget_degradation == "deep":
            verdict, reason = RouteVerdict.ROUTE_TO_DEEP, "BUDGET_GUARD_DEGRADE_CLOSED"
            extra = (
                f"{spent:.1f} ms of a {self._budget:.0f} ms budget already "
                "spent; this profile degrades closed — escalating to DEEP."
            )
        else:
            verdict, reason = RouteVerdict.FAST_ALLOW, "BUDGET_GUARD_DEGRADE_OPEN"
            extra = (
                f"{spent:.1f} ms of a {self._budget:.0f} ms budget already "
                "spent; degrading open to FAST_ALLOW and flagging for "
                "asynchronous re-verification."
            )
        return self._finish(
            interaction, profile, verdict, CascadeTier.COMPLEXITY_CLASSIFIER,
            reason, fast, None, threshold, wall_start, upstream_latency_ms,
            extra=extra, within_budget=False,
        )

    def _finish(
        self,
        interaction: Interaction,
        profile: ApplicationProfile,
        verdict: RouteVerdict,
        tier: CascadeTier,
        reason_code: str,
        fast: FastPathResult,
        prediction: ComplexityPrediction | None,
        threshold: float,
        wall_start: float,
        upstream_latency_ms: float,
        *,
        extra: str = "",
        within_budget: bool | None = None,
    ) -> RoutingDecision:
        classifier_ms = 0.0  # measured separately below if a prediction was made
        total_ms = (time.perf_counter() - wall_start) * 1000.0
        if within_budget is None:
            within_budget = (upstream_latency_ms + total_ms) <= self._budget

        det_signals = fast.signals
        if reason_code == "DETERMINISTIC_HARD_BOUNDARY":
            det_signals = tuple(
                s for s in fast.signals if s.startswith(("PII_", "BLOCKED_"))
            )

        return RoutingDecision(
            interaction_id=interaction.interaction_id,
            application=profile.name,
            verdict=verdict,
            tier_reached=tier,
            reason_code=reason_code,
            explanation=f"[{tier.name}] {verdict.value}: {extra}".strip(),
            predicted_complexity_score=(
                None if prediction is None else prediction.score
            ),
            applied_threshold=round(threshold, 4),
            deterministic_signals=det_signals,
            fast_path_latency_ms=round(fast.latency_ms, 4),
            classifier_latency_ms=round(
                max(0.0, total_ms - fast.latency_ms) if prediction is not None else 0.0,
                4,
            ),
            total_latency_ms=round(total_ms, 4),
            within_latency_budget=within_budget,
            turn_index=interaction.turn_index,
        )


# --------------------------------------------------------------------------- #
# Manual smoke — `python -m verification.cascade_router`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from verification.routing_models import build_default_mf_router

    router = CascadeRouter(classifier=build_default_mf_router())

    demos = [
        Interaction("D1", "customer_support", "What are your support hours?",
                    "We are open Monday to Friday, 9am to 5pm.",
                    "Support hours: Mon-Fri 09:00-17:00."),
        Interaction("D2", "customer_support", "Confirm details for ACC-227763.",
                    "Karan Mehta, karan.mehta@example-test.com.", "",
                    action_type="external_communication"),
        Interaction("D3", "financial_agent", "Move my balance to the new fund?",
                    "The new fund has returned around 14% and is probably a good "
                    "choice for most investors.",
                    "Fund X 1-year return: 6.2%.", action_type="advice"),
        Interaction("D4", "internal_knowledge_assistant", "Where is the Q3 doc?",
                    "In the Product drive under Planning / 2026 / Q3.",
                    "Product drive > Planning > 2026 > Q3 > roadmap.pdf"),
    ]
    for d in demos:
        r = router.route(d)
        print(f"\n{r.interaction_id} [{r.application}]  {r.verdict.value} "
              f"(tier {r.tier_reached.name}, {r.reason_code})")
        print(f"  score/thr {r.predicted_complexity_score}/{r.applied_threshold}  "
              f"bypass_semantics={r.bypass_semantics}  "
              f"{r.total_latency_ms:.3f} ms")

    print("\n--- incremental (agentic) session ---")
    state: RoutingState | None = None
    turns = [
        Interaction("S1", "customer_support", "hi", "Hello! How can I help?",
                    "greeting", turn_index=0),
        Interaction("S2", "customer_support", "", "Tool: order #4471 status = SHIPPED",
                    "order 4471 SHIPPED", turn_index=1, is_tool_output=True),
        Interaction("S3", "customer_support", "when will it arrive",
                    "It is probably arriving within 2 to 5 business days, though "
                    "delays around holidays are possible.", "", turn_index=2),
    ]
    for t in turns:
        d, state = router.route_incremental(t, state)
        print(f"  turn {t.turn_index}: {d.verdict.value:13s} {d.reason_code:28s} "
              f"session_cx={state.session_complexity:.2f}")
