"""
Cascade Router promotion — synthesise shadow telemetry and train the
MatrixFactorizationRouter to replicate the authoritative router's decisions.
====================================================================

This is step 2 of the shadow -> calibrate -> promote path for the Tiered
Cascade Router. It proves the training loop end to end, fully offline and
deterministically:

    1. build a labelled dataset  — the A-H demo scenarios plus a sample of the
       synthetic production traffic;
    2. collect ground truth      — route every interaction through the real
       ``VerificationRouter`` (which accounts for consequence, criticality,
       evidence quality and detector disagreement) and record its FAST/DEEP
       call as the label;
    3. train                     — ``fit_matrix_factorization_embeddings`` over
       ``(routing_document, label)`` pairs with the stdlib
       ``HashingEmbeddingBackend`` — no PyTorch, no model download;
    4. evaluate                  — score a held-out split with the trained
       router and check its ``predict_complexity`` scores now clear the
       escalation threshold for exactly the interactions the authoritative
       router sends DEEP.

Why a *routing document*, not the bare prompt
--------------------------------------------
Scenarios A (clean), B (contradiction) and F (cost anomaly) share the prompt
"What is the refund policy?" but route FAST / DEEP / DEEP respectively — the
routing signal lives in the *response* and the *action / cost metadata*, not
the user's question. So the text embedded for routing is a compact document:
``[prompt] ... [response] ... [meta] action/amount/tools/retries [flags] ...``.
That mirrors how a production routing model is fed (prompt + candidate
response + request metadata), and it is what the cascade would pass as the
router's input text once promoted.

Run:  ``python -m scripts.train_mf_router``
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from common.timing import Stopwatch
from data.generator import generate_interactions
from data.schemas import Interaction
from settings import load_settings
from tests import scenarios
from verification.cascade_router import Interaction as CascadeInteraction
from verification.routing_models import (
    EmbeddingRoutingSample,
    HashingEmbeddingBackend,
    MatrixFactorizationRouter,
    RoutingSample,
    fit_matrix_factorization_embeddings,
    select_cost_optimal_threshold,
)
from verification.router import VerificationRouter
from verification.schemas import VerificationPath

_TS = datetime(2026, 8, 21, 12, 0, 0)
_OPERATING_THRESHOLD = 0.50   # the cascade's default cost-optimal escalation point


# --------------------------------------------------------------------------- #
# 1 + 2 — labelled dataset from the authoritative router
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LabelledRow:
    interaction_id: str
    routing_document: str
    router_deep: bool          # ground truth: did VerificationRouter route DEEP?
    scenario: str | None       # set for the A-H demo scenarios


def build_routing_document(it: Interaction) -> str:
    """Compact text the routing model embeds — prompt + response + metadata."""
    amount = float(getattr(it, "action_amount_inr", 0.0) or 0.0)
    tool_calls = int(getattr(it, "tool_calls", 0) or 0)
    retries = int(getattr(it, "retry_count", 0) or 0)
    action = getattr(it.action_type, "value", str(it.action_type))

    flags: list[str] = []
    if amount >= 50_000:
        flags.append("high_value_action")
    if tool_calls >= 5 or retries >= 3:
        flags.append("operational_anomaly")
    if action not in ("information", "informational", "none", ""):
        flags.append("side_effecting_action")

    return (
        f"[prompt] {it.prompt} "
        f"[response] {it.response or ''} "
        f"[meta] action={action} amount={amount:.0f} tools={tool_calls} retries={retries} "
        f"[flags] {' '.join(flags) or 'none'}"
    )


def collect_labelled_dataset(
    *, n_synth: int = 80, seed: int = 11, config: dict | None = None
) -> list[LabelledRow]:
    """
    Route the demo scenarios + a deterministic sample of synthetic production
    traffic through the real ``VerificationRouter`` and label each by its
    FAST/DEEP call. This is the "shadow telemetry" the promotion needs.
    """
    config = config or load_settings()
    router = VerificationRouter(config)

    catalogue: list[tuple[str, str | None, Interaction]] = [
        (f"SCEN-{name}", name, factory())
        for name, factory in scenarios.ALL_SINGLE_TURN.items()
    ]
    synthetic = generate_interactions(config, random.Random(seed))[:n_synth]
    catalogue += [(it.interaction_id, None, it) for it in synthetic]

    rows: list[LabelledRow] = []
    for iid, scenario, interaction in catalogue:
        _, _, _, report = router.route(interaction, Stopwatch(), {})
        rows.append(
            LabelledRow(
                interaction_id=iid,
                routing_document=build_routing_document(interaction),
                router_deep=report.verification_path is VerificationPath.DEEP,
                scenario=scenario,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# 3 + 4 — train and evaluate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Split:
    train: list[LabelledRow]
    holdout: list[LabelledRow]


def split_dataset(rows: list[LabelledRow], *, holdout_every: int = 4) -> Split:
    """Deterministic interleaved split — every Nth row is held out."""
    train = [r for i, r in enumerate(rows) if i % holdout_every != 0]
    holdout = [r for i, r in enumerate(rows) if i % holdout_every == 0]
    return Split(train=train, holdout=holdout)


@dataclass(frozen=True, slots=True)
class EvalResult:
    n: int
    accuracy: float
    deep_recall: float
    deep_precision: float
    always_deep_accuracy: float   # majority-class baseline


def _evaluate(
    router: MatrixFactorizationRouter, rows: list[LabelledRow], threshold: float
) -> EvalResult:
    tp = fp = tn = fn = 0
    deep_true = 0
    for r in rows:
        deep_true += r.router_deep
        score = router.predict_complexity(
            CascadeInteraction("eval", "customer_support", r.routing_document, "", "")
        ).score
        predicted_deep = score >= threshold
        tp += predicted_deep and r.router_deep
        fp += predicted_deep and not r.router_deep
        tn += (not predicted_deep) and (not r.router_deep)
        fn += (not predicted_deep) and r.router_deep
    n = len(rows)
    return EvalResult(
        n=n,
        accuracy=(tp + tn) / n,
        deep_recall=tp / (tp + fn) if (tp + fn) else 1.0,
        deep_precision=tp / (tp + fp) if (tp + fp) else 1.0,
        always_deep_accuracy=deep_true / n,
    )


@dataclass(frozen=True, slots=True)
class TrainingReport:
    dataset_size: int
    deep_fraction: float
    train: EvalResult
    holdout: EvalResult
    calibrated_threshold: float
    operating_threshold: float
    scenario_scores: dict[str, tuple[str, float, bool]]   # name -> (auth, score, aligned)
    router: MatrixFactorizationRouter


def train_and_evaluate(
    rows: list[LabelledRow],
    *,
    holdout_every: int = 4,
    embedding_dim: int = 64,
    latent_dim: int = 8,
    epochs: int = 300,
    learning_rate: float = 0.08,
    config: dict | None = None,
) -> TrainingReport:
    split = split_dataset(rows, holdout_every=holdout_every)
    embedder = HashingEmbeddingBackend(dim=embedding_dim)

    params = fit_matrix_factorization_embeddings(
        [
            EmbeddingRoutingSample(
                prompt=r.routing_document, deep_would_change_decision=r.router_deep
            )
            for r in split.train
        ],
        embedder,
        latent_dim=latent_dim,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    router = MatrixFactorizationRouter(params, embedder=embedder)

    # calibrate the escalation threshold on the training scores (safety-first)
    train_scores = [
        RoutingSample(
            deep_would_change_decision=r.router_deep,
            complexity_score=router.predict_complexity(
                CascadeInteraction("cal", "customer_support", r.routing_document, "", "")
            ).score,
        )
        for r in split.train
    ]
    calib = select_cost_optimal_threshold(
        train_scores,
        deep_verification_cost=1.0,
        missed_risk_penalty=1.5,
        max_missed_risk_rate=0.15,
    )

    config = config or load_settings()
    auth_router = VerificationRouter(config)
    scenario_scores: dict[str, tuple[str, float, bool]] = {}
    for name, factory in scenarios.ALL_SINGLE_TURN.items():
        interaction = factory()
        _, _, _, report = auth_router.route(interaction, Stopwatch(), {})
        auth_deep = report.verification_path is VerificationPath.DEEP
        score = router.predict_complexity(
            CascadeInteraction(
                "scen", "customer_support",
                build_routing_document(interaction), "", "",
            )
        ).score
        aligned = (score >= _OPERATING_THRESHOLD) == auth_deep
        scenario_scores[name] = (report.verification_path.value, round(score, 4), aligned)

    deep = sum(r.router_deep for r in rows)
    return TrainingReport(
        dataset_size=len(rows),
        deep_fraction=deep / len(rows),
        train=_evaluate(router, split.train, _OPERATING_THRESHOLD),
        holdout=_evaluate(router, split.holdout, _OPERATING_THRESHOLD),
        calibrated_threshold=calib.threshold,
        operating_threshold=_OPERATING_THRESHOLD,
        scenario_scores=scenario_scores,
        router=router,
    )


# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - operator entry point
    print("Collecting shadow telemetry from the authoritative VerificationRouter ...")
    rows = collect_labelled_dataset(n_synth=80, seed=11)
    report = train_and_evaluate(rows)

    print(
        f"\nDataset: {report.dataset_size} interactions "
        f"({report.deep_fraction:.0%} routed DEEP by the authoritative router)\n"
    )
    for name, res in (("train", report.train), ("holdout", report.holdout)):
        print(
            f"  {name:8s}  accuracy {res.accuracy:.3f}   "
            f"DEEP recall {res.deep_recall:.3f}   DEEP precision {res.deep_precision:.3f}   "
            f"(always-DEEP baseline {res.always_deep_accuracy:.3f})"
        )
    print(
        f"\n  operating threshold {report.operating_threshold:.2f}   "
        f"cost-optimal (calibrated on train) {report.calibrated_threshold:.2f}\n"
    )
    print("  per-scenario (trained MF router vs authoritative):")
    for name, (auth, score, aligned) in report.scenario_scores.items():
        print(
            f"    {name:22s} authoritative={auth:5s}  mf_score={score:.3f}  "
            f"{'ALIGNED' if aligned else 'MISS'}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
