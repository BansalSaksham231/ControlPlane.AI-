# 🛡️ ControlPlane.ai

**A real-time, risk-adaptive oversight layer for enterprise AI.**
*Accenture Innovation Challenge 2026 — production system.*

Enterprises run many AI use cases at once — a customer chatbot, an internal knowledge
assistant, an agent that can move money — each with a different risk tolerance. Governing
them identically is wrong. ControlPlane.ai sits around a foundation model's inputs and
outputs and, for **every interaction**, answers:

1. **What could be wrong?** — performance / responsibility / cost detectors
2. **How confident are we?** — per-detector confidence
3. **What evidence supports that?** — retrieved evidence, matched spans, rule traces
4. **What happens if the AI is wrong?** — consequence engine (severity ≠ likelihood)
5. **Which application is this?** — per-application policy profiles
6. **What policy applies?** — configurable policy engine
7. **What intervention is appropriate?** — `ALLOW → ANNOTATE → VERIFY → HUMAN_REVIEW → BLOCK`
8. **Should a human intervene?** — human-review routing
9. **What happened earlier in this session?** — multi-turn risk accumulation
10. **Can the decision be audited and evaluated?** — decision trace + feedback + metrics

> **Architecture note.** Every detector is a **transparent, deterministic heuristic**
> (regex, TF-IDF, lexical NLI, lexicons). This is a demonstrable architecture, **not** a
> production AI-safety guarantee. All data is synthetic; all metrics are measured on
> that synthetic data. Prototype cost rates are illustrative, not real billing.

---

## Quick start (2 commands)

```bash
pip install -r requirements.txt
python run_app.py                 # opens the web UI at http://localhost:8501
```

Then open **http://localhost:8501** in your browser (the launcher also tries to open it
for you). `run_app.py` generates the synthetic data on first run, then starts the UI.

> If you start Streamlit yourself and it prints `http://0.0.0.0:8501`, that is a *bind*
> address — browse to **http://localhost:8501** instead.

Other modes:

```bash
python run_app.py --api           # FastAPI backend + Swagger docs at :8000/docs
python run_app.py --both          # API + UI together
python run_app.py --demo          # run the end-to-end demo in the terminal and exit
```

Windows PowerShell: the same commands work as-is (no shell scripts, no bash).

**Two front ends ship with the project:**

| Front end | Command | What it is |
|---|---|---|
| **Streamlit** (`streamlit_app.py`) | `python run_app.py` | The full 10-tab demo UI. Runs the decision pipeline **in-process** — no API needed. This is the primary judge demo surface. |
| **Next.js** (`frontend/`) | `cd frontend && npm install && npm run dev` | A production-shaped React/TypeScript Enterprise Command Center + incident-investigation UI. Talks to the FastAPI backend over HTTP (`NEXT_PUBLIC_API_URL`). |

**Full stack in one command** (needs Docker): `docker compose up --build` → Next.js `:3000`, API `:8000`, Streamlit `:8501`, PostgreSQL.

---

## 1. Round 2 problem

An enterprise runs several AI use cases at once — a customer chatbot, an internal
knowledge assistant, a decision-support system — each with different risk tolerance,
latency budget, data sensitivity and failure consequences. Risks overlap (a fabricated
detail about a person is *simultaneously* a hallucination, a privacy risk and
potentially consequential). There is rarely real-time ground truth, so the system must
distinguish **supported / contradicted / unverified**. Over-flagging causes alert
fatigue; under-flagging creates liability. Multi-turn conversations compound risk, and
agents increasingly take actions rather than just generate text.

ControlPlane.ai is the control layer for exactly that situation.

---

## 2. Architecture

```
                         AI Interaction
                              │
                       ControlPlane API  /  Web UI
                              │
        ┌─────────────────────┼─────────────────────┐   (independent —
        ▼                     ▼                     ▼    parallelizable
  Performance            Responsibility            Cost   in production)
   Detector                Detector              Detector
   evidence-weighted     PII · toxicity ·      efficiency + typed
   RISK + CONFIDENCE     bias signal            anomalies
        └─────────────────────┼─────────────────────┘
                              ▼
                  Claim / Action Criticality    (does it matter if wrong?
                              │                  amplifies performance risk)
                              ▼
                     Risk Fusion Engine     (weighted + severity pull; RISK + CONFIDENCE)
                              │
                              ▼
                     Consequence Engine     (5-factor severity from action metadata)
                              │
                              ▼
                 Session Risk Accumulation  (bounded, decaying, multi-turn)
                              │
                              ▼
                       Policy Engine         (per-application profiles; confidence-aware;
                              │               structured reason codes)
                              ▼
                      Decision Engine   →   FinalDecision  +  DecisionTrace
                              │              (decision · risk · confidence ·
                              │               reason_codes · what-moved-the-decision)
              ┌───────────────┼───────────────┬──────────────────┐
              ▼               ▼               ▼                  ▼
           Audit           Feedback     Evaluation / Monitoring   Policy simulation
                                                                 + Counterfactuals
```

**Non-negotiable rule:** detectors only ever receive a production-shaped `Interaction`.
Ground-truth / evaluation fields (`ground_truth_*`, `expected_decision`,
`final_outcome`, …) live on `EvaluationCase` and are read **only** by the evaluation
code. Tests explicitly guard against leakage.

---

## 3. Risk dimensions

| Dimension | Question | Signal |
|---|---|---|
| **Performance** | Is the response grounded in the evidence? | `SUPPORTED` / `CONTRADICTED` / `PARTIALLY_SUPPORTED` / `UNVERIFIED`, grounding score, performance risk |
| **Responsibility** | Does it leak PII, or contain toxic / biased language? | `pii_risk`, `toxicity_risk`, `bias_risk`, redacted findings |
| **Cost** | Is this interaction operationally abnormal? | estimated cost, anomaly indicators (tokens, latency, retries, tool calls) |

`UNVERIFIED` ≠ `CONTRADICTED`. With no usable evidence, claims are `NO_EVIDENCE` and the
response is `UNVERIFIED` with *moderate* risk — never the risk of a confirmed
hallucination.

---

## 4. Performance Detector (`detectors/performance/`)

```
response → claim extraction → context chunking → TF-IDF retrieval
        → document ranking → lexical NLI → per-claim result
        → response aggregation → PerformanceResult
```

- `chunker.py` — claim extraction + context chunking (stdlib only)
- `embeddings.py` — `TfidfEmbeddingBackend` (scikit-learn) behind an `EmbeddingBackend` ABC
- `nli.py` — original `LexicalNLIBackend` (unchanged) **and** `CoverageNLIBackend`
  (directional token coverage + negation / numeric polarity), used by the detector
- `detector.py` — `PerformanceDetector`

**Evidence-weighted scoring (Round 2).** Each claim gets an `evidence_strength`
(retrieval similarity + NLI confidence + evidence availability) and a `claim_risk`:

| claim outcome | claim_risk |
|---|---|
| confident contradiction over strong evidence | `≈ 0.55 + 0.30·nli_conf`, scaled by retrieval strength → up to ~0.95 |
| supported | `≤ 0.08`, shrinking with support strength |
| retrieved but ambiguous (NEUTRAL) | `0.42 · (0.6 + 0.4·(1 − evidence_strength))` — more risk when evidence is weak |
| no usable evidence | `0.50` — the flat "we could not check" penalty |

The response-level `performance_risk` is a **criticality-weighted mean** of the claim
risks (a claim whose text names money / an action / an entity weighs more), with a
floor of `0.62 + 0.12·n` whenever `n` contradictions are present so "one hallucination
is enough" still holds.

**Risk ≠ confidence.** `PerformanceResult` reports `performance_risk`, `confidence`,
`uncertainty = 1 − confidence` and `evidence_quality` separately. A confident
contradiction is high-risk / high-confidence; an unverifiable claim on unrelated
context is elevated-risk / **low**-confidence.

No model downloads, no network. `method` is reported as `tfidf+lexical_nli`.

## 4a. Risk vs Confidence (throughout the pipeline)

Every layer keeps two distinct numbers:

* **Risk** — *how dangerous does this interaction look?* (`overall_risk`)
* **Confidence** — *how sure is ControlPlane that the risk assessment is right?*
  (`decision_confidence`, `fusion.confidence`)

Confidence is derived from evidence availability, retrieval strength, NLI confidence,
detector agreement and dimension spread — never set equal to risk, never inflated.

Policy consequence: a would-be **BLOCK** that rests on weak evidence
(`risk ≥ 0.60`, `confidence ≤ 0.45`, no evidence-backed hard override) is routed to
**HUMAN_REVIEW** instead — `LOW_CONFIDENCE_HIGH_RISK`. `DecisionEngine(confidence_aware=False)`
turns this reasoning off (used by the ablation study).

## 4b. Claim / Action Criticality (`criticality/`)

*"Not all hallucinations have equal consequences."* `action_criticality ∈ [0,1]` is a
transparent weighted blend of five **production-visible** factors (financial impact from
`action_amount_inr`, irreversibility / sensitivity / automation from `action_type`,
blast radius from `affected_entities`). Per-claim criticality additionally scans the
claim text for money amounts, action verbs and named entities.

The decision engine uses it to **amplify performance risk before fusion** —
`eff = risk + (1−risk)·risk·headroom·0.6`, engaging only once criticality is at least
"moderate" — and to emit reason codes (`HIGH_FINANCIAL_IMPACT`, `IRREVERSIBLE_ACTION`,
`AUTOMATED_EXTERNAL_ACTION`, `HIGH_BLAST_RADIUS`). A big consequential action always
gets at least a `VERIFY` (`HIGH_CRITICALITY_ACTION`).

## 5. Responsibility Detector (`detectors/responsibility/`)

- **PII** (`pii.py`) — regex + entity heuristics for email, phone, card (Luhn),
  government IDs, account / employee IDs, addresses, and names *when they co-occur with
  other sensitive data*. **Context-aware severity (Round 2):** an identifier exposed in
  an *outward-facing* response (`external_communication` / `account_cancellation`) is
  escalated to `CRITICAL`. Every finding carries a `severity_rationale`, keeps the raw
  span for the audit record and a redacted form for everything else; the detector emits
  a fully redacted response.
- **Toxicity** (`toxicity.py`) — small auditable category lexicons (threats, hate,
  harassment, severe profanity), each finding with category / severity / confidence /
  evidence. Detection is kept separate from the block decision.
- **Bias** (`bias.py`) — heuristic patterns reported as a **`POTENTIAL_BIAS_SIGNAL`**
  requiring human review — never as established discrimination.

The unified result also emits reason codes (`CRITICAL_PII`, `PII_EXPOSURE`, `TOXICITY`,
`POTENTIAL_BIAS_SIGNAL`, `HIGH_RESPONSIBILITY_RISK`).

## 6. Cost Detector (`detectors/cost/`)

Transparent additive estimate (prototype INR rates in config):

```
estimated_cost = input_tokens/1k·input_rate + output_tokens/1k·output_rate
               + tool_calls·tool_rate       + retries·retry_rate
```

Anomalies are flagged against `max(static baseline, empirical p80 baseline)` per
`(application, model)`; tool-call / retry counts use absolute caps.

**Efficiency & typed anomalies (Round 2).** `CostResult` adds `cost_efficiency_score`
(1.0 = on par with baseline; penalised by retry waste, tool overhead, oversized
output), `retry_inefficiency`, `cost_per_success_inr`, and named `anomaly_types`:
`TOKEN_SPIKE`, `RETRY_SPIKE`, `TOOL_LOOP`, `LATENCY_SPIKE`, `COST_PER_SUCCESS_SPIKE`.
A cost anomaly with no safety signal is still capped for operational review, never
safety-blocked (`COST_ONLY_CAP`).

## 7. Consequence Engine (`consequence/`)

Risk (likelihood of being wrong) and consequence (severity if wrong) stay **separate**.
Five factors — `financial_impact`, `reversibility`, `sensitivity`, `blast_radius`,
`action_automation` — are derived from action metadata and weighted (weights in config,
kept identical to the data generator) into `consequence_score ∈ [0,1]` with a
`severity_band` and an explanation naming the top contributors.

## 8. Risk Fusion (`fusion/`)

Documented, configurable — never a silent average:

```
weighted = Σ wᵢ·riskᵢ
max_dim  = max(riskᵢ)
pull     = severity_pull      if max_dim ≥ severity_trigger  else  severity_pull_low
blended  = (1 − pull)·weighted + pull·max_dim
if max_dim ≥ severity_floor_trigger:  blended = max(blended, severity_floor_value)
```

The `pull` term is the **conservative safety rule**: `performance = 0.95,
responsibility = 0.05, cost = 0.05` still fuses to `overall_risk ≈ 0.80`, not `0.35`.

## 9. Policy Engine (`policy/`)

Per-application profiles in `config/settings.yaml` (`customer_support`,
`internal_knowledge_assistant`, `decision_support`, `default`). A small composable rule
set — **not** a sprawl of `if risk > 0.5` conditionals:

1. **Risk bands** map `overall_risk` → base tier
2. **Consequence escalation** raises the tier when consequence is high, *or* when a
   single factor (financial impact / irreversibility) is extreme
3. **`HIGH_CRITICALITY_ACTION`** — a large consequential action always gets at least `VERIFY`
4. **Minimum-tier rules**: contradiction, unverified evidence, high responsibility risk,
   low detector confidence
5. **Hard overrides**: critical PII, severe toxicity
6. **Cost-only cap**: an operational anomaly with no safety signal is capped for
   operational review, not safety-blocked
7. **`LOW_CONFIDENCE_HIGH_RISK`** — a weak-evidence BLOCK is downgraded to HUMAN_REVIEW

Every rule emits a `RuleTraceEntry`, and the engine maps them to **structured
`reason_codes`** (`common/reason_codes.py` — `CONTRADICTED_EVIDENCE`, `CRITICAL_PII`,
`HIGH_FINANCIAL_IMPACT`, `MULTI_RISK`, `SESSION_ESCALATION`, `LOW_CONFIDENCE_HIGH_RISK`,
… each with a one-line description).

## 9a. Structured reason codes & "what moved the decision"

Every `FinalDecision` exposes `decision`, `triggered_rules` (internal rule ids),
**`reason_codes`** (canonical product-level reasons) and a generated `explanation`.
`DecisionTrace.decision_drivers` lists exactly which policy rules moved the tier and by
how much — e.g. *"HIGH_CONSEQUENCE: ANNOTATE→VERIFY"*.

## 9b. Policy simulation & counterfactuals (`simulation/`)

Both run the **real pipeline** — nothing fabricates an alternate decision.

* `simulate_policies(engine, interaction, profiles)` — the same interaction under every
  application policy profile. Same contradicted response → `customer_support` VERIFY,
  `decision_support` HUMAN_REVIEW.
* `compare_decisions(engine, interaction, {"action_amount_inr": 100})` — the same
  interaction with a few **whitelisted** production fields changed; returns the tier
  change plus which rules / reason codes stopped and started firing. Ground-truth fields
  are rejected.

Exposed as `POST /simulate-policy` and `POST /counterfactual`, and as two Web-UI tabs.

## 10. Decision Engine (`decision/`)

Orchestrates the whole pipeline once (detectors → criticality → fusion → consequence →
session → policy) and produces:

- **`FinalDecision`** — compact, API-facing (three risks, `ConsequenceFactors`,
  `overall_risk`, `decision_confidence`, `triggered_rules`, `reason_codes`, human-readable
  `explanation`, timestamp)
- **`DecisionTrace`** — full replayable audit bundle incl. `criticality`,
  `criticality_weighted_performance_risk`, `decision_drivers`; `.audit_summary()` and
  `.redacted_dump()` mask raw PII

The three detectors are independent; `DecisionEngine(parallel_detectors=True)` runs them
in a thread pool with byte-identical results (all deterministic). It is off by default
(no benefit at initial scale) and documents the production-parallel architecture.

**Why this is not just a classifier**

| performance_risk | consequence | → decision |
|---|---|---|
| 0.74 (contradiction) | 0.10 | **VERIFY** |
| 0.74 (contradiction) | 0.90 | **HUMAN_REVIEW** |
| 0.95 + responsibility 0.72 | any | **BLOCK** |

Same likelihood, different consequence → different intervention.

## 11. Session / multi-turn risk (`session/`)

```
on record:  cumulative ← decay·cumulative + (1−decay)·turn_risk
on decide:  session_risk = history_weight·cumulative + current_weight·turn_risk
            + escalation_bump  if ≥ N high-risk turns in the window
            adjusted_risk = max(turn_risk, session_risk)   # only ever raises
```

In the demo, four borderline (`VERIFY`) turns in one session escalate to
`HUMAN_REVIEW`. Everything is clamped to `max_session_risk`.

## 12. Feedback (`feedback/`)

A clean data contract — not self-training. Records system decision, optional reviewer
decision, outcome (`approved` / `modified` / `rejected`), reason, reviewer, timestamp;
optionally mirrored to JSONL. `aggregate()` returns override rate, approval rate, and a
system-vs-reviewer tier confusion matrix.

## 13. Evaluation (`evaluation/`)

- `metrics.py` — precision / recall / F1 / FPR / FNR, confusion matrix, threshold sweep
- `evaluation.py` — runs the pipeline over 150 `EvaluationCase`s (detectors still get
  only `Interaction` fields) and scores against ground truth: per-detector metrics,
  decision confusion vs the generator baseline, intervention distribution,
  **abstention / unverified rate**, coverage, mean latency, plus (Round 2)
  **risk-vs-confidence buckets** (`high_risk_high_confidence` … `low_risk_low_confidence`),
  `contradiction_precision/recall`, `criticality_band_distribution`, `reason_code_frequency`
- `ablation.py` — `performance_only` / `responsibility_only` / `cost_only` /
  `fused_only` / `fused_plus_consequence` / **`risk_only_no_confidence`** / `full_pipeline`

Representative measured numbers (`seed=42`, 150 evaluation cases):

| detector | P | R | F1 | FPR |
|---|---|---|---|---|
| performance (contradiction) | 1.00 | 0.47 | 0.64 | 0.00 |
| PII / toxicity / bias / cost | 1.00 | 1.00 | 1.00 | 0.00 |

> **Evaluation note on the 1.00 / 1.00 row.** These synthetic PII / toxicity / bias cases
> are built from a small set of fixed templates whose surface forms — email/phone/
> account-ID shapes, lexicon phrases — are exactly what the corresponding detector's
> regex / lexicon was written to match. A perfect score here demonstrates the detector
> correctly recognises the formats it targets; it is **not** evidence that it
> generalises to disguised, paraphrased, or novel-format real-world PII / toxicity /
> bias, and should not be read next to the hallucination row as if the two numbers were
> comparable evidence of robustness. The 0.47-recall hallucination row is the more
> representative signal of how these heuristic detectors behave on genuinely unseen
> input; production-grade PII / toxicity / bias detection would need the same honest,
> held-out evaluation the performance detector already gets (see Limitations).

Abstention (UNVERIFIED) rate ≈ 0.72 — the lexical detector honestly abstains rather than
guessing. ~32% of decisions land in the *high-risk / low-confidence* bucket — exactly
where the confidence-aware policy matters.

Ablation: every flat single-score mode has high precision but low recall (best flat
F1 ≈ 0.82, recall ≈ 0.69 — it misses risky cases it cannot see). The full pipeline
reaches **F1 ≈ 0.88, recall ≈ 0.98** while escalating **0%** of clean traffic to a human
and assigning a proportionate tier rather than a binary flag. Turning confidence-
awareness off routes ~10 more cases to BLOCK/HUMAN_REVIEW that would otherwise be sent
to a human for verification.

> Evaluation note: on this synthetic evaluation set the lexical NLI produces
> *confident* contradictions and *low-risk* unverifieds, so `risk_only_no_confidence`
> and `full_pipeline` differ on only a handful of cases. The `LOW_CONFIDENCE_HIGH_RISK`
> rule is unit-tested directly and would matter more with a calibrated NLI backend.

## 14. API (`api/`)

Thin FastAPI layer over `ControlPlaneService` (all logic stays in the engines).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + counters |
| `POST` | `/check` | one interaction → `FinalDecision` (`include_trace=true` for a redacted trace) |
| `GET` | `/session/{id}` | current session risk state |
| `DELETE` | `/session/{id}` | reset a session |
| `GET` | `/audit/{interaction_id}` | redacted audit summary of a past decision |
| `POST` | `/feedback` | record a human review outcome |
| `GET` | `/feedback/summary` | aggregated feedback |
| `POST` | `/simulate-policy` | one interaction under several application profiles |
| `POST` | `/counterfactual` | re-run with whitelisted fields changed; diff the decision |

Swagger UI at `/docs` with worked `/check` examples.

## 14a. Incident Investigation & Governance (`investigation/`)

Turns a Command-Center incident into a full, auditable investigation and records the
**human governance** response.

```
Command Center incident
  → InvestigationService.investigate(interaction_id)
       = stored DecisionTrace
       + build_replay(trace)        (Incident Replay — reconstruction)
       + build_explanation(trace)   (Explainability — presentation)
       + governance history
  → IncidentInvestigation
```

- **Read-only reconstruction.** `investigate()` never re-runs a detector, the decision
  engine, fusion, policy, consequence, criticality or verification — it reads the stored
  trace. (`tests/test_investigation.py` monkeypatches all nine to raise and the
  investigation still succeeds.)
- **The automated ControlPlane decision is immutable.** A governance action —
  `ACKNOWLEDGE`, `APPROVE_DECISION`, `MODIFY_DECISION`, `REJECT_DECISION`, `ESCALATE`,
  `CLOSE` — records the human outcome. `MODIFY_DECISION` stores the reviewer's proposed
  tier *alongside* the unchanged `original_decision`; `trace.final_decision.decision` is
  never overwritten.
- **Workflow state** (`OPEN → ACKNOWLEDGED → REVIEWED → ESCALATED → CLOSED`) is
  governance state, not risk state — a `BLOCK` that is `CLOSED` is still a `BLOCK`.
- **Counterfactuals are explicitly-labelled SIMULATIONS** run by the existing
  `simulation.engine` (change a production-visible field, see the decision/rules move).
  They never modify the stored decision or production policy.
- **Reviewer feedback is a governance signal, not ground truth** — it does not change
  evaluation metrics or detector thresholds.
- **PII-safe at every stage** — only structured fields + the already-redacted
  replay/explanation text; no `matched_text`, no raw response.
- **API:** `GET /investigation/{id}`, `GET /investigation/{id}/history`,
  `POST /investigation/{id}/action`, `POST /investigation/{id}/counterfactual`.
- **Demo:** `python demo.py --investigation`.

## 14b. Governance Intelligence & Closed-Loop Monitoring (`governance/`)

Completes the PS loop — **traffic → detection → decision → monitoring → incident →
human review → governance signal → analysis → calibration RECOMMENDATION**. A
**read-only analytics layer** over data that already exists (`DecisionTrace`,
`GovernanceAction`, `FeedbackRecord`, `calibration.select` output).

It answers: *are we over- or under-flagging? which applications generate the most
risk? which detectors drive incidents? are reviewers overriding us? which policy
rules cause the most intervention? is confidence deteriorating? is FAST/DEEP routing
changing? should a policy or threshold be reviewed?*

- **`analytics.py`** — traffic / decision distribution / risk / confidence / FAST-DEEP /
  detector contribution / policy-rule behaviour / human-governance status / **reviewer
  disagreement** (automated `original_decision` kept strictly separate from
  `reviewer_decision`) + per-application comparison.
- **`signals.py`** — a unified `GovernanceSignal` with source provenance
  (`reviewer_override` / `feedback_modified` / `feedback_rejected` / `feedback_approved`).
- **`insights.py`** — deterministic insights: `HIGH_OVERRIDE_RATE`,
  `HIGH_HUMAN_REVIEW_RATE`, `LOW_CONFIDENCE_PATTERN`, `RULE_DOMINANCE`,
  `DEEP_ROUTING_CONCENTRATION`, `RISK_CONCENTRATION` — each with severity, supporting
  metrics, a recommended human action, and example incident ids.
- **`trends.py`** — sequence-based first-half/second-half comparison; cautious labels
  `TREND` / `SIGNAL` / `POTENTIAL_DRIFT`.
- **`recommendations.py`** — the calibration bridge. Consumes analytics + (optionally)
  a real `calibration.sweep` + `calibration.select` run. Disposition is always
  `RECOMMENDED_FOR_EVALUATION` / `REVIEW_REQUIRED` — **never `APPLIED`**. It never
  writes `config/settings.yaml`.

Guarantees (tests): never imports the evaluation package, never reads a ground-truth
label, never re-runs a detector / decision engine / policy, never writes production
config, never exposes raw PII, and always names reviewer feedback as a *governance
signal, not ground truth*.

- **API:** `GET /governance/overview`, `/governance/applications`,
  `/governance/applications/{application}`, `/governance/insights`, `/governance/signals`,
  `/governance/trends`, `/governance/recommendations`.
- **Demo:** `python demo.py --governance`.

## 14c. Closed-Loop Adaptive Guardrails & Incident Intelligence (`incident/`, `adaptive/`)

Moves ControlPlane from *detect → decide → explain → monitor* to
*… → learn from operational signals → identify recurring problems → simulate
improvements → recommend a safer configuration → **require human approval***.
A **safe adaptive governance loop** — nothing here autonomously changes production
policy or `config/settings.yaml`.

**`incident/` — Incident Intelligence**
- `store.py` / `IncidentRecord` — PII-safe view of each flagged interaction + a
  reviewer/feedback *governance signal* overlay (no `matched_text`, no ground truth).
- `clustering.py` — deterministic grouping by a transparent structured **signature**
  (`application | dimension | tier | verification | reason codes | tier-changing rules |
  consequence band | criticality band`). No LLM, no embeddings, no randomness.
- `patterns.py` — 10 recurring-failure pattern types with a `detection_confidence`
  (confidence in the *pattern*, not correctness of any AI response).
- `drift.py` — historical vs recent window, global + per-detector + per-application +
  per-policy-rule scopes. `STABLE` / `TREND` / `POTENTIAL_DRIFT` — *"operational drift
  signal, not proof of model degradation"*.
- `attribution.py` — observed-association percentages ("dominant contributor",
  "observed alongside", "associated with" — never "caused by").
- reviewer-override patterns (e.g. `BLOCK -> HUMAN_REVIEW ×N`) — *"a governance signal,
  NOT evidence the automated decision was incorrect"*.

**`adaptive/` — Adaptive Guardrails**
- `recommendations.py` — patterns → `AdaptiveRecommendation` (`REVIEW_POLICY` /
  `REVIEW_VERIFICATION_THRESHOLD` / `INVESTIGATE_DRIFT` / `REVIEW_APPLICATION` /
  `REVIEW_DETECTOR` / `NO_ACTION`), deterministic content-hash IDs.
- `counterfactual.py` — **reuses** `calibration.sweep` + `calibration.select` (no second
  simulation framework). CURRENT vs CANDIDATE recall / precision / FPR / missed-risk /
  FAST / DEEP / human-review / latency. **Safety is evaluated FIRST** — a candidate that
  fails a safety constraint is never chosen; "no safe candidate" is a valid outcome.
- `approval.py` — approval gate (no auth). Approving records
  **`APPROVED_FOR_EVALUATION`** — there is **no `DEPLOYED` / `APPLIED_TO_PRODUCTION`
  status and no auto-deployment path**.
- `AdaptiveGovernanceReport` — `production_configuration_status = "UNCHANGED"`.

Guarantees (tests): neither package imports the evaluation package or reads ground truth,
re-runs a detector / decision engine / policy, or writes `config/settings.yaml`; no raw
PII appears anywhere; every report is deterministic.

- **API:** `GET /incidents`, `/incidents/{id}`, `/incidents/patterns`, `/incidents/drift`,
  `/adaptive/report`, `/adaptive/recommendations`, `/adaptive/recommendations/{id}`,
  `POST /adaptive/recommendations/{id}/approve`, `POST /adaptive/recommendations/{id}/reject`.
- **Demo:** `python demo.py --adaptive` (scenarios K–R).

## 14d. Enterprise Command Center (`enterprise/`)

A **read-only presentation / orchestration layer** — no new risk formula, no
decision logic, no simulation framework. It assembles judge-facing enterprise
views from the reports the earlier phases already produce (`monitoring`,
`governance`, `incident`, `adaptive`) plus stored `DecisionTrace` /
`GovernanceAction` / `FeedbackRecord` objects. It never re-runs a detector /
`DecisionEngine` / verification pass.

- `views.py` — executive KPI strip, risk posture (reuses the monitoring trend
  direction), per-application risk matrix (posture `LOW` / `MODERATE` / `HIGH` is a
  **band label over existing metrics, not a new score**), Performance /
  Responsibility / Cost / Consequence heatmap ("N/A" where unavailable), live
  decision feed (newest first, always `STORED_TRACE`), executive summary / story
  mode, static architecture map.
- `timeline.py` — governance audit timeline from stored objects only
  (DECISION → INCIDENT → REVIEWER_FEEDBACK → PATTERN → DRIFT → RECOMMENDATION →
  COUNTERFACTUAL → APPROVAL). Where no chronological timestamp exists the event is
  shown in causal workflow order and says so — timestamps are never fabricated.
- **What-If policy playground** — explores an *existing* calibration control
  (`deep_verification_risk_threshold` / `fast_path_min_confidence`) by **reusing**
  `adaptive.counterfactual` (`calibration.sweep` + `calibration.select`). CURRENT vs
  CANDIDATE on recall / precision / FPR / missed-risk / FAST / DEEP / human-review /
  latency; **safety is evaluated FIRST**; returns "No safe configuration
  recommended — constraints were NOT relaxed." rather than relaxing a constraint.
  The interpretation string is generated from the actual metric deltas.
- `demo.py` — `run_enterprise_demo`: a deterministic, bounded **9-step** demo over
  the real pipeline (AI traffic → risk detection → FAST/DEEP → control decision →
  incident intelligence → pattern/drift → governance recommendation → counterfactual
  safety check → human approval). Every number is produced by the actual system;
  ends `APPROVED_FOR_EVALUATION` with `production_configuration_status = "UNCHANGED"`.
- `service.py` — `EnterpriseService`: the dashboard bundle is cheap (no calibration
  on load); the expensive counterfactual runs **only** on an explicit What-If / demo
  request.

Guarantees (tests): `enterprise/` never imports the evaluation package or reads a
ground-truth field; never constructs or calls a detector / `DecisionEngine` /
`PolicyEngine` / verification router (source-scanned and monkeypatched to raise);
never writes the production configuration file; exposes no raw PII / `matched_text`
in any view, API JSON or Streamlit screen; every view is deterministic for a given
stored state.

- **API (read-only):** `GET /command-center`, `GET /applications`,
  `GET /applications/{application}` (404), `GET /governance/timeline`
  (`?interaction_id=`), `GET /incidents/{id}/investigation` (404; related incidents
  = shared deterministic structured risk signature).
- **Demo:** `python demo.py --enterprise` (concise judge-ready transcript; also in
  `--all`).

## 15. Web UI (`streamlit_app.py`)

A single Streamlit app running the **real pipeline in-process**. Tabs:

- **Check a response** — form + one-click scenario loader (A–F, H) + an executive result
  screen: **RISK and CONFIDENCE side by side**, per-dimension + criticality metrics,
  reason-code chips, **"why did ControlPlane decide this?"** explainability panel, and
  breakdown tabs for performance / responsibility / cost / consequence / **criticality** /
  policy trace / fusion
- **Policy simulation** — the same interaction under every application profile
- **Counterfactual** — change `action_amount_inr` / `action_type` / `affected_entities`
  and see the decision (and which rules) change
- **Multi-turn session** — watch risk accumulate and escalate
- **Audit** — look up any past decision by interaction ID (PII redacted)
- **Feedback** — approve / modify / reject the last decision
- **🛰️ Command Center** — enterprise risk monitoring from real traces ("Populate demo
  operational traffic"): executive health, application risk map, **🚨 Incident Command**,
  risk drivers, FAST/DEEP verification, trends & operational shifts, governance feedback.
  Select an incident → **Investigate Incident** → the full **🔎 Incident Investigation**
  workspace (executive assessment → why → risk / evidence / consequence / criticality →
  decision path → counterfactual → reviewer governance action → governance history;
  the automated decision stays immutable).
- **🧭 Governance** — closed-loop governance intelligence: executive overview, application
  comparison, policy intelligence, reviewer signals (with a prominent *automated ≠
  reviewer signal ≠ ground truth* banner), governance insights, trend monitoring, and
  calibration recommendations (with a prominent *RECOMMENDATION ONLY — NOT APPLIED*
  banner).
- **🧠 Adaptive Intelligence** — incident patterns → drift → root-cause/attribution →
  adaptive recommendation → counterfactual (CURRENT vs CANDIDATE, safety highlighted) →
  **APPROVE FOR EVALUATION / REJECT** (no "DEPLOY NOW"). Prominent banner: *RECOMMENDATION
  ONLY … no auto-deployment path … config/settings.yaml is never modified.*
- **🏢 Executive** — the judge-facing consolidated command center: executive summary /
  story mode → A KPI strip → B risk posture → C application risk matrix → D risk
  heatmap → live control feed (select → stored trace → explainability → replay, no
  detector re-run) → governance audit timeline → **▶ RUN ENTERPRISE DEMO** (9 steps) →
  🔬 **What-If policy playground** (CURRENT vs CANDIDATE + SAFETY PASS/FAIL +
  metric-derived interpretation) → 🛠️ technical architecture (visual only). No
  "DEPLOY NOW" / "APPLY TO PRODUCTION" button; approval stays `APPROVED_FOR_EVALUATION`;
  *"Production configuration remains unchanged."*

## 16. Dashboard (`dashboard/`)

`dashboard/metrics.py` aggregates real `DecisionTrace` outputs (no fabricated numbers).
`dashboard/app.py` is a standalone Streamlit monitoring view; the same metrics are also
in the main UI's Command Center tab (which supersedes it).

## 17. Next.js Enterprise Command Center (`frontend/`)

A production-shaped React/TypeScript front end — an alternative to the Streamlit UI for
enterprise deployment. **Next.js 14 (App Router) · TypeScript (strict) · Tailwind CSS ·
SWR · Recharts · lucide-react.**

- **`/dashboard`** — Command Center: responsive metric cards (FAST-path %, DEEP-path %,
  bypass savings, critical-floor escalations), risk-distribution chart, decision mix,
  flagged-incident table. Data from `GET /monitoring/operational`.
- **`/incident/[id]`** — investigation workspace: "Original Decision (immutable)" vs
  "Effective Governed Decision" header, interaction context + extracted claims +
  multi-turn session memory (left), human governance & override form + append-only
  governance history (right).
- **`/check`** — live governance console: submit an interaction, see the governed
  decision.

The API client (`src/lib/api.ts`) is a typed Axios instance with an `X-API-Key`
request interceptor; a server-side `/api/*` proxy route keeps the key out of the
browser bundle in production. Run: `cd frontend && npm install && npm run dev`
(needs the API at `NEXT_PUBLIC_API_URL`, default `http://127.0.0.1:8000`).

## 18. Persistent storage (`database/`) — optional

By default all session / audit / governance state is **in-process** (see Limitations).
Setting `CONTROLPLANE_PERSISTENCE=1` (and `pip install -r requirements-db.txt`) swaps
in a **SQLAlchemy 2.0** persistence layer:

- **SQLite** by default (`sqlite:///./controlplane.db`, zero config) — or any
  `CONTROLPLANE_DATABASE_URL` (e.g. `postgresql+psycopg://…` for the Docker stack).
- ORM models: `DbInteraction`, `DbDecisionTrace`, `DbSessionState`, `DbGovernanceAction`.
  Decision traces and governance actions are **append-only** (enforced by ORM
  `before_update` / `before_delete` guards — mirroring the immutable `DecisionTrace`).
- The `database/` package is **import-isolated**: nothing in the runtime graph imports
  it unless persistence is switched on. `SessionManager` and `GovernanceStore` take an
  optional injected store; the in-memory path is byte-for-byte unchanged. Without
  SQLAlchemy installed the DB tests skip and everything else runs identically.

## 19. API-key security & CORS (`api/security.py`)

- `verify_api_key` is a **global FastAPI dependency** that is a **no-op until
  `CONTROLPLANE_API_KEY` is set**. When set, every request needs an `X-API-Key` header
  (`/health` and `/docs` stay open). Default = no auth (local dev, tests).
- `CORSMiddleware` allow-list from `CONTROLPLANE_CORS_ORIGINS` (default `*`).
- A global exception handler returns a clean JSON 500 instead of a stack trace.

---

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ (developed on 3.13). No GPU, no model downloads, no network, no secrets.

## Usage

```bash
python run_generator.py                      # → data/generated/*.csv  (6000 + 150 rows)
python demo.py --all                         # scenarios A–J + evaluation + ablation
python demo.py --enterprise                   # judge-ready 9-step enterprise demo
python demo.py --all --save demo_output.txt  # + save the transcript
python -m pytest -q                          # 790 tests

python run_app.py                            # web UI  (http://localhost:8501)
python run_app.py --api                      # API + docs (http://localhost:8000/docs)
streamlit run streamlit_app.py               # web UI directly
streamlit run dashboard/app.py               # standalone monitoring dashboard
uvicorn api.app:app --reload                 # API directly

cd frontend && npm install && npm run dev    # Next.js Command Center (http://localhost:3000)
```

## Example `/check` request

```bash
curl -s -X POST http://localhost:8000/check -H "content-type: application/json" -d "{
  \"application\": \"customer_support\",
  \"session_id\": \"SESSION-001\",
  \"context\": \"Customer asked to confirm contact details for account ACC-227763.\",
  \"response\": \"The contact details for account ACC-227763 are: Karan Mehta, email karan.mehta@example-test.com, phone +91-940847221.\",
  \"action_type\": \"information\"
}"
```

## Example decision response

```json
{
  "interaction_id": "INT-API-…",
  "decision": {
    "performance_risk": 0.55,
    "responsibility_risk": 0.72,
    "cost_risk": 0.0,
    "consequence": {
      "financial_impact": 0.0, "reversibility": 0.05, "sensitivity": 0.2,
      "blast_radius": 0.01, "action_automation": 0.9, "consequence_score": 0.14
    },
    "overall_risk": 0.62,
    "decision": "BLOCK",
    "decision_confidence": 0.95,
    "triggered_rules": ["HIGH_RESPONSIBILITY_RISK", "CRITICAL_PII", "AUTOMATED_ACTION"],
    "reason_codes": ["CRITICAL_PII", "PII_EXPOSURE", "HIGH_RESPONSIBILITY_RISK"],
    "explanation": "BLOCK (base VERIFY) because the response exposes critical personal data. Risk 0.62 at confidence 0.95 (uncertainty 0.05). Key signals: responsibility: pii; consequence low, criticality low. Reason codes: CRITICAL_PII, PII_EXPOSURE, HIGH_RESPONSIBILITY_RISK.",
    "timestamp": "2026-08-28T00:00:00Z"
  }
}
```

*(Values come from the running implementation — see `demo_output.txt` for a full transcript.)*

---

## ⏱️ 5-minute judge demo

```bash
pip install -r requirements.txt
python run_app.py
```

Open **http://localhost:8501**, then, in the **Check a response** tab:

1. **A · Clean** → **CHECK AI RESPONSE** → **ALLOW** (risk ~0.02, confidence ~0.90).
2. **B · Hallucination** → **CHECK** → **VERIFY** — performance status `CONTRADICTED`,
   reason code `HIGH_PERFORMANCE_RISK`; consequence is low so it is VERIFY, not a block.
3. **C · PII leakage** → **CHECK** → **BLOCK** — `CRITICAL_PII`, human review = YES; open the
   **Responsibility** breakdown for the *redacted* findings and the severity rationale.
4. **D · High consequence** → **CHECK** → **VERIFY** — a plausible answer, but
   `HIGH_CRITICALITY_ACTION` / `HIGH_FINANCIAL_IMPACT` fire on a ₹480,000 refund. Note the
   **RISK is low but CONFIDENCE is also low** — the two numbers are shown side by side.
5. **E · Multi-risk** → **CHECK** → **BLOCK** — reason codes `MULTI_RISK`, `CRITICAL_PII`,
   `IRREVERSIBLE_ACTION`; **"what moved the decision"** shows each rule.
6. **F · Cost anomaly** → **CHECK** → **VERIFY** — `anomaly_types` = RETRY_SPIKE / LATENCY_SPIKE
   / …; `COST_ONLY_CAP` keeps it an operational review, not a safety block.
7. **H · Low confidence** → **CHECK** → **VERIFY** — risk only ~0.28, confidence ~0.35: the
   pipeline neither ALLOWs (there is risk + consequence) nor BLOCKs (weak evidence).
8. **Policy simulation** tab → **Simulate** — the same interaction: `customer_support` →
   VERIFY, `decision_support` → HUMAN_REVIEW.
9. **Counterfactual** tab → set `action_amount_inr` to `100` → **Compare** — VERIFY → ALLOW,
   `HIGH_CONSEQUENCE` / `HIGH_FINANCIAL_IMPACT` stop firing.
10. **Multi-turn session** tab → **Add a borderline (hallucination) turn** ×4 → cumulative
    risk rises, **Escalated → YES**, decision moves VERIFY → **HUMAN_REVIEW**.
11. **Monitoring** → **Populate with 150 demo interactions** → distributions, reason-code
    counts, latency, estimated cost.
12. **Audit** tab → paste an `interaction_id` → full redacted decision trail with
    reason codes and confidence band.

Terminal-only alternative: `python demo.py --all` (transcript in `demo_output.txt`) —
covers scenarios **A–J** (I = policy counterfactual, J = consequence counterfactual).

---

## Deployment

The web UI is self-contained (pipeline runs in-process), so any Python-app host works.

| Platform | How |
|---|---|
| **Full stack (local / VM)** | `docker compose up --build` — brings up `web` (Next.js `:3000`), `api` (FastAPI `:8000`, persists to Postgres), `ui` (Streamlit `:8501`) and `db` (`postgres:16`) on one network. |
| **Streamlit Community Cloud** | Point it at this repo; it auto-detects `streamlit_app.py`. Build command not needed (data is generated lazily on first request). *Easiest for the demo.* |
| **Render** | `render.yaml` blueprint included — deploys the UI and the API as two free web services. |
| **Docker single-service** (Fly.io / Railway / Cloud Run / HF Spaces) | `api.Dockerfile` (multi-stage, non-root, Uvicorn), `ui.Dockerfile` (Streamlit), `frontend.Dockerfile` (Next standalone). The legacy root `Dockerfile` serves the Streamlit UI on `$PORT`. |
| **Railway / Heroku-style** | `Procfile` included (`web` = UI, `api` = FastAPI). |

No secrets are required. `.env.example` lists the few optional tuning variables; never
commit a real `.env`.

**Deployment status in this environment:** deployment configuration is prepared and the
app is verified running locally (`streamlit` health `200`, API health `200`). A public
URL was **not** provisioned here because no deployment-platform credentials are available
in this environment. To go live: push the repo and pick a row from the table above (the
Streamlit Community Cloud path needs only a GitHub repo + a few clicks).

---

## Limitations

- **Lexical NLI is a system backend.** TF-IDF retrieval + rule-based `CoverageNLIBackend`.
  ~53% recall on synthetic contradiction cases (0 false positives); abstains
  (`UNVERIFIED`) on most paraphrased-but-grounded responses; cannot follow synonymy,
  multi-hop inference or entity-swap contradictions without lexical cues. It also
  produces *confident* contradictions and *low-risk* unverifieds, so genuine
  "high-risk / low-confidence" cases (where the `LOW_CONFIDENCE_HIGH_RISK` rule bites)
  are rarer than a calibrated model would produce.
- **Responsibility heuristics need stronger production models.** Regex / lexicons tuned
  to synthetic data; real-world recall would be materially lower. **Bias detection is a
  signal, not absolute truth** — every bias output is a `POTENTIAL_BIAS_SIGNAL` for
  human interpretation.
- **Cost model needs calibration.** Prototype INR rates are illustrative, not real
  billing; the efficiency score is a heuristic.
- **Criticality uses production-visible fields only** (action metadata + response text) —
  it is transparent and documented, not a learned or hidden feature.
- **Synthetic data is not production data.** `expected_decision` in the dataset is itself
  a simple baseline heuristic, not an oracle; claim-level ground truth is not available,
  so contradiction metrics are reported at the response level.
- **Session / audit / feedback state is in-memory by default** (process-local). An
  opt-in SQLAlchemy persistence layer (`CONTROLPLANE_PERSISTENCE=1`, SQLite or
  PostgreSQL — see §18) provides durable, append-only storage; investigation reads
  still use the in-process audit cache within a running process, so a fully
  DB-driven cross-process investigation path is future work.
- **Policy configuration would need enterprise governance.** The profiles here are
  system defaults; real deployments need change control and per-geography packs.
- **No regulatory claims.** Having a feature is not compliance with any regulation.
- **Detectors run sequentially here** (they are independent and `parallel_detectors=True`
  demonstrates concurrency, but at initial scale it saves <1 ms). Production would run
  them concurrently and stream a partial decision for latency-critical paths.

## Future work

- Swap `CoverageNLIBackend` / `TfidfEmbeddingBackend` for a small fine-tuned NLI model and
  a sentence-embedding retriever behind the same interfaces.
- Presidio-class PII recognition; a reviewed toxicity model; a fairness toolkit for bias.
- Real provider cost integration and per-tenant budget enforcement.
- Shared, TTL'd session + audit stores; signed append-only audit log.
- Calibrate fusion / policy thresholds from the feedback store; per-geography policy packs.
- Parallel detector execution and streaming partial decisions.

---

## Repository layout

```
streamlit_app.py   interactive web UI (runs the pipeline in-process)
run_app.py          launcher (UI / API / both / demo)
demo.py             end-to-end terminal demo  →  demo_output.txt
run_generator.py    synthetic data generator

# --- decision pipeline (evaluation order) ---
data/           schemas + deterministic synthetic data generator (6000 + 150)
detectors/
  performance/    grounding / hallucination (claim extraction → TF-IDF → lexical NLI)
  responsibility/ pii · toxicity · bias (context-aware severity, regex + lexicon)
  cost/           operational / cost anomaly (efficiency + typed anomalies)
criticality/    claim / action criticality — amplifies performance risk
fusion/         risk fusion engine (weighted + severity pull; RISK + CONFIDENCE)
consequence/    5-factor consequence engine (severity ≠ likelihood)
session/        multi-turn risk accumulation, ContextualSnapshot, critical floor
verification/   tiered cascade router (FastPath → deterministic gates → MF router); FAST/DEEP
policy/         configurable per-application policy engine → InterventionTier
decision/       pipeline orchestrator, FinalDecision + immutable DecisionTrace, replay

# --- services over stored traces (never re-run the pipeline) ---
explainability/ human-readable "why did ControlPlane decide this?" builder
monitoring/     OperationalMonitoringReport, incident classification
investigation/  incident investigation + append-only GovernanceStore
governance/     read-only governance analytics / insights / recommendations
incident/       incident clustering / patterns / drift / attribution
adaptive/       closed-loop guardrail recommendations + approval gate (no auto-deploy)
enterprise/     Enterprise Command Center view assembly (read-only)
simulation/     policy simulation + counterfactual analysis
calibration/    threshold sweep + safety-constrained configuration selection
evaluation/     metrics, evaluation harness, ablation (evaluation-only ground truth)

# --- infrastructure ---
api/            FastAPI app (37 routes) + ControlPlaneService + API-key security + CORS
database/       SQLAlchemy 2.0 opt-in persistence (import-isolated; SQLite / PostgreSQL)
frontend/       Next.js 14 (App Router) + TypeScript + Tailwind — Command Center UI
dashboard/      legacy standalone Streamlit monitoring view (redundant with Command Center)
common/         reason_codes.py · timing.py
config/         settings.yaml (single source of decision thresholds)
scripts/        operator scripts (MF router training loop)
tests/          790 tests — scenarios A–R, every detector, policy, fusion, session,
                verification, investigation, governance, adaptive, enterprise,
                full-pipeline integration, API, Streamlit UI, DB layer, no-leakage guards

Dockerfile · api.Dockerfile · ui.Dockerfile · frontend.Dockerfile · docker-compose.yml
render.yaml · Procfile · .env.example · .streamlit/config.toml
```

## Latency

Per-detector timing is recorded on every run (`DecisionTrace.stage_latency_ms`); the
Web UI shows total pipeline latency. On a laptop the full pipeline is a few milliseconds
per interaction (mostly TF-IDF vectorisation). The three detectors share no state and
are wired to run concurrently — `DecisionEngine(parallel_detectors=True)` executes them
in a thread pool with byte-identical output. It is **off by default** because at this
scale the fan-out costs more than it saves; in production the detectors would run
concurrently and a partial decision could stream while the slowest detector finishes.
No models are loaded per request — the engine parses config and fits the cost baseline
once at construction.
