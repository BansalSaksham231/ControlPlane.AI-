/**
 * TypeScript contracts mapping the ControlPlane.ai FastAPI schemas.
 *
 * Source of truth (Python):
 *   decision/schemas.py        -> InterventionTier, FinalDecision, DecisionTrace
 *   session/schemas.py         -> ContextualSnapshot, CriticalEvent
 *   monitoring/schemas.py      -> OperationalMonitoringReport, VerificationSummary, MultiTurnSummary
 *   investigation/schemas.py   -> GovernanceAction, IncidentInvestigation
 *   explainability/schemas.py  -> ExplainabilitySummary, SessionMemoryExplanation
 *
 * These mirror `model_dump(mode="json")` output. Optional / nullable Python
 * fields (`float | None`, `default=None`) are `T | null` here.
 */

// ---------------------------------------------------------------------------
// enums
// ---------------------------------------------------------------------------

export type InterventionTier =
  | "ALLOW"
  | "ANNOTATE"
  | "VERIFY"
  | "HUMAN_REVIEW"
  | "BLOCK";

export const INTERVENTION_TIERS: InterventionTier[] = [
  "ALLOW",
  "ANNOTATE",
  "VERIFY",
  "HUMAN_REVIEW",
  "BLOCK",
];

export type VerificationPath = "FAST" | "DEEP";

export type Application =
  | "customer_support"
  | "internal_knowledge_assistant"
  | "decision_support";

export type ActionType =
  | "information"
  | "refund"
  | "account_update"
  | "account_cancellation"
  | "external_communication"
  | "recommendation";

export type GovernanceActionType =
  | "ACKNOWLEDGE"
  | "APPROVE_DECISION"
  | "MODIFY_DECISION"
  | "REJECT_DECISION"
  | "ESCALATE"
  | "CLOSE";

export type InvestigationStatus =
  | "OPEN"
  | "ACKNOWLEDGED"
  | "REVIEWED"
  | "ESCALATED"
  | "CLOSED";

// ---------------------------------------------------------------------------
// /check
// ---------------------------------------------------------------------------

export interface Consequence {
  financial_impact: number;
  reversibility: number;
  sensitivity: number;
  blast_radius: number;
  action_automation: number;
  consequence_score: number;
}

export interface FinalDecision {
  performance_risk: number;
  responsibility_risk: number;
  cost_risk: number;
  consequence: Consequence;
  decision: InterventionTier;
  overall_risk: number;
  decision_confidence: number;
  explanation: string;
  triggered_rules: string[];
  reason_codes: string[];
  verification_path: VerificationPath;
  timestamp: string;
}

export interface CheckRequest {
  application: Application;
  response: string;
  context?: string;
  prompt?: string;
  session_id?: string;
  interaction_id?: string;
  action_type?: ActionType;
  action_amount_inr?: number;
  affected_entities?: number;
  include_trace?: boolean;
}

export interface CheckResponse {
  interaction_id: string;
  decision: FinalDecision;
  trace: Record<string, unknown> | null;
}

// ---------------------------------------------------------------------------
// /audit/{id}
// ---------------------------------------------------------------------------

export interface ResponsibilityFinding {
  category: string;
  subtype: string;
  severity: string;
  redacted_text: string;
}

export interface AuditTrace {
  interaction_id: string;
  timestamp: string;
  application: string;
  action_type: string;
  decision: InterventionTier;
  overall_risk: number;
  performance_risk: number;
  responsibility_risk: number;
  cost_risk: number;
  consequence_score: number;
  decision_confidence: number;
  triggered_rules: string[];
  reason_codes: string[];
  decision_confidence_band: string;
  action_criticality: number;
  decision_drivers: string[];
  decision_path: string[];
  verification_path: VerificationPath;
  verification_latency_ms: number;
  performance_status: string;
  responsibility_findings: ResponsibilityFinding[];
  cost_anomalies: unknown[];
  explanation: string;
}

// ---------------------------------------------------------------------------
// monitoring — OperationalMonitoringReport
// ---------------------------------------------------------------------------

export interface RiskBucket {
  bucket_name: string;
  min_risk: number;
  max_risk: number;
  count: number;
  percentage: number | null;
}

export interface RiskDistribution {
  buckets: RiskBucket[];
  total: number;
}

export interface MonitoringSnapshot {
  total_interactions: number;
  allow_count: number;
  annotate_count: number;
  verify_count: number;
  human_review_count: number;
  block_count: number;
  allow_rate: number;
  annotate_rate: number;
  verify_rate: number;
  human_review_rate: number;
  block_rate: number;
  fast_path_rate: number;
  deep_path_rate: number;
  high_consequence_rate: number;
  high_criticality_rate: number;
  dominant_risk_dimension: string | null;
}

export interface VerificationSummary {
  fast_count: number;
  deep_count: number;
  fast_rate: number;
  deep_rate: number;
  deep_trigger_reason_counts: Record<string, number>;
  average_fast_latency_ms: number | null;
  average_deep_latency_ms: number | null;
  average_total_verification_latency_ms: number | null;
  p95_total_verification_latency_ms: number | null;
  semantic_bypass_count: number;
  semantic_bypass_rate_of_deep: number;
  estimated_bypass_compute_saved_ms: number | null;
}

export interface MultiTurnSummary {
  total_sessions: number;
  multi_turn_sessions: number;
  sessions_hitting_critical_floor: number;
  critical_floor_session_rate: number | null;
  critical_floor_events: number;
}

export interface IncidentSummary {
  interaction_id: string;
  application: string;
  timestamp: string;
  action_type: string;
  decision: InterventionTier;
  overall_risk: number;
  confidence: number;
  dominant_dimension: string | null;
  reason_codes: string[];
  verification_path: VerificationPath;
  consequence_score: number;
  criticality: number;
  requires_human_review: boolean;
  severity: "CRITICAL" | "HIGH" | "MEDIUM";
  triggers: string[];
  severity_rationale: string;
}

export interface IncidentDigest {
  total: number;
  incident_rate: number;
  by_severity: Record<string, number>;
  modified: number;
  rejected: number;
  override_count: number;
  override_rate: number | null;
  approval_rate: number | null;
  note: string;
}

export interface ApplicationRiskSummary {
  application: string;
  interaction_count: number;
  average_risk: number;
  high_risk_rate: number;
  block_rate: number;
  human_review_rate: number;
  deep_path_rate: number;
  dominant_dimension: string | null;
}

export interface OperationalMonitoringReport {
  generated_at: string | null;
  total_interactions: number;
  snapshot: MonitoringSnapshot;
  risk_distribution: RiskDistribution;
  applications: ApplicationRiskSummary[];
  reason_codes: { reason_code: string; count: number; share_of_interventions: number }[];
  verification: VerificationSummary;
  multi_turn: MultiTurnSummary;
  incidents: IncidentSummary[];
  incident_digest: IncidentDigest;
  notes: string[];
}

// ---------------------------------------------------------------------------
// session — ContextualSnapshot
// ---------------------------------------------------------------------------

export interface CriticalEvent {
  turn_index: number;
  interaction_id: string;
  decision: string;
  trigger: string;
  risk_at_event: number;
}

export interface ContextualSnapshot {
  turns_recorded: number;
  pii_entity_keys: string[];
  reason_code_counts: Record<string, number>;
  tier_changing_rules: string[];
  peak_performance_risk: number;
  peak_responsibility_risk: number;
  peak_cost_risk: number;
  critical_floor: number;
  critical_events: CriticalEvent[];
}

/** explainability.schemas.SessionMemoryExplanation */
export interface SessionMemoryExplanation {
  turns_recorded: number;
  has_critical_history: boolean;
  critical_floor: number;
  critical_floor_applied: boolean;
  peak_performance_risk: number;
  peak_responsibility_risk: number;
  peak_cost_risk: number;
  pii_entity_keys: string[];
  reason_code_counts: Record<string, number>;
  critical_events: CriticalEvent[];
  explanation: string;
}

// ---------------------------------------------------------------------------
// investigation — GovernanceAction + IncidentInvestigation
// ---------------------------------------------------------------------------

export interface GovernanceAction {
  action_id: string;
  interaction_id: string;
  timestamp: string;
  actor: string;
  action: GovernanceActionType;
  comment: string;
  previous_status: InvestigationStatus;
  new_status: InvestigationStatus;
  original_decision: string;
  reviewer_decision: string | null;
}

export interface ReplayClaim {
  claim: string;
  status: string;
  claim_risk: number;
  evidence_strength: number;
  retrieval_similarity: number;
  nli_label: string | null;
  nli_confidence: number | null;
  top_evidence: string;
}

export interface ReplayInteraction {
  interaction_id: string;
  timestamp: string;
  application: string;
  action_type: string;
  model: string | null;
  response: string;
}

export interface IncidentReplay {
  found: boolean;
  replay_note: string;
  interaction: ReplayInteraction;
  verification: Record<string, unknown>;
  risk_signals: Record<string, unknown>;
  claims: ReplayClaim[];
  decision_path: string[];
  tier_transitions: unknown[];
  final_decision: Record<string, unknown>;
  explanation: string;
}

export interface ExplainabilitySummary {
  decision: InterventionTier;
  overall_risk: number;
  decision_confidence: number;
  verification_path: VerificationPath;
  human_review_required: boolean;
  primary_reasons: string[];
  decision_drivers: string[];
  decision_path: { from_tier?: string; to_tier?: string; reason?: string; explanation?: string }[];
  session_memory: SessionMemoryExplanation | null;
  explanation: string;
  notes: string[];
}

export interface IncidentInvestigation {
  found: boolean;
  interaction_id: string;
  incident: IncidentSummary | null;
  replay: IncidentReplay;
  explanation: ExplainabilitySummary;
  original_decision: InterventionTier;
  requires_human_review: boolean;
  investigation_status: InvestigationStatus;
  available_actions: GovernanceActionType[];
  governance_history: GovernanceAction[];
  latest_reviewer_decision: string | null;
  effective_governed_decision: InterventionTier | "";
  is_overridden: boolean;
  notes: string[];
}

// ---------------------------------------------------------------------------
// governance override
// ---------------------------------------------------------------------------

export interface GovernanceOverrideRequest {
  interaction_id: string;
  action_type: "APPROVE" | "MODIFY" | "REJECT" | GovernanceActionType;
  new_tier?: InterventionTier | null;
  justification?: string;
  reviewer_id?: string;
}

export interface GovernanceOverrideResponse {
  interaction_id: string;
  original_decision: InterventionTier;
  effective_governed_decision: InterventionTier;
  is_overridden: boolean;
  governance_history: GovernanceAction[];
}

// ---------------------------------------------------------------------------
// /health
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  detectors: string[];
  checks_served: number;
  active_sessions: number;
}
