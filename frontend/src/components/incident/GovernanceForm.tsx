"use client";

import { useId, useMemo, useState } from "react";
import { CheckCircle2, Scale, Loader2, History } from "lucide-react";

import { ApiError, postGovernanceOverride } from "@/lib/api";
import {
  INTERVENTION_TIERS,
  type GovernanceAction,
  type IncidentInvestigation,
  type InterventionTier,
} from "@/lib/types";
import {
  cn,
  formatTimestamp,
  tierBadgeClasses,
  toDateTimeAttr,
} from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Label, Select, Textarea } from "@/components/ui/field";

type ReviewerActionType = "APPROVE" | "MODIFY" | "REJECT";

interface GovernanceFormProps {
  investigation: IncidentInvestigation;
  /** re-fetch the investigation after a successful override */
  onSubmitted: () => void | Promise<unknown>;
}

export function GovernanceForm({
  investigation,
  onSubmitted,
}: GovernanceFormProps) {
  const fieldId = useId();
  const [actionType, setActionType] = useState<ReviewerActionType>("MODIFY");
  const [newTier, setNewTier] = useState<InterventionTier>(
    investigation.original_decision,
  );
  const [justification, setJustification] = useState("");
  const [reviewerId, setReviewerId] = useState("admin");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const requiresTier = actionType === "MODIFY";
  const requiresJustification = actionType !== "APPROVE";
  const justificationMissing =
    requiresJustification && justification.trim().length === 0;

  const canSubmit = useMemo(
    () => !isSubmitting && !justificationMissing,
    [isSubmitting, justificationMissing],
  );

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);
    setIsSubmitting(true);
    try {
      const result = await postGovernanceOverride({
        interaction_id: investigation.interaction_id,
        action_type: actionType,
        new_tier: requiresTier ? newTier : null,
        justification: justification.trim(),
        reviewer_id: reviewerId.trim() || "admin",
      });
      setSuccessMessage(
        `Recorded. Effective governed decision is now ${result.effective_governed_decision}` +
          (result.is_overridden ? " (override in effect)." : "."),
      );
      setJustification("");
      await onSubmitted();
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError
          ? `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`
          : "Failed to record governance action.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2" className="flex items-center gap-2">
            <Scale className="h-4 w-4" aria-hidden="true" /> Human Governance
            &amp; Override
          </CardTitle>
          <CardDescription>
            Records a reviewer outcome on the append-only governance track. The
            automated DecisionTrace is never modified.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor={`${fieldId}-action`}>Governance action</Label>
              <Select
                id={`${fieldId}-action`}
                value={actionType}
                onChange={(event) =>
                  setActionType(event.target.value as ReviewerActionType)
                }
              >
                <option value="APPROVE">
                  APPROVE — agree with the automated call
                </option>
                <option value="MODIFY">
                  MODIFY — I would have chosen a different tier
                </option>
                <option value="REJECT">
                  REJECT — the automated call was wrong
                </option>
              </Select>
            </div>

            {requiresTier && (
              <div className="space-y-1.5">
                <Label htmlFor={`${fieldId}-tier`}>New intervention tier</Label>
                <Select
                  id={`${fieldId}-tier`}
                  value={newTier}
                  onChange={(event) =>
                    setNewTier(event.target.value as InterventionTier)
                  }
                >
                  {INTERVENTION_TIERS.map((tier) => (
                    <option key={tier} value={tier}>
                      {tier}
                    </option>
                  ))}
                </Select>
              </div>
            )}

            <div className="space-y-1.5">
              <Label htmlFor={`${fieldId}-justification`}>
                Justification{" "}
                {requiresJustification && (
                  <span className="text-tier-block" aria-hidden="true">
                    *
                  </span>
                )}
              </Label>
              <Textarea
                id={`${fieldId}-justification`}
                value={justification}
                onChange={(event) => setJustification(event.target.value)}
                placeholder="Why is this override warranted? This is retained in the audit log."
                aria-required={requiresJustification}
                aria-invalid={justificationMissing || undefined}
                aria-describedby={`${fieldId}-justification-hint`}
              />
              <p
                id={`${fieldId}-justification-hint`}
                className="text-[11px] text-muted-foreground"
              >
                {requiresJustification
                  ? "Required for MODIFY and REJECT."
                  : "Optional for APPROVE."}
              </p>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor={`${fieldId}-reviewer`}>Reviewer ID</Label>
              <Input
                id={`${fieldId}-reviewer`}
                value={reviewerId}
                onChange={(event) => setReviewerId(event.target.value)}
              />
            </div>

            {errorMessage && (
              <p
                role="alert"
                className="rounded-md border border-tier-block/40 bg-tier-block/10 p-2 text-xs text-tier-block"
              >
                {errorMessage}
              </p>
            )}
            {successMessage && (
              <p
                role="status"
                className="flex items-start gap-1.5 rounded-md border border-tier-allow/40 bg-tier-allow/10 p-2 text-xs text-tier-allow"
              >
                <CheckCircle2
                  className="mt-0.5 h-3.5 w-3.5 shrink-0"
                  aria-hidden="true"
                />
                {successMessage}
              </p>
            )}

            <Button
              type="submit"
              disabled={!canSubmit}
              className="w-full sm:w-auto"
            >
              {isSubmitting && (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              )}
              Submit governance action
            </Button>
          </form>
        </CardContent>
      </Card>

      <GovernanceHistoryCard history={investigation.governance_history} />
    </div>
  );
}

function GovernanceHistoryCard({ history }: { history: GovernanceAction[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="flex items-center gap-2">
          <History className="h-4 w-4" aria-hidden="true" /> Governance History
        </CardTitle>
        <CardDescription>
          Append-only — {history.length} action(s) recorded
        </CardDescription>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No governance actions recorded yet.
          </p>
        ) : (
          <ol className="space-y-3">
            {history.map((action) => (
              <li
                key={action.action_id}
                className="rounded-md border border-border p-3"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Badge className="bg-muted text-muted-foreground">
                    {action.action}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {action.previous_status} → {action.new_status}
                  </span>
                  {action.reviewer_decision && (
                    <Badge
                      className={cn(
                        "text-[11px]",
                        tierBadgeClasses(action.reviewer_decision),
                      )}
                    >
                      reviewer: {action.reviewer_decision}
                    </Badge>
                  )}
                </div>
                {action.comment && (
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    “{action.comment}”
                  </p>
                )}
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {action.actor} ·{" "}
                  <time dateTime={toDateTimeAttr(action.timestamp)}>
                    {formatTimestamp(action.timestamp)}
                  </time>{" "}
                  · original decision {action.original_decision}
                </p>
              </li>
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  );
}
