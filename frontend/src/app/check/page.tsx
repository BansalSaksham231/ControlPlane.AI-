"use client";

import { useId, useState } from "react";
import Link from "next/link";
import { Loader2, Play, ArrowUpRight } from "lucide-react";

import { ApiError, checkInteraction } from "@/lib/api";
import type { Application, ActionType, CheckResponse } from "@/lib/types";
import { cn, formatNumber, formatPercent, tierBadgeClasses } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Label, Select, Textarea } from "@/components/ui/field";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const APPLICATIONS: Application[] = [
  "customer_support",
  "internal_knowledge_assistant",
  "decision_support",
];
const ACTION_TYPES: ActionType[] = [
  "information",
  "refund",
  "account_update",
  "account_cancellation",
  "external_communication",
  "recommendation",
];

export default function LiveGovernancePage() {
  const fieldId = useId();
  const [application, setApplication] = useState<Application>("customer_support");
  const [actionType, setActionType] = useState<ActionType>("information");
  const [context, setContext] = useState(
    "Refunds are allowed within 30 days of purchase.",
  );
  const [modelResponse, setModelResponse] = useState(
    "You can absolutely get a full refund any time, no conditions.",
  );
  const [sessionId, setSessionId] = useState("web-console");
  const [actionAmount, setActionAmount] = useState("0");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<CheckResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      const response = await checkInteraction({
        application,
        action_type: actionType,
        context,
        response: modelResponse,
        session_id: sessionId.trim() || "web-console",
        action_amount_inr: Number(actionAmount) || 0,
      });
      setResult(response);
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError
          ? `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`
          : "Request failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-lg font-semibold sm:text-xl">Live Governance</h1>
        <p className="text-sm text-muted-foreground">
          Send an interaction through the decision engine (POST /check)
        </p>
      </header>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle as="h2">Interaction</CardTitle>
            <CardDescription>Model output + grounding context</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} noValidate className="space-y-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor={`${fieldId}-app`}>Application</Label>
                  <Select
                    id={`${fieldId}-app`}
                    value={application}
                    onChange={(event) =>
                      setApplication(event.target.value as Application)
                    }
                  >
                    {APPLICATIONS.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor={`${fieldId}-action`}>Action type</Label>
                  <Select
                    id={`${fieldId}-action`}
                    value={actionType}
                    onChange={(event) =>
                      setActionType(event.target.value as ActionType)
                    }
                  >
                    {ACTION_TYPES.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </Select>
                </div>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor={`${fieldId}-context`}>Context / evidence</Label>
                <Textarea
                  id={`${fieldId}-context`}
                  value={context}
                  onChange={(event) => setContext(event.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor={`${fieldId}-response`}>Model response</Label>
                <Textarea
                  id={`${fieldId}-response`}
                  value={modelResponse}
                  onChange={(event) => setModelResponse(event.target.value)}
                  aria-required="true"
                />
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label htmlFor={`${fieldId}-session`}>Session ID</Label>
                  <Input
                    id={`${fieldId}-session`}
                    value={sessionId}
                    onChange={(event) => setSessionId(event.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor={`${fieldId}-amount`}>Action amount (INR)</Label>
                  <Input
                    id={`${fieldId}-amount`}
                    type="number"
                    value={actionAmount}
                    onChange={(event) => setActionAmount(event.target.value)}
                  />
                </div>
              </div>

              {errorMessage && (
                <p
                  role="alert"
                  className="rounded-md border border-tier-block/40 bg-tier-block/10 p-2 text-xs text-tier-block"
                >
                  {errorMessage}
                </p>
              )}

              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Play className="h-4 w-4" aria-hidden="true" />
                )}
                Run check
              </Button>
            </form>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle as="h2">Decision</CardTitle>
            <CardDescription>
              {result
                ? `Interaction ${result.interaction_id}`
                : "Submit an interaction to see the governed decision"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <output className="block">
              {!result ? (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  No result yet.
                </p>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <Badge
                      className={cn(
                        "px-2.5 py-1 text-sm",
                        tierBadgeClasses(result.decision.decision),
                      )}
                    >
                      {result.decision.decision}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {result.decision.verification_path} path
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <ResultStat
                      label="Overall risk"
                      value={formatNumber(result.decision.overall_risk)}
                    />
                    <ResultStat
                      label="Confidence"
                      value={formatPercent(result.decision.decision_confidence)}
                    />
                    <ResultStat
                      label="Performance"
                      value={formatNumber(result.decision.performance_risk)}
                    />
                    <ResultStat
                      label="Responsibility"
                      value={formatNumber(result.decision.responsibility_risk)}
                    />
                  </div>

                  {result.decision.reason_codes.length > 0 && (
                    <ul className="flex flex-wrap gap-1.5">
                      {result.decision.reason_codes.map((code) => (
                        <li key={code}>
                          <Badge className="bg-muted text-[11px] text-muted-foreground">
                            {code}
                          </Badge>
                        </li>
                      ))}
                    </ul>
                  )}

                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {result.decision.explanation}
                  </p>

                  <Link
                    href={`/incident/${encodeURIComponent(result.interaction_id)}`}
                    className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                  >
                    Open investigation workspace{" "}
                    <ArrowUpRight className="h-3 w-3" aria-hidden="true" />
                  </Link>
                </div>
              )}
            </output>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function ResultStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border p-2">
      <span className="block text-[11px] text-muted-foreground">{label}</span>
      <span className="block text-sm font-semibold tabular-nums">{value}</span>
    </div>
  );
}
