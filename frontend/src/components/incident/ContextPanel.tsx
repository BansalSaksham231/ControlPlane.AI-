import { AlertOctagon, Layers3, MessageSquare } from "lucide-react";

import type {
  IncidentInvestigation,
  ReplayClaim,
  SessionMemoryExplanation,
} from "@/lib/types";
import { cn, formatNumber, formatTimestamp, toDateTimeAttr } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function ContextPanel({
  investigation,
}: {
  investigation: IncidentInvestigation;
}) {
  const { replay, explanation } = investigation;
  const { interaction } = replay;

  return (
    <div className="space-y-4">
      {/* interaction */}
      <Card>
        <CardHeader>
          <CardTitle as="h2" className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" aria-hidden="true" /> Interaction
          </CardTitle>
          <CardDescription>
            {interaction.application} · {interaction.action_type} ·{" "}
            <time dateTime={toDateTimeAttr(interaction.timestamp)}>
              {formatTimestamp(interaction.timestamp)}
            </time>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <LabelledBlock label="Model response (redacted)">
            <p className="whitespace-pre-wrap rounded-md bg-muted/50 p-3 text-sm">
              {interaction.response || "—"}
            </p>
          </LabelledBlock>
          {explanation.primary_reasons.length > 0 && (
            <ul className="flex flex-wrap gap-2">
              {explanation.primary_reasons.map((reason) => (
                <li key={reason}>
                  <Badge className="bg-muted text-muted-foreground">
                    {reason}
                  </Badge>
                </li>
              ))}
            </ul>
          )}
          <p className="text-xs leading-relaxed text-muted-foreground">
            {explanation.explanation}
          </p>
        </CardContent>
      </Card>

      {/* claims */}
      <Card>
        <CardHeader>
          <CardTitle as="h2" className="flex items-center gap-2">
            <Layers3 className="h-4 w-4" aria-hidden="true" /> Extracted Claims
          </CardTitle>
          <CardDescription>
            Grounding of each factual claim against retrieved evidence
          </CardDescription>
        </CardHeader>
        <CardContent>
          {replay.claims.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No claims were extracted for this interaction.
            </p>
          ) : (
            <ul className="space-y-3">
              {replay.claims.map((claim, index) => (
                <ClaimListItem key={index} claim={claim} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {/* multi-turn session memory */}
      <SessionMemoryCard memory={explanation.session_memory} />
    </div>
  );
}

function ClaimListItem({ claim }: { claim: ReplayClaim }) {
  const statusClasses =
    claim.status === "SUPPORTED"
      ? "bg-tier-allow/15 text-tier-allow border-tier-allow/30"
      : claim.status === "CONTRADICTED"
        ? "bg-tier-block/15 text-tier-block border-tier-block/30"
        : "bg-tier-verify/15 text-tier-verify border-tier-verify/30";

  return (
    <li className="rounded-md border border-border p-3">
      <p className="text-sm">{claim.claim}</p>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        <Badge className={statusClasses}>{claim.status}</Badge>
        <span className="text-muted-foreground">
          claim risk {formatNumber(claim.claim_risk)}
        </span>
        <span className="text-muted-foreground">
          evidence {formatNumber(claim.evidence_strength)}
        </span>
        {claim.nli_label && (
          <span className="text-muted-foreground">
            NLI {claim.nli_label} ({formatNumber(claim.nli_confidence)})
          </span>
        )}
      </div>
    </li>
  );
}

function SessionMemoryCard({
  memory,
}: {
  memory: SessionMemoryExplanation | null;
}) {
  if (!memory) {
    return (
      <Card>
        <CardHeader>
          <CardTitle as="h2">Multi-Turn Session Memory</CardTitle>
          <CardDescription>
            Single-turn / stateless interaction — no session history contributed
            to this decision.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card
      className={cn(
        memory.critical_floor_applied && "border-tier-block/40 bg-tier-block/5",
      )}
    >
      <CardHeader>
        <CardTitle as="h2" className="flex items-center gap-2">
          {memory.critical_floor_applied && (
            <AlertOctagon className="h-4 w-4 text-tier-block" aria-hidden="true" />
          )}
          Multi-Turn Session Memory
        </CardTitle>
        <CardDescription>
          {memory.turns_recorded} turn(s) recorded ·{" "}
          {memory.has_critical_history
            ? "critical history present"
            : "no critical history"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {memory.critical_floor_applied && (
          <p className="rounded-md border border-tier-block/40 bg-tier-block/10 p-3 text-xs text-tier-block">
            <strong>Non-decaying critical floor active.</strong> A prior critical
            violation holds this session at a minimum risk of{" "}
            {formatNumber(memory.critical_floor)} — it does not decay and raised
            the effective risk of the current turn.
          </p>
        )}

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <PeakStat label="Critical floor" value={formatNumber(memory.critical_floor)} />
          <PeakStat
            label="Peak performance"
            value={formatNumber(memory.peak_performance_risk)}
          />
          <PeakStat
            label="Peak responsibility"
            value={formatNumber(memory.peak_responsibility_risk)}
          />
          <PeakStat label="Peak cost" value={formatNumber(memory.peak_cost_risk)} />
        </div>

        {memory.critical_events.length > 0 && (
          <LabelledBlock label="Critical events">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[420px] text-xs">
                <caption className="sr-only">
                  Turns in this session that crossed a critical boundary
                </caption>
                <thead className="text-left text-muted-foreground">
                  <tr>
                    <th scope="col" className="pb-1 pr-3 font-medium">Turn</th>
                    <th scope="col" className="pb-1 pr-3 font-medium">Trigger</th>
                    <th scope="col" className="pb-1 pr-3 font-medium">Decision</th>
                    <th scope="col" className="pb-1 font-medium">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {memory.critical_events.map((event) => (
                    <tr
                      key={event.turn_index}
                      className="border-t border-border/60"
                    >
                      <th scope="row" className="py-1 pr-3 text-left font-normal tabular-nums">
                        {event.turn_index}
                      </th>
                      <td className="py-1 pr-3">{event.trigger}</td>
                      <td className="py-1 pr-3">{event.decision}</td>
                      <td className="py-1 tabular-nums">
                        {formatNumber(event.risk_at_event)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </LabelledBlock>
        )}

        {memory.pii_entity_keys.length > 0 && (
          <LabelledBlock label="PII entities seen in session (redacted)">
            <ul className="flex flex-wrap gap-1.5">
              {memory.pii_entity_keys.map((key) => (
                <li key={key}>
                  <Badge className="bg-muted font-mono text-[11px] text-muted-foreground">
                    {key}
                  </Badge>
                </li>
              ))}
            </ul>
          </LabelledBlock>
        )}

        {Object.keys(memory.reason_code_counts).length > 0 && (
          <LabelledBlock label="Recurring reason codes">
            <ul className="flex flex-wrap gap-1.5">
              {Object.entries(memory.reason_code_counts).map(([code, count]) => (
                <li key={code}>
                  <Badge className="bg-muted text-[11px] text-muted-foreground">
                    {code} ×{count}
                  </Badge>
                </li>
              ))}
            </ul>
          </LabelledBlock>
        )}
      </CardContent>
    </Card>
  );
}

function PeakStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border p-2">
      <span className="block text-[11px] text-muted-foreground">{label}</span>
      <span className="block text-sm font-semibold tabular-nums">{value}</span>
    </div>
  );
}

function LabelledBlock({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      {children}
    </div>
  );
}
