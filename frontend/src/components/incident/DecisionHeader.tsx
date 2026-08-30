import { Lock, GitBranch, ArrowRight } from "lucide-react";

import type { InterventionTier } from "@/lib/types";
import { cn, tierBadgeClasses } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface DecisionHeaderProps {
  interactionId: string;
  originalDecision: InterventionTier;
  effectiveDecision: InterventionTier | "";
  isOverridden: boolean;
}

export function DecisionHeader({
  interactionId,
  originalDecision,
  effectiveDecision,
  isOverridden,
}: DecisionHeaderProps) {
  const effective = effectiveDecision || originalDecision;

  return (
    <Card>
      <CardContent className="p-4 sm:p-5">
        <header className="flex flex-col gap-1">
          <span className="font-mono text-xs text-muted-foreground">
            {interactionId}
          </span>
          <h1 className="text-lg font-semibold sm:text-xl">
            Incident Investigation
          </h1>
        </header>

        <dl className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-[1fr_auto_1fr] sm:items-center">
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <dt className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Lock className="h-3.5 w-3.5" aria-hidden="true" /> Original
              Decision (immutable)
            </dt>
            <dd className="mt-2">
              <Badge
                className={cn(
                  "px-2.5 py-1 text-sm",
                  tierBadgeClasses(originalDecision),
                )}
              >
                {originalDecision}
              </Badge>
            </dd>
          </div>

          <div aria-hidden="true" className="hidden sm:block">
            <ArrowRight className="mx-auto h-5 w-5 text-muted-foreground" />
          </div>

          <div
            className={cn(
              "rounded-lg border p-3",
              isOverridden
                ? "border-tier-review/40 bg-tier-review/5"
                : "border-border bg-muted/40",
            )}
          >
            <dt className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <GitBranch className="h-3.5 w-3.5" aria-hidden="true" /> Effective
              Governed Decision
            </dt>
            <dd className="mt-2">
              <Badge
                className={cn("px-2.5 py-1 text-sm", tierBadgeClasses(effective))}
              >
                {effective}
              </Badge>
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                {isOverridden
                  ? "A human reviewer override is in effect. The automated decision to the left is unchanged."
                  : "No override recorded — matches the automated decision."}
              </p>
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
