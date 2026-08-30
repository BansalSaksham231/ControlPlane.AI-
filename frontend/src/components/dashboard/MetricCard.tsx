import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export type MetricTone = "default" | "positive" | "warning" | "critical";

export interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  icon?: LucideIcon;
  tone?: MetricTone;
}

const toneIconClasses: Record<MetricTone, string> = {
  default: "text-muted-foreground",
  positive: "text-tier-allow",
  warning: "text-tier-verify",
  critical: "text-tier-block",
};

/**
 * One KPI in the dashboard metrics row. Exposed to assistive tech as a labelled
 * group so the metric name and its value are announced together.
 */
export function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "default",
}: MetricCardProps) {
  return (
    <Card role="group" aria-label={label}>
      <CardContent className="flex flex-col gap-1 p-4 sm:p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </span>
          {Icon && (
            <Icon className={cn("h-4 w-4", toneIconClasses[tone])} aria-hidden="true" />
          )}
        </div>
        <p className="text-2xl font-semibold tabular-nums tracking-tight sm:text-3xl">
          {value}
        </p>
        {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}

export function MetricCardSkeleton() {
  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-4 sm:p-5">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-3 w-32" />
      </CardContent>
    </Card>
  );
}
