"use client";

import { Gauge, Layers, Timer, ShieldAlert, RefreshCw } from "lucide-react";

import { useCommandCenterMetrics } from "@/lib/hooks";
import type { OperationalMonitoringReport } from "@/lib/types";
import { formatDuration, formatPercent } from "@/lib/format";
import { Button } from "@/components/ui/button";
import { LoadingRegion } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/common/states";
import { MetricCard, MetricCardSkeleton } from "@/components/dashboard/MetricCard";
import {
  RiskDistributionChart,
  RiskDistributionChartSkeleton,
} from "@/components/dashboard/RiskDistributionChart";
import { IncidentTable } from "@/components/dashboard/IncidentTable";

export default function DashboardPage() {
  const { report, error, isLoading, refresh } = useCommandCenterMetrics();

  return (
    <div className="space-y-6 animate-fade-in">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold sm:text-xl">Command Center</h1>
          <p className="text-sm text-muted-foreground">
            Live operational monitoring over the governed audit log
            {report ? ` · ${report.total_interactions} interaction(s)` : ""}
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => refresh()}
          disabled={isLoading}
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" /> Refresh
        </Button>
      </header>

      {error ? (
        <ErrorState error={error} onRetry={() => refresh()} />
      ) : (
        <>
          {/* -------- metrics row: 1 / 2 / 4 columns -------- */}
          <section aria-labelledby="metrics-heading">
            <h2 id="metrics-heading" className="sr-only">
              Key performance indicators
            </h2>
            {isLoading || !report ? (
              <LoadingRegion
                label="Loading key performance indicators"
                className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
              >
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
                <MetricCardSkeleton />
              </LoadingRegion>
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="FAST Path %"
                  value={formatPercent(report.verification.fast_rate, 1)}
                  hint={`${report.verification.fast_count} of ${
                    report.verification.fast_count + report.verification.deep_count
                  } verifications`}
                  icon={Timer}
                  tone="positive"
                />
                <MetricCard
                  label="DEEP Path %"
                  value={formatPercent(report.verification.deep_rate, 1)}
                  hint={`${report.verification.deep_count} deep verification(s)`}
                  icon={Layers}
                  tone="warning"
                />
                <MetricCard
                  label="Bypass Savings"
                  value={formatDuration(
                    report.verification.estimated_bypass_compute_saved_ms,
                  )}
                  hint={`${report.verification.semantic_bypass_count} semantic bypass(es) · ${formatPercent(
                    report.verification.semantic_bypass_rate_of_deep,
                  )} of DEEP`}
                  icon={Gauge}
                  tone="positive"
                />
                <MetricCard
                  label="Critical Floor Escalations"
                  value={String(
                    report.multi_turn.sessions_hitting_critical_floor,
                  )}
                  hint={`${report.multi_turn.critical_floor_events} event(s) across ${report.multi_turn.multi_turn_sessions} multi-turn session(s)`}
                  icon={ShieldAlert}
                  tone={
                    report.multi_turn.sessions_hitting_critical_floor > 0
                      ? "critical"
                      : "default"
                  }
                />
              </div>
            )}
          </section>

          {/* -------- visualisation + decision mix: stack on mobile, split lg -------- */}
          <section
            aria-labelledby="distribution-heading"
            className="grid grid-cols-1 gap-4 lg:grid-cols-5"
          >
            <h2 id="distribution-heading" className="sr-only">
              Risk distribution and decision mix
            </h2>
            <div className="lg:col-span-3">
              {isLoading || !report ? (
                <RiskDistributionChartSkeleton />
              ) : (
                <RiskDistributionChart distribution={report.risk_distribution} />
              )}
            </div>
            <div className="lg:col-span-2">
              {isLoading || !report ? (
                <RiskDistributionChartSkeleton />
              ) : (
                <DecisionMixCard report={report} />
              )}
            </div>
          </section>

          <section aria-labelledby="incidents-heading">
            <h2 id="incidents-heading" className="sr-only">
              Flagged incidents
            </h2>
            {isLoading || !report ? (
              <RiskDistributionChartSkeleton />
            ) : (
              <IncidentTable incidents={report.incidents} />
            )}
          </section>
        </>
      )}
    </div>
  );
}

function DecisionMixCard({ report }: { report: OperationalMonitoringReport }) {
  const { snapshot } = report;
  const rows = [
    { label: "ALLOW", count: snapshot.allow_count, rate: snapshot.allow_rate },
    { label: "ANNOTATE", count: snapshot.annotate_count, rate: snapshot.annotate_rate },
    { label: "VERIFY", count: snapshot.verify_count, rate: snapshot.verify_rate },
    {
      label: "HUMAN_REVIEW",
      count: snapshot.human_review_count,
      rate: snapshot.human_review_rate,
    },
    { label: "BLOCK", count: snapshot.block_count, rate: snapshot.block_rate },
  ];

  return (
    <article className="flex h-full flex-col rounded-lg border border-border bg-card p-4 sm:p-5">
      <h3 className="text-sm font-semibold">Decision Mix</h3>
      <p className="text-xs text-muted-foreground">
        Intervention tiers over {report.total_interactions} interaction(s)
      </p>
      <ul className="mt-4 space-y-3">
        {rows.map((row) => (
          <li key={row.label} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium">{row.label}</span>
              <span className="tabular-nums text-muted-foreground">
                {row.count} · {formatPercent(row.rate)}
              </span>
            </div>
            <progress
              className="cp-meter h-1.5 w-full"
              value={row.rate}
              max={1}
              aria-label={`${row.label} share: ${formatPercent(row.rate)}`}
            />
          </li>
        ))}
      </ul>
    </article>
  );
}
