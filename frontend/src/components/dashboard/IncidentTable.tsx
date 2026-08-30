import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

import type { IncidentSummary } from "@/lib/types";
import {
  cn,
  formatNumber,
  formatTimestamp,
  tierBadgeClasses,
  toDateTimeAttr,
} from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const severityBadgeClasses: Record<string, string> = {
  CRITICAL: "bg-tier-block/15 text-tier-block border-tier-block/30",
  HIGH: "bg-tier-review/15 text-tier-review border-tier-review/30",
  MEDIUM: "bg-tier-verify/15 text-tier-verify border-tier-verify/30",
};

interface IncidentTableProps {
  incidents: IncidentSummary[];
  headingLevel?: "h2" | "h3";
}

export function IncidentTable({
  incidents,
  headingLevel = "h2",
}: IncidentTableProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as={headingLevel}>Flagged Incidents</CardTitle>
        <CardDescription>
          Interactions crossing an operational attention threshold — open one to
          investigate
        </CardDescription>
      </CardHeader>
      <CardContent>
        {incidents.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            No incidents in the current window.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <caption className="sr-only">
                Flagged incidents, most recent first
              </caption>
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    Interaction
                  </th>
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    Application
                  </th>
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    Decision
                  </th>
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    Risk
                  </th>
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    Severity
                  </th>
                  <th scope="col" className="pb-2 pr-4 font-medium">
                    When
                  </th>
                  <th scope="col" className="pb-2">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {incidents.map((incident) => (
                  <tr
                    key={incident.interaction_id}
                    className="border-b border-border/60 last:border-0"
                  >
                    <th
                      scope="row"
                      className="py-2.5 pr-4 text-left font-mono text-xs font-normal"
                    >
                      {incident.interaction_id}
                    </th>
                    <td className="py-2.5 pr-4 text-muted-foreground">
                      {incident.application}
                    </td>
                    <td className="py-2.5 pr-4">
                      <Badge className={tierBadgeClasses(incident.decision)}>
                        {incident.decision}
                      </Badge>
                    </td>
                    <td className="py-2.5 pr-4 tabular-nums">
                      {formatNumber(incident.overall_risk)}
                    </td>
                    <td className="py-2.5 pr-4">
                      <Badge
                        className={cn(
                          severityBadgeClasses[incident.severity] ??
                            "bg-muted text-muted-foreground",
                        )}
                      >
                        {incident.severity}
                      </Badge>
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-muted-foreground">
                      <time dateTime={toDateTimeAttr(incident.timestamp)}>
                        {formatTimestamp(incident.timestamp)}
                      </time>
                    </td>
                    <td className="py-2.5 text-right">
                      <Link
                        href={`/incident/${encodeURIComponent(incident.interaction_id)}`}
                        aria-label={`Investigate incident ${incident.interaction_id}`}
                        className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
                      >
                        Investigate{" "}
                        <ArrowUpRight className="h-3 w-3" aria-hidden="true" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
