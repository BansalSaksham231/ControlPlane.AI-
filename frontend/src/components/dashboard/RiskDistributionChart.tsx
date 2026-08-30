"use client";

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { RiskDistribution } from "@/lib/types";
import { RISK_BUCKET_COLOR, formatNumber } from "@/lib/format";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface RiskDistributionChartProps {
  distribution: RiskDistribution;
  /** Heading level for the card title, to keep the page outline correct. */
  headingLevel?: "h2" | "h3";
}

export function RiskDistributionChart({
  distribution,
  headingLevel = "h2",
}: RiskDistributionChartProps) {
  const rows = distribution.buckets.map((bucket) => ({
    name: bucket.bucket_name,
    count: bucket.count,
    range: `${bucket.min_risk.toFixed(2)}–${Math.min(bucket.max_risk, 1).toFixed(2)}`,
    sharePercent: bucket.percentage ?? 0,
  }));

  const summary = rows
    .map((r) => `${r.name}: ${r.count}`)
    .join(", ");

  return (
    <Card>
      <CardHeader>
        <CardTitle as={headingLevel}>Enterprise Risk Distribution</CardTitle>
        <CardDescription>
          {distribution.total} governed interaction(s), bucketed by overall risk
        </CardDescription>
      </CardHeader>
      <CardContent>
        <figure className="m-0">
          <div
            className="h-64 w-full"
            role="img"
            aria-label={`Bar chart of interaction count by risk bucket. ${summary}.`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={rows}
                margin={{ top: 8, right: 8, bottom: 8, left: -16 }}
              >
                <XAxis
                  dataKey="name"
                  tickLine={false}
                  axisLine={false}
                  fontSize={11}
                />
                <YAxis
                  allowDecimals={false}
                  tickLine={false}
                  axisLine={false}
                  fontSize={11}
                  width={32}
                />
                <Tooltip
                  cursor={{ fill: "hsl(var(--muted))", opacity: 0.4 }}
                  contentStyle={{
                    fontSize: 12,
                    borderRadius: 8,
                    border: "1px solid hsl(var(--border))",
                    background: "hsl(var(--card))",
                  }}
                  formatter={(value: number, _name, item) => [
                    `${value} interaction(s) · ${(item.payload.sharePercent as number).toFixed(1)}%`,
                    `risk ${item.payload.range}`,
                  ]}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={72}>
                  {rows.map((row) => (
                    <Cell
                      key={row.name}
                      fill={RISK_BUCKET_COLOR[row.name] ?? "#8892a6"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* accessible data table equivalent */}
          <figcaption className="sr-only">
            <table>
              <caption>Interaction count by risk bucket</caption>
              <thead>
                <tr>
                  <th scope="col">Risk bucket</th>
                  <th scope="col">Risk range</th>
                  <th scope="col">Interactions</th>
                  <th scope="col">Share</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name}>
                    <th scope="row">{row.name}</th>
                    <td>{row.range}</td>
                    <td>{row.count}</td>
                    <td>{formatNumber(row.sharePercent, 1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </figcaption>
        </figure>
      </CardContent>
    </Card>
  );
}

export function RiskDistributionChartSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-3 w-64" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-64 w-full" />
      </CardContent>
    </Card>
  );
}
