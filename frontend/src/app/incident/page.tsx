"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Search } from "lucide-react";

import { useCommandCenterMetrics } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/field";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ErrorState } from "@/components/common/states";
import { IncidentTable } from "@/components/dashboard/IncidentTable";
import { LoadingRegion, Skeleton } from "@/components/ui/skeleton";

export default function IncidentIndexPage() {
  const router = useRouter();
  const [interactionId, setInteractionId] = useState("");
  const { report, error, isLoading, refresh } = useCommandCenterMetrics();

  return (
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-lg font-semibold sm:text-xl">Incident Investigation</h1>
        <p className="text-sm text-muted-foreground">
          Open a flagged incident, or look up any interaction by ID
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle as="h2">Look up an interaction</CardTitle>
          <CardDescription>
            Enter an interaction ID to open its investigation workspace
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              const trimmed = interactionId.trim();
              if (trimmed)
                router.push(`/incident/${encodeURIComponent(trimmed)}`);
            }}
          >
            <Label htmlFor="interaction-id-lookup" className="sr-only">
              Interaction ID
            </Label>
            <Input
              id="interaction-id-lookup"
              value={interactionId}
              onChange={(event) => setInteractionId(event.target.value)}
              placeholder="e.g. SEED-02"
              className="font-mono"
            />
            <Button type="submit" className="shrink-0">
              <Search className="h-4 w-4" aria-hidden="true" /> Investigate
            </Button>
          </form>
        </CardContent>
      </Card>

      {error ? (
        <ErrorState error={error} onRetry={() => refresh()} />
      ) : isLoading || !report ? (
        <LoadingRegion label="Loading incidents">
          <Skeleton className="h-64 w-full" />
        </LoadingRegion>
      ) : (
        <IncidentTable incidents={report.incidents} headingLevel="h2" />
      )}
    </div>
  );
}
