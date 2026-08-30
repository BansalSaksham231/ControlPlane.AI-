"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";

import { useInvestigation } from "@/lib/hooks";
import { LoadingRegion, Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";
import { ErrorState } from "@/components/common/states";
import { DecisionHeader } from "@/components/incident/DecisionHeader";
import { ContextPanel } from "@/components/incident/ContextPanel";
import { GovernanceForm } from "@/components/incident/GovernanceForm";

export default function IncidentPage() {
  const params = useParams<{ id: string }>();
  const interactionId = decodeURIComponent(params.id);
  const { investigation, error, isLoading, refresh } =
    useInvestigation(interactionId);

  return (
    <div className="space-y-5 animate-fade-in">
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" /> Back to Command
        Center
      </Link>

      {error ? (
        <ErrorState
          error={error}
          onRetry={() => refresh()}
          title={`Could not load investigation for ${interactionId}`}
        />
      ) : isLoading || !investigation ? (
        <LoadingRegion label={`Loading investigation for ${interactionId}`}>
          <IncidentSkeleton />
        </LoadingRegion>
      ) : (
        <>
          <DecisionHeader
            interactionId={investigation.interaction_id}
            originalDecision={investigation.original_decision}
            effectiveDecision={investigation.effective_governed_decision}
            isOverridden={investigation.is_overridden}
          />

          {/* two-column: stacks below lg */}
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[3fr_2fr]">
            <section aria-labelledby="context-heading">
              <h2 id="context-heading" className="sr-only">
                Interaction context
              </h2>
              <ContextPanel investigation={investigation} />
            </section>
            <section aria-labelledby="governance-heading">
              <h2 id="governance-heading" className="sr-only">
                Governance and override
              </h2>
              <GovernanceForm
                investigation={investigation}
                onSubmitted={() => refresh()}
              />
            </section>
          </div>

          <Card>
            <CardContent className="p-4 text-[11px] leading-relaxed text-muted-foreground">
              <ul className="space-y-1">
                {investigation.notes.map((note, index) => (
                  <li key={index}>• {note}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function IncidentSkeleton() {
  return (
    <div className="space-y-5">
      <Skeleton className="h-40 w-full" />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[3fr_2fr]">
        <div className="space-y-4">
          <Skeleton className="h-56 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
        <div className="space-y-4">
          <Skeleton className="h-72 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      </div>
    </div>
  );
}
