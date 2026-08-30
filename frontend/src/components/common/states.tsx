"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

import { ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export function ErrorState({
  error,
  onRetry,
  title = "Could not load data",
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const message =
    error instanceof ApiError
      ? `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`
      : error instanceof Error
        ? error.message
        : "Unknown error";

  return (
    <Card className="border-tier-block/30 bg-tier-block/5" role="alert">
      <CardContent className="flex flex-col items-start gap-3 p-5">
        <div className="flex items-center gap-2 text-tier-block">
          <AlertTriangle className="h-4 w-4" aria-hidden="true" />
          <span className="text-sm font-semibold">{title}</span>
        </div>
        <p className="text-xs text-muted-foreground">{message}</p>
        <p className="text-xs text-muted-foreground">
          Check that the FastAPI backend is running and{" "}
          <code className="rounded bg-muted px-1">NEXT_PUBLIC_API_URL</code> is
          correct.
        </p>
        {onRetry && (
          <Button size="sm" variant="outline" onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" /> Retry
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <Card>
      <CardContent className="p-8 text-center text-sm text-muted-foreground">
        {message}
      </CardContent>
    </Card>
  );
}
