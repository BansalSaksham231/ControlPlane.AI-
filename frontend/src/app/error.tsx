"use client";

import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/common/states";

export default function AppError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto max-w-lg space-y-4 py-10">
      <ErrorState error={error} title="Something went wrong" />
      <Button variant="outline" onClick={reset}>
        Try again
      </Button>
    </div>
  );
}
