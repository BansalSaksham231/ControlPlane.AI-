"use client";

import { Menu, CircleCheck, CircleAlert, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import { useHealth } from "@/lib/hooks";

export function Header({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  const { health, error, isLoading } = useHealth();

  const status: "ok" | "down" | "loading" = isLoading
    ? "loading"
    : error || health?.status !== "ok"
      ? "down"
      : "ok";

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <button
        type="button"
        onClick={onOpenSidebar}
        aria-label="Open navigation menu"
        className="rounded-md p-2 text-muted-foreground hover:bg-muted lg:hidden"
      >
        <Menu className="h-5 w-5" aria-hidden="true" />
      </button>

      <div className="flex flex-col leading-tight">
        <span className="text-sm font-semibold">Enterprise AI Risk Command Center</span>
        <span className="hidden text-[11px] text-muted-foreground sm:block">
          Real-time governance over model interactions
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2 text-xs">
        <span
          role="status"
          aria-live="polite"
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-medium",
            status === "ok" && "border-tier-allow/30 bg-tier-allow/10 text-tier-allow",
            status === "down" && "border-tier-block/30 bg-tier-block/10 text-tier-block",
            status === "loading" && "border-border bg-muted text-muted-foreground",
          )}
        >
          {status === "loading" && (
            <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
          )}
          {status === "ok" && <CircleCheck className="h-3 w-3" aria-hidden="true" />}
          {status === "down" && <CircleAlert className="h-3 w-3" aria-hidden="true" />}
          {status === "loading"
            ? "Connecting…"
            : status === "ok"
              ? `API online · v${health?.version ?? "?"}`
              : "API unreachable"}
        </span>
        {health && status === "ok" && (
          <span className="hidden text-muted-foreground md:inline">
            {health.checks_served} checks · {health.active_sessions} sessions
          </span>
        )}
      </div>
    </header>
  );
}
