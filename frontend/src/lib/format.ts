/**
 * Presentation formatters — pure, null-safe, locale-aware where it matters.
 *
 * Every formatter renders a stable em-dash ("—") for missing values so the UI
 * never shows "null", "NaN" or "undefined".
 */

import type { InterventionTier } from "./types";

const EMPTY = "—";

function isMissing(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || Number.isNaN(value);
}

/** A 0–1 ratio as a percentage, e.g. `formatPercent(0.421, 1)` -> "42.1%". */
export function formatPercent(value: number | null | undefined, fractionDigits = 0): string {
  return isMissing(value) ? EMPTY : `${(value * 100).toFixed(fractionDigits)}%`;
}

/** A plain number with fixed precision, e.g. risk scores. */
export function formatNumber(value: number | null | undefined, fractionDigits = 2): string {
  return isMissing(value) ? EMPTY : value.toFixed(fractionDigits);
}

/** A millisecond duration, promoted to seconds past 1000ms. */
export function formatDuration(milliseconds: number | null | undefined): string {
  if (isMissing(milliseconds)) return EMPTY;
  return milliseconds >= 1000
    ? `${(milliseconds / 1000).toFixed(2)} s`
    : `${milliseconds.toFixed(1)} ms`;
}

/** An ISO-8601 timestamp in the viewer's locale; echoes the input if unparseable. */
export function formatTimestamp(isoTimestamp: string | null | undefined): string {
  if (!isoTimestamp) return EMPTY;
  const parsed = new Date(isoTimestamp);
  return Number.isNaN(parsed.getTime()) ? isoTimestamp : parsed.toLocaleString();
}

/** Machine-readable value for a `<time dateTime>` attribute (falls back to the raw string). */
export function toDateTimeAttr(isoTimestamp: string | null | undefined): string | undefined {
  if (!isoTimestamp) return undefined;
  const parsed = new Date(isoTimestamp);
  return Number.isNaN(parsed.getTime()) ? isoTimestamp : parsed.toISOString();
}

// ---------------------------------------------------------------------------
// decision-tier palette
// ---------------------------------------------------------------------------

/** Tailwind classes for a decision-tier badge (semantic colour, not the accent). */
export function tierBadgeClasses(tier: InterventionTier | string): string {
  switch (tier) {
    case "ALLOW":
      return "bg-tier-allow/15 text-tier-allow border-tier-allow/30";
    case "ANNOTATE":
      return "bg-tier-annotate/15 text-tier-annotate border-tier-annotate/30";
    case "VERIFY":
      return "bg-tier-verify/15 text-tier-verify border-tier-verify/40";
    case "HUMAN_REVIEW":
      return "bg-tier-review/15 text-tier-review border-tier-review/40";
    case "BLOCK":
      return "bg-tier-block/15 text-tier-block border-tier-block/40";
    default:
      return "bg-muted text-muted-foreground border-border";
  }
}

/** Hex values for charts / SVG fills, keyed by decision tier. */
export const TIER_COLOR: Record<string, string> = {
  ALLOW: "#31ab74",
  ANNOTATE: "#2496c9",
  VERIFY: "#f5a201",
  HUMAN_REVIEW: "#ee6f28",
  BLOCK: "#dc3a3a",
};

/** Hex values for charts, keyed by risk-distribution bucket name. */
export const RISK_BUCKET_COLOR: Record<string, string> = {
  LOW: "#31ab74",
  MODERATE: "#f5a201",
  HIGH: "#ee6f28",
  CRITICAL: "#dc3a3a",
};
