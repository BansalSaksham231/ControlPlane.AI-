import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * `cn` — merge conditional class names, de-duplicating conflicting Tailwind
 * utilities (the later class wins).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// Formatters and the tier palette live in ./format. Re-exported here so existing
// `@/lib/utils` imports keep working; prefer importing from `@/lib/format`.
export {
  formatPercent,
  formatNumber,
  formatDuration,
  formatTimestamp,
  toDateTimeAttr,
  tierBadgeClasses,
  TIER_COLOR,
  RISK_BUCKET_COLOR,
} from "./format";
