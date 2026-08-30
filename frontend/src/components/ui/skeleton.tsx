import { cn } from "@/lib/utils";

/**
 * A loading placeholder. Decorative by default (`aria-hidden`); wrap a group of
 * skeletons in an element with `role="status"` and an accessible label so
 * screen-reader users hear that content is loading.
 */
export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden="true"
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  );
}

/** Wrapper that announces a loading region to assistive tech. */
export function LoadingRegion({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div role="status" aria-busy="true" aria-live="polite" className={className}>
      <span className="sr-only">{label}</span>
      {children}
    </div>
  );
}
