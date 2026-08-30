import { Skeleton } from "@/components/ui/skeleton";

export default function IncidentLoading() {
  return (
    <div className="space-y-5">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-40 w-full" />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[3fr_2fr]">
        <Skeleton className="h-96 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    </div>
  );
}
