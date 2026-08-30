import { MetricCardSkeleton } from "@/components/dashboard/MetricCard";
import { RiskDistributionChartSkeleton } from "@/components/dashboard/RiskDistributionChart";
import { LoadingRegion, Skeleton } from "@/components/ui/skeleton";

/** Route-level Suspense fallback for the dashboard segment. */
export default function DashboardLoading() {
  return (
    <LoadingRegion label="Loading the Command Center" className="space-y-6">
      <Skeleton className="h-8 w-56" />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
        <MetricCardSkeleton />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <RiskDistributionChartSkeleton />
        </div>
        <div className="lg:col-span-2">
          <RiskDistributionChartSkeleton />
        </div>
      </div>
    </LoadingRegion>
  );
}
