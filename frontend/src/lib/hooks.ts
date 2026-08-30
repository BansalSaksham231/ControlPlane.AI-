"use client";

/**
 * SWR data hooks. Each wraps one api.ts function with a stable cache key,
 * sensible revalidation defaults and typed return shapes. Components read
 * `isLoading` to render skeletons and `error` (an ApiError) for error states.
 */

import useSWR, { SWRConfiguration } from "swr";

import {
  ApiError,
  getCommandCenterMetrics,
  getGovernanceHistory,
  getHealth,
  getInvestigation,
} from "./api";
import type {
  HealthResponse,
  IncidentInvestigation,
  OperationalMonitoringReport,
} from "./types";

const DEFAULTS: SWRConfiguration = {
  revalidateOnFocus: false,
  shouldRetryOnError: false,
  dedupingInterval: 5_000,
};

export function useHealth() {
  const { data, error, isLoading } = useSWR<HealthResponse, ApiError>(
    "health",
    getHealth,
    { ...DEFAULTS, refreshInterval: 30_000 },
  );
  return { health: data, error, isLoading };
}

export function useCommandCenterMetrics() {
  const { data, error, isLoading, mutate } = useSWR<
    OperationalMonitoringReport,
    ApiError
  >("command-center-metrics", getCommandCenterMetrics, {
    ...DEFAULTS,
    refreshInterval: 20_000,
  });
  return { report: data, error, isLoading, refresh: mutate };
}

export function useInvestigation(interactionId: string | null) {
  const { data, error, isLoading, mutate } = useSWR<
    IncidentInvestigation,
    ApiError
  >(
    interactionId ? ["investigation", interactionId] : null,
    () => getInvestigation(interactionId as string),
    DEFAULTS,
  );
  return { investigation: data, error, isLoading, refresh: mutate };
}

export function useGovernanceHistory(interactionId: string | null) {
  const { data, error, isLoading, mutate } = useSWR(
    interactionId ? ["governance-history", interactionId] : null,
    () => getGovernanceHistory(interactionId as string),
    DEFAULTS,
  );
  return { history: data, error, isLoading, refresh: mutate };
}
