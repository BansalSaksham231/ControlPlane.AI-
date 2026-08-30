/**
 * ControlPlane.ai API client.
 *
 * A single configured Axios instance with:
 *   - base URL from NEXT_PUBLIC_API_URL (or same-origin "/api" when a BFF
 *     proxy is configured via API_PROXY_TARGET in next.config.mjs)
 *   - a request interceptor that injects the X-API-Key header
 *   - a response interceptor that normalises FastAPI error payloads
 *     ({ "detail": ... }) into a typed ApiError
 *
 * SECURITY NOTE (Principal Architect): NEXT_PUBLIC_API_KEY ships in the browser
 * bundle. For anything beyond local dev, set API_PROXY_TARGET + a server-only
 * CONTROLPLANE_API_KEY and let the Next.js rewrite inject the header — the
 * client then talks to same-origin "/api" and no secret reaches the browser.
 */

import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  InternalAxiosRequestConfig,
} from "axios";

import type {
  AuditTrace,
  CheckRequest,
  CheckResponse,
  GovernanceAction,
  GovernanceOverrideRequest,
  GovernanceOverrideResponse,
  HealthResponse,
  IncidentInvestigation,
  OperationalMonitoringReport,
} from "./types";

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

const RAW_BASE = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
/** Same-origin proxy is used when NEXT_PUBLIC_API_URL is not set. */
export const API_BASE_URL = RAW_BASE || "/api";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

// ---------------------------------------------------------------------------
// typed error
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function toApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const axErr = error as AxiosError<{ detail?: unknown }>;
    const status = axErr.response?.status ?? 0;
    const detail = axErr.response?.data?.detail ?? axErr.message;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ")
          : `Request failed with status ${status}`;
    return new ApiError(message, status, detail);
  }
  return new ApiError(
    error instanceof Error ? error.message : "Unknown error",
    0,
    error,
  );
}

// ---------------------------------------------------------------------------
// axios instance + interceptors
// ---------------------------------------------------------------------------

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20_000,
  headers: { "Content-Type": "application/json" },
});

/** @deprecated use {@link apiClient} */
export const http = apiClient;

apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  if (API_KEY) {
    config.headers.set("X-API-Key", API_KEY);
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(toApiError(error)),
);

async function requestJson<T>(config: AxiosRequestConfig): Promise<T> {
  const { data } = await apiClient.request<T>(config);
  return data;
}

// ---------------------------------------------------------------------------
// endpoints
// ---------------------------------------------------------------------------

/** GET /health */
export function getHealth(): Promise<HealthResponse> {
  return requestJson({ url: "/health", method: "GET" });
}

/** POST /check — governs a single AI interaction. */
export function checkInteraction(body: CheckRequest): Promise<CheckResponse> {
  return requestJson({ url: "/check", method: "POST", data: body });
}

/**
 * GET /monitoring/operational — the full OperationalMonitoringReport
 * (FAST/DEEP split, semantic-bypass savings, multi-turn critical floor,
 * risk distribution, incident digest).
 */
export function getCommandCenterMetrics(): Promise<OperationalMonitoringReport> {
  return requestJson({ url: "/monitoring/operational", method: "GET" });
}

/** GET /audit/{id} — the redacted decision trace summary for one interaction. */
export function getAuditTrace(interactionId: string): Promise<AuditTrace> {
  return requestJson({
    url: `/audit/${encodeURIComponent(interactionId)}`,
    method: "GET",
  });
}

/**
 * GET /investigation/{id} — the full incident investigation: immutable replay,
 * explainability, session memory and the append-only governance history.
 */
export function getInvestigation(
  interactionId: string,
): Promise<IncidentInvestigation> {
  return requestJson({
    url: `/investigation/${encodeURIComponent(interactionId)}`,
    method: "GET",
  });
}

/** GET /governance/actions/{id} — append-only governance history. */
export function getGovernanceHistory(
  interactionId: string,
): Promise<{
  interaction_id: string;
  original_decision: string;
  effective_governed_decision: string;
  is_overridden: boolean;
  history: GovernanceAction[];
}> {
  return requestJson({
    url: `/governance/actions/${encodeURIComponent(interactionId)}`,
    method: "GET",
  });
}

/**
 * POST /governance/action — record a human override on the append-only
 * governance track. The automated DecisionTrace is never mutated.
 */
export function postGovernanceOverride(
  body: GovernanceOverrideRequest,
): Promise<GovernanceOverrideResponse> {
  return requestJson({ url: "/governance/action", method: "POST", data: body });
}

export const api = {
  getHealth,
  checkInteraction,
  getCommandCenterMetrics,
  getAuditTrace,
  getInvestigation,
  getGovernanceHistory,
  postGovernanceOverride,
};

export type Api = typeof api;
