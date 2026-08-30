/**
 * BFF proxy — the recommended production path.
 *
 * When the client base URL is "/api" (i.e. NEXT_PUBLIC_API_URL is unset), every
 * request lands here. This route runs on the server, so it can attach the
 * secret CONTROLPLANE_API_KEY without ever exposing it to the browser.
 *
 * Configure:
 *   API_PROXY_TARGET=http://api:8000     (upstream FastAPI base URL, server-only)
 *   CONTROLPLANE_API_KEY=<secret>        (server-only; optional)
 *
 * If API_PROXY_TARGET is unset this returns 503 with a hint, so a misconfigured
 * deployment fails loudly instead of silently hitting the wrong host.
 */

import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const UPSTREAM = process.env.API_PROXY_TARGET?.replace(/\/$/, "");
const API_KEY = process.env.CONTROLPLANE_API_KEY ?? "";

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

async function proxy(req: NextRequest, path: string[]): Promise<NextResponse> {
  if (!UPSTREAM) {
    return NextResponse.json(
      { detail: "API_PROXY_TARGET is not configured on the frontend server." },
      { status: 503 },
    );
  }

  const search = req.nextUrl.search;
  const url = `${UPSTREAM}/${path.map(encodeURIComponent).join("/")}${search}`;

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers.set(key, value);
  });
  if (API_KEY) headers.set("X-API-Key", API_KEY);

  const method = req.method.toUpperCase();
  const body =
    method === "GET" || method === "HEAD" ? undefined : await req.arrayBuffer();

  const upstreamRes = await fetch(url, {
    method,
    headers,
    body,
    redirect: "manual",
    cache: "no-store",
  });

  const resHeaders = new Headers();
  upstreamRes.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) resHeaders.set(key, value);
  });

  return new NextResponse(upstreamRes.body, {
    status: upstreamRes.status,
    statusText: upstreamRes.statusText,
    headers: resHeaders,
  });
}

type Ctx = { params: { path: string[] } };

export const GET = (req: NextRequest, { params }: Ctx) => proxy(req, params.path);
export const POST = (req: NextRequest, { params }: Ctx) => proxy(req, params.path);
export const PUT = (req: NextRequest, { params }: Ctx) => proxy(req, params.path);
export const PATCH = (req: NextRequest, { params }: Ctx) =>
  proxy(req, params.path);
export const DELETE = (req: NextRequest, { params }: Ctx) =>
  proxy(req, params.path);
