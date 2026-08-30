# ControlPlane.ai — Frontend

Production-ready enterprise dashboard for the ControlPlane.ai governance engine.
**Next.js 14 (App Router) · TypeScript · Tailwind CSS · SWR · Recharts.**

## Quick start

```bash
cd frontend
cp .env.local.example .env.local        # point NEXT_PUBLIC_API_URL at the FastAPI backend
npm install
npm run dev                             # http://localhost:3000  (redirects to /dashboard)
```

The FastAPI backend must be running (default `http://127.0.0.1:8000`):

```bash
# from the repo root
CONTROLPLANE_PERSISTENCE=1 CONTROLPLANE_CORS_ORIGINS="*" \
  python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## Directory structure

```
frontend/
├── package.json                 # next, react, typescript, tailwindcss, swr, recharts, lucide-react, axios
├── next.config.mjs              # output: "standalone" for slim Docker
├── tailwind.config.ts          # token-based theme + decision-tier semantic palette
├── tsconfig.json               # strict, @/* -> src/*
├── .env.local.example
└── src/
    ├── app/
    │   ├── layout.tsx           # <AppShell> + <SWRProvider>, font, metadata
    │   ├── globals.css          # CSS custom-property design tokens (light + dark)
    │   ├── page.tsx             # redirect -> /dashboard
    │   ├── error.tsx            # route error boundary
    │   ├── not-found.tsx
    │   ├── api/[...path]/route.ts   # BFF proxy (server-side X-API-Key injection)
    │   ├── dashboard/
    │   │   ├── page.tsx         # Command Center (Task 3)
    │   │   └── loading.tsx      # Suspense skeleton fallback
    │   ├── incident/
    │   │   ├── page.tsx         # incident list + ID lookup
    │   │   └── [id]/
    │   │       ├── page.tsx     # Investigation workspace (Task 4)
    │   │       └── loading.tsx
    │   └── check/page.tsx       # Live Governance console (POST /check)
    ├── components/
    │   ├── shell/               # AppShell, Sidebar (collapsible/drawer), Header (global status)
    │   ├── ui/                  # card, button, badge, skeleton, field (shadcn-style primitives)
    │   ├── common/states.tsx    # ErrorState / EmptyState
    │   ├── dashboard/           # MetricCard, RiskDistributionChart, DecisionMix, IncidentTable
    │   └── incident/            # DecisionHeader, ContextPanel, GovernanceForm
    ├── lib/
    │   ├── types.ts             # TS contracts mirroring the Python schemas
    │   ├── api.ts               # Axios client + interceptors + typed wrappers
    │   ├── hooks.ts             # SWR data hooks
    │   └── utils.ts             # cn(), formatters, tier palette
    └── providers/swr-provider.tsx
```

## Responsive design

Mobile-first. Every layout uses Tailwind breakpoint prefixes (`sm:` `md:` `lg:`):

| Region                | mobile          | `sm` (≥640) | `lg` (≥1024)          |
| --------------------- | --------------- | ----------- | --------------------- |
| Sidebar               | hamburger drawer| drawer      | persistent + collapsible rail |
| Dashboard metric row  | 1 col           | 2 col       | 4 col                 |
| Dashboard chart + mix | stacked         | stacked     | 3/5 + 2/5 split       |
| Incident workspace    | 1 col           | 1 col       | 3fr / 2fr two-column  |
| Tables                | horizontal scroll container |  |                       |

## API key / security

`src/lib/api.ts` injects `X-API-Key` from `NEXT_PUBLIC_API_KEY` via an Axios
request interceptor. **`NEXT_PUBLIC_*` is inlined into the browser bundle** — fine
for local dev where the backend has no key, but for production set:

```bash
API_PROXY_TARGET=http://api:8000     # server-only
CONTROLPLANE_API_KEY=<secret>        # server-only
# leave NEXT_PUBLIC_API_URL unset -> client uses same-origin "/api"
```

The client then talks to `/api/*`, `src/app/api/[...path]/route.ts` proxies to the
backend and attaches the key server-side. No secret reaches the browser.

## Docker

```bash
# from repo root — builds api + ui (Streamlit) + web (this app) + postgres
docker compose up --build
#   web -> http://localhost:3000   api -> http://localhost:8000   ui -> http://localhost:8501
```

## Scripts

| command             | purpose                        |
| ------------------- | ------------------------------ |
| `npm run dev`       | dev server (HMR)               |
| `npm run build`     | production build               |
| `npm start`         | serve the production build     |
| `npm run typecheck` | `tsc --noEmit`                 |
| `npm run lint`      | `next lint`                    |
