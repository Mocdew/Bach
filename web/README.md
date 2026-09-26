# Recoup Console — web

React + TypeScript + Vite single-page app over the `recoup` console API.

```
recoup/api/schemas.py  ──(FastAPI)──►  openapi.json  ──(openapi-typescript)──►  src/api/schema.d.ts
      ML side owns                        committed                                 frontend compiles against
```

## Who owns what

| | owner | lives in |
|---|---|---|
| response shapes (the contract) | agreed together, written by ML | `recoup/api/schemas.py` |
| anything that computes — reads jobs' files, re-runs the planner | ML | `recoup/api/service.py` |
| routes | ML (thin: validate, call service, map errors) | `recoup/api/app.py` |
| everything that displays or navigates | frontend | `web/src/` |

A schema change is a PR that touches `schemas.py` **and** the regenerated
`openapi.json` + `src/api/schema.d.ts`. `tests/test_ops.py::test_api_contract_matches_the_generated_frontend_types`
fails if they drift, so the frontend never compiles against a contract the
server no longer honours.

## Run

```bash
pip install -e ..[api]                                  # once, from the repo root
python -m recoup.ops serve --workdir ../var/demo        # API on :8080 (after `recoup-ops demo`)
npm ci
npm run dev                                             # UI on :5173, /api proxied to :8080
```

Other scripts:

| script | does |
|---|---|
| `npm run gen:api` | regenerate `openapi.json` + `src/api/schema.d.ts` from Python (`PYTHON=` to pick the interpreter) |
| `npm run typecheck` | `tsc -b` |
| `npm run build` | typecheck + build to `dist/` — `recoup-ops serve` serves it at `/` |

No data yet? `recoup-ops demo --fresh` from the repo root makes a synthetic
world and runs 45 simulated days of the rollout (≈35 min); `recoup-ops synth &&
recoup-ops retrain` is enough for a first look (≈1 min).

## Stack, and why

- **Vite + React + TS**, no SSR: one operator on a LAN/Tailscale; the
  production server is a Python process on an old laptop, so no Node there.
- **openapi-fetch** + generated types: typed requests and responses, zero
  hand-written API types.
- **TanStack Query**: caching and polling. The batch jobs run hourly, so
  queries refetch every 60 s; the simulation status polls every 1.5 s only
  while a step is running.
- **TanStack Table v8** (headless) for the two server-paged tables. v9 had
  just shipped a new API when this was built; v8 is the documented,
  widely-known one.
- **Observable Plot** for charts. Colours come from the CSS tokens at render
  time (`src/theme.ts`), so charts follow light/dark with the page.
- **Plain CSS on the operator console's tokens** (`deploy/dashboard/index.html`)
  rather than Tailwind: the design system already existed as CSS variables.
- **Hash routing** (`src/router.ts`): the API serves `dist/` as static files,
  so `#/invoices/inv_123` survives a reload with no server rewrites.

## Screens

| route | shows | reads |
|---|---|---|
| `#/overview` | gate verdict + interval + ESS, KPIs, gate timeline, where first retries landed by logging mode, decisions per day | `/api/overview`, `/api/gate/history`, `/api/decisions/coverage`, `/api/decisions/daily` |
| `#/decisions` | the audit log, filterable by mode / policy / action / attempt / invoice | `/api/decisions` |
| `#/invoices[/id]` | invoices by status; drawer with retry timeline, the planner's best schedule, value and Thompson probability per first bucket, logged decisions | `/api/invoices`, `/api/invoices/{id}` |
| `#/model` | holdout metrics, coefficients with ±1.96·SE, versions | `/api/model*` |
| `#/simulation` | synthetic only: advance the clock (plan + execute), realised vs the ladder's oracle value | `/api/sim*` |

The API's interactive docs are at `/docs` on the running server.
