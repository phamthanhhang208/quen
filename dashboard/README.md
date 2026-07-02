# Quên · memory — dashboard

Plain admin dashboard for the Quên memory engine (spec §7). Five panels:
Memories (table + detail), Dream log, Recall trace, Vitals.

Stack: Vite + React 18 + TypeScript + Tailwind CSS 3.4 + hand-vendored
shadcn-style components + recharts + SWR (3s polling). Default shadcn zinc
look, light theme, no animations.

## Run

```sh
cd dashboard
npm install
npm run dev          # http://localhost:5173
```

The UI talks to the engine at `VITE_API_BASE` (default
`http://localhost:8000`). Start the engine first:

```sh
uvicorn quen.api:create_app --factory --port 8000
```

To point at another engine:

```sh
VITE_API_BASE=http://my-host:9000 npm run dev
```

Production build (type-checks first):

```sh
npm run build        # tsc -b && vite build → dist/
```

## Dev proxy note

`vite.config.ts` ships a dev proxy mapping `/api/*` → `http://localhost:8000`
(prefix stripped). If the engine does not enable CORS, set
`VITE_API_BASE=/api` and the browser stays same-origin:

```sh
VITE_API_BASE=/api npm run dev
```

## Implementation notes

- `src/lib/types.ts` is the FROZEN API contract mirroring `src/quen/api.py`
  DTOs — do not edit it independently.
- `src/lib/fsrs.ts` mirrors only the FSRS-4.5 forgetting curve
  (`retrievability`, `daysUntilRetrievability`); constants come from
  `GET /config` with hardcoded fallbacks. Live-R cells re-render every second
  (`useNow`) so decay is visible.
- `src/components/ui/` are hand-vendored shadcn-style components (zinc
  Tailwind classes + cva + `cn()`); **no Radix** — Tabs and Switch are plain
  controlled React components with the shadcn API surface.
- The engine being down is handled everywhere via SWR error state (subtle
  "engine offline" banner); panels render friendly empty states.
