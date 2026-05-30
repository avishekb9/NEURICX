# NEURICX — Live Economic Intelligence (thin vertical slice, Phase 1)

Real backend substance for the NEURICX prototype (previously a skeleton web design
with no computability). This is a **live-data service**, not a static page.

## What it does (today, real)

```
GDELT DOC 2.0 (keyless live news)
   → Gemini 2.5 Flash classification (tango-filter relevance pattern)
       → contagion channel (Trade/Financial/Geopolitical/Behavioural/Monetary Policy)
       → relevance 0–1 · sentiment (risk-on/off/neutral) · location + lat/lon
   → ranked feed + per-channel summary + risk-off "stress index"
   → JSON API + dark dashboard
```

Channel taxonomy is identical to the **contagion-channels** framework, so news
classification lines up with the research portfolio's structural analysis.

## Run

```bash
cd ivy-fineco/neuricx/server
node neuricx-server.mjs          # reads GOOGLE_API_KEY from ../../../.env.local
# dashboard: http://127.0.0.1:3100/
# api:       http://127.0.0.1:3100/api/neuricx/intel?q=(trade OR tariff)&n=25&days=3
# health:    http://127.0.0.1:3100/health
```

Env: `PORT` (3100) · `HOST` (127.0.0.1) · `NEURICX_CACHE_TTL_MIN` (30).

## Design decisions (Karpathy-aligned)

- **GDELT, not NewsAPI** — keyless, real, current, no key dependency. OR'd terms
  must be wrapped in `()`; rate-limited to 1 req / 5 s (both handled in
  `fetchGdelt`).
- **Gemini geotag, not Maps** — the project `GOOGLE_API_KEY` is restricted to
  `aiplatform` + `generativelanguage`; Maps Geocode returns `REQUEST_DENIED`
  (verified 2026-05-30). When a Maps key is provisioned, replace the lat/lon
  fields in `classifyArticles` with a real geocode call — the dashboard already
  renders coordinates.
- **`node:https` only** — no npm install, no framework. One file each
  (`server/neuricx-server.mjs`, `web/index.html`).
- **Standalone on :3100** — deliberately NOT folded into the 11k-line VERALABS
  proxy (separation of concerns per the mjstudio security-lockdown).
- **Graceful degradation** — Gemini failure → unclassified articles still shown
  + warning. Never crashes.
- **Disk-cached last-good pull** — every successful pull is mirrored to
  `server/.cache/last-good.json`. It loads on startup (warm after restart) and is
  served as `stale:true` (amber badge) when a live fetch fails (e.g. GDELT 429),
  so the dashboard shows the last good data instead of zeros. Tune freshness with
  `NEURICX_CACHE_TTL_MIN` (default 30).

## NOT deployable to GitHub Pages

This needs a running Node process + the `GOOGLE_API_KEY` server-side. Do **not**
push the server or key to the public `econstellar` repo. The public showcase
links to it as a "live demo (requires local backend)" or it gets hosted on a
real VM/Cloud Run later (see Phase 2).

## Phase 2 (deferred — needs PI go-ahead + keys)

- Real Maps/Geolocation key → swap Gemini geotag for Geocoding API + render an
  actual map (Leaflet/MapLibre).
- Financial market data (FRED/AlphaVantage/yfinance) → overlay news events on
  the G20/commodity series the frameworks analyze.
- Vertex agentic (not just Gemini REST) → multi-step "given this news cluster,
  which contagion channel is activating and which markets are exposed?"
- Image-gen (Imagen 4) → generate dashboard UI/UX design variants for polish.
- Caching to disk + scheduled cron refresh → always-warm snapshot.
- Wire into the research-engine snapshot as a `live_intel` section.

## Files

```
neuricx/
├── server/neuricx-server.mjs   # GDELT + Gemini pipeline, JSON API, static host
└── web/index.html              # dark dashboard (KPIs · channels · ranked feed)
```
