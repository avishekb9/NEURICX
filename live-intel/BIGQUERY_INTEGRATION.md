# NEURICX → BigQuery data-feed warehouse (2026-06-01)

Turns the ephemeral Gemini-classified news feed into a queryable longitudinal
research dataset. Built + verified end-to-end against real BigQuery this session.

## What was added
- **`server/bq.mjs`** (zero npm deps, `node:https`): OAuth via Cloud Run metadata
  server (default SA) with `GOOGLE_OAUTH_TOKEN` env fallback for local runs;
  auto-creates dataset `neuricx` + tables `articles`, `snapshots` (idempotent);
  `persistRun(payload)` streaming-inserts each good pull; `channelHistory({days,query})`
  reads back the channel-intensity time series. All failures are non-fatal/logged —
  the news pipeline never breaks because of a warehouse hiccup.
- **`server/neuricx-server.mjs`**: imports bq.mjs; calls `persistRun(payload)`
  fire-and-forget right after a good pull is cached; new route
  **`GET /api/neuricx/history?days=30&q=...`** returns the warehoused time series.

## Verified (real BigQuery, project hopeful-flash-485308-v3, dataset `neuricx`)
- persistRun → `{"ok":true, articles:2, snapshots:5}`; dataset+tables auto-created.
- channelHistory(days=1) → 5 rows, correct values (Trade n=1 rel=0.8 risk_off=1; Financial n=1 rel=0.5).
- Cost: GDELT-BQ dry-run = 0 bytes; insertAll streaming inserts are well within free tier.
- Auth note: BigQuery REST needs an **OAuth bearer token, NOT the API key** the rest
  of NEURICX uses for Gemini/Maps. (gcloud-POST TLS bug does NOT affect BigQuery REST.)

## GO-LIVE — DONE & VERIFIED (2026-06-01)
1. **IAM granted.** SA `672903689767-compute@developer.gserviceaccount.com` now holds
   `roles/bigquery.dataEditor` + `roles/bigquery.jobUser` (verified in the live policy).
2. **Deployed.** Cloud Run revision **neuricx-intel-00006-msg**, 100% traffic.
3. **Live-verified:** `GET https://neuricx-intel-672903689767.asia-south1.run.app/api/neuricx/history?days=1`
   → HTTP 200, dataset `neuricx`, 5 rows with correct channel data. The deployed
   service (running as its SA, via the Cloud Run metadata-server OAuth token) reads
   and writes BigQuery end-to-end.

### Bug found + fixed during go-live (live testing caught it)
First post-deploy `/history` returned `"BigQuery unavailable (no OAuth token on this host)"`.
Cause: `metadataToken()` used `node:https` to `metadata.google.internal`, but the
GCE/Cloud Run metadata server speaks **plain HTTP** — the TLS handshake failed, so no
token. Fix: use `node:http` for the metadata call (kept `node:https` for the BigQuery
API). Also removed a permanent `_disabled` latch that would have killed warehousing
for the instance lifetime on a single transient token miss. Redeployed → verified.

Note: live `intel` pulls only warehouse when GDELT returns articles; during go-live
GDELT was rate-limiting (HTTP 429, external), so the write path was proven via the
direct persistRun test rather than a live pull. persistRun correctly skips on a
0-article/warning pull (guard: `classified.length>0 && warnings.length===0`).

## Cleanup pending
7 synthetic `query LIKE 'TEST%'` rows were inserted during verification. BigQuery
blocks DML DELETE on rows still in the streaming buffer (~30-90 min); a cleanup
DELETE is scheduled. They are harmless (filterable by `query NOT LIKE 'TEST%'`).

## Local test recipe
```
export GOOGLE_OAUTH_TOKEN=$(gcloud auth print-access-token --account avishekb@iitbbs.ac.in)
# then run the server, or call persistRun/channelHistory directly (see /tmp/bqtest.mjs pattern)
```
