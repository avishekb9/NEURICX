// bq.mjs — BigQuery data-feed warehouse for NEURICX (zero npm deps, node:https).
//
// Persists each good Gemini-classified news pull as flat rows so the ephemeral
// live feed becomes a queryable longitudinal research dataset. Auth uses an
// OAuth bearer token (BigQuery does NOT accept an API key):
//   1. Cloud Run / GCE metadata server (default SA) — zero config in production
//   2. GOOGLE_OAUTH_TOKEN env (for local runs: `gcloud auth print-access-token`)
// If no token is obtainable, every call no-ops with a logged warning, so the
// news pipeline never breaks because of a warehouse hiccup.
//
// Dataset/table are auto-created (idempotent) on first successful insert.

import { request as httpsRequest } from "node:https";
import { request as httpRequest } from "node:http";  // metadata server is plain HTTP

const PROJECT  = process.env.GCP_PROJECT || "hopeful-flash-485308-v3";
const DATASET  = process.env.BQ_DATASET || "neuricx";
const LOCATION = process.env.BQ_LOCATION || "asia-south1";
const ARTICLES = "articles";   // one row per classified article
const SNAPS    = "snapshots";  // one row per pipeline run (channel summary)

let _ensured = false;          // dataset+tables created this process?

// ── auth: metadata-server OAuth token (cached ~50 min) ────────────────────────
let _tok = null, _tokExp = 0;
function metadataToken() {
  return new Promise((resolve) => {
    // NOTE: the GCE/Cloud Run metadata server speaks plain HTTP, not HTTPS.
    const req = httpRequest(
      { host: "metadata.google.internal", path: "/computeMetadata/v1/instance/service-accounts/default/token",
        headers: { "Metadata-Flavor": "Google" }, timeout: 4000 },
      (r) => { let b = ""; r.on("data", d => b += d); r.on("end", () => {
        try { const j = JSON.parse(b); resolve(j.access_token ? j : null); } catch { resolve(null); } }); }
    );
    req.on("error", () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
    req.end();
  });
}
async function getToken() {
  if (process.env.GOOGLE_OAUTH_TOKEN) return process.env.GOOGLE_OAUTH_TOKEN;
  if (_tok && Date.now() < _tokExp) return _tok;
  const t = await metadataToken();
  if (!t) return null;
  _tok = t.access_token; _tokExp = Date.now() + Math.max(0, (t.expires_in - 120)) * 1000;
  return _tok;
}

// ── generic BigQuery REST POST ────────────────────────────────────────────────
function bqRequest(method, path, body) {
  return new Promise((resolve) => {
    getToken().then((tok) => {
      if (!tok) return resolve({ ok: false, status: 0, error: "no-oauth-token" });
      const data = body ? Buffer.from(JSON.stringify(body)) : null;
      const u = new URL(`https://bigquery.googleapis.com${path}`);
      const req = httpsRequest(u, {
        method,
        headers: { "Authorization": `Bearer ${tok}`, "Content-Type": "application/json",
                   ...(data ? { "Content-Length": data.length } : {}) },
        timeout: 30000,
      }, (r) => {
        let b = ""; r.on("data", d => b += d);
        r.on("end", () => { let j = {}; try { j = JSON.parse(b); } catch {}
          resolve({ ok: r.statusCode >= 200 && r.statusCode < 300, status: r.statusCode, json: j }); });
      });
      req.on("error", (e) => resolve({ ok: false, status: 0, error: e.message }));
      req.on("timeout", () => { req.destroy(); resolve({ ok: false, status: 0, error: "timeout" }); });
      if (data) req.write(data); req.end();
    });
  });
}

// ── schema (idempotent create; 404 on insert triggers a one-time ensure) ──────
const ARTICLE_SCHEMA = { fields: [
  { name: "run_id",        type: "STRING" },
  { name: "generated_at",  type: "TIMESTAMP" },
  { name: "query",         type: "STRING" },
  { name: "url",           type: "STRING" },
  { name: "title",         type: "STRING" },
  { name: "source_country",type: "STRING" },
  { name: "channel",       type: "STRING" },
  { name: "relevance",     type: "FLOAT" },
  { name: "sentiment",     type: "STRING" },
  { name: "location",      type: "STRING" },
  { name: "lat",           type: "FLOAT" },
  { name: "lon",           type: "FLOAT" },
]};
const SNAP_SCHEMA = { fields: [
  { name: "run_id",        type: "STRING" },
  { name: "generated_at",  type: "TIMESTAMP" },
  { name: "query",         type: "STRING" },
  { name: "article_count", type: "INTEGER" },
  { name: "stress_index",  type: "FLOAT" },
  { name: "mean_relevance",type: "FLOAT" },
  { name: "channel",       type: "STRING" },
  { name: "channel_count", type: "INTEGER" },
  { name: "channel_mean_relevance", type: "FLOAT" },
  { name: "channel_risk_off",       type: "INTEGER" },
]};

async function ensureSchema() {
  if (_ensured) return true;
  // dataset (ignore 409 already-exists)
  await bqRequest("POST", `/bigquery/v2/projects/${PROJECT}/datasets`,
    { datasetReference: { datasetId: DATASET, projectId: PROJECT }, location: LOCATION });
  const t1 = await bqRequest("POST", `/bigquery/v2/projects/${PROJECT}/datasets/${DATASET}/tables`,
    { tableReference: { projectId: PROJECT, datasetId: DATASET, tableId: ARTICLES }, schema: ARTICLE_SCHEMA });
  const t2 = await bqRequest("POST", `/bigquery/v2/projects/${PROJECT}/datasets/${DATASET}/tables`,
    { tableReference: { projectId: PROJECT, datasetId: DATASET, tableId: SNAPS }, schema: SNAP_SCHEMA });
  // 200 (created) or 409 (exists) both fine; a hard auth/permission failure is not
  const okish = (r) => r.ok || r.status === 409;
  _ensured = okish(t1) && okish(t2);
  return _ensured;
}

// ── streaming insert via tabledata.insertAll ──────────────────────────────────
function insertAll(table, rows) {
  return bqRequest("POST",
    `/bigquery/v2/projects/${PROJECT}/datasets/${DATASET}/tables/${table}/insertAll`,
    { kind: "bigquery#tableDataInsertAllRequest", skipInvalidRows: true, ignoreUnknownValues: true, rows });
}

// PUBLIC: persist one good pipeline payload. Fire-and-forget; never throws.
export async function persistRun(payload) {
  try {
    const tok = await getToken();
    if (!tok) { console.log("  [bq] no OAuth token this attempt — skipping persist (will retry next run)"); return { ok: false, skipped: "no-token" }; }
    if (!(await ensureSchema())) { console.log("  [bq] schema ensure failed — skipping persist"); return { ok: false, skipped: "schema" }; }

    const runId = `${payload.generated_at_utc}|${payload.query}`.slice(0, 256);
    const ts = payload.generated_at_utc;
    const artRows = (payload.articles || []).map((a) => ({ json: {
      run_id: runId, generated_at: ts, query: payload.query,
      url: a.url || null, title: a.title || null, source_country: a.source_country || null,
      channel: a.channel || null, relevance: a.relevance ?? null, sentiment: a.sentiment || null,
      location: a.location || null, lat: a.lat ?? null, lon: a.lon ?? null,
    }}));
    const sm = payload.summary || { by_channel: {} };
    const snapRows = Object.entries(sm.by_channel || {}).map(([ch, b]) => ({ json: {
      run_id: runId, generated_at: ts, query: payload.query,
      article_count: payload.article_count ?? null, stress_index: sm.stress_index ?? null,
      mean_relevance: sm.mean_relevance ?? null, channel: ch,
      channel_count: b.count ?? null, channel_mean_relevance: b.mean_relevance ?? null,
      channel_risk_off: b.risk_off ?? null,
    }}));

    const r1 = artRows.length ? await insertAll(ARTICLES, artRows) : { ok: true };
    const r2 = snapRows.length ? await insertAll(SNAPS, snapRows) : { ok: true };
    const ok = r1.ok && r2.ok;
    console.log(`  [bq] persisted run: ${artRows.length} articles + ${snapRows.length} channel rows → ${PROJECT}.${DATASET} (${ok ? "ok" : "partial"})`);
    return { ok, articles: artRows.length, snapshots: snapRows.length };
  } catch (e) {
    console.log(`  [bq] persist error (non-fatal): ${e.message}`);
    return { ok: false, error: e.message };
  }
}

// PUBLIC: read back the channel-intensity time series (analytics endpoint).
export async function channelHistory({ days = 30, query = null } = {}) {
  const tok = await getToken();
  if (!tok) return { error: "BigQuery unavailable (no OAuth token on this host)" };
  const where = [`generated_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL ${parseInt(days,10)} DAY)`];
  if (query) where.push(`query = @q`);
  const sql =
    `SELECT TIMESTAMP_TRUNC(generated_at, HOUR) AS bucket, channel,
            SUM(channel_count) AS n, AVG(channel_mean_relevance) AS mean_rel, SUM(channel_risk_off) AS risk_off
     FROM \`${PROJECT}.${DATASET}.${SNAPS}\`
     WHERE ${where.join(" AND ")}
     GROUP BY bucket, channel ORDER BY bucket DESC, channel LIMIT 2000`;
  const body = { query: sql, useLegacySql: false,
    ...(query ? { queryParameters: [{ name: "q", parameterType: { type: "STRING" }, parameterValue: { value: query } }] } : {}) };
  const r = await bqRequest("POST", `/bigquery/v2/projects/${PROJECT}/queries`, body);
  if (!r.ok) return { error: `BigQuery query failed (HTTP ${r.status})`, detail: r.json?.error?.message };
  const fields = (r.json.schema?.fields || []).map(f => f.name);
  const rows = (r.json.rows || []).map(row => Object.fromEntries(row.f.map((c, i) => [fields[i], c.v])));
  return { project: PROJECT, dataset: DATASET, days, query, row_count: rows.length, rows };
}
