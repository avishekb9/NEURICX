#!/usr/bin/env node
// NEURICX live news + geo intelligence — thin vertical slice (Phase 1)
//
// Pipeline:  GDELT (real, keyless news)  →  Gemini channel-classification
//            (tango-filter relevance pattern)  →  Gemini geotag  →  JSON API
//            + static dashboard.
//
// Why this shape (Karpathy: simplicity + honest deps):
//  - GDELT DOC 2.0 needs no key and returns live econ/financial/geo articles.
//  - Geocoding uses Gemini, NOT Maps: the project's GOOGLE_API_KEY is restricted
//    to aiplatform + generativelanguage (Maps geocode returns REQUEST_DENIED,
//    verified 2026-05-30). A real Maps key can replace geotagViaGemini() later.
//  - node:https only — this box sets NODE_OPTIONS=--no-experimental-fetch, so
//    global fetch is unavailable; no npm install required.
//  - Standalone on :3100, NOT folded into the 11k-line VERALABS proxy
//    (separation of concerns per the mjstudio security-lockdown rule).
//
// Channel taxonomy mirrors the contagion-channels framework exactly:
//   Trade · Financial · Geopolitical · Behavioural · Monetary Policy
//
// Usage:  GOOGLE_API_KEY=... node neuricx-server.mjs   (reads ../../../.env.local if unset)
// Env:    PORT (default 3100) · HOST (default 127.0.0.1) · NEURICX_CACHE_TTL_MIN (default 30)

import { createServer } from "node:http";
import { request as httpsRequest } from "node:https";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = join(__dirname, "..", "web");
const ENV_PATH = join(__dirname, "..", "..", "..", ".env.local");

// ── config ───────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || "3100", 10);
const HOST = process.env.HOST || "127.0.0.1";
const CACHE_TTL_MS = parseInt(process.env.NEURICX_CACHE_TTL_MIN || "30", 10) * 60_000;
const GEMINI_MODEL = "gemini-2.5-flash";

function loadApiKey() {
  if (process.env.GOOGLE_API_KEY) return process.env.GOOGLE_API_KEY;
  if (existsSync(ENV_PATH)) {
    const line = readFileSync(ENV_PATH, "utf8").split(/\r?\n/).find(l => l.startsWith("GOOGLE_API_KEY="));
    if (line) return line.slice("GOOGLE_API_KEY=".length).replace(/^["']|["']$/g, "").trim();
  }
  return null;
}
const API_KEY = loadApiKey();

const CHANNELS = ["Trade", "Financial", "Geopolitical", "Behavioural", "Monetary Policy"];

// ── tiny https helper (no global fetch on this box) ───────────────────────────
function httpsGetJson(url, { timeoutMs = 25_000 } = {}) {
  return new Promise((resolve, reject) => {
    const req = httpsRequest(url, { method: "GET", headers: { "User-Agent": "NEURICX/0.1 (research)" } }, res => {
      let body = "";
      res.on("data", c => (body += c));
      res.on("end", () => {
        if (res.statusCode < 200 || res.statusCode >= 300) return reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 200)}`));
        try { resolve(JSON.parse(body)); } catch (e) { reject(new Error(`bad JSON: ${e.message}`)); }
      });
    });
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error("timeout")));
    req.end();
  });
}

// ── geocoding (Google Maps Geocoding API — real coords, replaces Gemini guess) ─
// The GOOGLE_API_KEY's restrictions were extended to geocoding-backend on
// 2026-05-30, so the same key now authorizes Geocoding. Results are cached per
// location string (location names repeat heavily across a news pull, and the
// cache persists for the process lifetime to keep Maps calls minimal).
const geocodeCache = new Map(); // "Mumbai" -> {lat,lon} | null
async function geocode(place) {
  if (!place || place === "Global") return null;
  if (geocodeCache.has(place)) return geocodeCache.get(place);
  if (!API_KEY) return null;
  try {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(place)}&key=${API_KEY}`;
    const data = await httpsGetJson(url, { timeoutMs: 12_000 });
    let result = null;
    if (data.status === "OK" && data.results?.[0]?.geometry?.location) {
      const loc = data.results[0].geometry.location;
      result = { lat: loc.lat, lon: loc.lng };
    }
    geocodeCache.set(place, result); // cache misses too (incl. null) to avoid re-querying
    return result;
  } catch {
    return null; // never let a geocode failure break the pipeline
  }
}

function httpsPostJson(url, payload, { timeoutMs = 40_000 } = {}) {
  return new Promise((resolve, reject) => {
    const data = Buffer.from(JSON.stringify(payload));
    const u = new URL(url);
    const req = httpsRequest(u, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": data.length },
    }, res => {
      let body = "";
      res.on("data", c => (body += c));
      res.on("end", () => {
        if (res.statusCode < 200 || res.statusCode >= 300) return reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 300)}`));
        try { resolve(JSON.parse(body)); } catch (e) { reject(new Error(`bad JSON: ${e.message}`)); }
      });
    });
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error("timeout")));
    req.write(data);
    req.end();
  });
}

// ── stage 1: GDELT ingestion (real, keyless) ──────────────────────────────────
// GDELT constraints (verified 2026-05-30):
//   1. OR'd terms MUST be wrapped in parentheses: "(a OR b OR c)".
//   2. Rate limit: one request per 5 seconds, else a plain-text scolding.
// gdeltLastCall enforces (2); normalizeGdeltQuery enforces (1).
let gdeltLastCall = 0;
function normalizeGdeltQuery(q) {
  const t = q.trim();
  // if it contains OR and isn't already fully parenthesized, wrap it
  if (/\bOR\b/i.test(t) && !/^\(.*\)$/.test(t)) return `(${t})`;
  return t;
}
async function fetchGdelt(query, { maxRecords = 25, timespanDays = 3 } = {}) {
  const wait = 5100 - (Date.now() - gdeltLastCall);
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
  gdeltLastCall = Date.now();
  const url = "https://api.gdeltproject.org/api/v2/doc/doc?query=" +
    encodeURIComponent(normalizeGdeltQuery(query)) +
    `&mode=ArtList&maxrecords=${maxRecords}&format=json&timespan=${timespanDays}d&sort=DateDesc`;
  const data = await httpsGetJson(url);
  return (data.articles || []).map(a => ({
    title: a.title,
    url: a.url,
    domain: a.domain,
    language: a.language,
    source_country: a.sourcecountry,
    seendate: a.seendate,
  }));
}

// ── stage 2: Gemini classification (tango-filter relevance pattern) ───────────
// One batched call classifies all headlines → channel + relevance + sentiment +
// primary location. Mirrors tango-filter's "score every candidate in one Gemini
// pass" approach, retargeted from venues to econ/financial/geo news.
function buildClassifyPrompt(articles) {
  const list = articles.map((a, i) => `${i}. [${a.source_country || "?"}] ${a.title}`).join("\n");
  return `You are an economic-news triage engine for a financial-contagion research platform.
For EACH numbered headline below, return a JSON object with:
  "i": the index (integer),
  "channel": exactly one of ${JSON.stringify(CHANNELS)} (the dominant transmission channel),
  "relevance": 0.0-1.0 (how relevant to systemic financial / macro-economic risk; 0 = celebrity/sport/noise),
  "sentiment": one of "risk-on","risk-off","neutral",
  "location": the single most relevant country or city named or implied (string; "Global" if none),
  "lat": approximate latitude of that location (number),
  "lon": approximate longitude of that location (number).
Return ONLY a JSON array of these objects, no prose, no markdown fences.

Headlines:
${list}`;
}

async function classifyArticles(articles) {
  if (!API_KEY) throw new Error("GOOGLE_API_KEY not available");
  if (!articles.length) return [];
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${API_KEY}`;
  const payload = {
    contents: [{ parts: [{ text: buildClassifyPrompt(articles) }] }],
    generationConfig: { temperature: 0.1, responseMimeType: "application/json" },
  };
  const res = await httpsPostJson(url, payload);
  const text = res?.candidates?.[0]?.content?.parts?.[0]?.text || "[]";
  let parsed;
  try { parsed = JSON.parse(text); } catch { parsed = []; }
  const byIndex = new Map(parsed.map(p => [p.i, p]));
  // merge classification back onto articles; default safely if a row is missing
  return articles.map((a, i) => {
    const c = byIndex.get(i) || {};
    return {
      ...a,
      channel: CHANNELS.includes(c.channel) ? c.channel : "Financial",
      relevance: typeof c.relevance === "number" ? Math.max(0, Math.min(1, c.relevance)) : 0,
      sentiment: ["risk-on", "risk-off", "neutral"].includes(c.sentiment) ? c.sentiment : "neutral",
      location: c.location || "Global",
      lat: typeof c.lat === "number" ? c.lat : null,
      lon: typeof c.lon === "number" ? c.lon : null,
      classified: byIndex.has(i),
    };
  });
}

// ── aggregation into a NEURICX snapshot ───────────────────────────────────────
function summarize(articles) {
  const byChannel = {};
  for (const ch of CHANNELS) byChannel[ch] = { count: 0, mean_relevance: 0, risk_off: 0 };
  let relSum = 0;
  for (const a of articles) {
    const b = byChannel[a.channel];
    b.count++; b.mean_relevance += a.relevance;
    if (a.sentiment === "risk-off") b.risk_off++;
    relSum += a.relevance;
  }
  for (const ch of CHANNELS) {
    const b = byChannel[ch];
    if (b.count) b.mean_relevance = Math.round((b.mean_relevance / b.count) * 1000) / 1000;
  }
  // simple "stress index": share of risk-off weighted by relevance, 0-1
  const stress = articles.length
    ? Math.round((articles.filter(a => a.sentiment === "risk-off").reduce((s, a) => s + a.relevance, 0) /
        Math.max(1, articles.reduce((s, a) => s + a.relevance, 0))) * 1000) / 1000
    : 0;
  return { by_channel: byChannel, mean_relevance: articles.length ? Math.round((relSum / articles.length) * 1000) / 1000 : 0, stress_index: stress };
}

// ── cache (memory + disk-backed last-good) ────────────────────────────────────
// The Map only ever holds GOOD results (cacheSet is called only on success), so
// it doubles as the "last good pull" store. We mirror it to a single disk file
// so the last good pull survives a restart AND can be served as a stale fallback
// when GDELT throttles or is down.
const cache = new Map(); // key -> { at, payload }
const CACHE_DIR = join(__dirname, ".cache");
const CACHE_FILE = join(CACHE_DIR, "last-good.json");

function loadDiskCache() {
  try {
    if (!existsSync(CACHE_FILE)) return;
    const obj = JSON.parse(readFileSync(CACHE_FILE, "utf8"));
    for (const [k, v] of Object.entries(obj)) cache.set(k, v);
    console.log(`  loaded ${cache.size} cached query(ies) from disk`);
  } catch (e) {
    console.log(`  disk-cache load skipped: ${e.message}`);
  }
}
function persistDiskCache() {
  try {
    mkdirSync(CACHE_DIR, { recursive: true });
    writeFileSync(CACHE_FILE, JSON.stringify(Object.fromEntries(cache), null, 2));
  } catch (e) {
    console.log(`  disk-cache write failed: ${e.message}`);
  }
}

function cacheGet(key) { // fresh (within TTL) only
  const e = cache.get(key);
  if (e && Date.now() - e.at < CACHE_TTL_MS) return e.payload;
  return null;
}
function cacheGetAny(key) { // last good, ignoring TTL — for stale fallback
  return cache.get(key) || null;
}
function cacheSet(key, payload) {
  cache.set(key, { at: Date.now(), payload });
  persistDiskCache();
}

// ── pipeline ──────────────────────────────────────────────────────────────────
async function runPipeline(query, opts) {
  const key = `${query}|${opts.maxRecords}|${opts.timespanDays}`;
  const cached = cacheGet(key);
  if (cached) return { ...cached, cached: true };

  const warnings = [];
  let articles = [];
  try {
    articles = await fetchGdelt(query, opts);
  } catch (e) {
    warnings.push(`GDELT fetch failed: ${e.message}`);
  }
  let classified = articles.map(a => ({ ...a, channel: "Financial", relevance: 0, sentiment: "neutral", location: "Global", lat: null, lon: null, classified: false }));
  if (articles.length) {
    try {
      classified = await classifyArticles(articles);
    } catch (e) {
      warnings.push(`Gemini classification failed (showing unclassified): ${e.message}`);
    }
  }
  // Replace Gemini-estimated coords with real Geocoding API coords where possible.
  // De-dup locations first so we make at most one Maps call per distinct place.
  let geotag = "Gemini-estimated (Maps geocoding unavailable)";
  if (API_KEY && classified.length) {
    const places = [...new Set(classified.map(a => a.location).filter(p => p && p !== "Global"))];
    const coords = {};
    for (const p of places) coords[p] = await geocode(p); // geocodeCache makes repeats free
    let real = 0;
    classified = classified.map(a => {
      const g = coords[a.location];
      if (g) { real++; return { ...a, lat: g.lat, lon: g.lon, geocoded: true }; }
      return { ...a, geocoded: false };
    });
    if (real > 0) geotag = `Google Geocoding API (${real}/${classified.length} located) + Gemini fallback`;
  }
  // sort by relevance desc so the dashboard leads with what matters
  classified.sort((a, b) => b.relevance - a.relevance);
  const payload = {
    engine: "NEURICX",
    version: "0.2.0",
    generated_at_utc: new Date().toISOString(),
    query,
    options: opts,
    article_count: classified.length,
    summary: summarize(classified),
    articles: classified,
    sources: { news: "GDELT DOC 2.0 (keyless)", classifier: `Gemini ${GEMINI_MODEL}`, geotag },
    warnings,
    cached: false,
  };
  // Only cache a genuinely good result — never cache an empty/failed fetch, so a
  // transient GDELT rate-limit or outage doesn't pin the dashboard to zero.
  if (classified.length > 0 && warnings.length === 0) {
    cacheSet(key, payload);
    return payload;
  }
  // Bad/empty fetch (e.g. GDELT 429): serve the last good pull for this query
  // as STALE rather than showing zeros, if we have one on disk/memory.
  const lastGood = cacheGetAny(key);
  if (lastGood) {
    return {
      ...lastGood.payload,
      cached: true,
      stale: true,
      stale_as_of: lastGood.payload.generated_at_utc,
      warnings: [...payload.warnings, `Live fetch unavailable — showing last good pull from ${lastGood.payload.generated_at_utc}`],
    };
  }
  return payload;
}

// ── HTTP server ───────────────────────────────────────────────────────────────
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };

const server = createServer(async (req, res) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  const send = (code, type, body) => { res.writeHead(code, { "Content-Type": type, "Access-Control-Allow-Origin": "*" }); res.end(body); };

  // health
  if (u.pathname === "/health") {
    return send(200, "application/json", JSON.stringify({ ok: true, api_key: !!API_KEY, model: GEMINI_MODEL, cache_entries: cache.size, disk_cache: existsSync(CACHE_FILE) }));
  }

  // live intelligence API
  if (u.pathname === "/api/neuricx/intel") {
    const query = u.searchParams.get("q") || "economy OR financial OR trade OR tariff OR inflation OR central bank";
    const opts = {
      maxRecords: Math.min(50, Math.max(5, parseInt(u.searchParams.get("n") || "25", 10))),
      timespanDays: Math.min(14, Math.max(1, parseInt(u.searchParams.get("days") || "3", 10))),
    };
    try {
      const payload = await runPipeline(query, opts);
      return send(200, "application/json", JSON.stringify(payload));
    } catch (e) {
      return send(500, "application/json", JSON.stringify({ error: e.message }));
    }
  }

  // static dashboard
  let p = u.pathname === "/" ? "/index.html" : u.pathname;
  const filePath = join(WEB_DIR, p.replace(/\.\.+/g, ""));
  if (existsSync(filePath) && filePath.startsWith(WEB_DIR)) {
    return send(200, MIME[extname(filePath)] || "application/octet-stream", readFileSync(filePath));
  }
  send(404, "text/plain", "not found");
});

loadDiskCache();
server.listen(PORT, HOST, () => {
  console.log(`✓ NEURICX live-intel server on http://${HOST}:${PORT}`);
  console.log(`  dashboard:  http://${HOST}:${PORT}/`);
  console.log(`  api:        http://${HOST}:${PORT}/api/neuricx/intel?q=...&n=25&days=3`);
  console.log(`  health:     http://${HOST}:${PORT}/health`);
  console.log(`  GOOGLE_API_KEY: ${API_KEY ? "loaded" : "MISSING — classification will fall back to unclassified"}`);
});
