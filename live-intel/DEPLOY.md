# NEURICX live-intel — Cloud Run deployment

Make NEURICX an independent, public GCP-hosted engine.

## One-time: enable APIs (free; only usage is billed)

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project hopeful-flash-485308-v3
```

(`aiplatform` + `generativelanguage` are already enabled — NEURICX uses the
latter via `GOOGLE_API_KEY`.)

## Deploy

```bash
cd neuricx
bash deploy.sh
```

This runs `gcloud run deploy neuricx-intel --source .` → Cloud Build builds the
image, pushes to Artifact Registry, deploys to Cloud Run (asia-south1), and
prints the public URL.

Settings (in `deploy.sh`): `--allow-unauthenticated` (public site),
`--min-instances 0` (scale-to-zero = no idle cost), `--max-instances 2`,
`256Mi` / 1 CPU. `GOOGLE_API_KEY` is passed as an env var (not baked into the
image, not in git).

## Cost / abuse bounding

The server's GDELT throttle (`gdeltLastCall`, 1 req / 5 s, process-global) caps
the whole instance's uncached fetch rate. Since every uncached query must clear
GDELT before any Gemini call, max Gemini spend is ~12 calls/min/instance × 2
instances = trivial on research credits. The 30-min cache + disk last-good
further reduce calls. If you want a hard ceiling, lower `--max-instances` to 1.

## Hardening (optional, later)

- Move `GOOGLE_API_KEY` to Secret Manager:
  `gcloud secrets create neuricx-key --data-file=- <<< "$KEY"` then deploy with
  `--set-secrets GOOGLE_API_KEY=neuricx-key:latest` (needs `secretmanager.googleapis.com`).
- Restrict ingress or add Identity-Aware Proxy if it should not be fully public.
- Persistent cache: mount a GCS bucket via gcsfuse, or swap the disk cache for
  Firestore/GCS (the `.cache/` dir is ephemeral per Cloud Run instance).

## Phase 2 — real Maps geocoding

The dashboard already renders lat/lon (currently Gemini-estimated). To use the
Geocoding API: the project has `geocoding-backend.googleapis.com` enabled, but
the `GOOGLE_API_KEY` is **key-restricted** to aiplatform + generativelanguage,
so it returns `REQUEST_DENIED`. Fix in GCP Console → Credentials → that key →
API restrictions → add **Geocoding API** (or mint a separate, Maps-only key).
Then swap the lat/lon assignment in `server/neuricx-server.mjs::classifyArticles`
for a Geocoding call.
