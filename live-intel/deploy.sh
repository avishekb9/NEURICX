#!/usr/bin/env bash
# Deploy NEURICX live-intel to Cloud Run (hopeful-flash-485308-v3, asia-south1).
# Reads GOOGLE_API_KEY from the env or ../../.env.local. Run from neuricx/.
#
#   bash deploy.sh
#
# Prereqs (enable once): run.googleapis.com, artifactregistry.googleapis.com,
# cloudbuild.googleapis.com  (see DEPLOY.md).
set -euo pipefail

PROJECT="${GCP_PROJECT:-hopeful-flash-485308-v3}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE="${SERVICE:-neuricx-intel}"

KEY="${GOOGLE_API_KEY:-}"
if [ -z "$KEY" ] && [ -f ../../.env.local ]; then
  KEY=$(grep '^GOOGLE_API_KEY=' ../../.env.local | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [ -z "$KEY" ]; then echo "ERROR: GOOGLE_API_KEY not found"; exit 1; fi

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "HOST=0.0.0.0,GOOGLE_API_KEY=${KEY}" \
  --min-instances 0 \
  --max-instances 2 \
  --memory 256Mi \
  --cpu 1 \
  --timeout 120

echo "Deployed. URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)'
