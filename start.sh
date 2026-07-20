#!/usr/bin/env bash
# Cold-start entrypoint for the RunPod serverless worker.
#
# 1) Ensure model weights exist on the mounted network volume (idempotent — skips files already
#    present, so only the FIRST cold start after a fresh volume pays the download cost).
# 2) Launch the serverless handler.
#
# If you choose to BAKE weights into the image instead (set MODEL_CACHE_DIR to a baked path at
# build time and run download_models.py in the Dockerfile), this download step becomes a no-op.
set -euo pipefail

CACHE_DIR="${MODEL_CACHE_DIR:-/runpod-volume/models}"
mkdir -p "$CACHE_DIR"

# LEAN Tier-B image: only the LTX-Video weights are needed (no avatar Tier-A models).
if [ ! -f "$CACHE_DIR/.ltx_ready" ] \
   || [ ! -d "$CACHE_DIR/ltx-video" ]; then
    echo "[start] fetching LTX-Video weights into $CACHE_DIR (first cold start) ..."
    python /app/download_models.py ltx || echo "[start] WARN: model fetch reported errors; handler will retry lazily"
    touch "$CACHE_DIR/.ltx_ready" || true
else
    echo "[start] weights present in $CACHE_DIR (warm cache)"
fi

echo "[start] launching serverless handler"
exec python -u /app/rp_handler.py
