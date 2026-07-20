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

# Tier-A avatar image: fetch MuseTalk (lip-sync) + warm XTTS (voice clone). No LTX here.
if [ ! -f "$CACHE_DIR/.avatar_ready" ] \
   || [ ! -d "$CACHE_DIR/musetalk" ]; then
    echo "[start] fetching MuseTalk + XTTS weights into $CACHE_DIR (first cold start) ..."
    python /app/download_models.py musetalk xtts || echo "[start] WARN: model fetch reported errors; handler will retry lazily"
    touch "$CACHE_DIR/.avatar_ready" || true
else
    echo "[start] weights present in $CACHE_DIR (warm cache)"
fi

echo "[start] launching serverless handler"
exec python -u /app/rp_handler.py
