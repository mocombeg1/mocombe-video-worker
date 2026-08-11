#!/usr/bin/env bash

# ── LICENCE HOLD (2026-08-11) ─────────────────────────────────────────────────────────────
# Non-commercial models were found inside this pipeline. New renders are blocked until
# outside counsel answers. See hallo_licence_hold.sh for the finding and how to lift it.
source "$(dirname "$0")/hallo_licence_hold.sh"
# PARALLEL 558-clip avatar-variant batch. The A100 (80GB) fits ~4-5 concurrent Hallo renders
# (~10GB each), so K workers ~= Kx throughput vs serial (~9min/clip). Needed to finish 558 in
# the ~2-day Thunder window.
#
# Requires WAVs already generated (batch_hallo_558.sh Phase 1, or run gen here first).
# Each worker takes every K-th job from a shared balanced slot-major job list -> disjoint slices,
# so NO global pkill (that would kill siblings). Resumable (skip existing). Per-render timeout only.
#
# Usage: bash batch_hallo_558_parallel.sh [K]   (default K=4)
set -uo pipefail
cd /home/ubuntu/hallo || exit 1
K=${1:-4}
ROSTER=/home/ubuntu/agents.tsv
PORTRAITS=/home/ubuntu/portraits
OUT=/home/ubuntu/out
WAVDIR=/home/ubuntu/wav
PNGDIR=/home/ubuntu/png
JOBS=/home/ubuntu/jobs.txt
HPY=/home/ubuntu/hallo-venv/bin/python
mkdir -p "$OUT" "$PNGDIR"

# Build the balanced job list once (slot-major: one of each ethnicity at _1, then _2, then _3).
SLOTS="white_1 black_1 hispanic_1 east_asian_1 south_asian_1 middle_eastern_1 \
       white_2 black_2 hispanic_2 east_asian_2 south_asian_2 middle_eastern_2 \
       white_3 black_3 hispanic_3 east_asian_3 south_asian_3 middle_eastern_3"
AGENTS=$(cut -f1 "$ROSTER")
: > "$JOBS"
for slot in $SLOTS; do for id in $AGENTS; do echo "$id $slot" >> "$JOBS"; done; done
echo "$(wc -l < "$JOBS") jobs, K=$K workers"

# Per-worker CWD isolates Hallo's ./.cache (audio_preprocess, face crops) so parallel renders
# never clobber each other; weights+configs are shared read-only via symlink.
for w in $(seq 0 $((K-1))); do
  mkdir -p /home/ubuntu/hw$w
  ln -sfn /home/ubuntu/hallo/pretrained_models /home/ubuntu/hw$w/pretrained_models
  ln -sfn /home/ubuntu/hallo/configs           /home/ubuntu/hw$w/configs
done

worker() {
  local wid=$1 n=0
  cd /home/ubuntu/hw$wid || return
  sleep $(( wid * 40 ))   # stagger model-load spikes so concurrent inits don't kill each other
  while IFS=' ' read -r id slot; do
    n=$((n+1))
    [ $(( (n-1) % K )) -eq "$wid" ] || continue      # this worker owns every K-th job
    local WAV=$WAVDIR/${id}.wav
    local SRC=$PORTRAITS/${id}/${slot}.webp
    local OUTMP4=$OUT/greet_${id}_${slot}_hallo.mp4
    [ -s "$OUTMP4" ] && continue
    [ -s "$WAV" ] || { echo "w$wid FAIL $id/$slot (no wav)"; continue; }
    [ -s "$SRC" ] || { echo "w$wid FAIL $id/$slot (no portrait)"; continue; }
    local PNG=$PNGDIR/${id}_${slot}.png
    [ -s "$PNG" ] || ffmpeg -nostdin -y -loglevel error -i "$SRC" "$PNG" </dev/null 2>/dev/null
    echo "w$wid RENDER $id/$slot $(date +%H:%M:%S)"
    timeout 1200 "$HPY" /home/ubuntu/hallo/scripts/inference.py --config configs/inference/default.yaml \
      --source_image "$PNG" --driving_audio "$WAV" --output "$OUTMP4" </dev/null >/dev/null 2>>/home/ubuntu/hw$wid/err.log
    [ -s "$OUTMP4" ] && echo "w$wid OK $id/$slot $(date +%H:%M:%S)" || echo "w$wid FAIL $id/$slot (render) [see hw$wid/err.log]"
  done < "$JOBS"
}

echo "=== parallel batch start $(date) K=$K ==="
for w in $(seq 0 $((K-1))); do
  worker "$w" >> /home/ubuntu/batch558p.log 2>&1 &
done
wait
echo "=== parallel batch done $(date) ==="
