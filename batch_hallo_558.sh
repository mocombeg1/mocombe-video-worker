#!/usr/bin/env bash
# Resumable 558-clip avatar-VARIANT greeting batch (31 agents x 18 photoreal variants).
# One voice+greeting per agent (agents.tsv: id<TAB>voice<TAB>greeting), reused across that
# agent's 18 portraits. Balanced SLOT-MAJOR order: renders variant-slot k for ALL agents before
# slot k+1, so a partial run still leaves every agent with broad ethnicity coverage.
#
# Handles Thunder contended-GPU stalls: pkill wedged inference before each attempt, hard 15-min
# cap, one retry, continues past any single failure. Resumable (skip if output exists).
# BUGFIX carried over: read the manifest on FD 3, ffmpeg -nostdin, inference </dev/null so nothing
# eats the roster stdin (the leading-char-strip bug).
set -uo pipefail
cd /home/ubuntu/hallo || exit 1
ROSTER=/home/ubuntu/agents.tsv
PORTRAITS=/home/ubuntu/portraits
OUT=/home/ubuntu/out
LOG=/home/ubuntu/batch558.log
KPY=/home/ubuntu/kokoro-venv/bin/python
HPY=/home/ubuntu/hallo-venv/bin/python
mkdir -p "$OUT" /home/ubuntu/wav /home/ubuntu/png
echo "=== batch558 start $(date) ===" >> "$LOG"

# wait for kokoro to be ready (voice model warm)
for i in $(seq 1 60); do grep -q KOKORO_READY /home/ubuntu/kokoro_ready.txt 2>/dev/null && break; sleep 10; done

# ── Phase 1: one WAV per agent (fast) ────────────────────────────────────────────────
while IFS=$'\t' read -r id voice greeting <&3; do
  [ -z "$id" ] && continue
  WAV=/home/ubuntu/wav/${id}.wav
  [ -s "$WAV" ] && { echo "WAV-SKIP $id" >> "$LOG"; continue; }
  RAW=/home/ubuntu/wav/${id}_raw.wav
  printf '%s' "$greeting" | "$KPY" /home/ubuntu/gen_voice.py "$voice" "$RAW" >>"$LOG" 2>&1
  [ -s "$RAW" ] && ffmpeg -nostdin -y -loglevel error -i "$RAW" -filter:a atempo=1.08 "$WAV" 2>>"$LOG"
  [ -s "$WAV" ] && echo "WAV-OK $id" >> "$LOG" || echo "WAV-FAIL $id" >> "$LOG"
done 3< "$ROSTER"

# ── Phase 2: render matrix in balanced slot-major order ───────────────────────────────
# slot order: one of each ethnicity at _1, then _2, then _3
SLOTS="white_1 black_1 hispanic_1 east_asian_1 south_asian_1 middle_eastern_1 \
       white_2 black_2 hispanic_2 east_asian_2 south_asian_2 middle_eastern_2 \
       white_3 black_3 hispanic_3 east_asian_3 south_asian_3 middle_eastern_3"

# agent id list (order of roster)
AGENTS=$(cut -f1 "$ROSTER")

for slot in $SLOTS; do
  for id in $AGENTS; do
    WAV=/home/ubuntu/wav/${id}.wav
    SRC=$PORTRAITS/${id}/${slot}.webp
    OUTMP4=$OUT/greet_${id}_${slot}_hallo.mp4
    [ -s "$OUTMP4" ] && { echo "SKIP $id/$slot (done)" >> "$LOG"; continue; }
    [ -s "$WAV" ] || { echo "FAIL $id/$slot (no wav)" >> "$LOG"; continue; }
    [ -s "$SRC" ] || { echo "FAIL $id/$slot (no portrait)" >> "$LOG"; continue; }
    PNG=/home/ubuntu/png/${id}_${slot}.png
    [ -s "$PNG" ] || ffmpeg -nostdin -y -loglevel error -i "$SRC" "$PNG" 2>>"$LOG"
    for attempt in 1 2; do
      pkill -9 -f inference.py 2>/dev/null; sleep 3
      echo "RENDER $id/$slot attempt $attempt $(date)" >> "$LOG"
      timeout 900 "$HPY" scripts/inference.py --config configs/inference/default.yaml \
        --source_image "$PNG" --driving_audio "$WAV" --output "$OUTMP4" </dev/null >> "$LOG" 2>&1
      [ -s "$OUTMP4" ] && { echo "OK $id/$slot $(date)" >> "$LOG"; break; }
      echo "RETRY $id/$slot (attempt $attempt failed)" >> "$LOG"
    done
    [ -s "$OUTMP4" ] || echo "FAIL $id/$slot (render)" >> "$LOG"
  done
done
echo "=== batch558 done $(date) ===" >> "$LOG"
