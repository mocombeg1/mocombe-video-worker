#!/usr/bin/env bash
# Resumable, self-healing 31-agent Hallo greeting batch. Each manifest.txt line = id|voice|greeting.
# Runs UNATTENDED on the A100 (setsid). Handles Thunder's contended-GPU stalls: kills any wedged
# render before each attempt + a hard 15-min cap + one retry, and CONTINUES past any single failure.
cd /home/ubuntu/hallo || exit 1
MANIFEST=/home/ubuntu/manifest.txt
LOG=/home/ubuntu/batch.log
KPY=/home/ubuntu/kokoro-venv/bin/python
HPY=/home/ubuntu/hallo-venv/bin/python
echo "=== batch start $(date) ===" >> "$LOG"
# wait for the kokoro install to finish (may still be running)
for i in $(seq 1 60); do grep -q KOKORO_READY /home/ubuntu/kokoro_ready.txt 2>/dev/null && break; sleep 10; done
while IFS='|' read -r id voice greeting; do
  [ -z "$id" ] && continue
  OUT=/home/ubuntu/${id}_hallo.mp4
  # resumable: skip if already rendered (either naming)
  if [ -s "$OUT" ] || [ -s "/home/ubuntu/greet_${id}_hallo.mp4" ]; then echo "SKIP $id (done)" >> "$LOG"; continue; fi
  PNG=/home/ubuntu/${id}.png
  [ -s "$PNG" ] || ffmpeg -y -loglevel error -i "/home/ubuntu/portraits_src/${id}.webp" "$PNG" 2>>"$LOG"
  WAV=/home/ubuntu/${id}_108.wav
  if [ ! -s "$WAV" ]; then
    RAW=/home/ubuntu/${id}_raw.wav
    printf '%s' "$greeting" | "$KPY" /home/ubuntu/gen_voice.py "$voice" "$RAW" >>"$LOG" 2>&1
    [ -s "$RAW" ] && ffmpeg -y -loglevel error -i "$RAW" -filter:a atempo=1.08 "$WAV" 2>>"$LOG"
  fi
  if [ ! -s "$WAV" ] || [ ! -s "$PNG" ]; then echo "FAIL $id (no voice/png)" >> "$LOG"; continue; fi
  for attempt in 1 2; do
    pkill -9 -f inference.py 2>/dev/null; sleep 3
    echo "RENDER $id attempt $attempt $(date)" >> "$LOG"
    timeout 900 "$HPY" scripts/inference.py --config configs/inference/default.yaml \
      --source_image "$PNG" --driving_audio "$WAV" --output "$OUT" >> "$LOG" 2>&1
    [ -s "$OUT" ] && { echo "OK $id $(date)" >> "$LOG"; break; }
    echo "RETRY $id (attempt $attempt failed)" >> "$LOG"
  done
  [ -s "$OUT" ] || echo "FAIL $id (render)" >> "$LOG"
done < "$MANIFEST"
echo "=== batch done $(date) ===" >> "$LOG"
