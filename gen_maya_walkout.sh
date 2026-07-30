#!/usr/bin/env bash
# Render the photoreal FULL-BODY Maya walkout clips for the door-walkout tour (task #14).
# Output: one transparent-alpha webm per action -> /brand/maya/<action>.webm, consumed by
# frontend/src/components/layout/maya-walkout/maya-presenter.tsx (per-action clip player).
#
# Pipeline (per Garood's choice — "photoreal rendered walk (GPU)"):
#   1) RealVisXL  -> ONE consistent full-body Maya still on a flat CHROMA background (green),
#      so every action clip is the SAME Maya and the background is keyable to transparent.
#   2) LTX i2v    -> animate that still into a short loop per action (walk-in-place, point, wave,
#      idle/talk). Same source still => consistent character; the CSS glide provides travel.
#   3) ffmpeg     -> colorkey the green -> WEBM with alpha (VP9 yuva420p). Full-body figure, no box.
#
# ⚠️ Photoreal full-body motion is the finicky part — RUN THE `walk` ACTION FIRST and eyeball it
# before rendering the rest (Garood asked to approve a test clip first). Needs a GPU box with
# RealVisXL (:8500) + LTX (video server :8090) — e.g. the freed Thunder A100 after the 558 batch,
# or the RunPod prod box off-hours. Set IMG_URL / VIDEO_URL + their tokens below.
set -uo pipefail
BASE=${BASE:-/home/ubuntu/maya_walkout}
IMG_URL=${IMG_URL:-http://127.0.0.1:8500}        # RealVisXL /generate  (X-Auth)
IMG_TOK=${IMG_TOK:-}
VIDEO_URL=${VIDEO_URL:-http://127.0.0.1:8090}     # LTX render server /render (Bearer) — needs i2v
VIDEO_TOK=${VIDEO_TOK:-}
KEYCOLOR=${KEYCOLOR:-0x00b140}                    # chroma green
ACTIONS=${ACTIONS:-"walk point wave talk idle"}   # override to just "walk" for the approval test
mkdir -p "$BASE"

# 1) full-body Maya still on green (consistent identity + keyable bg)
STILL=$BASE/maya_base.png
if [ ! -s "$STILL" ]; then
  echo "[1/3] RealVisXL full-body Maya still..."
  curl -s -m 120 -X POST "$IMG_URL/generate" -H "X-Auth: $IMG_TOK" -H 'Content-Type: application/json' \
    -d '{"prompt":"RAW photo, full body of a friendly professional businesswoman, mid 30s, navy blazer and white blouse, standing facing camera, natural warm smile, studio lighting, sharp focus, on a flat chroma-key green screen background, 85mm","negative_prompt":"cropped, close-up, headshot, multiple people, text, watermark","width":768,"height":1152,"steps":30,"guidance":5.0}' \
    -o "$STILL" -w "  still http=%{http_code}\n"
fi
[ -s "$STILL" ] || { echo "no base still — check IMG_URL/IMG_TOK"; exit 1; }

# 2) + 3) per action: LTX i2v -> mp4 -> colorkey -> alpha webm
MOTION_walk="the woman walking in place toward the viewer, natural confident stride, arms swinging gently"
MOTION_point="the woman gesturing, extending one arm to point to the side, presenting"
MOTION_wave="the woman smiling and waving hello with one hand"
MOTION_talk="the woman talking to the viewer, friendly, subtle head and hand movement"
MOTION_idle="the woman standing, subtle breathing and blinking, relaxed"

for A in $ACTIONS; do
  eval "PROMPT=\$MOTION_$A"
  MP4=$BASE/${A}.mp4; WEBM=$BASE/${A}.webm
  [ -s "$WEBM" ] && { echo "skip $A (done)"; continue; }
  echo "[2/3] LTX i2v: $A ..."
  # submit i2v render (server.py accepts an init image for image->video seeding)
  RESP=$(curl -s -m 30 -X POST "$VIDEO_URL/render" -H "Authorization: Bearer $VIDEO_TOK" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT, full body, green screen background\",\"seconds\":3,\"init_image_path\":\"$STILL\"}")
  JOB=$(echo "$RESP" | grep -oE '"id":"[a-f0-9]+"' | cut -d'"' -f4)
  [ -z "$JOB" ] && { echo "  submit failed: $RESP"; continue; }
  for i in $(seq 1 40); do
    S=$(curl -s -m 12 "$VIDEO_URL/render/$JOB" -H "Authorization: Bearer $VIDEO_TOK")
    echo "$S" | grep -q '"succeeded"' && { URL=$(echo "$S" | grep -oE 'http[^"]+\.mp4'); curl -s -m 60 "$URL" -H "Authorization: Bearer $VIDEO_TOK" -o "$MP4"; break; }
    echo "$S" | grep -qE '"failed"|"error"' && { echo "  render failed: $S"; break; }
    sleep 8
  done
  [ -s "$MP4" ] || { echo "  no mp4 for $A"; continue; }
  echo "[3/3] chroma-key -> alpha webm: $A ..."
  # key the green to transparent; VP9 alpha (yuva420p). despill via a slight similarity/blend.
  ffmpeg -nostdin -y -loglevel error -i "$MP4" \
    -vf "colorkey=${KEYCOLOR}:0.30:0.12,format=yuva420p" \
    -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 30 -an "$WEBM" 2>/dev/null
  echo "  $A -> $([ -s "$WEBM" ] && du -h "$WEBM" | cut -f1 || echo FAILED)"
done
echo "=== done. Deploy: place $BASE/*.webm at core-crm-backend/public/brand/maya/<action>.webm ==="
