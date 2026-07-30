#!/usr/bin/env bash
# Render the PHOTOREAL Maya walkout clips with MimicMotion (pose-guided — no morphing).
# For each action: ref image (Maya still) + a driving POSE video -> photoreal Maya doing that
# motion -> background removed (rembg) -> transparent-alpha webm at /brand/maya/<action>.webm.
#
# INPUTS you must supply (driving pose videos — a real person doing the motion, front-facing, full
# body; MimicMotion extracts DWPose from them):
#   $DRIVE/walk.mp4  $DRIVE/point.mp4  $DRIVE/wave.mp4  $DRIVE/idle.mp4
# Sourcing: MimicMotion ships demo driving clips in assets/ (a good WALK/dance to start); for
# point/wave/idle, a short phone clip or a stock loop works. Front view, ~3-5s, person centered.
set -uo pipefail
MM="$HOME/MimicMotion"
VENV="$HOME/mm-venv"; PY="$VENV/bin/python"
REF=${REF:-$HOME/maya_walk/base.png}          # RealVisXL full-body Maya still
DRIVE=${DRIVE:-$HOME/maya_drive}               # driving pose videos live here
OUT=${OUT:-$HOME/maya_out}
ACTIONS=${ACTIONS:-"walk point wave idle"}     # override to "walk" for the approval test
mkdir -p "$OUT"
cd "$MM" || exit 1
$VENV/bin/pip install -q rembg onnxruntime 2>/dev/null || true

[ -s "$REF" ] || { echo "no ref image at $REF"; exit 1; }

for A in $ACTIONS; do
  DV="$DRIVE/${A}.mp4"; RAW="$OUT/${A}_raw.mp4"; WEBM="$OUT/${A}.webm"
  [ -s "$WEBM" ] && { echo "skip $A"; continue; }
  [ -s "$DV" ] || { echo "MISSING driving video $DV — skip $A"; continue; }
  echo "=== MimicMotion: $A (ref=$(basename "$REF") drive=$(basename "$DV")) ==="
  # MimicMotion inference. NOTE: verify the exact CLI/config against the cloned repo — recent
  # versions use a yaml (configs/*.yaml with ref_video_path / ref_image_path) driven by
  # inference.py. This writes a per-action config then runs it.
  cat > "/tmp/mm_${A}.yaml" <<YAML
base_model_path: models/SVD/stable-video-diffusion-img2vid-xt-1-1
ckpt_path: models/MimicMotion_1-1.pth
test_case:
  - ref_video_path: $DV
    ref_image_path: $REF
    num_frames: 72
    resolution: 576
    frames_overlap: 6
    num_inference_steps: 25
    noise_aug_strength: 0
    guidance_scale: 2.0
    sample_stride: 2
    fps: 15
    seed: 42
YAML
  $PY inference.py --inference_config "/tmp/mm_${A}.yaml" >/dev/null 2>&1 || { echo "  render failed for $A"; continue; }
  # MimicMotion writes to outputs/ — grab the newest mp4
  M=$(ls -t outputs/*.mp4 2>/dev/null | head -1)
  [ -s "$M" ] && cp "$M" "$RAW" || { echo "  no output mp4 for $A"; continue; }
  echo "  removing background (rembg) -> alpha webm ..."
  # rembg per-frame alpha then encode VP9 with alpha
  TMP=$(mktemp -d); ffmpeg -nostdin -y -loglevel error -i "$RAW" "$TMP/f_%04d.png"
  $VENV/bin/python - "$TMP" <<'PY'
import sys,glob,os
from rembg import remove
from PIL import Image
d=sys.argv[1]
for f in sorted(glob.glob(os.path.join(d,'f_*.png'))):
    Image.open(f).convert('RGBA').save(f)  # ensure RGBA
    with open(f,'rb') as fh: data=fh.read()
    out=remove(data)
    with open(f,'wb') as fh: fh.write(out)
PY
  ffmpeg -nostdin -y -loglevel error -framerate 15 -i "$TMP/f_%04d.png" \
    -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 30 -an "$WEBM" 2>/dev/null
  rm -rf "$TMP"
  echo "  $A -> $([ -s "$WEBM" ] && du -h "$WEBM"|cut -f1 || echo FAILED)"
done
echo "=== done. Deploy: core-crm-backend/public/brand/maya/<action>.webm (pull off-box first!) ==="
