#!/usr/bin/env bash
# Pose-guided full-body avatar animation (MimicMotion) — the CRM's "motion" render engine.
# Called by rp_handler._motion:   render_motion.sh <ref_image> <driving_pose.mp4> <out.webm>
#
# Animates the reference portrait with the driving clip's motion WITHOUT morphing the body (the
# failure mode of the generative/LTX path). Output is a transparent-alpha VP9 webm so the clip
# drops onto any background (the 3D office doorway / guided-tour presenter).
#
# ── One-time provisioning on the motion box (RENDER_MODE=motion, its own venv) ────────────────
#   $MM   = MimicMotion repo (git clone https://github.com/Tencent/MimicMotion)
#   $VENV = mm-venv (py3.10): torch==2.2.2+cu121 torchvision==0.17.2 diffusers accelerate omegaconf
#           einops decord av matplotlib "huggingface_hub==0.23.4" rembg onnxruntime fastapi uvicorn
#   PINS THAT MATTER (all bit us — keep them):
#     - opencv-python-headless==4.9.0.80  +  numpy==1.26.4   (opencv-5 pulls numpy-2 -> breaks torch)
#     - patch $VENV/.../torchvision/io/video.py:  frame.pict_type = "NONE"  ->  = 0   (new PyAV
#       needs an int, else TypeError at the final mp4 write — the render itself is fine)
#   $MM/models: SVD-xt-1.1 from the UNGATED mirror `vdo/stable-video-diffusion-img2vid-xt-1-1`
#     (the stabilityai repo is gated) + MimicMotion_1-1.pth + DWPose.
set -uo pipefail
REF="$1"; POSE="$2"; OUT="$3"
MM="${MOTION_MM_DIR:-/root/maya/MimicMotion}"
VENV="${MOTION_VENV:-/root/maya/mm-venv}"
PY="$VENV/bin/python"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/cfg.yaml" <<YAML
base_model_path: models/SVD/stable-video-diffusion-img2vid-xt-1-1
ckpt_path: models/MimicMotion_1-1.pth
test_case:
  - ref_video_path: $POSE
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

cd "$MM" || { echo "no MimicMotion dir at $MM" >&2; exit 1; }
"$PY" inference.py --inference_config "$WORK/cfg.yaml" --output_dir "$WORK" 1>&2 || { echo "inference failed" >&2; exit 2; }
RAW=$(ls -t "$WORK"/*.mp4 2>/dev/null | head -1)
[ -s "$RAW" ] || { echo "no output mp4 from inference" >&2; exit 3; }

# ── background removal -> transparent-alpha webm ──────────────────────────────────────────────
FR="$WORK/frames"; mkdir -p "$FR"
ffmpeg -nostdin -y -loglevel error -i "$RAW" "$FR/f_%04d.png"
"$PY" - "$FR" <<'PY' 1>&2
import sys, glob, os
from rembg import remove
from PIL import Image
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "f_*.png"))):
    Image.open(f).convert("RGBA").save(f)
    with open(f, "rb") as fh: data = fh.read()
    with open(f, "wb") as fh: fh.write(remove(data))
PY
ffmpeg -nostdin -y -loglevel error -framerate 15 -i "$FR/f_%04d.png" \
  -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 30 -an "$OUT" 2>/dev/null
[ -s "$OUT" ] || { echo "webm encode failed" >&2; exit 4; }
exit 0
