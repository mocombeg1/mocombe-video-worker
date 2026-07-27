#!/usr/bin/env bash
# LivePortrait bootstrap for a FRESH Thunder A100 — the "living portrait" layer that gives an
# agent's face LIFE (head motion + eye blinks) before MuseTalk fixes the lips. Proven working
# 2026-07-28. Pairs with gpu_bootstrap_musetalk.sh. Box does NOT persist — pull outputs off-box.
#
#   bash gpu_bootstrap_liveportrait.sh
#
# THE GREETING PIPELINE (per agent):
#   1. LivePortrait: portrait.png + a driving video (head motion + blinks) -> living_face.mp4
#   2. MuseTalk:     living_face.mp4 + greeting.wav (clear voice, ~1.08x pace) -> final greeting
#      (MuseTalk overwrites the mouth for tight lip-sync; LivePortrait's upper-face life survives)
set -euo pipefail

LP="$HOME/LivePortrait"
VENV="$HOME/liveportrait-venv"

sudo apt-get install -y -qq ffmpeg git
[ -d "$LP" ] || git clone -q https://github.com/KwaiVGI/LivePortrait "$LP"

# CRITICAL: python3.10, NOT the box default 3.12 — torch 2.0.1+cu118 has no cp312 wheel.
[ -d "$VENV" ] || python3.10 -m venv "$VENV"
PIP="$VENV/bin/pip"; PY="$VENV/bin/python"
$PIP install -q --upgrade pip
$PIP install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd "$LP" && $PIP install -q -r requirements.txt
# CRITICAL: pin huggingface_hub<1.0 — hub 1.x removed the `huggingface-cli download` syntax and
# the download silently prints help text instead of fetching (weights end up empty).
$PIP install -q "huggingface_hub==0.23.4"

$PY -c "import torch; assert torch.cuda.is_available(); print('LP torch CUDA OK')"

# weights (~1.9GB) from REAL huggingface.co (no mirror)
unset HF_ENDPOINT
$VENV/bin/huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights \
  --exclude "*.git*" "README.md" "docs"
[ -f pretrained_weights/liveportrait/base_models/warping_module.pth ] && echo "LP WEIGHTS OK" || { echo "LP weights MISSING"; exit 1; }

echo "== LivePortrait ready. Animate a portrait to life (CALM settings): =="
echo "  # driving: CONTINUOUS ~15s clip d10.mp4 (blinks + calm head motion; d0 is 3s -> loops, avoid)."
echo "  # CRITICAL: --driving-multiplier 0.6 — at the default 1.0 the eyes DART side-to-side (gaze"
echo "  #   transfer too strong; Garood rejected 1.0). 0.6 keeps blinks + gentle head motion, no darting."
echo "  #   (drop toward 0.5 if a given driving still darts; --animation-region {exp,pose,lip,eyes,all} exists"
echo "  #    if you need finer control, but multiplier 0.6 was the fix.)"
echo "  $PY inference.py -s ~/PORTRAIT.png -d assets/examples/driving/d10.mp4 -o ~/lp_out \\"
echo "     --driving-multiplier 0.6 --flag-stitching --flag-crop-driving-video"
echo "  # -> ~/lp_out/PORTRAIT--d10.mp4 (living face). Feed THAT as MuseTalk video_path (see musetalk bootstrap)."
