#!/usr/bin/env bash
# One-command MimicMotion setup (Tencent pose-guided human animation) — the engine for the
# PHOTOREAL Maya walkout (task #14). Unlike LTX i2v (which morphs the body), MimicMotion follows a
# DRIVING POSE VIDEO so the reference person walks/points/waves cleanly with no warping.
#
#   ref image (RealVisXL full-body Maya) + driving pose video (a person walking) -> photoreal Maya
#   doing that exact motion. Then chroma-key/rembg -> transparent webm for the tour presenter.
#
# Needs a GPU box with ~16-20GB free VRAM (SVD-based). Run on the freed Thunder A100 AFTER the 558
# avatar batch, or a fresh box — NOT the RunPod prod box (only ~8GB free alongside vLLM+RealVis+LTX).
# Box does NOT persist — pull outputs off immediately.
set -euo pipefail
MM="$HOME/MimicMotion"
VENV="$HOME/mm-venv"

sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg git git-lfs

[ -d "$MM" ] || git clone https://github.com/Tencent/MimicMotion "$MM"
# py3.10 preferred (torch cu wheels). Reuse the box default if 3.10 missing.
PY310=$(command -v python3.10 || command -v python3)
[ -d "$VENV" ] || "$PY310" -m venv "$VENV"
PIP="$VENV/bin/pip"; PY="$VENV/bin/python"
$PIP install -q --upgrade pip

cd "$MM"
# torch first (SVD needs a recent torch; cu121 build is proven on the Thunder A100 hosts)
$PIP install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
# MimicMotion deps (diffusers SVD pipeline + DWPose). requirements.txt pins are usually loose.
$PIP install -q -r requirements.txt || $PIP install -q \
  diffusers==0.27.2 transformers==4.39.2 accelerate omegaconf einops decord \
  opencv-python-headless onnxruntime matplotlib av "huggingface_hub==0.23.4"

# ── weights ──────────────────────────────────────────────────────────────────
# 1) SVD base image-to-video (image encoder + unet)  2) MimicMotion checkpoint  3) DWPose models.
mkdir -p models
unset HF_ENDPOINT
export HF_HOME="$HOME/.cache/huggingface"
# SVD xt 1.1 base (used by MimicMotion)
$VENV/bin/huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --local-dir models/SVD/stable-video-diffusion-img2vid-xt-1-1 --local-dir-use-symlinks False || \
  echo "WARN: SVD download (may need HF token acceptance) — see model card"
# MimicMotion checkpoint
$VENV/bin/huggingface-cli download tencent/MimicMotion MimicMotion_1-1.pth --local-dir models || \
  { echo "fallback: git-lfs pull"; }
# DWPose (pose extractor) — dwpose onnx models
$VENV/bin/huggingface-cli download yzd-v/DWPose --local-dir models/DWPose --local-dir-use-symlinks False || true

$PY -c "import torch,diffusers,cv2,onnxruntime; print('MM_IMPORTS_OK', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "== MimicMotion ready. Render: bash gen_maya_mimicmotion.sh  (needs ref image + driving pose video) =="
