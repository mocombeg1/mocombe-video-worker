#!/usr/bin/env bash
# One-shot setup of the GENERATIVE (Tier-B / LTX text-to-video) worker on a persistent GPU box —
# e.g. the Thunder Compute A100 80GB. Run it ON the instance (ssh in first):
#
#   ssh ubuntu@216.81.200.233 -p 32144 -i <your_key>     # or: tnr connect gd2vgc3k
#   curl -fsSL https://raw.githubusercontent.com/mocombeg1/mocombe-video-worker/main/setup_a100_gen.sh | bash
#
# This is the SAME engine that made the product clips on RunPod — run here it needs no API key.
# Idempotent. LTX weights (~14 GB) cache under $MODEL_CACHE_DIR. Prints the clip-batch command at the end.
set -euo pipefail

export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/models}"
export HF_HOME="$MODEL_CACHE_DIR/hf"
export TORCH_HOME="$MODEL_CACHE_DIR/torch"
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$MODEL_CACHE_DIR"

echo "==> [1/5] system packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 curl wget

echo "==> [2/5] worker code (main branch)"
cd "$HOME"
if [ ! -d mocombe-video-gen ]; then
    git clone https://github.com/mocombeg1/mocombe-video-worker.git mocombe-video-gen
fi
cd "$HOME/mocombe-video-gen"
git fetch origin && git checkout main && git pull --ff-only

echo "==> [3/5] python venv + generative deps (isolated from the avatar venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
# torch for A100 (sm_80). cu128 matches the repo; fall back to the default index for older drivers.
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
  || pip install torch torchvision torchaudio
# Tier-B generative stack (mirror the lean Dockerfile: let pip resolve one consistent set).
pip install runpod==1.7.0 requests==2.32.3 diffusers==0.32.2 transformers==4.46.3 \
    accelerate==1.1.1 sentencepiece==0.2.0 protobuf==5.28.3 safetensors==0.4.4 \
    huggingface_hub==0.26.2 imageio==2.36.0 imageio-ffmpeg==0.5.1 opencv-python-headless==4.10.0.84
# keep the matched torch trio (a dep above can nudge it)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall || true

echo "==> [4/5] download LTX-Video weights -> $MODEL_CACHE_DIR"
python download_models.py ltx || echo "WARN: LTX fetch reported errors; re-run to retry"

echo "==> [5/5] GPU self-check"
python run_local.py diag || true

cat <<EOF

==================== GENERATIVE SETUP COMPLETE ====================
Env:     $HOME/mocombe-video-gen/.venv
Weights: $MODEL_CACHE_DIR/ltx-video

Make ALL 31 Mocombe mega-menu clips right here on the A100:
  cd ~/mocombe-video-gen && source .venv/bin/activate
  export MODEL_CACHE_DIR=$MODEL_CACHE_DIR HF_HOME=$HF_HOME
  python generate_clips_local.py              # -> ./clips/<sect>/<slug>.mp4 (+ .jpg poster)
  python generate_clips_local.py --only personal   # or one section at a time
  python generate_clips_local.py --skip-existing   # resume if interrupted

Then, from your LOCAL machine, copy the whole tree into the website. The layout already
matches what site-addons.js loads (/Images/mega/<sect>/<slug>.mp4), so no renaming:
  scp -P 32144 -i <key> -r 'ubuntu@216.81.200.233:~/mocombe-video-gen/clips/*' \\
      "C:/Users/Garoo/OneDrive/Desktop/MocombeFinancial.com_Website/Images/mega/"
==================================================================
EOF
