#!/usr/bin/env bash
# One-shot setup of the Tier-A avatar worker (MuseTalk + XTTS voice clone) on a persistent GPU box
# — e.g. the Thunder Compute A100 80GB. Run it ON the instance (ssh in first), NOT locally:
#
#   ssh ubuntu@216.81.200.233 -p 32144 -i <your_key>     # or: tnr connect gd2vgc3k
#   curl -fsSL https://raw.githubusercontent.com/mocombeg1/mocombe-video-worker/tier-a-avatar/bootstrap_a100.sh | bash
#
# Idempotent: re-running skips work already done. Weights (~8 GB) cache under $MODEL_CACHE_DIR.
# After it finishes it prints a one-line test command.
set -euo pipefail

export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/models}"
export HF_HOME="$MODEL_CACHE_DIR/hf"
export TORCH_HOME="$MODEL_CACHE_DIR/torch"
export COQUI_TOS_AGREED=1
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$MODEL_CACHE_DIR"

echo "==> [1/6] system packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev git ffmpeg libsndfile1 libgl1 libglib2.0-0 curl wget

echo "==> [2/6] worker code (tier-a-avatar branch)"
cd "$HOME"
if [ ! -d mocombe-avatar-worker ]; then
    git clone https://github.com/mocombeg1/mocombe-video-worker.git mocombe-avatar-worker
fi
cd "$HOME/mocombe-avatar-worker"
git fetch origin && git checkout tier-a-avatar && git pull --ff-only

echo "==> [3/6] python venv + deps (isolated from any generative venv)"
python3 -m venv .venv-avatar
# shellcheck disable=SC1091
source .venv-avatar/bin/activate
pip install --upgrade pip "setuptools<81" wheel
# torch for A100 (sm_80). cu128 matches the repo; fall back to the default index if the box's driver
# is older. A100 kernels ship in every recent build, so either works.
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
  || pip install torch torchvision torchaudio
PIP_CONSTRAINT=<(printf 'setuptools<81\nwheel\n') pip install -r requirements.txt

echo "==> [4/6] MuseTalk repo + numpy-2 alias sweep"
[ -d MuseTalk ] || git clone https://github.com/TMElyralab/MuseTalk.git
rm -rf MuseTalk/models
ln -s "$MODEL_CACHE_DIR/musetalk" MuseTalk/models
find MuseTalk -name '*.py' -exec sed -i -E \
    's/\bnp\.float\b/float/g; s/\bnp\.int\b/int/g; s/\bnp\.bool\b/bool/g; s/\bnp\.object\b/object/g; s/\bnp\.str\b/str/g' {} + || true
# keep our matched torch trio (coqui-tts / MuseTalk reqs can nudge it)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall || true
pip install --force-reinstall --no-deps "numpy>=2.0,<2.3" || true

echo "==> [5/6] download weights (MuseTalk 6-source set + XTTS) -> $MODEL_CACHE_DIR"
python download_models.py musetalk xtts || echo "WARN: some weights failed; re-run this script to retry"

echo "==> [6/6] GPU self-check"
python run_local.py diag || true

cat <<EOF

==================== SETUP COMPLETE ====================
Env is at:  $HOME/mocombe-avatar-worker/.venv-avatar
Weights at: $MODEL_CACHE_DIR

Test a talking-avatar render (swap in a real portrait URL + your text):
  cd ~/mocombe-avatar-worker && source .venv-avatar/bin/activate
  export MODEL_CACHE_DIR=$MODEL_CACHE_DIR HF_HOME=$HF_HOME COQUI_TOS_AGREED=1
  python run_local.py avatar \\
    --reference_url "https://YOUR_PORTRAIT.jpg" \\
    --script "Hello, I am your Mocombe Financial agent." \\
    --out agent.mp4

Then copy agent.mp4 back to your machine to view it.
========================================================
EOF
