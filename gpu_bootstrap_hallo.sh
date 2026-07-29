#!/usr/bin/env bash
# One-command HALLO bootstrap (Fudan diffusion talking-head) — the APPROVED avatar-greeting engine
# (Garood signed off 2026-07-28: "her face is finally moving naturally, this is the one").
# Hallo generates the WHOLE face in motion (natural blinks/brows/head) from ONE photo + audio —
# the thing SadTalker/MuseTalk/LivePortrait could not do. Photoreal, self-hosted, ~minutes/clip.
# Box does NOT persist (see feedback) — pull every output off-box immediately.
#
#   bash gpu_bootstrap_hallo.sh            # then render (see bottom)
#
# PROVEN-WORKING pins captured from the box 2026-07-28 (Thunder A100, cu12 host):
set -euo pipefail
HALLO="$HOME/hallo"
VENV="$HOME/hallo-venv"

sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg git

[ -d "$HALLO" ] || git clone https://github.com/fudan-generative-vision/hallo "$HALLO"
# Python 3.10 is REQUIRED (torch cu wheels; box default 3.12 has no matching wheels).
[ -d "$VENV" ] || python3.10 -m venv "$VENV"
PIP="$VENV/bin/pip"; PY="$VENV/bin/python"
$PIP install -q --upgrade pip

cd "$HALLO"
# torch first (this exact build worked: 2.2.2 + cu121; xformers is the cu118 build, still fine)
$PIP install -q torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
$PIP install -q -e .   # hallo's own deps
# Pin the versions proven on the box (avoids the diffusers/transformers drift that breaks Hallo)
$PIP install -q diffusers==0.27.2 transformers==4.39.2 accelerate==0.28.0 insightface==0.7.3 \
                mediapipe==0.10.14 audio-separator==0.17.2 omegaconf einops
# insightface needs onnxruntime at import (face crop). CPU build is plenty — runs once per image.
$PIP install -q onnxruntime==1.17.3
# CRITICAL: huggingface_hub 0.23.4 — hub 1.x removed `huggingface-cli download` and silently
# prints CLI help instead of downloading (empty weights). Same trap that bit MuseTalk/LivePortrait.
$PIP install -q "huggingface_hub==0.23.4"

$PY -c "import torch,diffusers,transformers,accelerate,insightface,omegaconf,einops,audio_separator; print('IMPORTS_OK', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── weights (~11GB) from REAL huggingface.co (NOT hf-mirror) ─────────────────────────────────
unset HF_ENDPOINT
export HF_HOME="$HOME/.cache/huggingface"
# Prefer AX42-staged tarball if present (minutes vs the full HF pull). Set AX42=root@37.27.129.228.
if [ "${AX42:-}" ] && ssh -o ConnectTimeout=8 "$AX42" "test -f /root/gpu-assets/hallo-models.tgz" 2>/dev/null; then
  scp "$AX42:/root/gpu-assets/hallo-models.tgz" /tmp/hallo.tgz && mkdir -p pretrained_models && tar xzf /tmp/hallo.tgz -C pretrained_models/
else
  $VENV/bin/huggingface-cli download fudan-generative-ai/hallo --local-dir pretrained_models --local-dir-use-symlinks False
fi
# expected tree: audio_separator config.json face_analysis hallo motion_module sd-vae-ft-mse stable-diffusion-v1-5 wav2vec
[ -d pretrained_models/hallo ] && [ -d pretrained_models/motion_module ] && echo "HALLO WEIGHTS OK ($(du -sh pretrained_models | cut -f1))" || { echo "WEIGHTS INCOMPLETE"; exit 1; }

# stage weights to AX42 for next time (best-effort)
if [ "${AX42:-}" ]; then
  ( tar czf /tmp/hallo.tgz -C pretrained_models . && ssh -o ConnectTimeout=8 "$AX42" "mkdir -p /root/gpu-assets" && \
    scp /tmp/hallo.tgz "$AX42:/root/gpu-assets/hallo-models.tgz" && echo "staged to AX42" ) || echo "AX42 stage skipped"
fi

cat <<'USAGE'
== Hallo ready. Render an agent greeting (~minutes/clip on A100): ==
  cd ~/hallo
  ~/hallo-venv/bin/python scripts/inference.py \
     --config configs/inference/default.yaml \
     --source_image ~/PORTRAIT.png \
     --driving_audio ~/GREETING.wav \
     --output ~/AGENT_hallo.mp4
  # Source = a clear, roughly frontal face. Audio = clear voice (~1.08x pace).
  # If the GPU wedges (0% util + "capacity busy" on an empty card): pkill -9 -f inference.py, relaunch.
  # THEN scp the mp4 off-box immediately (local + AX42) — the A100 does not persist.
USAGE
