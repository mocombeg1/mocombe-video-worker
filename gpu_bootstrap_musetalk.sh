#!/usr/bin/env bash
# One-command MuseTalk bootstrap for a FRESH Thunder A100 (the box does NOT persist across
# stop/start — see feedback: always pull outputs off-box immediately). This is the exact
# WORKING recipe proven 2026-07-27 (render = 36s once set up). Run as ubuntu on a clean box.
#
#   bash gpu_bootstrap_musetalk.sh
#
# It will: create the venv, install the fiddly mmlab stack correctly, and fetch the 4.1GB of
# weights from the REAL huggingface.co (the repo's own download_weights.sh silently fails
# because it hardpins the hf-mirror.com endpoint which is unreachable from Thunder's US host).
set -euo pipefail

MUSETALK_DIR="$HOME/MuseTalk"
VENV="$HOME/musetalk-venv"

# ── 0. system dep ───────────────────────────────────────────────────────────────
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg git wget

# ── 1. repo + venv ──────────────────────────────────────────────────────────────
[ -d "$MUSETALK_DIR" ] || git clone https://github.com/TMElyralab/MuseTalk "$MUSETALK_DIR"
[ -d "$VENV" ] || python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
$PIP install -q --upgrade pip

# ── 2. torch (cu118) + core deps ────────────────────────────────────────────────
$PIP install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 || true
$PIP install -q -r "$MUSETALK_DIR/requirements.txt" || true

# ── 3. THE mmlab fix (this is what breaks every fresh build) ─────────────────────
#   numpy<2 (mmcv/torch built for 1.x; numpy 2.x → "numpy.core.multiarray failed to import")
$PIP install -q "numpy<2" --force-reinstall                 # -> numpy 1.26.4
$PIP install -q -U openmim
$VENV/bin/mim install -q "mmcv==2.0.1" "mmengine"
#   chumpy blocks mmpose's build isolation → install it first without build isolation
$PIP install -q --no-build-isolation chumpy
$VENV/bin/mim install -q "mmdet==3.1.0" "mmpose==1.1.0"

# verify the stack
$PY -c "import numpy,mmcv,mmdet,mmpose,torch; print('mmlab OK', numpy.__version__, 'cuda', torch.cuda.is_available())"

# ── 4. weights — bypass the broken mirror endpoint ──────────────────────────────
#   The repo's download_weights.sh pins HF_ENDPOINT=hf-mirror.com (unreachable from Thunder US);
#   it silently grabs only the 45MB resnet and reports success. Fetch from real huggingface.co.
cd "$MUSETALK_DIR"
mkdir -p models
unset HF_ENDPOINT
$PIP install -q "huggingface_hub[cli]" gdown
export HF_HOME="$HOME/.cache/huggingface"
# If AX42 has a staged tarball, prefer it (minutes vs ~45min). Set AX42=root@37.27.129.228 to use.
if [ "${AX42:-}" ] && ssh -o ConnectTimeout=8 "$AX42" "test -f /root/gpu-assets/musetalk-models.tgz" 2>/dev/null; then
  echo "Staging weights from AX42..."; scp "$AX42:/root/gpu-assets/musetalk-models.tgz" /tmp/mt.tgz && tar xzf /tmp/mt.tgz -C models/
else
  echo "Fetching weights from huggingface.co..."
  $VENV/bin/huggingface-cli download TMElyralab/MuseTalk --local-dir models --local-dir-use-symlinks False
  $VENV/bin/huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae --local-dir-use-symlinks False
  $VENV/bin/huggingface-cli download openai/whisper-tiny --local-dir models/whisper --local-dir-use-symlinks False
  # face-parse 79999_iter.pth: gdrive is quota-blocked → camenduru mirror
  $VENV/bin/huggingface-cli download camenduru/MuseTalk 79999_iter.pth --local-dir models/face-parse-bisent --local-dir-use-symlinks False || \
    wget -q -O models/face-parse-bisent/79999_iter.pth "https://huggingface.co/camenduru/MuseTalk/resolve/main/79999_iter.pth"
fi
[ -f models/musetalkV15/unet.pth ] && echo "UNET OK" || { echo "UNET MISSING — weight fetch failed"; exit 1; }

# ── 5. stage weights to AX42 for next time (best-effort) ─────────────────────────
if [ "${AX42:-}" ]; then
  ( tar czf /tmp/mt.tgz -C models . && ssh -o ConnectTimeout=8 "$AX42" "mkdir -p /root/gpu-assets" && \
    scp /tmp/mt.tgz "$AX42:/root/gpu-assets/musetalk-models.tgz" && echo "Weights staged to AX42" ) || echo "AX42 stage skipped (unreachable)"
fi

echo "== MuseTalk ready. Render a greeting: =="
echo "  printf 'task_0:\\n video_path: \"\$HOME/PORTRAIT.png\"\\n audio_path: \"\$HOME/VOICE.wav\"\\n' > $MUSETALK_DIR/job.yaml"
echo "  cd $MUSETALK_DIR && $PY -m scripts.inference --inference_config job.yaml --result_dir results/out \\"
echo "     --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15"
echo "  # then IMMEDIATELY scp the result mp4 off-box (local + AX42). ~36s render."
