#!/usr/bin/env bash
# =============================================================================
# gpu_bootstrap_production.sh
# ONE-COMMAND production GPU stack for a fresh, DEDICATED Ubuntu GPU box
# (bare-metal: /dev/nvidia* present — NOT a Thunder prototyping proxy).
#
# Stands up the exact three co-resident servers the CRM production path speaks to,
# the way they run today on the Thunder A100 (Thunder deuz9bf6, per
# insurance-agency-crm/COORDINATION.md line 96 — "vLLM(:8000)+RealVisXL(:8500)+
# video(:8090), 53.9GB used/27GB free"):
#
#   1. vLLM brain        :8000   Qwen/Qwen2.5-32B-Instruct-AWQ  (served as qwen2.5-32b)
#   2. RealVisXL image   :8500   SG161222/RealVisXL_V5.0        (Creative-Studio images)
#   3. LTX render server :8090   Lightricks/LTX-Video           (server.py RENDER_MODE=generative)
#
# All three bind the SAME ports the AX42 SSH tunnels expect, so the cutover is
# "re-point the tunnels at this box" (see Reports/GPU_Cutover_Runbook.md) with no
# CRM code change. The render server (#3) also satisfies the worker-pool worker
# contract (PR #315): POST /render + GET /render/{id} + GET /health, Bearer auth.
#
# USAGE (run ON the dedicated box, as a sudo-capable user — 'ubuntu' assumed):
#   export AX42=root@37.27.129.228          # optional: pull staged weight tarballs (fast path)
#   export PUBLIC_HOST=<this box's reachable IP or 127.0.0.1 if only via tunnel>
#   curl -fsSL https://raw.githubusercontent.com/mocombeg1/mocombe-video-worker/main/gpu_bootstrap_production.sh | bash
#     - or -  scp it over and:  bash gpu_bootstrap_production.sh
#
# IDEMPOTENT: re-running skips installs/weights already present and just relaunches.
# Comment-heavy on purpose — every non-obvious step is a lesson paid for on Thunder.
# =============================================================================
set -euo pipefail

# ---- Tunables (override via env) --------------------------------------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
VLLM_NAME="${VLLM_NAME:-qwen2.5-32b}"     # MUST match the CRM's LLM_MODEL (hyphen, not colon!)
VLLM_UTIL="${VLLM_UTIL:-0.70}"            # 0.70 keeps positive KV headroom alongside image+video
VLLM_MAXLEN="${VLLM_MAXLEN:-8192}"
VLLM_PORT="${VLLM_PORT:-8000}"
REALVIS_PORT="${REALVIS_PORT:-8500}"
VIDEO_PORT="${VIDEO_PORT:-8090}"
REALVIS_MODEL="${REALVIS_MODEL:-SG161222/RealVisXL_V5.0}"
REALVIS_VAE="${REALVIS_VAE:-madebyollin/sdxl-vae-fp16-fix}"
PUBLIC_HOST="${PUBLIC_HOST:-127.0.0.1}"   # what PUBLIC_BASE_URL uses for /media links on the video server
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$HOME/models}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_DIR/hf}"
export DEBIAN_FRONTEND=noninteractive
USER_NAME="$(whoami)"
mkdir -p "$MODEL_CACHE_DIR" "$HF_HOME"

log(){ echo "==> $*"; }

# -----------------------------------------------------------------------------
# 0) SANITY: this script is for a TRULY DEDICATED box. On Thunder prototyping,
#    /dev/nvidia* is ABSENT (CUDA is proxied to a shared pool) — the whole point
#    of the migration is to get OFF that. Warn loudly but don't hard-fail (a
#    Thunder *Production*-mode box also lacks the files yet is dedicated+SLA'd).
# -----------------------------------------------------------------------------
if ls /dev/nvidia* >/dev/null 2>&1; then
  log "[0/8] /dev/nvidia* present — dedicated silicon confirmed."
else
  echo "!!  WARNING: /dev/nvidia* NOT found. If this is Thunder *prototyping*, STOP —"
  echo "!!  it will hang under pool congestion (the exact defect we're migrating away from)."
  echo "!!  Acceptable ONLY on Thunder *Production* mode (dedicated+SLA, files still virtualized)."
  echo "!!  Continuing in 10s; Ctrl-C to abort."; sleep 10
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "!!  nvidia-smi failed — install the NVIDIA driver first, then re-run."; exit 1; }

# -----------------------------------------------------------------------------
# 1) System packages (Python 3.10 REQUIRED — the cu12 torch wheels have no 3.12
#    build; this bit MuseTalk/LivePortrait/Hallo). deadsnakes provides 3.10 on 22.04+.
# -----------------------------------------------------------------------------
log "[1/8] system packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends software-properties-common curl wget git ffmpeg \
    libgl1 libglib2.0-0 openssl ca-certificates
if ! command -v python3.10 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
fi
sudo apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev \
    python3-venv python3-pip python3-dev

# vLLM/torch on some driver images ship libcuda.so.1 but not the linker name libcuda.so.
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
  sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so && log "symlinked libcuda.so"
fi

# -----------------------------------------------------------------------------
# 2) Worker code (this repo) — supplies server.py + rp_handler.py + download_models.py
# -----------------------------------------------------------------------------
log "[2/8] worker code"
cd "$HOME"
[ -d mocombe-video-gen ] || git clone https://github.com/mocombeg1/mocombe-video-worker.git mocombe-video-gen
cd "$HOME/mocombe-video-gen" && git fetch origin && git checkout main && git pull --ff-only
VIDEO_DIR="$HOME/mocombe-video-gen"

# =============================================================================
# 3) vLLM BRAIN (:8000) — venv, weights, launch flags (from gpu-node/thunder-vllm-setup.sh)
# =============================================================================
log "[3/8] vLLM brain venv + install"
if [ ! -x "$HOME/vllm-env/bin/vllm" ]; then
  python3.10 -m venv "$HOME/vllm-env"
  "$HOME/vllm-env/bin/pip" install --upgrade pip setuptools wheel >/dev/null
  "$HOME/vllm-env/bin/pip" install vllm huggingface_hub
else
  log "vLLM venv present — skipping install"
fi
# Persisted API key (same value the CRM orchestrator sends as LLM_API_KEY).
[ -f "$HOME/vllm_token.env" ] || { echo "VLLM_KEY=$(openssl rand -hex 24)" > "$HOME/vllm_token.env"; chmod 600 "$HOME/vllm_token.env"; }
# Prefetch the AWQ weights so first boot doesn't stall the health check (~19GB).
"$HOME/vllm-env/bin/huggingface-cli" download "$VLLM_MODEL" --local-dir "$MODEL_CACHE_DIR/qwen-awq" >/dev/null 2>&1 \
  || log "WARN: Qwen prefetch reported errors; vLLM will pull lazily on first boot"

# =============================================================================
# 4) RealVisXL IMAGE SERVER (:8500) — venv + reconstructed FastAPI server.
#    NOTE: realvis_server.py is NOT committed to any repo — it lived only on the
#    Thunder box. This is a faithful reconstruction of the contract the CRM speaks
#    (creative_gen.js renderSelfHosted + rp_handler._realvis_keyframe):
#      POST /generate  {prompt,negative_prompt,width,height,steps,guidance}  X-Auth: <token>
#         -> image/png bytes
#      GET  /health -> {ok, gpu}
#    If you have the original realvis_server.py, drop it in ~/realvis and it wins.
# =============================================================================
log "[4/8] RealVisXL image server venv + code + weights"
if [ ! -x "$HOME/imgvenv/bin/python" ]; then
  python3.10 -m venv "$HOME/imgvenv"
  "$HOME/imgvenv/bin/pip" install --upgrade pip wheel >/dev/null
  # torch 2.5.1+cu124 + diffusers 0.39.0 = the proven RealVisXL_V5.0 stack (per HANDOFF_3D_Office.md).
  "$HOME/imgvenv/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
    || "$HOME/imgvenv/bin/pip" install torch torchvision
  "$HOME/imgvenv/bin/pip" install "diffusers==0.39.0" "transformers==4.46.3" accelerate safetensors \
    "huggingface_hub==0.34.0" fastapi "uvicorn[standard]" pillow
fi
[ -f "$HOME/realvis_token.env" ] || { echo "REALVIS_TOKEN=$(openssl rand -hex 24)" > "$HOME/realvis_token.env"; chmod 600 "$HOME/realvis_token.env"; }

mkdir -p "$HOME/realvis"
if [ ! -f "$HOME/realvis/realvis_server.py" ]; then
cat > "$HOME/realvis/realvis_server.py" <<'PYEOF'
#!/usr/bin/env python3
"""Reconstructed RealVisXL_V5.0 image server (SDXL). Matches the CRM contract:
POST /generate {prompt,negative_prompt,width,height,steps,guidance}  header X-Auth: <token> -> PNG."""
import io, os
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

MODEL   = os.environ.get("REALVIS_MODEL", "SG161222/RealVisXL_V5.0")
VAE     = os.environ.get("REALVIS_VAE", "madebyollin/sdxl-vae-fp16-fix")
TOKEN   = os.environ.get("REALVIS_TOKEN", "")
_pipe = None

def _load():
    global _pipe
    if _pipe is None:
        vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
        _pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.float16,
                                                          use_safetensors=True, variant="fp16")
        _pipe = _pipe.to("cuda")
    return _pipe

app = FastAPI(title="RealVisXL image server")

class Gen(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 5.0

@app.get("/health")
def health():
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    except Exception as e:
        gpu = f"unavailable: {e}"
    return {"ok": True, "gpu": gpu, "model": MODEL, "loaded": _pipe is not None}

@app.post("/generate")
def generate(req: Gen, x_auth: str | None = Header(default=None)):
    if TOKEN and x_auth != TOKEN:
        raise HTTPException(401, "bad or missing X-Auth")
    pipe = _load()
    img = pipe(prompt=req.prompt[:300], negative_prompt=(req.negative_prompt or "")[:300],
               width=req.width, height=req.height, num_inference_steps=req.steps,
               guidance_scale=req.guidance).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
PYEOF
fi
# Prefetch RealVisXL + fp16 VAE (~7GB) so the first image isn't a cold 3-minute download.
HF_HOME="$HF_HOME" "$HOME/imgvenv/bin/python" - <<PYEOF || log "WARN: RealVisXL prefetch errored; will pull on first request"
from huggingface_hub import snapshot_download
for r in ("$REALVIS_MODEL", "$REALVIS_VAE"):
    try: snapshot_download(r, allow_patterns=["*.safetensors","*.json","*.txt"]); print("cached", r)
    except Exception as e: print("skip", r, e)
PYEOF

# =============================================================================
# 5) LTX / RENDER SERVER (:8090) — generative video venv + weights.
#    This is the repo's own setup_a100_gen.sh path, driven by server.py.
#    huggingface_hub is pinned to 0.26.2 here (snapshot_download is fine); the
#    0.23.4 CLI-download trap only bites `huggingface-cli download` (Hallo path).
# =============================================================================
log "[5/8] LTX generative venv + LTX weights"
cd "$VIDEO_DIR"
if [ ! -x "$VIDEO_DIR/.venv/bin/python" ]; then
  python3.10 -m venv .venv
  ./.venv/bin/pip install --upgrade pip setuptools wheel >/dev/null
  ./.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128 || ./.venv/bin/pip install torch torchvision torchaudio
  ./.venv/bin/pip install runpod==1.7.0 requests==2.32.3 diffusers==0.32.2 transformers==4.46.3 \
      accelerate==1.1.1 sentencepiece==0.2.0 protobuf==5.28.3 safetensors==0.4.4 \
      "huggingface_hub==0.34.0" imageio==2.36.0 imageio-ffmpeg==0.5.1 opencv-python-headless==4.10.0.84 \
      fastapi "uvicorn[standard]"
  ./.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128 --force-reinstall || true
fi
MODEL_CACHE_DIR="$MODEL_CACHE_DIR" HF_HOME="$HF_HOME" ./.venv/bin/python download_models.py ltx \
  || log "WARN: LTX fetch reported errors; re-run to retry (server retries lazily too)"
[ -f "$VIDEO_DIR/render.key" ] || { openssl rand -hex 32 > "$VIDEO_DIR/render.key"; chmod 600 "$VIDEO_DIR/render.key"; }

# -----------------------------------------------------------------------------
#    Optional AX42-staged fast path: if a prior box tarred the weights to AX42,
#    pull them instead of re-downloading 40GB from HuggingFace.
# -----------------------------------------------------------------------------
if [ "${AX42:-}" ] && ssh -o ConnectTimeout=8 "$AX42" "test -d /root/gpu-assets" 2>/dev/null; then
  for t in qwen-awq realvis ltx-video; do
    if ssh -o ConnectTimeout=8 "$AX42" "test -f /root/gpu-assets/$t.tgz" 2>/dev/null && [ ! -d "$MODEL_CACHE_DIR/$t" ]; then
      log "pulling AX42-staged $t.tgz"; scp "$AX42:/root/gpu-assets/$t.tgz" /tmp/$t.tgz \
        && mkdir -p "$MODEL_CACHE_DIR/$t" && tar xzf /tmp/$t.tgz -C "$MODEL_CACHE_DIR/$t" || log "WARN: $t stage pull failed"
    fi
  done
fi

# =============================================================================
# 6) SUPERVISORS — one restart-loop per server. These self-heal a crashed server
#    (the Thunder box had no systemd; a bare-metal box does, but the loops are the
#    canonical relaunch logic start_gpu_servers.sh calls, kept identical for parity).
# =============================================================================
log "[6/8] supervisor daemons"
mkdir -p "$HOME/logs"

cat > "$HOME/vllm_daemon.sh" <<EOF
#!/usr/bin/env bash
set -a; . "$HOME/vllm_token.env"; set +a
export HF_HOME="$HF_HOME"
while true; do
  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu VLLM_USE_FLASHINFER_SAMPLER=0 \\
  "$HOME/vllm-env/bin/vllm" serve "$VLLM_MODEL" \\
    --quantization awq_marlin --max-model-len $VLLM_MAXLEN --max-num-seqs 16 --max-num-batched-tokens 8192 \\
    --gpu-memory-utilization $VLLM_UTIL --enforce-eager --host 0.0.0.0 --port $VLLM_PORT \\
    --api-key "\$VLLM_KEY" --served-model-name "$VLLM_NAME" >> "$HOME/logs/vllm.log" 2>&1
  echo "[\$(date)] vLLM exited — restarting in 5s" >> "$HOME/logs/vllm.log"; sleep 5
done
EOF

cat > "$HOME/realvis_daemon.sh" <<EOF
#!/usr/bin/env bash
set -a; . "$HOME/realvis_token.env"; set +a
export HF_HOME="$HF_HOME" REALVIS_MODEL="$REALVIS_MODEL" REALVIS_VAE="$REALVIS_VAE"
while true; do
  "$HOME/imgvenv/bin/uvicorn" realvis_server:app --host 0.0.0.0 --port $REALVIS_PORT \\
    --app-dir "$HOME/realvis" >> "$HOME/logs/realvis.log" 2>&1
  echo "[\$(date)] RealVisXL exited — restarting in 5s" >> "$HOME/logs/realvis.log"; sleep 5
done
EOF

cat > "$HOME/video_daemon.sh" <<EOF
#!/usr/bin/env bash
export RENDER_KEY="\$(cat "$VIDEO_DIR/render.key")"
export PUBLIC_BASE_URL="http://$PUBLIC_HOST:$VIDEO_PORT"
export RENDER_MODE=generative
export MODEL_CACHE_DIR="$MODEL_CACHE_DIR" HF_HOME="$HF_HOME"
export REALVIS_URL="http://127.0.0.1:$REALVIS_PORT"   # keyframe seeding hits the local image server
export REALVIS_TOKEN="\$(. "$HOME/realvis_token.env"; echo \$REALVIS_TOKEN)"
cd "$VIDEO_DIR"
while true; do
  ./.venv/bin/uvicorn server:app --host 0.0.0.0 --port $VIDEO_PORT >> "$HOME/logs/video.log" 2>&1
  echo "[\$(date)] video server exited — restarting in 5s" >> "$HOME/logs/video.log"; sleep 5
done
EOF
chmod +x "$HOME"/{vllm_daemon,realvis_daemon,video_daemon}.sh

# One-shot relaunch helper (parity with Thunder's start_gpu_servers.sh; systemd calls the daemons).
cat > "$HOME/start_gpu_servers.sh" <<EOF
#!/usr/bin/env bash
# Relaunch all three production servers (idempotent; kills stale then starts).
for s in vllm realvis video; do pkill -f "\${s}_daemon.sh" 2>/dev/null || true; done; sleep 2
setsid "$HOME/vllm_daemon.sh"    < /dev/null &
setsid "$HOME/realvis_daemon.sh" < /dev/null &
setsid "$HOME/video_daemon.sh"   < /dev/null &
echo "launched vllm(:$VLLM_PORT) realvis(:$REALVIS_PORT) video(:$VIDEO_PORT) — logs in $HOME/logs" | tee -a "$HOME/gpu_autostart.log"
EOF
chmod +x "$HOME/start_gpu_servers.sh"

# =============================================================================
# 7) BOOT PERSISTENCE — systemd units (a dedicated Ubuntu box HAS systemd, unlike
#    Thunder's s6). One unit per server so they restart independently.
# =============================================================================
log "[7/8] systemd units (start on boot)"
mk_unit(){  # $1 name  $2 daemon-script  $3 human desc
  sudo tee "/etc/systemd/system/$1.service" >/dev/null <<EOF
[Unit]
Description=$3
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=/usr/bin/env bash $HOME/$2
Restart=always
RestartSec=5
# GPU jobs can run long; don't let systemd kill on a slow model load.
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
}
mk_unit gpu-vllm    vllm_daemon.sh    "Mocombe vLLM brain (qwen2.5-32b)"
mk_unit gpu-realvis realvis_daemon.sh "Mocombe RealVisXL image server"
mk_unit gpu-video   video_daemon.sh   "Mocombe LTX/render server (server.py)"
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-vllm gpu-realvis gpu-video

# =============================================================================
# 8) HEALTH — give models time to load, then probe each port.
# =============================================================================
log "[8/8] waiting for services to warm (models load lazily; up to ~3 min)"
VKEY="$(. "$HOME/vllm_token.env"; echo "$VLLM_KEY")"
for i in $(seq 1 36); do
  v=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $VKEY" "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null || echo 000)
  r=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$REALVIS_PORT/health" 2>/dev/null || echo 000)
  d=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$VIDEO_PORT/health" 2>/dev/null || echo 000)
  echo "   vllm=$v realvis=$r video=$d  (attempt $i/36)"
  [ "$v" = "200" ] && [ "$r" = "200" ] && [ "$d" = "200" ] && break
  sleep 5
done

cat <<EOF

================= PRODUCTION STACK BOOTSTRAP COMPLETE =================
Services (bound to the ports the AX42 tunnels expect):
  vLLM brain     http://127.0.0.1:$VLLM_PORT/v1   model=$VLLM_NAME   key: ~/vllm_token.env
  RealVisXL      http://127.0.0.1:$REALVIS_PORT/generate            token: ~/realvis_token.env (X-Auth)
  LTX render     http://127.0.0.1:$VIDEO_PORT/render                key: ~/mocombe-video-gen/render.key (Bearer)

Manage:   sudo systemctl status gpu-vllm gpu-realvis gpu-video
Logs:     tail -f ~/logs/{vllm,realvis,video}.log
Relaunch: ~/start_gpu_servers.sh

NEXT (see insurance-agency-crm/Reports/GPU_Cutover_Runbook.md):
  Re-point the 3 AX42 systemd tunnels at THIS box's SSH endpoint, then verify a
  Maya chat + an image gen + a render from the CRM. The three secrets above are
  what the CRM must carry: LLM_API_KEY, SELFHOST_IMAGE_TOKEN, AVATAR_RENDER_KEY.
======================================================================
EOF
