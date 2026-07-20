# RunPod serverless worker — Tier A talking avatar (voice-clone lip-sync).  [avatar branch]
#
# Powers the CRM's "clone myself" feature: MuseTalk (lip-sync, primary/only engine — SadTalker is
# intentionally omitted; rp_handler.lipsync() uses MuseTalk exclusively) driven by Coqui XTTS-v2
# (multilingual TTS + zero-shot voice clone). NO diffusers/LTX here — the Tier-B generative stack
# conflicts with coqui-tts's transformers/tokenizers/numpy pins and lives in the lean Tier-B worker
# on `main`. Keeping the two stacks in SEPARATE images is what makes both actually work.
#
# CUDA 12.8 / cu128 targets sm_120 (RTX PRO 6000 Blackwell) down through sm_75/80/86/89/90.
# Weights (MuseTalk UNet/VAE/whisper/dwpose + XTTS) download to the mounted network volume
# (/runpod-volume/models) on first cold start via download_models.py; warm starts hit the cache.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REPO_DIR=/app \
    MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/models/hf \
    TORCH_HOME=/runpod-volume/models/torch \
    COQUI_TOS_AGREED=1

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ffmpeg libsndfile1 libgl1 libglib2.0-0 ca-certificates wget \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip "setuptools<81" wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (CUDA 12.8 / cu128 build — carries sm_120 Blackwell kernels) ---
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Prepend pip's matched cu12 lib dirs so torch 2.11.0+cu128's ABI-matched cupti/cudnn/cublas win
# the linker search over the base image's system copies ("undefined symbol: cuptiActivityEnableDriverApi").
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}"

# --- Python deps (RunPod SDK, coqui-tts, whisper, IO) ---
COPY requirements.txt /app/requirements.txt
# openai-whisper's build imports pkg_resources (removed in setuptools 81+). Constrain the PEP517
# build env to a setuptools that still ships it.
RUN printf 'setuptools<81\nwheel\n' > /app/build-constraints.txt
ENV PIP_CONSTRAINT=/app/build-constraints.txt
RUN pip install -r /app/requirements.txt

# --- Clone MuseTalk (lip-sync). NEEDS-GPU-VERIFY: pin to a commit you validate on the GPU so the
#     inference CLI/flags in rp_handler.lipsync_musetalk stay stable. ---
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
# Install MuseTalk's own requirements, stripping torch/torchvision/torchaudio so they can't downgrade
# our cu128 build (some pins are cu113 / no Blackwell kernels). || true: remaining pins may conflict;
# the core deps install. NEEDS-GPU-VERIFY: hard-pin survivors once validated.
RUN grep -vEi '^(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' /app/MuseTalk/requirements.txt > /tmp/mt-reqs.txt 2>/dev/null || true; \
    pip install -r /tmp/mt-reqs.txt || true

# FINAL torch re-pin (LAST pip step): coqui-tts / MuseTalk reqs can silently bump the torch trio.
# Force the matched cu128 trio back (NOT --no-deps: let pip re-resolve torch's transitive nvidia-cu12
# wheels so cupti/cudnn/cublas stay ABI-consistent).
RUN pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# torchcodec: coqui-tts audio IO on torch>=2.9 needs it; the default PyPI wheel links CUDA 13 ->
# load failure on a CUDA-12 image. Install the cu128 build to match torch.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchcodec

# NUMPY 2: scipy/scikit-image resolved by coqui-tts are built for numpy>=2; forcing numpy 1.26 breaks
# their imports. numpy 2.2.x is the combination that renders.
RUN pip install --force-reinstall --no-deps "numpy>=2.0,<2.3"

# MuseTalk source predates numpy 2 (np.float/np.int/np.bool aliases removed). Sweep them.
RUN find /app/MuseTalk -name '*.py' -exec sed -i -E \
        's/\bnp\.float\b/float/g; s/\bnp\.int\b/int/g; s/\bnp\.bool\b/bool/g; s/\bnp\.object\b/object/g; s/\bnp\.str\b/str/g' {} + || true

# MuseTalk expects its weights under ./models; symlink to the network-volume cache download_models.py fills.
RUN rm -rf /app/MuseTalk/models \
    && ln -s /runpod-volume/models/musetalk /app/MuseTalk/models

# --- Worker code ---
COPY rp_handler.py      /app/rp_handler.py
COPY download_models.py /app/download_models.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
