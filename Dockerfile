# RunPod serverless worker — LEAN Tier B (generative text/image-to-video via LTX-Video).
#
# WHY THIS IS TIER-B-ONLY: the combined Tier-A(avatar)+Tier-B(generative) image is dependency-
# conflicted — coqui-tts / MuseTalk / SadTalker force an old numpy + tokenizers==0.15.2, which breaks
# transformers 4.46 / diffusers' LTX pipeline (LTXPipeline import, then the T5 tokenizer, then a
# numpy/pybind ABI error on transformers.models.t5). For the Mocombe product-card clips we only need
# Tier B (t2v), so this image installs ONLY the generative stack and lets pip resolve ONE consistent
# set of transformers + tokenizers + numpy. The talking-avatar (Tier A) path in rp_handler.py is
# import-lazy and simply unused here. To rebuild the full avatar worker, keep the combined Dockerfile
# on a separate branch/endpoint.
#
# CUDA 12.8 / cu128 targets sm_120 (RTX PRO 6000 Blackwell) down through sm_75/80/86/89/90.
# Weights are NOT baked in — download_models.py fetches LTX-Video to the mounted network volume
# (/runpod-volume/models) on first cold start; warm starts hit the cache. See start.sh + README.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REPO_DIR=/app \
    MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/models/hf \
    TORCH_HOME=/runpod-volume/models/torch

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates wget \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (CUDA 12.8 / cu128 build — carries sm_120 Blackwell kernels) ---
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# The nvidia/cuda base ships a system libcupti that can shadow the pip cu12 wheel matching torch
# 2.11.0+cu128's ABI ("undefined symbol: cuptiActivityEnableDriverApi" on `import torch`). Prepend
# pip's matched cu12 lib dirs so they win the dynamic-linker search.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}"

# --- Tier B generative stack. NO --no-deps: let pip resolve ONE consistent set (this is the whole
#     point of the lean image). diffusers/transformers pull their own matching tokenizers; numpy is
#     resolved to a version compatible with torch 2.11 + transformers 4.46 (no forced old numpy that
#     caused the "() -> handle" T5 ABI failure in the combined image). protobuf covers the T5
#     sentencepiece tokenizer conversion; imageio/opencv back diffusers' export_to_video. ---
RUN pip install \
        runpod==1.7.0 \
        requests==2.32.3 \
        diffusers==0.32.2 \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        sentencepiece==0.2.0 \
        protobuf==5.28.3 \
        safetensors==0.4.4 \
        huggingface_hub==0.26.2 \
        imageio==2.36.0 \
        imageio-ffmpeg==0.5.1 \
        opencv-python-headless==4.10.0.84

# Final torch re-pin (LAST pip step) — force the matched cu128 trio back in case a dep above nudged
# torch/vision/audio to a mismatched wheel. Not --no-deps: torch's compiled .so links its transitive
# nvidia-cu12 wheels, and a --no-deps re-pin can let one drift and reintroduce the cupti symbol error.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# --- Worker code ---
COPY rp_handler.py      /app/rp_handler.py
COPY download_models.py /app/download_models.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

# start.sh ensures the LTX weights exist on the volume (idempotent), then launches the handler.
CMD ["/app/start.sh"]
