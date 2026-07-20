# RunPod serverless worker — Tier A talking-avatar + Tier B generative (text/image-to-video).
# CUDA 12.8 + cuDNN runtime, Python 3.10, ffmpeg, MuseTalk (primary) + SadTalker (still fallback)
# + Coqui XTTS-v2 (TTS / voice clone) for Tier A; LTX-Video (diffusers) for Tier B.
#
# BLACKWELL / sm_120: this image targets the CUDA 12.8 / cu128 toolchain so it runs on the NVIDIA
# RTX PRO 6000 Blackwell (sm_120) as well as sm_75/80/86/90/100. cu124-and-earlier PyTorch wheels
# carry no sm_120 kernels and fail with "CUDA error: no kernel image is available for execution on
# the device" on matmul — see the torch install below.
#
# DESIGN: keep the image LEAN. Large model weights are NOT baked in — they are downloaded at cold
# start onto a mounted RunPod NETWORK VOLUME at /runpod-volume/models (see README + download_models.py
# + start.sh). That keeps the image ~ a few GB and makes warm-cache cold starts fast.
#
# Build context = REPO ROOT (RunPod's GitHub builder uses the repo root). COPY paths below are
# repo-root-relative (gpu-video-worker/...), and a root .dockerignore trims the context to just
# this folder so the build is fast.
#   docker build -f Dockerfile -t <user>/mocombe-avatar-worker:latest .
#   docker push  <user>/mocombe-avatar-worker:latest
# RunPod Serverless settings: Dockerfile Path = Dockerfile (this repo's root) (leave Build Context
# at the repo root / default).

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
        git ffmpeg libsndfile1 libgl1 libglib2.0-0 \
        ca-certificates wget \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip "setuptools<81" wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (CUDA 12.8 / cu128 build — carries sm_120 Blackwell kernels) ---
# cu128 wheels ship kernels for sm_75/80/86/90/100/120, so the same image runs on RTX 4090 (sm_89)
# through RTX PRO 6000 Blackwell (sm_120) without "no kernel image is available for execution on the
# device". These exact versions are the ones verified against this GPU class (torch 2.11.0+cu128).
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# The nvidia/cuda base image ships its OWN system libcupti (CUDA Profiling Tools Interface),
# which the dynamic linker can find before the pip-installed nvidia-cuda-cupti-cu12 wheel that
# actually matches torch 2.11.0+cu128's compiled ABI -> "undefined symbol:
# cuptiActivityEnableDriverApi, version libcupti.so.12" on `import torch`. Prepend pip's own
# matched cu12 library dirs to LD_LIBRARY_PATH so they win the search over any system copies.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}"

# --- Python deps (RunPod SDK, TTS, IO, shared model deps) ---
COPY requirements.txt /app/requirements.txt
# openai-whisper's wheel build imports pkg_resources, which setuptools 81+ removed. PIP_CONSTRAINT
# applies to PEP517 build-isolation envs too, so this forces the build to use a setuptools that
# still ships pkg_resources. (Belt-and-suspenders with the main-env pin above.)
RUN printf 'setuptools<81\nwheel\n' > /app/build-constraints.txt
ENV PIP_CONSTRAINT=/app/build-constraints.txt
RUN pip install -r /app/requirements.txt

# --- Clone the two lip-sync repos. Pin commits so handler.py's invocation stays stable. ---
# NEEDS-GPU-VERIFY: pin these to the commits you validate on the GPU. Leaving them at default
# branch tips risks the inference CLI/flags drifting out from under handler.py.
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk \
    && git clone https://github.com/OpenTalker/SadTalker.git /app/SadTalker

# Install each repo's own requirements (face-detection, 3DMM, dlib-free deps, etc.).
# CRITICAL: strip torch/torchvision/torchaudio from these files first — SadTalker pins torch==1.12.1
# (cu113, no Blackwell kernels), which would silently DOWNGRADE our cu128 build and reintroduce the
# "no kernel image" failure. We keep our torch and let the rest install. Use || true defensively:
# remaining pins may still conflict; the core packages install. NEEDS-GPU-VERIFY: hard-pin survivors.
RUN grep -vEi '^(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' /app/MuseTalk/requirements.txt  > /tmp/mt-reqs.txt 2>/dev/null || true; \
    pip install -r /tmp/mt-reqs.txt || true
RUN grep -vEi '^(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' /app/SadTalker/requirements.txt > /tmp/st-reqs.txt 2>/dev/null || true; \
    pip install -r /tmp/st-reqs.txt || true

# FINAL torch re-pin (must be the LAST pip step). coqui-tts / tensorflow / jax above can silently
# bump the torch trio to a mismatched wheel. Force the matched cu128 trio back so the installed
# torch/torchvision/torchaudio all stay version-consistent and keep the sm_120 kernels.
# NOTE: deliberately NOT --no-deps here (unlike a plain re-pin) — torch's compiled .so links
# against its transitive nvidia-cuda-cupti-cu12/cudnn/cublas/nccl wheels, and an earlier
# --no-deps re-pin let one of those drift to a mismatched version after MuseTalk/SadTalker/
# coqui-tts installs, producing "undefined symbol: cuptiActivityEnableDriverApi" on `import
# torch`. Letting pip re-resolve the full dependency set restores the matched cu128 set.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# torchcodec: coqui-tts audio IO on torch>=2.9 requires it, and the PyPI default wheel links
# CUDA 13 (libnvrtc.so.13) -> load failure on a CUDA-12 image. Install the cu128 build to match torch.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchcodec

# NUMPY 2 (verified on the Blackwell pod): scipy/scikit-image resolved by coqui-tts are built for
# numpy>=2, and forcing numpy 1.26 breaks scipy imports (np.long). numpy 2.2.x + the SadTalker
# source patches below is the combination that actually renders. (numpy<2 guidance is obsolete.)
RUN pip install --force-reinstall --no-deps "numpy>=2.0,<2.3"

# TIER B (generative / LTX-Video) FINAL re-pin — must come after the MuseTalk/SadTalker requirement
# installs above. SadTalker's 2023 requirements.txt pins an OLD diffusers, which silently DOWNGRADES
# our diffusers==0.32.2 below 0.31 and removes LTXPipeline -> the handler's `_get_ltx_t2v()` dies with
# "ImportError: cannot import name 'LTXPipeline' from 'diffusers'". Re-pin the generative trio LAST.
# --no-deps so the carefully-matched torch/numpy set above is left untouched (diffusers/transformers
# would otherwise try to re-resolve numpy/torch and reintroduce the sm_120 / ABI failures).
# tokenizers is pinned explicitly: transformers 4.46.3 requires tokenizers>=0.20,<0.21, but the
# avatar repos / coqui-tts drag in tokenizers==0.15.2, and a --no-deps transformers reinstall won't
# bump it -> "tokenizers>=0.20,<0.21 is required ... but found 0.15.2" when diffusers imports the
# LTX pipeline (which loads transformers' T5 tokenizer). Pin the matching tokenizers here too.
RUN pip install --no-deps --force-reinstall \
        diffusers==0.32.2 transformers==4.46.3 tokenizers==0.20.3 accelerate==1.1.1

# basicsr (via SadTalker/GFPGAN) imports torchvision.transforms.functional_tensor, which was
# removed in torchvision 0.17+ -> ImportError on `from gfpgan import GFPGANer`. Rewrite the import
# to the current location (rgb_to_grayscale lives in torchvision.transforms.functional now).
RUN find /usr/local/lib/python3.10/dist-packages/basicsr -name '*.py' \
        -exec sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' {} + || true

# SadTalker source predates numpy 2 / OpenCV 5. patch_sadtalker.py applies the four fixes proven
# on the Blackwell pod (numpy alias sweep, VisibleDeprecationWarning removal, ragged trans_params
# array, seamlessClone paste-rect clamp). MuseTalk gets the alias sweep only.
COPY patch_sadtalker.py /app/patch_sadtalker.py
RUN python /app/patch_sadtalker.py /app/SadTalker || true
RUN find /app/MuseTalk -name '*.py' -exec sed -i -E \
        's/\bnp\.float\b/float/g; s/\bnp\.int\b/int/g; s/\bnp\.bool\b/bool/g; s/\bnp\.object\b/object/g; s/\bnp\.str\b/str/g' {} + || true

# MuseTalk expects its weights under ./models and SadTalker under ./checkpoints. We symlink those
# to the network-volume cache so the repos find weights download_models.py placed there.
RUN rm -rf /app/MuseTalk/models /app/SadTalker/checkpoints \
    && ln -s /runpod-volume/models/musetalk            /app/MuseTalk/models \
    && ln -s /runpod-volume/models/sadtalker/checkpoints /app/SadTalker/checkpoints

# --- Worker code ---
COPY rp_handler.py     /app/rp_handler.py
COPY download_models.py /app/download_models.py
COPY start.sh          /app/start.sh
RUN chmod +x /app/start.sh

# start.sh ensures weights exist on the volume (idempotent), then launches the serverless handler.
CMD ["/app/start.sh"]
