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
# openai-whisper is an sdist: pip must RUN its setup.py, which imports pkg_resources (deleted in
# setuptools 81+). A constraints file does not save you here — pip's build isolation builds each
# sdist in a FRESH environment and installs the newest setuptools into it, ignoring both the
# image's `setuptools<81` and PIP_CONSTRAINT. Observed 2026-08-13:
#   ModuleNotFoundError: No module named 'pkg_resources'
#   ERROR: Failed to build 'openai-whisper' when getting requirements to build wheel
# `--no-build-isolation` makes the build use THIS image's interpreter and its pinned
# setuptools<81 (installed above), which is the version whisper's setup.py expects.
RUN printf 'setuptools<81\nwheel\n' > /app/build-constraints.txt
ENV PIP_CONSTRAINT=/app/build-constraints.txt
RUN pip install --no-build-isolation -r /app/requirements.txt

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

# NO numpy override here. There used to be a `pip install --no-deps "numpy>=2.0,<2.3"` at this
# point, carried over from the Tier-B/LTX image where scikit-image wanted numpy 2. It is wrong for
# THIS image and was the actual defect behind a render that died 200s in:
#
#   coqui-tts 0.26.0 depends on numpy<2.0 and >=1.25.2      <- pip, stating it plainly
#
# requirements.txt had resolved the whole stack correctly for numpy 1; this step then dropped
# numpy 2 underneath it with --no-deps, so pip never saw the conflict and the build stayed green
# while the image was broken. It surfaced as thinc's `numpy.dtype size changed` and then cv2's
# `_ARRAY_API not found` — two symptoms, one cause. numpy is pinned in requirements.txt now.

# PROVE the numpy ABI is coherent AT BUILD TIME. A broken combination must fail here — loudly, in a
# build log, for free — rather than in a render that has already burned GPU seconds and a user's
# patience. Imports the exact module whose failure took the endpoint down, plus the stack that
# reaches it. If this line ever goes red, the build is telling you the truth about the image.
RUN python - <<'PY'
import numpy
print("BUILD-CHECK numpy", numpy.__version__, "from", numpy.__file__, flush=True)
# numpy 1.x is REQUIRED here, not merely tolerated: coqui-TTS pins numpy<2.0, and coqui-TTS is the
# whole point of the avatar image. Assert it so a future numpy-2 "upgrade" fails here rather than
# in a render.
assert numpy.__version__.startswith("1."), f"coqui-TTS requires numpy 1.x, got {numpy.__version__}"
import thinc.backends.numpy_ops   # the exact import that raised at runtime
import spacy, cv2, scipy, librosa
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer  # the caller that pulled spacy in
print("BUILD-CHECK numpy/thinc/spacy/cv2/scipy/librosa/TTS all import cleanly", flush=True)
PY

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
