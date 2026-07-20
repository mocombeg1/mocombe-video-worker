"""
Idempotent model-weight fetcher for the Tier A talking-avatar worker.

Downloads into MODEL_CACHE_DIR (default /runpod-volume/models) so the heavy weights live on a
RunPod NETWORK VOLUME, not in the image. Safe to run:
  - at IMAGE BUILD time onto a baked path (set MODEL_CACHE_DIR=/app/models), OR
  - at COLD START into the mounted volume (the Dockerfile CMD can call this before the handler).

Re-running is cheap: huggingface_hub.snapshot_download / hf_hub_download skip files already present.

Approx download sizes (one-time, cached thereafter):
  MuseTalk (UNet + VAE + whisper + face parsing/detection)  ~3-4 GB
  SadTalker (3DMM + checkpoints + GFPGAN/enhancer)           ~2-3 GB
  Coqui XTTS-v2 (downloaded lazily by TTS on first synth)    ~1.8 GB  (into HF_HOME)
  whisper (tiny/base used by MuseTalk feature extractor)     <0.2 GB
  LTX-Video (Tier B: transformer + VAE + T5 text encoder)    ~13-14 GB
Total network-volume footprint after first warm-up: ~22-24 GB. Provision a >= 30 GB volume
(bumped from the Tier-A-only 20 GB now that Tier B's T5 text encoder is included).
"""

import os
import sys

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
os.environ.setdefault("HF_HOME", os.path.join(CACHE_DIR, "hf"))


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def fetch_musetalk():
    """
    MuseTalk weights live in the HF repo TMElyralab/MuseTalk (v1.5 layout: models/musetalkV15/...,
    plus sd-vae, whisper, dwpose, face-parse-bisent, syncnet). We mirror the whole repo's `models`
    tree into CACHE_DIR/musetalk so the inference script finds them via the paths handler.py passes.

    NEEDS-GPU-VERIFY: confirm the exact repo subpaths against the MuseTalk commit pinned in the
    Dockerfile. MuseTalk also documents a download_weights.sh — if the repo layout shifts, prefer
    running that script instead of this function.
    """
    from huggingface_hub import snapshot_download

    dst = _ensure(os.path.join(CACHE_DIR, "musetalk"))
    print(f"[models] MuseTalk -> {dst}", flush=True)
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=dst,
        allow_patterns=["*.pth", "*.json", "*.bin", "*.safetensors", "*.txt", "*.yaml"],
        local_dir_use_symlinks=False,
    )
    # Auxiliary backbones MuseTalk depends on (downloaded into the same tree by its script).
    # sd-vae-ft-mse (VAE) and whisper are pulled lazily by the inference code via HF_HOME on first
    # run; nothing else to do here.


def fetch_sadtalker():
    """
    SadTalker checkpoints (vico-st / OpenTalker/SadTalker on HF). Land them under
    CACHE_DIR/sadtalker/checkpoints which handler.py passes as --checkpoint_dir.

    NEEDS-GPU-VERIFY: SadTalker historically shipped weights via gdown/its download script
    (scripts/download_models.sh). The HF mirror 'vinthony/SadTalker-V002rc' / 'OpenTalker/SadTalker'
    is the no-gdown path; confirm filenames (mapping_*.pth.tar, SadTalker_V0.0.2_*.safetensors,
    GFPGANv1.4, etc.) for the pinned commit.
    """
    from huggingface_hub import snapshot_download

    dst = _ensure(os.path.join(CACHE_DIR, "sadtalker", "checkpoints"))
    print(f"[models] SadTalker -> {dst}", flush=True)
    snapshot_download(
        repo_id="vinthony/SadTalker-V002rc",
        local_dir=dst,
        allow_patterns=["*.safetensors", "*.pth", "*.pth.tar", "*.tar", "*.json"],
        local_dir_use_symlinks=False,
    )


def fetch_ltx():
    """
    LTX-Video (Tier B generative) — diffusers-format repo (transformer/vae/text_encoder/tokenizer/
    scheduler subfolders), NOT the monolithic single-file checkpoints also hosted in the same repo.
    Mirrors the whole diffusers-format tree into CACHE_DIR/ltx-video; handler._ltx_source() prefers
    this local copy and falls back to pulling straight from the Hub if it's absent.

    Sizeable: the T5 text encoder is the bulk of this download (~9 GB). Total ~13-14 GB.
    """
    from huggingface_hub import snapshot_download

    dst = _ensure(os.path.join(CACHE_DIR, "ltx-video"))
    print(f"[models] LTX-Video -> {dst}", flush=True)
    snapshot_download(
        repo_id="Lightricks/LTX-Video",
        local_dir=dst,
        allow_patterns=[
            "model_index.json",
            "scheduler/*", "tokenizer/*",
            "transformer/*.safetensors", "transformer/*.json",
            "vae/*.safetensors", "vae/*.json",
            "text_encoder/*.safetensors", "text_encoder/*.json",
        ],
        local_dir_use_symlinks=False,
    )


def warm_xtts():
    """
    Trigger Coqui XTTS-v2 weight download into HF_HOME so the first real job is fast. Cheap to skip
    if it fails at build time (it will download lazily on first synth). ~1.8 GB.
    """
    try:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.utils.manage import ModelManager

        print("[models] XTTS-v2 (Coqui) ...", flush=True)
        ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    except Exception as e:
        print(f"[models] XTTS warm skipped ({e}); will download lazily on first job", flush=True)


def main():
    _ensure(CACHE_DIR)
    which = sys.argv[1:] or ["musetalk", "sadtalker", "xtts", "ltx"]
    if "musetalk" in which:
        fetch_musetalk()
    if "sadtalker" in which:
        fetch_sadtalker()
    if "xtts" in which:
        warm_xtts()
    if "ltx" in which:
        fetch_ltx()
    print("[models] done.", flush=True)


if __name__ == "__main__":
    main()
