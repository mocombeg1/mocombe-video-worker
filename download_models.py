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
    Download EVERY weight MuseTalk v1.5 inference needs into CACHE_DIR/musetalk (which the Dockerfile
    symlinks to MuseTalk/models). This mirrors MuseTalk's own download_weights.sh exactly — the main
    UNet is NOT enough; inference also loads sd-vae, whisper, dwpose, syncnet and the face-parse
    (bisenet) backbones, each from a different source:

      musetalkV15/{unet.pth, musetalk.json}              <- TMElyralab/MuseTalk
      sd-vae/{config.json, diffusion_pytorch_model.bin}  <- stabilityai/sd-vae-ft-mse
      whisper/{config.json, pytorch_model.bin, preprocessor_config.json} <- openai/whisper-tiny
      dwpose/dw-ll_ucoco_384.pth                          <- yzd-v/DWPose
      syncnet/latentsync_syncnet.pt                       <- ByteDance/LatentSync
      face-parse-bisent/79999_iter.pth                    <- Google Drive (gdown)
      face-parse-bisent/resnet18-5c106cde.pth             <- download.pytorch.org

    hf_hub_download / urlretrieve skip files already present, so re-running is cheap (idempotent).
    """
    from huggingface_hub import hf_hub_download
    import urllib.request

    root = _ensure(os.path.join(CACHE_DIR, "musetalk"))
    print(f"[models] MuseTalk (+ aux backbones) -> {root}", flush=True)

    def hf(repo, filename, subdir=None):
        target = _ensure(os.path.join(root, subdir)) if subdir else root
        hf_hub_download(repo_id=repo, filename=filename, local_dir=target,
                        local_dir_use_symlinks=False)

    # Main UNet (v1.5): filenames carry the musetalkV15/ prefix, local_dir=root preserves it.
    hf("TMElyralab/MuseTalk", "musetalkV15/musetalk.json")
    hf("TMElyralab/MuseTalk", "musetalkV15/unet.pth")
    # sd-vae-ft-mse
    hf("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae")
    hf("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "sd-vae")
    # whisper-tiny (MuseTalk's audio feature extractor)
    for f in ("config.json", "pytorch_model.bin", "preprocessor_config.json"):
        hf("openai/whisper-tiny", f, "whisper")
    # DWPose (face/pose detection)
    hf("yzd-v/DWPose", "dw-ll_ucoco_384.pth", "dwpose")
    # SyncNet (v1.5 lip-sync scoring)
    hf("ByteDance/LatentSync", "latentsync_syncnet.pt", "syncnet")

    # Face-parse (bisenet) — not on HF: resnet18 from pytorch.org, 79999_iter.pth from Google Drive.
    fp = _ensure(os.path.join(root, "face-parse-bisent"))
    resnet = os.path.join(fp, "resnet18-5c106cde.pth")
    if not os.path.exists(resnet):
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth", resnet)
    bisenet = os.path.join(fp, "79999_iter.pth")
    if not os.path.exists(bisenet):
        try:
            import gdown
            gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=bisenet, quiet=False)
        except Exception as e:  # noqa: BLE001
            print(f"[models] WARN: face-parse 79999_iter.pth fetch failed ({e}); "
                  "face-parsing may be degraded", flush=True)


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
