"""
RunPod serverless GPU worker — Tier A AI talking-avatar video.

Fulfills the contract consumed by core-crm-backend/src/video_render.js:

  CRM calls   POST  https://api.runpod.ai/v2/{endpoint}/run   body {"input": <job>}
  CRM polls   GET   https://api.runpod.ai/v2/{endpoint}/status/{id}
  CRM reads   output.video_url   OR   output.video_base64 / output.base64

INPUT  (event["input"]):
  {
    "type": "talking_avatar",     # required
    "reference_url": "https://...",# required — portrait image (.jpg/.png) or short clip (.mp4)
    "voice_sample_url": "https://.." # optional — 10-30s clean voice sample for cloning
    "script": "Hello from Mocombe",# required — text the avatar speaks
    "voice": "default",            # optional — XTTS speaker name when no voice_sample_url
    "result_bucket": "...",        # optional — if set AND R2/S3 env configured -> upload, return video_url
  }

  type == "generative" (Tier B — text/image-to-video, matches video_studio.js's shot renders):
  {
    "type": "generative",
    "mode": "t2v" | "i2v",          # informational only — i2v is auto-selected when image_url is set
    "prompt": "a calm ocean at sunset, gentle waves",   # required
    "image_url": "https://...",    # optional — seeds i2v (image-to-video) instead of pure t2v
    "seconds": 4,                  # optional, 1-8, default 4
    "result_bucket": "...",        # optional, same as above
  }

OUTPUT (success):
  { "video_base64": "<base64 mp4>", "mime": "video/mp4", "duration_seconds": <float> }
  ...or when an object store is configured and result_bucket is requested:
  { "video_url": "https://...", "mime": "video/mp4", "duration_seconds": <float>, "model_used": "..." }

OUTPUT (failure):
  { "error": "<human-readable message>" }

NOTE ON GPU VERIFICATION:
  Everything marked  # NEEDS-GPU-VERIFY  cannot be exercised on a CPU-only dev box. The MuseTalk /
  SadTalker invocation flags (CLI arg names, config paths, output filename) are written to the
  models' documented usage and WILL likely need one live iteration pass on a real RTX 4090 / A5000.
"""

import base64
import glob
import os
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlparse

import requests
import runpod

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------
# Weights are cached on a RunPod NETWORK VOLUME mounted at /runpod-volume (see README).
# We keep them off the image so the image stays lean and cold starts hit a warm cache.
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")
REPO_DIR = os.environ.get("REPO_DIR", "/app")
MUSETALK_DIR = os.path.join(REPO_DIR, "MuseTalk")
SADTALKER_DIR = os.path.join(REPO_DIR, "SadTalker")

# HuggingFace + Torch caches also live on the network volume so XTTS / whisper weights persist.
os.environ.setdefault("HF_HOME", os.path.join(CACHE_DIR, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_DIR, "torch"))
# Coqui TTS license auto-accept (XTTS-v2 is CPML; non-interactive in a worker).
os.environ.setdefault("COQUI_TOS_AGREED", "1")

MAX_SCRIPT_CHARS = 5000          # bound the TTS workload
DOWNLOAD_TIMEOUT = 120           # seconds for fetching reference / voice sample
SUBPROCESS_TIMEOUT = 540         # seconds for a single inference call (worker exec timeout governs overall)

# Tier B (generative) defaults. 704x480 is a native-ish LTX-Video resolution (both divisible by
# 32, as the model requires). LTX wants num_frames of the form 8k+1; FPS=24 is its documented
# default. Bounded to <= 8s so a single shot render stays well inside the endpoint's execution
# timeout and VRAM budget.
LTX_WIDTH = 704
LTX_HEIGHT = 480
LTX_FPS = 24
LTX_STEPS = 30
LTX_DEFAULT_SECONDS = 4
LTX_MAX_SECONDS = 8
LTX_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, low resolution"
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi", ".mkv")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _log(msg):
    print(f"[worker] {msg}", flush=True)


def _ext_from_url(url, default=".bin"):
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower() if ext else default


def _download(url, dest):
    """Stream a URL to a local file. Raises on non-2xx / empty body."""
    _log(f"download {url[:80]} -> {dest}")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
    if size == 0:
        raise ValueError("downloaded file was empty")
    return dest


def _run(cmd, cwd=None):
    """Run a subprocess, streaming output; raise with captured tail on failure."""
    _log("exec: " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({proc.returncode}): {tail}")
    return proc.stdout


def _ffprobe_duration(path):
    """Best-effort media duration in seconds (returns None if ffprobe unavailable)."""
    try:
        out = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ])
        return float(out.strip())
    except Exception:
        return None


def _newest(folder, exts):
    """Return the newest file in `folder` matching `exts`, or None."""
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# --------------------------------------------------------------------------------------
# Text-to-speech (Coqui XTTS-v2; clone when a voice sample is supplied)
# --------------------------------------------------------------------------------------
# We lazily build a single XTTS model and reuse it across warm invocations.
_TTS = None


def _get_tts():
    global _TTS
    if _TTS is not None:
        return _TTS
    # Imported lazily so the import cost (and any CUDA init) only happens on first real job.
    from TTS.api import TTS  # noqa: WPS433  (lazy import is intentional)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"loading XTTS-v2 on {device}")
    # NEEDS-GPU-VERIFY: first run downloads ~1.8GB XTTS weights into HF_HOME (network volume).
    _TTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _TTS


def synthesize_speech(script, out_wav, voice_sample=None, language="en"):
    """
    TTS `script` to `out_wav`.

    Primary: Coqui XTTS-v2.
      - voice_sample given  -> zero-shot voice CLONE from that wav.
      - no voice_sample     -> a built-in XTTS speaker.

    FALLBACK NOTE (Kokoro): if XTTS init fails on the target GPU (e.g. transformers/torch ABI
    mismatch), swap this body for Kokoro:
        from kokoro import KPipeline
        pipe = KPipeline(lang_code="a")              # 'a' = American English
        audio = next(pipe(script, voice="af_heart"))[2]
        import soundfile as sf; sf.write(out_wav, audio, 24000)
    Kokoro has no voice cloning (fixed speakers) but is small/fast and a reliable safety net.
    """
    tts = _get_tts()
    if voice_sample:
        _log("XTTS clone from supplied voice sample")
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language=language,
            file_path=out_wav,
        )
    else:
        # A neutral default speaker bundled with XTTS-v2.
        _log("XTTS default speaker (no voice sample)")
        tts.tts_to_file(
            text=script,
            speaker="Ana Florence",   # NEEDS-GPU-VERIFY: confirm a valid built-in speaker id
            language=language,
            file_path=out_wav,
        )
    if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
        raise RuntimeError("TTS produced no audio")
    return out_wav


# --------------------------------------------------------------------------------------
# Lip-sync — MuseTalk (primary, image OR clip) / SadTalker (still-photo fallback)
# --------------------------------------------------------------------------------------
def lipsync_musetalk(reference, audio_wav, work_dir):
    """
    Drive `reference` (image or short clip) with `audio_wav` using MuseTalk.

    MuseTalk's documented inference flow (realtime branch / inference script):
      whisper audio-feature extraction -> UNet latent prediction -> VAE decode -> ffmpeg mux wav.

    MuseTalk takes a small YAML listing (video_path, audio_path) pairs. We synthesize one and run
    the official inference entrypoint, then pick up the muxed mp4 from its results folder.

    NEEDS-GPU-VERIFY (high): the exact script path, flag names, config schema and result location
    below follow MuseTalk's README but vary by release/commit. Expect ONE iteration pass on a live
    GPU to lock the invocation. Pin a commit in the Dockerfile to keep this stable.
    """
    results_dir = os.path.join(work_dir, "musetalk_out")
    os.makedirs(results_dir, exist_ok=True)

    # MuseTalk drives a VIDEO. If the reference is a still portrait, synthesize a short static clip
    # (matched to the speech length, 25fps) for MuseTalk to animate.
    driver = reference
    if _is_still_image(reference):
        dur = _ffprobe_duration(audio_wav) or 5.0
        driver = os.path.join(work_dir, "driver.mp4")
        _run([
            "ffmpeg", "-y", "-loop", "1", "-i", reference, "-t", f"{dur:.2f}",
            "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", driver,
        ])

    # MuseTalk inference config (one task). Keys per MuseTalk's example configs/inference/test.yaml.
    cfg_path = os.path.join(work_dir, "musetalk_task.yaml")
    with open(cfg_path, "w") as f:
        f.write(
            "task_0:\n"
            f"  video_path: \"{driver}\"\n"
            f"  audio_path: \"{audio_wav}\"\n"
        )

    # MuseTalk v1.5 inference entrypoint. Weights resolved from CACHE_DIR (see download_models.py),
    # which MuseTalk expects under ./models. We symlink that in the Dockerfile / at cold start.
    # MuseTalk requires --ffmpeg_path (a DIRECTORY containing the ffmpeg binary), per its README.
    import shutil
    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg_dir = os.path.dirname(ffmpeg_bin) if ffmpeg_bin else "/usr/bin"
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", cfg_path,
        "--result_dir", results_dir,
        "--unet_model_path", os.path.join(CACHE_DIR, "musetalk", "musetalkV15", "unet.pth"),
        "--unet_config", os.path.join(CACHE_DIR, "musetalk", "musetalkV15", "musetalk.json"),
        "--version", "v15",
        "--ffmpeg_path", ffmpeg_dir,
    ]
    _run(cmd, cwd=MUSETALK_DIR)  # NEEDS-GPU-VERIFY

    out = _newest(results_dir, (".mp4",))
    if not out:
        raise RuntimeError("MuseTalk produced no mp4")
    return out


def lipsync_sadtalker(image, audio_wav, work_dir):
    """
    Single-still-photo fallback. SadTalker animates one portrait from audio.

    SadTalker's documented CLI:
      python inference.py --driven_audio a.wav --source_image p.png --result_dir out [--still --preprocess full]

    NEEDS-GPU-VERIFY (high): SadTalker writes a timestamped mp4 into a subfolder of --result_dir;
    we grab the newest mp4. Checkpoints come from CACHE_DIR/sadtalker (symlinked to ./checkpoints).
    """
    results_dir = os.path.join(work_dir, "sadtalker_out")
    os.makedirs(results_dir, exist_ok=True)
    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", audio_wav,
        "--source_image", image,
        "--result_dir", results_dir,
        "--still",
        "--preprocess", "full",
        "--checkpoint_dir", os.path.join(CACHE_DIR, "sadtalker", "checkpoints"),
    ]
    _run(cmd, cwd=SADTALKER_DIR)  # NEEDS-GPU-VERIFY

    out = _newest(results_dir, (".mp4",))
    if not out:
        raise RuntimeError("SadTalker produced no mp4")
    return out


def _is_still_image(path):
    return path.lower().endswith(IMAGE_EXTS)


def lipsync(reference, audio_wav, work_dir):
    """
    MuseTalk-only. SadTalker is intentionally disabled: it is 2023 code that requires numpy<1.24,
    which conflicts with the numpy>=1.25 that the Coqui XTTS voice engine needs in this same image.
    MuseTalk (2024) is the primary, higher-quality engine and is numpy-1.26 compatible. For a still
    image we synthesize a short static driving clip first (MuseTalk drives a video, not a lone photo).
    """
    return lipsync_musetalk(reference, audio_wav, work_dir)


# --------------------------------------------------------------------------------------
# Optional object-store upload (R2 / S3) for longer / production clips
# --------------------------------------------------------------------------------------
def maybe_upload(mp4_path):
    """
    If S3/R2 env is configured, upload and return a public/presigned URL; else None.

    Env (all optional — set them to enable the URL path):
      VIDEO_S3_BUCKET           bucket name (or set result_bucket in the job to override)
      VIDEO_S3_ENDPOINT         e.g. https://<accountid>.r2.cloudflarestorage.com  (R2) — omit for AWS S3
      VIDEO_S3_REGION           e.g. auto (R2) or us-east-1
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   credentials (boto3 standard)
      VIDEO_S3_PUBLIC_BASE      optional CDN/base; if set, returns {base}/{key} (no presign)
      VIDEO_S3_PRESIGN_SECS     presigned-URL TTL (default 86400)

    Returns the URL string, or None if not configured / on any failure (caller then uses base64).
    """
    bucket = os.environ.get("VIDEO_S3_BUCKET")
    if not bucket:
        return None
    try:
        import boto3  # lazy — only needed when uploading

        endpoint = os.environ.get("VIDEO_S3_ENDPOINT") or None
        region = os.environ.get("VIDEO_S3_REGION", "auto")
        s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region)
        key = f"creative/render_{int(time.time())}.mp4"
        s3.upload_file(mp4_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})

        public_base = os.environ.get("VIDEO_S3_PUBLIC_BASE")
        if public_base:
            return public_base.rstrip("/") + "/" + key
        ttl = int(os.environ.get("VIDEO_S3_PRESIGN_SECS", "86400"))
        return s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl
        )
    except Exception as e:
        _log(f"upload failed, falling back to base64: {e}")
        return None


# --------------------------------------------------------------------------------------
# Main job
# --------------------------------------------------------------------------------------
def _talking_avatar(job, work_dir):
    reference_url = job["reference_url"]
    voice_sample_url = job.get("voice_sample_url")
    script = job["script"].strip()[:MAX_SCRIPT_CHARS]
    language = job.get("language", "en")

    # 1) Fetch inputs
    ref_path = os.path.join(work_dir, "reference" + _ext_from_url(reference_url, ".png"))
    _download(reference_url, ref_path)
    if not (ref_path.lower().endswith(IMAGE_EXTS) or ref_path.lower().endswith(VIDEO_EXTS)):
        # Unknown extension: assume image (MuseTalk/SadTalker will still try).
        _log("reference has unknown extension; treating as image")

    voice_sample = None
    if voice_sample_url:
        voice_sample = os.path.join(work_dir, "voice" + _ext_from_url(voice_sample_url, ".wav"))
        _download(voice_sample_url, voice_sample)

    # 2) TTS
    audio_wav = os.path.join(work_dir, "speech.wav")
    synthesize_speech(script, audio_wav, voice_sample=voice_sample, language=language)

    # 3) Lip-sync -> mp4
    mp4 = lipsync(ref_path, audio_wav, work_dir)

    duration = _ffprobe_duration(mp4) or _ffprobe_duration(audio_wav)

    # 4) Return — prefer an uploaded URL when an object store is configured and requested,
    #    otherwise inline base64 (fine for short Tier A clips; video_render.js handles both).
    result = {"mime": "video/mp4", "model_used": "musetalk+xtts"}
    if duration:
        result["duration_seconds"] = round(duration, 2)

    want_url = bool(job.get("result_bucket") or os.environ.get("VIDEO_S3_BUCKET"))
    url = maybe_upload(mp4) if want_url else None
    if url:
        result["video_url"] = url
        return result

    with open(mp4, "rb") as f:
        result["video_base64"] = base64.b64encode(f.read()).decode("ascii")
    return result


# --------------------------------------------------------------------------------------
# Generative (Tier B) — text/image-to-video via LTX-Video (diffusers)
# --------------------------------------------------------------------------------------
# Unlike MuseTalk/SadTalker above (ad-hoc CLI invocations of research repos), LTX-Video is
# consumed through diffusers' own documented, stable pipeline API (LTXPipeline /
# LTXImageToVideoPipeline) — the call shape here is low-risk. What genuinely needs a live-GPU
# pass is VRAM headroom and per-shot timing at LTX_WIDTH/LTX_HEIGHT/num_frames on the actual
# target GPU class; tune those constants if a real render OOMs or is too slow for the endpoint's
# execution timeout.
_LTX_T2V = None
_LTX_I2V = None


def _ltx_source():
    """Prefer a pre-fetched network-volume copy (download_models.py fetch_ltx); else pull from HF."""
    local = os.path.join(CACHE_DIR, "ltx-video")
    return local if os.path.isdir(local) else "Lightricks/LTX-Video"


def _get_ltx_t2v():
    global _LTX_T2V
    if _LTX_T2V is not None:
        return _LTX_T2V
    from diffusers import LTXPipeline  # noqa: WPS433  (lazy import is intentional)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"loading LTX-Video (t2v) on {device}")
    _LTX_T2V = LTXPipeline.from_pretrained(_ltx_source(), torch_dtype=torch.bfloat16).to(device)
    return _LTX_T2V


def _get_ltx_i2v():
    global _LTX_I2V
    if _LTX_I2V is not None:
        return _LTX_I2V
    from diffusers import LTXImageToVideoPipeline  # noqa: WPS433
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"loading LTX-Video (i2v) on {device}")
    _LTX_I2V = LTXImageToVideoPipeline.from_pretrained(_ltx_source(), torch_dtype=torch.bfloat16).to(device)
    return _LTX_I2V


def _ltx_num_frames(seconds):
    """LTX-Video requires num_frames = 8k + 1. Round to the nearest valid value for the target duration."""
    k = max(1, round((seconds * LTX_FPS - 1) / 8))
    return 8 * k + 1


def _generative(job, work_dir):
    """
    Text-to-video (t2v), or image-to-video (i2v) when image_url is given, via LTX-Video. Matches
    the job shape core-crm-backend/src/video_render.js already sends for type:'generative'.
    """
    from diffusers.utils import export_to_video

    prompt = str(job.get("prompt") or "").strip()[:1000]
    if not prompt:
        raise RuntimeError("A prompt is required for a generative video")

    try:
        seconds = float(job.get("seconds") or LTX_DEFAULT_SECONDS)
    except (TypeError, ValueError):
        seconds = LTX_DEFAULT_SECONDS
    seconds = max(1, min(LTX_MAX_SECONDS, seconds))
    num_frames = _ltx_num_frames(seconds)

    image_url = job.get("image_url")
    gen_kwargs = dict(
        prompt=prompt,
        negative_prompt=LTX_NEGATIVE_PROMPT,
        width=LTX_WIDTH,
        height=LTX_HEIGHT,
        num_frames=num_frames,
        num_inference_steps=LTX_STEPS,
    )
    if image_url:
        from diffusers.utils import load_image

        ref_path = os.path.join(work_dir, "seed" + _ext_from_url(image_url, ".png"))
        _download(image_url, ref_path)
        pipe = _get_ltx_i2v()
        result = pipe(image=load_image(ref_path), **gen_kwargs)
    else:
        pipe = _get_ltx_t2v()
        result = pipe(**gen_kwargs)

    frames = result.frames[0]
    out_mp4 = os.path.join(work_dir, "generative_out.mp4")
    export_to_video(frames, out_mp4, fps=LTX_FPS)

    duration = _ffprobe_duration(out_mp4) or seconds
    result_out = {"mime": "video/mp4", "model_used": "ltx-video", "duration_seconds": round(duration, 2)}

    want_url = bool(job.get("result_bucket") or os.environ.get("VIDEO_S3_BUCKET"))
    url = maybe_upload(out_mp4) if want_url else None
    if url:
        result_out["video_url"] = url
        return result_out

    with open(out_mp4, "rb") as f:
        result_out["video_base64"] = base64.b64encode(f.read()).decode("ascii")
    return result_out


def _diag():
    """
    GPU self-diagnostic. Returns the exact card, its compute capability, the archs this torch
    build supports, and whether a trivial CUDA kernel actually runs. This isolates a torch/arch
    mismatch (matmul_ok == False) from a sub-dependency's compiled-op mismatch (matmul_ok == True
    but a real render still fails). Never raises.
    """
    info = {"type": "diag"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["device_capability"] = f"sm_{cap[0]}{cap[1]}"
            try:
                info["arch_list"] = torch.cuda.get_arch_list()
            except Exception as e:  # noqa: BLE001
                info["arch_list_error"] = str(e)
            try:
                x = torch.randn(64, 64, device="cuda")
                info["matmul_sample"] = round(float((x @ x).sum().item()), 3)
                info["matmul_ok"] = True
            except Exception as e:  # noqa: BLE001
                info["matmul_ok"] = False
                info["matmul_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        info["trace"] = traceback.format_exc()[-1500:]
    return info


def handler(event):
    """RunPod serverless entrypoint. Never raises — always returns a dict (output or {error})."""
    try:
        job = (event or {}).get("input") or {}
        job_type = job.get("type")

        if job_type == "diag":
            return _diag()

        if job_type == "generative":
            with tempfile.TemporaryDirectory(prefix="job_") as work_dir:
                return _generative(job, work_dir)

        if job_type != "talking_avatar":
            return {"error": f"Unknown render job type: {job_type!r}"}

        # Validate required fields (mirror video_render.js so the CRM gets matching messages).
        if not job.get("reference_url"):
            return {"error": "reference_url is required (a portrait image or short clip)."}
        if not str(job.get("script") or "").strip():
            return {"error": "A script is required."}

        with tempfile.TemporaryDirectory(prefix="job_") as work_dir:
            return _talking_avatar(job, work_dir)

    except requests.HTTPError as e:
        return {"error": f"Could not download an input URL: {e}"}
    except subprocess.TimeoutExpired:
        return {"error": "Render timed out (try a shorter script / smaller clip)."}
    except Exception as e:
        _log("UNHANDLED:\n" + traceback.format_exc())
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
