#!/usr/bin/env python3
"""
Always-on HTTP render service for the A100 — the production replacement for the RunPod endpoints.

Implements the contract the CRM already speaks (core-crm-backend/src/avatar_render.js):

  POST /render            Authorization: Bearer $RENDER_KEY
       { reference_url, voice_sample_url, script, format:"mp4" }   -> talking avatar
       { prompt, seconds?, image_url?, width?, height? }           -> generative
    -> 202 { "id": "<job id>", "status": "processing" }
  GET  /render/{id}       -> { status: "processing"|"succeeded"|"failed", video_url?, error? }
  GET  /media/{file}.mp4  -> the finished clip (what video_url points at)
  GET  /health            -> { ok, gpu, queue }   (unauthenticated, for uptime checks)

Jobs run one-at-a-time on a background worker thread: a single GPU cannot render two clips at
once, and queueing is far better than two jobs thrashing VRAM and both failing.

Run (generative venv — add the avatar venv separately, see RENDER_MODE below):
    cd ~/mocombe-video-gen && source .venv/bin/activate
    pip install fastapi uvicorn
    export RENDER_KEY="$(openssl rand -hex 32)"     # same value goes in the CRM's AVATAR_RENDER_KEY
    export PUBLIC_BASE_URL="http://216.81.200.233:8090"
    export MODEL_CACHE_DIR=$HOME/models HF_HOME=$HOME/models/hf
    uvicorn server:app --host 0.0.0.0 --port 8090

NOTE ON THE TWO VENVS: Tier-A (avatar) and Tier-B (generative) dependencies cannot coexist, so
run TWO instances of this file — one per venv, on different ports — and point the CRM at
whichever handles the job type. RENDER_MODE=generative|avatar|auto (default auto) makes an
instance reject job types its venv can't serve, instead of failing deep inside the model load.
"""
import os
import threading
import time
import uuid
from collections import OrderedDict
from queue import Queue

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rp_handler import handler

MEDIA_DIR = os.environ.get("RENDER_MEDIA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "media"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
RENDER_KEY = os.environ.get("RENDER_KEY", "")
RENDER_MODE = os.environ.get("RENDER_MODE", "auto").lower()
MAX_JOBS_RETAINED = 500

os.makedirs(MEDIA_DIR, exist_ok=True)

app = FastAPI(title="Mocombe A100 render service")

_jobs = OrderedDict()          # id -> {status, video_url, error, created}
_jobs_lock = threading.Lock()
_queue: "Queue[tuple]" = Queue()


class RenderRequest(BaseModel):
    # talking avatar
    reference_url: str | None = None
    voice_sample_url: str | None = None
    script: str | None = None
    language: str = "en"
    # generative
    prompt: str | None = None
    image_url: str | None = None
    seconds: int | None = None
    width: int | None = None
    height: int | None = None
    format: str = "mp4"
    # motion (pose-guided full-body animation — MimicMotion). Animates the reference portrait with a
    # library motion (walk/point/wave/idle/gesture) WITHOUT morphing the body, unlike generative/LTX.
    motion: str | None = None


def _auth(authorization: str | None):
    """Constant-ish time bearer check. An unset RENDER_KEY means the service refuses to serve at
    all rather than silently running wide open on a public IP."""
    if not RENDER_KEY:
        raise HTTPException(503, "RENDER_KEY is not set on the render service")
    if authorization != f"Bearer {RENDER_KEY}":
        raise HTTPException(401, "bad or missing bearer token")


def _set(job_id, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)
        while len(_jobs) > MAX_JOBS_RETAINED:
            _jobs.popitem(last=False)


def _worker():
    """Single consumer: one render at a time on the one GPU."""
    while True:
        job_id, payload = _queue.get()
        try:
            _set(job_id, status="processing")
            out = handler({"input": payload})
            if isinstance(out, dict) and out.get("error"):
                _set(job_id, status="failed", error=str(out["error"]))
                continue
            b64 = (out or {}).get("video_base64")
            if not b64:
                _set(job_id, status="failed", error="render produced no video")
                continue
            import base64

            # Honor the render's own container: motion returns a transparent-alpha webm (so the clip
            # drops onto any background — the 3D office / tour presenter); avatar/generative are mp4.
            mime = (out or {}).get("mime", "video/mp4")
            ext = "webm" if "webm" in mime else "mp4"
            name = f"{job_id}.{ext}"
            with open(os.path.join(MEDIA_DIR, name), "wb") as f:
                f.write(base64.b64decode(b64))
            _set(job_id, status="succeeded", video_url=f"{PUBLIC_BASE_URL}/media/{name}")
        except Exception as e:  # a bad job must never kill the worker thread
            _set(job_id, status="failed", error=f"{type(e).__name__}: {e}")
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.get("/health")
def health():
    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    except Exception as e:
        gpu = f"unavailable: {e}"
    return {"ok": True, "gpu": gpu, "mode": RENDER_MODE, "queue": _queue.qsize()}


@app.post("/render", status_code=202)
def start_render(req: RenderRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)

    if req.prompt:
        job_type = "generative"
        payload = {"type": "generative", "prompt": req.prompt}
        for k in ("image_url", "seconds", "width", "height"):
            v = getattr(req, k)
            if v is not None:
                payload[k] = v
    elif req.reference_url and req.motion:
        job_type = "motion"
        payload = {"type": "motion", "reference_url": req.reference_url, "motion": req.motion}
    elif req.reference_url and req.script:
        job_type = "avatar"
        payload = {
            "type": "talking_avatar",
            "reference_url": req.reference_url,
            "script": req.script,
            "language": req.language,
        }
        if req.voice_sample_url:
            payload["voice_sample_url"] = req.voice_sample_url
    else:
        raise HTTPException(400, "send {prompt} for generative, {reference_url, script} for a talking avatar, or {reference_url, motion} for pose-guided full-body motion")

    # Reject early if this instance's venv can't serve the type — a clear 400 beats an
    # ImportError five minutes into a model load.
    if RENDER_MODE != "auto" and RENDER_MODE != job_type:
        raise HTTPException(400, f"this instance runs RENDER_MODE={RENDER_MODE}; it cannot serve a {job_type} job")

    job_id = uuid.uuid4().hex
    _set(job_id, status="processing", created=time.time())
    _queue.put((job_id, payload))
    return {"id": job_id, "status": "processing"}


@app.get("/render/{job_id}")
def poll(job_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    return JSONResponse({k: v for k, v in job.items() if k != "created"})


@app.get("/media/{name}")
def media(name: str):
    # basename() so a crafted name can't escape MEDIA_DIR
    path = os.path.join(MEDIA_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    mt = "video/webm" if name.lower().endswith(".webm") else "video/mp4"
    return FileResponse(path, media_type=mt)
