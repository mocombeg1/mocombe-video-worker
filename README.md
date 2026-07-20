# GPU Video Worker — Tier A (talking avatar) + Tier B (generative)

Self-hosted, open-source-only RunPod **serverless** GPU worker. Tier A turns a portrait/clip + a
script into a lip-synced talking-avatar mp4; Tier B turns a text prompt (optionally seeded by an
image) into a short generative video clip (the job type Video Studio's shot-based storyboards use).
No paid APIs, no per-minute SaaS fee — you pay only for GPU seconds while a render runs
(scale-to-zero when idle).

It implements the exact contract the CRM client `core-crm-backend/src/video_render.js` already speaks,
so once this endpoint is live the CRM works **unchanged** — you just set two env vars.

```
CRM  --POST /v2/{endpoint}/run {input: job}-->  this worker (RunPod serverless)
CRM  --GET  /v2/{endpoint}/status/{id}------>   {output: {...}}  (polled until COMPLETED)
worker returns  video_base64 (short clips)  OR  video_url (when an R2/S3 bucket is configured)
CRM saves the mp4 into /media/creative and shows it in Creative Studio
```

Pipeline (Tier A): **Coqui XTTS-v2** (TTS / voice clone) -> **MuseTalk** lip-sync (primary; image
or clip) with **SadTalker** as the still-photo fallback -> ffmpeg mux -> mp4.

Pipeline (Tier B): **LTX-Video** (diffusers `LTXPipeline`/`LTXImageToVideoPipeline`) -> mp4 via
`diffusers.utils.export_to_video`.

---

## Files

| File | Purpose |
|---|---|
| `handler.py` | RunPod `handler(event)`: validate -> download -> TTS -> lip-sync -> base64/URL. |
| `Dockerfile` | CUDA 12.1 + cuDNN8, Python 3.10, ffmpeg, torch cu121, TTS, MuseTalk + SadTalker. |
| `requirements.txt` | Python deps (torch is installed separately in the Dockerfile from the cu121 index). |
| `download_models.py` | Idempotent weight fetch into the network-volume cache. |
| `start.sh` | Cold-start: ensure weights on the volume, then launch the handler. |

---

## Input / output contract (implemented)

**Input** (`event.input`):
```json
{
  "type": "talking_avatar",
  "reference_url": "https://crm.example.com/media/personas/garood.png",
  "voice_sample_url": "https://crm.example.com/media/personas/garood_voice.wav",
  "script": "Hello from Mocombe Financial Services.",
  "voice": "default"
}
```
- `type` — required. Both `talking_avatar` and `generative` are implemented.
- `reference_url` — required for `talking_avatar`. A portrait image (`.png/.jpg/...`) or a short
  clip (`.mp4/...`).
- `voice_sample_url` — optional (`talking_avatar`). 10-30s clean voice sample -> XTTS zero-shot
  **voice clone**. Omitted -> a built-in XTTS speaker.
- `script` — required for `talking_avatar`. Text the avatar speaks (bounded to 5000 chars).
- `prompt` — required for `generative`. Text description of the clip (bounded to 1000 chars).
- `image_url` — optional (`generative`). Seeds image-to-video instead of pure text-to-video.
- `seconds` — optional (`generative`). Clip length, 1-8s, default 4.
- `result_bucket` — optional (either type). When set (and R2/S3 env is configured) the worker
  uploads and returns `video_url` instead of base64.

**Output** (success, short clip):
```json
{ "video_base64": "<base64 mp4>", "mime": "video/mp4", "duration_seconds": 4.2 }
```
**Output** (success, when an object store is configured):
```json
{ "video_url": "https://...", "mime": "video/mp4", "duration_seconds": 4.2, "model_used": "musetalk+xtts" }
```
**Output** (failure): `{ "error": "<message>" }`

RunPod wraps this dict as `{"output": <dict>}` in the `/status` response. `video_render.js` reads
`output.video_url` OR `output.video_base64`/`output.base64` — all three are honored here.

---

## Model weights — sizes & cache location

Weights are **not** baked into the image. They download once into the mounted **network volume**
at `/runpod-volume/models` (set by `MODEL_CACHE_DIR`); subsequent cold starts hit the warm cache.

| Model | Repo | Approx size | Cached at |
|---|---|---|---|
| MuseTalk (UNet/VAE/whisper/dwpose/face-parse) | `TMElyralab/MuseTalk` | ~3-4 GB | `/runpod-volume/models/musetalk` |
| SadTalker (3DMM + checkpoints + enhancer) | `vinthony/SadTalker-V002rc` | ~2-3 GB | `/runpod-volume/models/sadtalker/checkpoints` |
| Coqui XTTS-v2 (TTS / clone) | Coqui `xtts_v2` | ~1.8 GB | `/runpod-volume/models/hf` |
| whisper (MuseTalk audio features) | OpenAI whisper | <0.2 GB | `/runpod-volume/models/hf` |
| LTX-Video (Tier B: transformer + VAE + T5 text encoder) | `Lightricks/LTX-Video` (diffusers-format subfolders only, not the monolithic checkpoints in the same repo) | ~13-14 GB | `/runpod-volume/models/ltx-video` |

Total after first warm-up: ~22-24 GB. **Provision a >= 30 GB network volume** (bumped from 20 GB
now that Tier B's T5 text encoder is included).

---

## Create the RunPod Serverless Endpoint (step by step)

### (a) Build/push the image — two options

**Option 1 — RunPod builds from GitHub (no local Docker / no GPU needed):**
1. Push this repo to GitHub.
2. RunPod console -> **Serverless** -> **New Endpoint** -> **Import Git Repository** (connect GitHub).
3. Select the repo; set **Dockerfile path** to `gpu-video-worker/Dockerfile` and **build context**
   to `gpu-video-worker/`. RunPod builds the image for you.

**Option 2 — build & push yourself (needs Docker; a GPU host is best for a test render):**
```bash
docker build -t <dockerhub-user>/mocombe-avatar-worker:latest gpu-video-worker
docker push  <dockerhub-user>/mocombe-avatar-worker:latest
```
Then in **New Endpoint**, choose **Docker Image** and enter that tag.

### (b) Endpoint hardware / scaling settings
- **GPU:** **24 GB** class (RTX 4090 / A5000) comfortably fits EITHER tier alone (~8-12 GB
  MuseTalk+XTTS, or ~8-10 GB LTX-Video in bf16). If this ONE endpoint serves both job types and a
  single warm worker can lazy-load both models across different requests, prefer a **40-48 GB**
  class (A6000/L40) for headroom — or run two separate endpoints (one per tier) to keep each on a
  24 GB card with no shared-VRAM risk.
- **Container disk:** **20-30 GB** (image + scratch; weights live on the volume, not here).
- **Network volume:** create/attach a **>= 30 GB** volume (see the weights table above — Tier B's
  T5 text encoder is the bulk of the increase from Tier A's original 20 GB). RunPod mounts it at
  **`/runpod-volume`**. Pick the SAME region as the endpoint.
- **Max workers:** **1-2** (raise later if you queue many renders).
- **Idle timeout:** **5-30 s** (scale-to-zero quickly; you only pay while rendering).
- **Execution timeout:** **~600 s** (a render is minutes; keep headroom — CRM polls up to ~10 min).
- **FlashBoot:** **ON** (faster warm starts).

### (c) Environment variables on the endpoint
Required: none (defaults work with a `/runpod-volume` network volume attached).

Optional:
```
MODEL_CACHE_DIR=/runpod-volume/models     # default; change only if baking weights into the image
HF_TOKEN=...                              # only if a gated HF repo ever requires it
# --- Optional R2/S3 output (returns video_url instead of base64; good for longer clips) ---
VIDEO_S3_BUCKET=...                       # enables the upload path
VIDEO_S3_ENDPOINT=https://<acct>.r2.cloudflarestorage.com   # R2; omit for AWS S3
VIDEO_S3_REGION=auto                      # 'auto' for R2, e.g. us-east-1 for S3
AWS_ACCESS_KEY_ID=...                     # boto3 standard credentials
AWS_SECRET_ACCESS_KEY=...
VIDEO_S3_PUBLIC_BASE=https://cdn.example.com   # optional: return {base}/{key} (no presign)
VIDEO_S3_PRESIGN_SECS=86400               # presigned URL TTL when no public base
```

### (d) Test request + expected output
RunPod console -> your endpoint -> **Requests** -> paste and **Run** (or use the `/run` API):
```json
{
  "input": {
    "type": "talking_avatar",
    "reference_url": "https://your-public-host/portrait.png",
    "script": "Hello from Mocombe"
  }
}
```
Expected `/status/{id}` once `COMPLETED`:
```json
{
  "status": "COMPLETED",
  "output": {
    "video_base64": "AAAAIGZ0eXBpc29t....",
    "mime": "video/mp4",
    "duration_seconds": 1.6
  }
}
```
Quick error check (omit `script`) -> `{"output": {"error": "A script is required."}}`.

> The `reference_url` must be reachable from RunPod (a public URL or your CRM's
> `PUBLIC_BASE_URL` + `/media/...`). `video_render.js` already rewrites app-relative media paths to
> absolute URLs before calling `/run`.

### (e) Wire the CRM
1. Copy the **Endpoint ID** from the RunPod endpoint page.
2. On the CRM host set:
   ```
   RUNPOD_API_KEY=<your RunPod API key>
   RUNPOD_VIDEO_ENDPOINT=<the Endpoint ID>
   ```
   (Garood adds the key himself; never commit it.) Restart the backend.

3. **Critical: unset `FAL_KEY`** (or this endpoint never gets used). `video_render.js`'s dispatch
   order checks fal.ai BEFORE RunPod for every job type — `if (falRender.available()) return
   falRender.render(job);` runs unconditionally, ahead of the RunPod branch. As long as `FAL_KEY`
   is set, ALL renders (including generative) keep going to fal.ai regardless of RunPod being
   configured — which, with fal.ai's balance exhausted, means renders keep failing the exact same
   way even after this worker is live. Remove `FAL_KEY` from the production env once this endpoint
   is verified working, then restart the backend again.

---

## NEEDS-VERIFICATION (must be tested on a real GPU)

This worker was authored to MuseTalk / SadTalker / Coqui documented usage on a **CPU-only** dev box.
It **cannot** be run here. Expect ONE live-GPU iteration pass. Specifically verify:

1. **MuseTalk invocation** (`handler.lipsync_musetalk`) — the inference module path
   (`scripts.inference`), flag names (`--inference_config`, `--unet_model_path`, `--version v15`),
   the task-YAML schema, and the results-folder mp4 location vary by commit. **Pin a MuseTalk commit**
   in the Dockerfile once it works.
2. **SadTalker invocation** (`handler.lipsync_sadtalker`) — `inference.py` flags and the
   timestamped output subfolder; confirm `--checkpoint_dir` is honored.
3. **Weight repos / paths** (`download_models.py`) — confirm `TMElyralab/MuseTalk` and
   `vinthony/SadTalker-V002rc` subpaths match what the inference scripts expect; some releases ship a
   `download_weights.sh` you may prefer to call instead. Confirm the `models`/`checkpoints` symlinks.
4. **XTTS speaker id** — the default `speaker="Ana Florence"` must be a valid built-in XTTS-v2
   speaker; adjust if the model rejects it. Voice-clone path (with `voice_sample_url`) is the main one.
5. **torch / repo requirement conflicts** — the Dockerfile installs each repo's `requirements.txt`
   with `|| true`; check what actually fails on the GPU and hard-pin versions so nothing silently
   downgrades torch off the cu121 build.
6. **VRAM / timing** — confirm a 24 GB GPU holds XTTS + MuseTalk together; tune endpoint
   execution/idle timeouts against real render durations.
7. **Cold-start weight download** — first cold start on a fresh volume downloads ~22-24 GB (Tier A
   + Tier B combined); confirm it completes within RunPod's startup window (or pre-warm by running
   one request of each type after attaching the volume).

The Kokoro TTS fallback (noted in `handler.synthesize_speech`) and the base64-vs-URL output paths are
both wired but only the active path is exercised per render.

## NEEDS-VERIFICATION — Tier B (generative)

Lower risk than Tier A above (LTX-Video is consumed through diffusers' own documented, stable
`LTXPipeline`/`LTXImageToVideoPipeline` API, not an ad-hoc research-repo CLI), but still genuinely
unverified on a real GPU — confirmed on this session's shared dev pod that `diffusers==0.32.2`
imports `LTXPipeline` cleanly and begins a real, correct Hub download (no code/dependency error);
the actual generation call was NOT completed end-to-end because that particular pod's local disk
was too small (a pre-existing 30GB container mostly consumed by an unrelated vLLM install — not a
constraint on a properly-sized dedicated endpoint). Specifically still verify on the real endpoint:

1. **VRAM at LTX_WIDTH/LTX_HEIGHT/num_frames** (`handler.py` constants) — tune down if a 24 GB card
   OOMs; LTX-Video's own model card documents lower-VRAM presets if needed.
2. **Per-shot render time** vs. the endpoint's execution timeout (~600s) and video_studio.js's own
   per-shot budget — a slow first inference (kernel autotuning) is normal; steady-state timing on
   warm workers is what matters.
3. **`transformers`/`accelerate`/`diffusers` version pins in requirements.txt** — installed
   alongside Tier A's coqui-tts/whisper stack in the same image; confirm no dependency resolution
   conflict on a real build (the Dockerfile's final torch/numpy force-repin steps already guard
   against the most likely collision, but this combination hasn't been build-tested).
4. **Output quality at the chosen resolution/step count** — 704x480 @ 30 steps is a reasonable
   starting point, not a benchmarked-optimal one; adjust `LTX_STEPS` for a quality/speed tradeoff
   once you can see real output.
