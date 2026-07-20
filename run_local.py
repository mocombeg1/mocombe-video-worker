#!/usr/bin/env python3
"""
Direct (non-serverless) runner for the worker on a persistent GPU box (e.g. the Thunder A100).

Runs the SAME rp_handler.handler() logic the RunPod serverless entrypoint uses, but from the CLI —
so generative (and, on the avatar branch, talking-avatar) renders work identically whether hosted
on RunPod or run here on the A100.

Set MODEL_CACHE_DIR before running (setup_a100_gen.sh does this) so weights resolve on local disk
instead of the RunPod /runpod-volume default.

  # generative text-to-video (the Mocombe product-clip engine)
  python run_local.py generative --prompt "a calm ocean at sunset, gentle waves" --seconds 4 --out clip.mp4

  # talking-avatar (only works in the avatar venv, which has MuseTalk + XTTS installed)
  python run_local.py avatar --reference_url https://.../portrait.jpg --script "Hi." --out agent.mp4

  # GPU self-diagnostic (card, compute capability, does a CUDA matmul actually run)
  python run_local.py diag
"""
import argparse
import base64
import json
import sys

from rp_handler import handler


def main():
    ap = argparse.ArgumentParser(description="Local runner for the avatar/generative worker.")
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generative", help="text/image-to-video (LTX)")
    g.add_argument("--prompt", required=True)
    g.add_argument("--image_url", help="seed image for image-to-video (optional)")
    g.add_argument("--seconds", type=int, default=4)
    g.add_argument("--out", default="generative_out.mp4")

    a = sub.add_parser("avatar", help="talking-avatar (MuseTalk + XTTS) — avatar venv only")
    a.add_argument("--reference_url", required=True, help="portrait image or short clip URL")
    a.add_argument("--script", required=True, help="text the avatar speaks")
    a.add_argument("--voice_sample_url", help="10-30s clean voice sample to CLONE (optional)")
    a.add_argument("--language", default="en")
    a.add_argument("--out", default="avatar_out.mp4")

    sub.add_parser("diag", help="GPU self-diagnostic")

    args = ap.parse_args()

    if args.mode == "generative":
        job = {"type": "generative", "prompt": args.prompt, "seconds": args.seconds}
        if args.image_url:
            job["image_url"] = args.image_url
    elif args.mode == "avatar":
        job = {"type": "talking_avatar", "reference_url": args.reference_url,
               "script": args.script, "language": args.language}
        if args.voice_sample_url:
            job["voice_sample_url"] = args.voice_sample_url
    else:  # diag
        job = {"type": "diag"}

    out = handler({"input": job})

    if args.mode == "diag":
        print(json.dumps(out, indent=2))
        return

    if isinstance(out, dict) and out.get("error"):
        print("ERROR:", out["error"], file=sys.stderr)
        sys.exit(1)

    if out.get("video_base64"):
        with open(args.out, "wb") as f:
            f.write(base64.b64decode(out["video_base64"]))
        print(f"OK -> {args.out}  ({out.get('duration_seconds')}s, {out.get('model_used')})")
    elif out.get("video_url"):
        print("OK (uploaded):", out["video_url"])
    else:
        print("no video in output:", list(out.keys()), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
