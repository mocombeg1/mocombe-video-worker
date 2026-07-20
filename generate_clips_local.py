#!/usr/bin/env python3
"""
Generate the Mocombe product-card + mega-menu clips ON THIS BOX (the A100), no API key needed —
calls rp_handler.handler() directly. Saves <slug>.mp4 into ./clips/ (override with --out-dir).

Run inside the generative venv on the A100 (setup_a100_gen.sh creates it):
  cd ~/mocombe-video-gen && source .venv/bin/activate
  export MODEL_CACHE_DIR=$HOME/models HF_HOME=$HOME/models/hf
  python generate_clips_local.py                 # all 24
  python generate_clips_local.py health auto life # a subset

Then copy the clips back to your machine's website folder:
  # from your LOCAL machine:
  scp -P 32144 -i <key> 'ubuntu@216.81.200.233:~/mocombe-video-gen/clips/*.mp4' \
      "C:/Users/Garoo/OneDrive/Desktop/MocombeFinancial.com_Website/Images/vid/"
"""
import argparse
import base64
import json
import os
import sys

from rp_handler import handler

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "clip_prompts.json")


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as f:
        data = json.load(f)
    suffix = data.get("_style_suffix", "")
    return {slug: f"{p}, {suffix}".strip(", ") for slug, p in data["clips"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", help="specific clip slugs (default: all)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "clips"))
    ap.add_argument("--seconds", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_prompts()
    slugs = args.slugs or list(prompts.keys())

    ok, fail = [], []
    for i, slug in enumerate(slugs, 1):
        if slug not in prompts:
            print(f"skip unknown slug: {slug}")
            continue
        print(f"\n=== [{i}/{len(slugs)}] {slug} ===\n{prompts[slug][:90]}...")
        out = handler({"input": {"type": "generative", "prompt": prompts[slug],
                                 "seconds": args.seconds}})
        dest = os.path.join(args.out_dir, f"{slug}.mp4")
        if isinstance(out, dict) and out.get("video_base64"):
            with open(dest, "wb") as f:
                f.write(base64.b64decode(out["video_base64"]))
            print(f"  OK -> {dest} ({os.path.getsize(dest)//1024} KB)")
            ok.append(slug)
        else:
            print(f"  FAIL: {out.get('error') if isinstance(out, dict) else out}")
            fail.append(slug)

    print(f"\nDONE. {len(ok)} ok, {len(fail)} failed. -> {args.out_dir}")
    if fail:
        print("  failed:", ", ".join(fail))


if __name__ == "__main__":
    main()
