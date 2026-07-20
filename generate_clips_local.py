#!/usr/bin/env python3
"""
Generate the Mocombe mega-menu hover-preview clips ON THIS BOX (the A100), no API key needed —
calls rp_handler.handler() directly.

Output layout mirrors exactly what site-addons.js loads, so the whole tree drops straight into
the site's Images/mega/ folder:

    clips/personal/life.mp4   + life.jpg    ->  /Images/mega/personal/life.mp4
    clips/business/group-dv.mp4 + .jpg      ->  /Images/mega/business/group-dv.mp4
    clips/enroll/aca.mp4      + aca.jpg     ->  /Images/mega/enroll/aca.mp4

A .jpg poster (first frame) is written next to each clip — site-addons.js sets it as the <video>
poster, so the pane shows a still instead of a grey box before playback starts.

Run inside the generative venv on the A100 (setup_a100_gen.sh creates it):
  cd ~/mocombe-video-gen && source .venv/bin/activate
  export MODEL_CACHE_DIR=$HOME/models HF_HOME=$HOME/models/hf
  python generate_clips_local.py                        # all of them
  python generate_clips_local.py personal/life personal/auto   # a subset
  python generate_clips_local.py --only personal        # one section
  python generate_clips_local.py --skip-existing        # resume an interrupted batch

Then copy the tree back to your machine's website folder:
  # from your LOCAL machine:
  scp -P 32144 -i <key> -r 'ubuntu@216.81.200.233:~/mocombe-video-gen/clips/*' \
      "C:/Users/Garoo/OneDrive/Desktop/MocombeFinancial.com_Website/Images/mega/"
"""
import argparse
import base64
import json
import os
import subprocess
import sys

from rp_handler import handler

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "clip_prompts.json")


def load_spec():
    with open(PROMPTS, "r", encoding="utf-8") as f:
        data = json.load(f)
    suffix = data.get("_style_suffix", "")
    render = data.get("_render", {})
    clips = {k: f"{p}, {suffix}".strip(", ") for k, p in data["clips"].items()}
    return clips, render


def write_poster(mp4_path):
    """Extract frame 1 as a .jpg poster. Non-fatal — the clip is the deliverable."""
    jpg = os.path.splitext(mp4_path)[0] + ".jpg"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
             "-frames:v", "1", "-q:v", "3", jpg],
            check=True,
        )
        return os.path.basename(jpg)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  (poster skipped: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", help="specific clip keys, e.g. personal/life (default: all)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "clips"))
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--only", help="render just one section: personal | business | enroll")
    ap.add_argument("--skip-existing", action="store_true", help="resume: skip clips already on disk")
    ap.add_argument("--no-poster", action="store_true")
    args = ap.parse_args()

    prompts, render = load_spec()
    width, height = render.get("width"), render.get("height")

    slugs = args.slugs or list(prompts.keys())
    if args.only:
        slugs = [s for s in slugs if s.startswith(args.only.rstrip("/") + "/")]

    ok, fail, skipped = [], [], []
    for i, slug in enumerate(slugs, 1):
        if slug not in prompts:
            print(f"skip unknown key: {slug}")
            continue
        dest = os.path.join(args.out_dir, *slug.split("/")) + ".mp4"
        if args.skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[{i}/{len(slugs)}] {slug} — exists, skipping")
            skipped.append(slug)
            continue

        print(f"\n=== [{i}/{len(slugs)}] {slug} ===\n{prompts[slug][:90]}...")
        job = {"type": "generative", "prompt": prompts[slug], "seconds": args.seconds}
        if width and height:
            job["width"], job["height"] = width, height

        try:
            out = handler({"input": job})
        except Exception as e:  # one bad prompt shouldn't kill a 31-clip batch
            print(f"  FAIL (exception): {e}")
            fail.append(slug)
            continue

        if isinstance(out, dict) and out.get("video_base64"):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(base64.b64decode(out["video_base64"]))
            msg = f"  OK -> {dest} ({os.path.getsize(dest)//1024} KB"
            if not args.no_poster:
                p = write_poster(dest)
                if p:
                    msg += f", +{p}"
            print(msg + ")")
            ok.append(slug)
        else:
            print(f"  FAIL: {out.get('error') if isinstance(out, dict) else out}")
            fail.append(slug)

    print(f"\nDONE. {len(ok)} ok, {len(fail)} failed, {len(skipped)} skipped -> {args.out_dir}")
    if fail:
        print("  failed:", ", ".join(fail))
        print("  re-run just those:  python generate_clips_local.py " + " ".join(fail))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
