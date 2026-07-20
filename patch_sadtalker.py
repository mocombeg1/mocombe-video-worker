"""
Idempotent SadTalker source patches for numpy 2.x / OpenCV 5 (verified on the Blackwell pod,
2026-07-05). Run AFTER cloning the SadTalker repo, BEFORE the first render:

    python patch_sadtalker.py [/path/to/SadTalker]     (default: $REPO_DIR/SadTalker)

Applied patches (each skipped if already applied):
  1. np.float / np.int / np.bool / np.object / np.str aliases -> builtins (removed in numpy 1.24).
  2. np.VisibleDeprecationWarning -> plain filterwarnings("ignore")   (removed in numpy 2).
  3. align_img trans_params ragged np.array([w0,h0,s,t[0],t[1]]) -> ravel to scalars
     (numpy 2 rejects inhomogeneous arrays; t entries are shape-(1,) arrays).
  4. paste_pic: clamp the seamlessClone paste rect to the frame bounds (2px inset) — OpenCV
     asserts (-215, Mat ROI) when a portrait's face crop touches the image edge.
"""

import os
import re
import sys


def _sub_file(path, pattern, repl, tag):
    if not os.path.exists(path):
        print(f"[patch] SKIP (missing): {path}")
        return
    src = open(path, encoding="utf-8").read()
    out = re.sub(pattern, repl, src)
    if out != src:
        open(path, "w", encoding="utf-8").write(out)
        print(f"[patch] applied {tag}: {path}")
    else:
        print(f"[patch] already ok {tag}: {path}")


def main(root):
    # 1) numpy alias sweep across the whole repo
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            p = os.path.join(dirpath, name)
            src = open(p, encoding="utf-8").read()
            out = re.sub(r"\bnp\.float\b(?!\d|_)", "float", src)
            out = re.sub(r"\bnp\.int\b(?!\d|_)", "int", out)
            out = re.sub(r"\bnp\.bool\b(?!\d|_)", "bool", out)
            out = re.sub(r"\bnp\.object\b(?!_)", "object", out)
            out = re.sub(r"\bnp\.str\b(?!_)", "str", out)
            if out != src:
                open(p, "w", encoding="utf-8").write(out)
                print(f"[patch] numpy aliases: {p}")

    # 2) VisibleDeprecationWarning (removed in numpy 2)
    _sub_file(
        os.path.join(root, "src/face3d/util/preprocess.py"),
        re.escape('warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)'),
        'warnings.filterwarnings("ignore")',
        "VisibleDeprecationWarning",
    )

    # 3) ragged trans_params array (numpy 2 strictness)
    _sub_file(
        os.path.join(root, "src/face3d/util/preprocess.py"),
        re.escape("trans_params = np.array([w0, h0, s, t[0], t[1]])"),
        "trans_params = np.array([w0, h0, s, float(np.ravel(t)[0]), float(np.ravel(t)[1])])",
        "trans_params-ravel",
    )

    # 4) seamlessClone paste-rect clamp
    pp = os.path.join(root, "src/utils/paste_pic.py")
    if os.path.exists(pp):
        src = open(pp, encoding="utf-8").read()
        anchor = "    tmp_path = str(uuid.uuid4())+'.mp4'"
        clamp = (
            "    # Clamp the paste rect to the frame bounds (2px inset) so cv2.seamlessClone's internal ROI\n"
            "    # (centered at `location`) never falls outside full_img — fixes the -215 Mat assert on\n"
            "    # portraits whose face crop touches the image edge.\n"
            "    oy1 = max(2, int(oy1)); oy2 = min(frame_h - 2, int(oy2))\n"
            "    ox1 = max(2, int(ox1)); ox2 = min(frame_w - 2, int(ox2))\n\n"
        )
        if "Clamp the paste rect" in src:
            print(f"[patch] already ok paste-clamp: {pp}")
        elif anchor in src:
            open(pp, "w", encoding="utf-8").write(src.replace(anchor, clamp + anchor, 1))
            print(f"[patch] applied paste-clamp: {pp}")
        else:
            print(f"[patch] WARN: paste_pic anchor not found (repo layout changed?): {pp}")

    print("[patch] done.")


if __name__ == "__main__":
    default_root = os.path.join(os.environ.get("REPO_DIR", "/app"), "SadTalker")
    main(sys.argv[1] if len(sys.argv) > 1 else default_root)
