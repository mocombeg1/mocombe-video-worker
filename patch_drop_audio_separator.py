#!/usr/bin/env python3
"""Drop the Kim_Vocal_2 audio separator from the Hallo inference path.

WHY
The model-licence review (2026-08-11) found that `pretrained_models/audio_separator/Kim_Vocal_2.onnx`,
which ships inside the Hallo weights bundle, carries NO STATED LICENCE. No licence is not a permissive
licence: nothing grants the right to use it in a paid product.

It also earns us nothing. Hallo runs it to isolate a voice from the driving audio before wav2vec sees
it — but our driving audio is Kokoro TTS: one clean synthetic voice, no music, no second speaker,
nothing to separate. So this is a removal, not a substitution, and there is no quality trade to argue
about.

HOW
Upstream already supports running without it. `hallo/datasets/audio_processor.py` reads:

    if audio_separator_model_name is not None:
        self.audio_separator = Separator(...)
        self.audio_separator.load_model(audio_separator_model_name)
    else:
        self.audio_separator = None
        print("Use audio directly without vocals seperator.")

and in preprocess():

    if self.audio_separator is not None:
        ... separate, then resample_audio(...) to the configured rate ...
    else:
        vocal_audio_file = wav_file

`scripts/inference.py` always passes a real path, so the None branch is unreachable as shipped. This
patch makes it pass None. The condition is `is not None`, so an empty string would NOT work — it has
to be the literal None, which is why this is a code patch and not a config edit.

⚠️ THE PART THAT IS NOT OPTIONAL — SAMPLE RATE
Read the else branch again: it assigns `vocal_audio_file = wav_file` and does NOT resample. The
resample to 16 kHz that wav2vec needs lives inside the separator branch. Our Kokoro WAVs are 24 kHz
(gen_voice.py writes 24000), so dropping the separator without resampling would feed 24 kHz audio to
a 16 kHz model and quietly wreck the mouth timing.

The batch scripts therefore now produce the driving WAV with `-ar 16000 -ac 1`. Do not remove that
flag while this patch is applied; the two changes are one change.

USAGE
    python3 patch_drop_audio_separator.py [/path/to/hallo]     # default: ~/hallo
Idempotent: re-running on a patched checkout reports "already patched" and exits 0.
"""
import os
import re
import sys

BEFORE = """        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),"""

AFTER = """        # LICENCE (2026-08-11): the bundled Kim_Vocal_2.onnx separator has no stated licence and
        # is not used. Passing None on both makes AudioProcessor take its documented
        # "Use audio directly without vocals seperator." branch. It must be literal None — the
        # upstream check is `is not None`, so "" would still build the Separator.
        # The driving WAV is produced at 16 kHz mono by the batch scripts, because the resample
        # that wav2vec needs lived inside the separator branch we are skipping.
        None,
        None,"""


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/hallo")
    target = os.path.join(root, "scripts", "inference.py")
    if not os.path.isfile(target):
        print(f"FAIL: {target} not found — is {root} a Hallo checkout?", file=sys.stderr)
        return 2

    with open(target, encoding="utf-8") as fh:
        src = fh.read()

    if "LICENCE (2026-08-11)" in src:
        print(f"already patched: {target}")
    elif BEFORE not in src:
        print("FAIL: the AudioProcessor call does not look the way this patch expects.",
              file=sys.stderr)
        print("      Upstream may have changed. Patch it by hand: the 5th and 6th positional",
              file=sys.stderr)
        print("      arguments to AudioProcessor must both be None.", file=sys.stderr)
        return 3
    else:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(src.replace(BEFORE, AFTER, 1))
        print(f"patched: {target}")

    # Prove it, rather than trusting the replace: the separator path must no longer be passed.
    with open(target, encoding="utf-8") as fh:
        now = fh.read()
    call = re.search(r"with AudioProcessor\((.*?)\) as audio_processor:", now, re.S)
    if not call:
        print("FAIL: could not re-read the AudioProcessor call to verify.", file=sys.stderr)
        return 4
    args = call.group(1)
    if "audio_separator_model_file" in args:
        print("FAIL: the separator path is still being passed.", file=sys.stderr)
        return 5
    print("verified: AudioProcessor is called with None for both separator arguments")

    # The weight itself should not be sitting on disk either — an unlicensed file we do not use is
    # still an unlicensed file. Removing it also turns any accidental un-patching into a loud
    # failure instead of a silent re-enable.
    onnx = os.path.join(root, "pretrained_models", "audio_separator", "Kim_Vocal_2.onnx")
    if os.path.isfile(onnx):
        os.remove(onnx)
        print(f"removed unlicensed weight: {onnx}")
    else:
        print("unlicensed weight not present (nothing to remove)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
