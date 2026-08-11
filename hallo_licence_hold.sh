#!/usr/bin/env bash
# HALLO RENDER HOLD — sourced by every Hallo greeting script. Refuses to run.
#
# WHY THIS EXISTS
# The pre-launch model-licence review (Reports/For Legal Counsel, document 8) did a
# dependency-level pass on this pipeline on 2026-08-11 and found two problems inside the weights
# bundle Hallo downloads, neither visible from Hallo's own MIT licence:
#
#   1. pretrained_models/face_analysis/models/*.onnx are INSIGHTFACE models. InsightFace grants MIT
#      for its CODE and states separately that "the training data containing the annotation (and the
#      models trained with these data) are available for non-commercial research purposes only."
#      They run on every render — the face crop of the source portrait.
#   2. pretrained_models/audio_separator/Kim_Vocal_2.onnx has NO STATED LICENCE at all.
#
# Provara Advantage is a paid product, so this is the same shape as the Creole voice finding:
# a non-commercial model inside something we sell. Whether that reaches the greeting clips ALREADY
# rendered and published is a question with outside counsel; whether we should keep producing MORE
# of them while they answer is not.
#
# WHAT THIS DOES, AND DOES NOT DO
#   - Stops NEW renders. Every batch script sources this and exits before touching the GPU.
#   - Does NOT delete or unpublish anything already rendered. That call belongs to counsel, and
#     destroying the output would also destroy the evidence of what was produced and when.
#
# WHEN IT CAN BE LIFTED — any one of:
#   a) counsel says the non-commercial models do not bite here;
#   b) a commercial licence is obtained from InsightFace (they publish a contact address for the
#      open-source recognition pack);
#   c) the pipeline is changed so neither model is used — note that Kim_Vocal_2 is almost free to
#      drop, because our driving audio is Kokoro TTS: one clean synthetic voice with nothing to
#      separate, so the separator earns nothing.
#
# Delete this file and the `source` lines when the hold is lifted, and say why in the commit.

if [ "${HALLO_LICENCE_HOLD_OVERRIDE:-}" = "i-have-counsel-approval" ]; then
  echo "[hallo-hold] OVERRIDE ACCEPTED — proceeding. This is recorded in your shell history and" >&2
  echo "[hallo-hold] should be recorded in COORDINATION.md too: who approved, and on what basis." >&2
else
  cat >&2 <<'HOLD'

  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  HALLO GREETING RENDERS ARE ON HOLD — LICENCE REVIEW                         │
  └──────────────────────────────────────────────────────────────────────────────┘

  This pipeline loads models that are NOT licensed for use in a paid product:

    * InsightFace face models (face_analysis/models/*.onnx)
        code = MIT, but the MODELS are "non-commercial research purposes only"
    * Kim_Vocal_2.onnx (audio_separator/)
        no stated licence at all

  Found 2026-08-11 in the dependency-level pass; Hallo's own MIT licence covers the
  wrapper, not these. See "8 - Model Licence Review.pdf" in Reports/For Legal Counsel.

  Nothing already rendered has been deleted. Only NEW renders are blocked, until
  outside counsel answers.

  If counsel has cleared it, re-run with:
      HALLO_LICENCE_HOLD_OVERRIDE=i-have-counsel-approval <your command>

HOLD
  exit 9
fi
