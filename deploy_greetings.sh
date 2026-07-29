#!/usr/bin/env bash
# Map rendered greeting clips into the URL layout the 3D office expects:
#   rendered  : <SRC>/greet_<id>_<variant>_hallo.mp4   (variant matrix, from batch_hallo_558*)
#   rendered  : <PREVIEW>/greet_<id>_hallo.mp4         (31 default clips)
#   served as : <DEST>/<id>/greet_<variant>.mp4        (per office-shared.tsx greetingSrcs())
#               <DEST>/<id>/greet.mp4                  (legacy default fallback)
#
# DEST is core-crm-backend/public/brand/agents (co-located with the portraits) OR an object-store
# staging dir. NOTE: 558 clips * ~730KB ~= 400MB — do NOT commit to git; serve from the app's
# static dir (rsync/tarball deploy, gitignored) or upload to R2/object store. Hosting is Garood's call.
#
# Usage: SRC=... PREVIEW=... DEST=... bash deploy_greetings.sh
set -uo pipefail
SRC=${SRC:-/c/Users/Garoo/3d-office-mockups/assets/variants}
PREVIEW=${PREVIEW:-/c/Users/Garoo/3d-office-mockups/assets/preview}
DEST=${DEST:-/c/Users/Garoo/3d-office-mockups/assets/greetings_served}
n=0
# variant clips: greet_<id>_<variant>_hallo.mp4  (variant may contain underscores, e.g. east_asian_2)
for f in "$SRC"/greet_*_hallo.mp4; do
  [ -s "$f" ] || continue
  b=$(basename "$f"); rest=${b#greet_}; rest=${rest%_hallo.mp4}   # <id>_<variant>
  # split at the LAST two underscore groups: variant = <eth>_<n>; id = the rest
  variant=$(echo "$rest" | grep -oE '[a-z]+(_[a-z]+)*_[0-9]+$')
  id=${rest%_$variant}
  [ -n "$variant" ] && [ -n "$id" ] || { echo "skip (unparsed): $b"; continue; }
  mkdir -p "$DEST/$id"; cp "$f" "$DEST/$id/greet_${variant}.mp4"; n=$((n+1))
done
# default clips: greet_<id>_hallo.mp4 -> <id>/greet.mp4
for f in "$PREVIEW"/greet_*_hallo.mp4; do
  [ -s "$f" ] || continue
  b=$(basename "$f"); id=${b#greet_}; id=${id%_hallo.mp4}
  # skip variant-named files that also match here
  echo "$id" | grep -qE '_[0-9]+$' && continue
  mkdir -p "$DEST/$id"; cp "$f" "$DEST/$id/greet.mp4"
done
echo "mapped $n variant clips -> $DEST/<id>/greet_<variant>.mp4"
echo "total served files: $(find "$DEST" -name '*.mp4' 2>/dev/null | wc -l)"
