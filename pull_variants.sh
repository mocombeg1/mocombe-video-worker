#!/usr/bin/env bash
# Pull every finished variant greeting off the ephemeral Thunder box to durable local storage.
# Run repeatedly (or on a loop). Only pulls clips not already local + non-zero on the box.
# Usage: bash pull_variants.sh   (env: HOST, PORT, KEY, DEST)
set -uo pipefail
HOST=${HOST:-198.145.126.210}
PORT=${PORT:-30310}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
DEST=${DEST:-/c/Users/Garoo/3d-office-mockups/assets/variants}
mkdir -p "$DEST"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15 ubuntu@$HOST"
remote_list=$($SSH 'cd ~/out 2>/dev/null && for f in greet_*_hallo.mp4; do [ -s "$f" ] && echo "$f"; done' 2>/dev/null)
[ -z "$remote_list" ] && { echo "no remote clips yet"; exit 0; }
new=0
for f in $remote_list; do
  if [ ! -s "$DEST/$f" ]; then
    scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no "ubuntu@$HOST:~/out/$f" "$DEST/$f" >/dev/null 2>&1 && new=$((new+1))
  fi
done
echo "local variants now: $(ls "$DEST"/greet_*_hallo.mp4 2>/dev/null | wc -l) (+$new new)"
