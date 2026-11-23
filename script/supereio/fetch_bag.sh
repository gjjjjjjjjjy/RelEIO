#!/usr/bin/env bash
# Download raw rosbags of the RPG Event Camera Dataset (DAVIS240C sequences) for SuperEIO into data/rosbags/rpgdavis/
# (git-ignored). Resumes a partial download, checks the size against the server, writes <seq>.bag.sha256.
#   bash script/supereio/fetch_bag.sh boxes_6dof [poster_6dof shapes_6dof ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"   # the repository checkout: first directory with .git above this script
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
DIR="$ROOT/data/rosbags/rpgdavis"
mkdir -p "$DIR"
[ "$#" -gt 0 ] || { echo "usage: $0 <sequence> [...]"; exit 1; }
for s in "$@"; do
  url="https://download.ifi.uzh.ch/rpg/web/datasets/davis/$s.bag"
  want="$(curl -sIL -m 30 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')"
  [ -n "$want" ] || { echo "$s: the server gave no size ($url)"; exit 1; }
  have="$(stat -c %s "$DIR/$s.bag" 2>/dev/null || echo 0)"
  if [ "$have" != "$want" ]; then
    echo "$s: downloading $((want / 1048576)) MB"
    curl -sSL -C - -o "$DIR/$s.bag" "$url"
  fi
  have="$(stat -c %s "$DIR/$s.bag")"
  [ "$have" = "$want" ] || { echo "$s.bag: size $have, server says $want"; exit 1; }
  (cd "$DIR" && sha256sum "$s.bag" | tee "$s.bag.sha256")
done
