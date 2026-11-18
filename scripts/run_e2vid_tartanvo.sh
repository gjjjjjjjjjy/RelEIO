#!/usr/bin/env bash
# Each cell command must run conversion, reconstruction and VO in full.
set -euo pipefail
: "${PYTHON:?set PYTHON to the matrix-runner Python executable}"
: "${RELEIO_COMMAND_TEMPLATE:?set the actual adapter argv template}"
: "${RELEIO_RUN_DIR:?set a fresh experiment output directory}"
: "${RELEIO_SEEDS:?set space-separated integer seeds or repeat IDs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
read -r -a seeds <<< "$RELEIO_SEEDS"
exec "$PYTHON" "$SCRIPT_DIR/run_matrix.py" --system E2VID-TartanVO \
  --command "$RELEIO_COMMAND_TEMPLATE" --out "$RELEIO_RUN_DIR" \
  --axes keep --seeds "${seeds[@]}" --scenes "$@"
