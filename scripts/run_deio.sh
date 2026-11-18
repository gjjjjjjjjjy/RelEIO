#!/usr/bin/env bash
# Supply the actual DEIO adapter command, including all stress parameters.
set -euo pipefail
: "${PYTHON:?set PYTHON to the matrix-runner Python executable}"
: "${RELEIO_COMMAND_TEMPLATE:?set the actual adapter argv template}"
: "${RELEIO_RUN_DIR:?set a fresh experiment output directory}"
: "${RELEIO_SEEDS:?set space-separated integer seeds or repeat IDs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
read -r -a seeds <<< "$RELEIO_SEEDS"
exec "$PYTHON" "$SCRIPT_DIR/run_matrix.py" --system DEIO \
  --command "$RELEIO_COMMAND_TEMPLATE" --out "$RELEIO_RUN_DIR" \
  --axes keep time patch --seeds "${seeds[@]}" --scenes "$@"
