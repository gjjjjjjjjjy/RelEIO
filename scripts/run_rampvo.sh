#!/usr/bin/env bash
# RelEIO --- run RAMP-VO (event+frame, independently trained). Needs intensity
# frames + events (DAVIS sequences). Isolated venv (see docs/INTEGRATION.md).
# Degradation is injected only on the event channel; frames are untouched.
set -euo pipefail
RAMPVO_ROOT=${RAMPVO_ROOT:?set RAMPVO_ROOT}; DATA_ROOT=${DATA_ROOT:?}
SEEDS=${SEEDS:-"1000 1001 1002 1003 1004"}; KEEPS=${KEEPS:-"1.0 0.3 0.1"}
for scene in "$@"; do for k in $KEEPS; do for s in $SEEDS; do
  RELEIO_EVENT_KEEP=$k RELEIO_SEED=$s \
  python "$RAMPVO_ROOT/evaluate.py" \
    --weights "$RAMPVO_ROOT/checkpoints/RAMPVO_MultiScale.pth" \
    --config_eval config_net/MultiScale_RPGDAVIS.json \
    --config_VO config_vo/default.yaml --results_path "results_${scene}_k${k}_s${s}"
done; done; done
