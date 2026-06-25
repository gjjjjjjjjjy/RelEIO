#!/usr/bin/env bash
# RelEIO --- run DEVO (event-only, IMU-off ablation; shares DEIO front-end).
# Same as run_deio.sh but with the IMU factor disabled (ENABLE_IMU=False),
# routing to the event-only tracker. See docs/INTEGRATION.md.
set -euo pipefail
DEIO_ROOT=${DEIO_ROOT:?}; DATA_ROOT=${DATA_ROOT:?}
SEEDS=${SEEDS:-"1000 1001 1002 1003 1004"}; KEEPS=${KEEPS:-"1.0 0.3 0.1"}
for scene in "$@"; do for k in $KEEPS; do for s in $SEEDS; do
  RELEIO_EVENT_KEEP=$k \
  python "$DEIO_ROOT/eval.py" --datapath "$DATA_ROOT" --scene "$scene" \
    --seed "$s" --enable_event --network DEVO.pth --trials 1 \
    --opts ENABLE_IMU False
done; done; done
