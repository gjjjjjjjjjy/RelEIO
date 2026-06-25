#!/usr/bin/env bash
# RelEIO --- run DEIO (event+IMU, primary system) under the three stress axes.
# Prereq: DEIO installed (see docs/INTEGRATION.md). Set DEIO_ROOT and DATA_ROOT.
set -euo pipefail
DEIO_ROOT=${DEIO_ROOT:?set DEIO_ROOT to your DEIO checkout}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your event+IMU datasets}
SEEDS=${SEEDS:-"1000 1001 1002 1003 1004"}

# Stress is injected via the RelEIO operators (stress_operators/event_stress.py),
# which the dataset iterators call before voxelization. Levels per axis:
KEEPS=${KEEPS:-"1.0 0.5 0.3 0.1"}      # event-density axis
TQS=${TQS:-"0 1000 5000 10000"}         # timestamp-precision axis (us)
PATCHES=${PATCHES:-"96 64 32 16"}       # visual-budget axis

# Example for one axis (event density). Repeat analogously for tq / patches.
for scene in "$@"; do
  for k in $KEEPS; do
    for s in $SEEDS; do
      RELEIO_EVENT_KEEP=$k \
      python "$DEIO_ROOT/eval.py" --datapath "$DATA_ROOT" --scene "$scene" \
        --seed "$s" --enable_event --network DEVO.pth --trials 1
    done
  done
done
# Per-run outcomes (status, MPE, stress evidence) are logged and aggregated into
# records/*.csv; see tools/ for aggregation. Each run is an independent process
# so a crash never interrupts the matrix.
