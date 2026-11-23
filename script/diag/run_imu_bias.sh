#!/usr/bin/env bash
# Cross-system IMU-bias check, DEIO side (revision/DIAG_PILOT_RUNBOOK.md section 13; design pending the authors).
#   DAVIS240C boxes_6dof and poster_6dof x {clean, gbias:<GBIAS>} x 10 seeds (3000-3009), setting no_tf32_override
#   (the container's TF32 override removed, mixed precision kept: the environment in which round 4 saw no numerical
#   failure) = 40 runs. gbias:<d> adds a constant gyro bias of (+d, -d, +d) deg/s, given in rad/s, to the IMU array
#   before the estimator (utils/eval_utils.py, utils/sensor_stress.py); every run prints an [imu-stress] line and the
#   ledger check fails a run whose applied bias differs from the requested one, or a clean run with any bias.
#   The SuperEIO side is script/supereio/run_imu_bias.sh (same sequences, same bias vector, rewritten rosbags).
# GBIAS must be given explicitly (deg/s per axis), so nothing runs before the size is agreed:
#   GBIAS=5 nohup bash script/diag/run_imu_bias.sh >/dev/null 2>&1 &
#   log: logs/runs/imu_bias_deio_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
GBIAS="${GBIAS:?set GBIAS (deg/s per axis) as agreed with the authors, e.g. GBIAS=5}"
ARM="no_tf32_override"
SEEDS=$(seq -s " " 3000 3009)
UNITS="rpgdavis/boxes_6dof@clean rpgdavis/boxes_6dof@gbias:$GBIAS rpgdavis/poster_6dof@clean rpgdavis/poster_6dof@gbias:$GBIAS"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/imu_bias_deio_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/4 environment"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
ln -sf "$(basename "$LOG")" "logs/runs/imu_bias_deio_latest.log"
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here: running setup (build + tests) first; progress in logs/runs/pilot_all_latest.log"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
fi
python tests/test_run_ledger.py > "logs/runs/test_imu_bias_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_imu_bias_${STAMP}.log"; fail "unit tests failed"; }
python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351 > "logs/runs/env_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_${STAMP}.log"; fail "Python environment differs from the earlier RTX 4090 rounds (see logs/runs/env_${STAMP}.log)"; }
echo "  DAVIS event selection (binary search vs the original mask, CPU):"
python script/diag/check_davis_slicing.py | sed 's/^/    /' || fail "DAVIS event selection differs from the original"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))
RM="logs/ledger/imu_bias_deio_${STAMP}"
RS="logs/ledger/imu_bias_deio_smoke_${STAMP}"
echo "  GPUs: $GPUS | setting $ARM | gbias $GBIAS deg/s per axis | units: $UNITS"

step "1/4 smoke: one boxes gbias run (seed 2999, own ledger, not counted); its [imu-stress] line must match"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units "rpgdavis/boxes_6dof@gbias:$GBIAS" --arms $ARM \
  --k 1 --seeds 2999 --cholinfo --ledger "$RS" || fail "smoke stopped (see $RS)"
python script/diag/precision_matrix.py --summary-only --ledger "$RS" > "${RS}_summary.txt"
grep -h "\[imu-stress\]" "$RS"/runs/*/run.log | head -n 1 | sed 's/^/    applied: /' || fail "no [imu-stress] line in the smoke run"
grep -q "EXCLUDED" "${RS}_summary.txt" && { grep -A3 "EXCLUDED" "${RS}_summary.txt"; fail "the smoke run failed its control check"; }

step "2/4 main: 4 units x 10 seeds, $ARM"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --k 10 --seeds $SEEDS --arms $ARM --cholinfo \
  --plan-out "${RM}_plan.json" --ledger "$RM" || fail "main stopped (see $RM)"

step "3/4 summary"
python script/diag/precision_matrix.py --summary-only --ledger "$RM" | tail -n 40

step "4/4 done"
echo "IMU-bias check, DEIO side, finished. Ledger: $RM (trajectories in runs/*/traj_est_tum.txt)"
