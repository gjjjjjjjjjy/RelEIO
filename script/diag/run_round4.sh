#!/usr/bin/env bash
# Round 4: the main matrix. Two designs:
#   DESIGN=core (default)  8 sequences x 4 conditions (clean, keep:0.1, tq:10000, patch:16) x 10 distinct seeds
#                          (2000-2009) x 2 settings = 640 runs, plan script/diag/plans/round4_core_dual_plan.json;
#                          no key block. About 26 GPU-hours: two 8-GPU instances about 1.7 h each.
#   DESIGN=full            the design below: 1600 + 180 runs, about 70 GPU-hours.
#   main block  8 sequences x 10 conditions (clean, shared by the three axes, + 9 degraded levels)
#               x 10 distinct seeds (2000-2009) x 2 settings = 1600 runs. Settings, mixed precision kept in both:
#                 default           the container's TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (cuBLAS matmul in TF32)
#                 no_tf32_override  that variable removed (stock PyTorch numerics)
#               both settings get the same seed in every round, interleaved on the same instance
#   key block   the 9 high-risk cells of round 3A x 20 more distinct seeds (2010-2029), no_tf32_override only
#               = 180 runs (K=30 there for that setting)
# Plans frozen in script/diag/plans/ (round4_main_dual_plan.json, round4_key_plan.json; the single-arm
# round4_main_plan.json is kept); the run stops if the plan generated here differs.
# First a smoke stage: one VECtor/corner keep:0.1 run per setting (seed 1999, not in the plans, own ledger);
# every run's recorded precision settings are checked, and a wrong one stops the script.
#
#   one instance:   nohup bash script/diag/run_round4.sh >/dev/null 2>&1 &
#   two instances:  PART=1 NPARTS=2 nohup bash script/diag/run_round4.sh ... ; PART=2 NPARTS=2 on the other
#                   (whole sequences per instance; the same sequences for both blocks)
#   log: logs/runs/round4_<all|1of2|2of2>_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PART="${PART:-1}"
NPARTS="${NPARTS:-1}"
DESIGN="${DESIGN:-core}"
MAIN_ARMS="default no_tf32_override"
case "$DESIGN" in
  core) MAIN_CELLS="clean keep:0.1 tq:10000 patch:16"; MAIN_PLAN=script/diag/plans/round4_core_dual_plan.json ;;
  full) MAIN_CELLS="main"; MAIN_PLAN=script/diag/plans/round4_main_dual_plan.json ;;
  *) echo "DESIGN must be core or full"; exit 1 ;;
esac
KEY_ARM="no_tf32_override"
SCENES="vector/desk_normal1 vector/corner_slow1 hku/vicon_dark1 tumvie/mocap-3d-trans fpv/indoor_forward_3 rpgdavis/boxes_6dof rpgdavis/poster_6dof rpgdavis/shapes_6dof"
KEY_UNITS="vector/corner_slow1@keep:0.1 vector/corner_slow1@keep:0.3 vector/corner_slow1@patch:16 vector/corner_slow1@patch:64
           fpv/indoor_forward_3@patch:64 rpgdavis/boxes_6dof@patch:32 rpgdavis/poster_6dof@patch:16
           rpgdavis/poster_6dof@patch:32 rpgdavis/shapes_6dof@patch:16"
MAIN_SEEDS=$(seq -s " " 2000 2009)
KEY_SEEDS=$(seq -s " " 2010 2029)
if [ "$NPARTS" = 1 ]; then TAG=all; else TAG="${PART}of${NPARTS}"; fi
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/round4_${DESIGN}_${TAG}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/4 environment"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
[ "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-}" = "1" ] \
  || fail "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is not 1 on this instance: the default setting would not be the TF32-on environment"
[ "${NVIDIA_TF32_OVERRIDE:-}" != "0" ] || fail "NVIDIA_TF32_OVERRIDE=0 disables TF32 globally: the default setting would not be TF32-on"
. script/diag/guard.sh
guard_idle
# the *_latest link moves only after the guard, so a refused second launch leaves it on the running one
ln -sf "$(basename "$LOG")" "logs/runs/round4_${TAG}_latest.log"   # the same link name for both designs
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-<unset>} in the shell; the no_tf32_override setting unsets it"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here: running setup (build + tests) first; progress in logs/runs/pilot_all_latest.log"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
fi
python tests/test_run_ledger.py > "logs/runs/test_round4_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_round4_${STAMP}.log"; fail "unit tests failed"; }
python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351 > "logs/runs/env_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_${STAMP}.log"; fail "Python environment differs from the earlier RTX 4090 rounds (see logs/runs/env_${STAMP}.log)"; }
python tests/test_event_slicing.py >> "logs/runs/test_round4_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_round4_${STAMP}.log"; fail "slicing tests failed"; }
echo "  IMU units of the preprocessed VECtor/HKU files (conversion vs raw, gyro vs ground-truth angular speed, CPU):"
python script/diag/check_imu_units.py | sed 's/^/    /' || fail "IMU unit check failed (see above)"
echo "  DAVIS event selection (binary search vs the original mask, 200 windows per sequence, CPU):"
python script/diag/check_davis_slicing.py | sed 's/^/    /' || fail "DAVIS event selection differs from the original"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))
RM="logs/ledger/round4_${DESIGN}_main_${TAG}_${STAMP}"
RK="logs/ledger/round4_key_${TAG}_${STAMP}"
RS="logs/ledger/round4_smoke_${TAG}_${STAMP}"
echo "  design $DESIGN | part $PART of $NPARTS | GPUs: $GPUS | main: $MAIN_ARMS on $MAIN_CELLS | key block: $([ "$DESIGN" = full ] && echo "$KEY_ARM" || echo none)"

COMMON="--cholinfo --part $PART/$NPARTS --part-scenes $SCENES"
step "1/5 smoke: one corner keep:0.1 run per setting (seed 1999, own ledger, not counted)"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units vector/corner_slow1@keep:0.1 --arms $MAIN_ARMS \
  --k 1 --seeds 1999 --cholinfo --ledger "$RS" || fail "smoke stopped (see $RS)"
python script/diag/precision_matrix.py --summary-only --ledger "$RS" > "${RS}_summary.txt"
grep -E "^   (default|no_tf32_override) " "${RS}_summary.txt" | sed 's/^/    recorded: /' || true

step "2/5 main block, both settings (frozen plan $MAIN_PLAN)"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --scenes $SCENES --cells $MAIN_CELLS --k 10 --seeds $MAIN_SEEDS \
  --arms $MAIN_ARMS $COMMON --plan-check "$MAIN_PLAN" --ledger "$RM" \
  || fail "main block stopped (see $RM)"

if [ "$DESIGN" = full ]; then
  step "3/5 key block, no_tf32_override (frozen plan script/diag/plans/round4_key_plan.json)"
  # shellcheck disable=SC2086
  python script/diag/precision_matrix.py --gpus $GPUS --units $KEY_UNITS --k 20 --seeds $KEY_SEEDS \
    --arms $KEY_ARM $COMMON --plan-check script/diag/plans/round4_key_plan.json --ledger "$RK" \
    || fail "key block stopped (see $RK)"
else
  step "3/5 key block: none in the core design"
fi

step "4/5 details"
python script/diag/pilot_details.py --ledger "$RM" --out "records/round4_main_${TAG}_details_${STAMP}.csv" | tail -n 8
[ -s "$RK/ledger.jsonl" ] && python script/diag/pilot_details.py --ledger "$RK" --out "records/round4_key_${TAG}_details_${STAMP}.csv" | tail -n 8
RKL=""; [ -s "$RK/ledger.jsonl" ] && RKL="$RK"
# shellcheck disable=SC2086
python script/diag/attribution_table.py --ledgers "$RM" $RKL --out "records/round4_${TAG}_attribution_${STAMP}.csv" \
  | sed -n '/^TOTAL:/,$p'

step "5/5 done"
echo "Round 4 ($DESIGN) part $PART of $NPARTS finished. Ledgers: $RM $RKL"
