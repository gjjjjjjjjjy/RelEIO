#!/usr/bin/env bash
# Round 2 of the precision diagnosis, on every visible GPU, in one command.
#   Part A  VECtor/desk, clean, seed 1000: default / amp_off / tf32_off / fp32_strict, 16 runs each
#           -> which half of fp32_strict removes the failures (mixed precision or TF32)
#   Part B  the other 7 sequences, clean, seeds 1000-1004: default / fp32_strict, 10 runs each
#           -> does the default-vs-fp32 difference hold beyond VECtor/desk
# Sets up the instance first if the probe-enabled cuda_ba is not installed (fresh instance).
#
#   cd <repository> && git pull
#   nohup bash script/diag/run_round2.sh >/dev/null 2>&1 &
#   tail -f logs/runs/round2_all_latest.log
#
# Two instances in parallel (same conditions, about half the time):
#   instance 1:  PART=1 nohup bash script/diag/run_round2.sh >/dev/null 2>&1 &   (Part A + corner, vicon_dark1)
#   instance 2:  PART=2 nohup bash script/diag/run_round2.sh >/dev/null 2>&1 &   (the other five sequences)
#   logs: logs/runs/round2_p1_latest.log, logs/runs/round2_p2_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PART="${PART:-all}"
B_ALL="vector/corner_slow1 hku/vicon_dark1 fpv/indoor_forward_3 rpgdavis/boxes_6dof rpgdavis/poster_6dof rpgdavis/shapes_6dof tumvie/mocap-3d-trans"
case "$PART" in
  all) RUN_A=1; B_SCENES="$B_ALL"; TAG=all ;;
  1)   RUN_A=1; B_SCENES="vector/corner_slow1 hku/vicon_dark1"; TAG=p1 ;;
  2)   RUN_A=0; B_SCENES="fpv/indoor_forward_3 rpgdavis/boxes_6dof rpgdavis/poster_6dof rpgdavis/shapes_6dof tumvie/mocap-3d-trans"; TAG=p2 ;;
  *)   echo "PART must be all, 1 or 2"; exit 1 ;;
esac
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/round2_${TAG}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/3 environment"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
# the *_latest link moves only after the guard, so a refused second launch leaves it on the running one
ln -sf "round2_${TAG}_${STAMP}.log" "logs/runs/round2_${TAG}_latest.log"
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here: running setup (build + tests) first"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
  tail -12 logs/runs/pilot_all_latest.log
fi
python tests/test_run_ledger.py > "logs/runs/test_round2_${STAMP}.log" 2>&1 || { tail -20 "logs/runs/test_round2_${STAMP}.log"; fail "unit tests failed"; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))
echo "  GPUs: $GPUS"

A="logs/ledger/round2A_${STAMP}"
B="logs/ledger/round2B_${TAG}_${STAMP}"
echo "  PART=$PART | Part A: $([ "$RUN_A" = 1 ] && echo yes || echo no) | Part B scenes: $B_SCENES"
if [ "$RUN_A" = 1 ]; then
  step "1/3 Part A: VECtor/desk, four precision policies"
  # shellcheck disable=SC2086
  python script/diag/precision_matrix.py --gpus $GPUS --scenes vector/desk_normal1 \
    --arms default amp_off tf32_off fp32_strict --k 16 --seeds 1000 --cholinfo --ledger "$A" \
    || fail "Part A stopped (see $A)"
fi

step "2/3 Part B: default vs fp32_strict on $B_SCENES"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --scenes $B_SCENES \
  --arms default fp32_strict --k 10 --seeds 1000 1001 1002 1003 1004 --cholinfo --ledger "$B" \
  || fail "Part B stopped (see $B)"

step "3/3 summaries"
if [ "$RUN_A" = 1 ]; then
  python script/diag/precision_matrix.py --summary-only --ledger "$A"
  python script/diag/pilot_details.py --ledger "$A" --out "records/round2A_details_${STAMP}.csv" | tail -12
fi
python script/diag/precision_matrix.py --summary-only --ledger "$B"
python script/diag/pilot_details.py --ledger "$B" --out "records/round2B_${TAG}_details_${STAMP}.csv" | tail -12
echo
echo "Round 2 (PART=$PART) finished. Ledgers: $([ "$RUN_A" = 1 ] && echo "$A ")$B"
