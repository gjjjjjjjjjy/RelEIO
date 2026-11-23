#!/usr/bin/env bash
# Round 3A: does the TF32 effect extend to degraded input? (revision/DIAG_PILOT_RUNBOOK.md section 11)
# Pre-selected from the frozen main matrix: every degraded cell with at least 2 of 5 failures (9 cells,
# 21 of 45 frozen runs failed), plus the clean condition of each of those 5 sequences.
#   default            container defaults (cuBLAS matmul TF32 on through TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)
#   no_tf32_override   the override variable removed (= stock PyTorch numerics), code unchanged
# 14 units x 2 arms x 10 runs (seeds 1000-1004, each twice) = 280 runs, one 8-GPU instance, about 50 min.
#
#   cd <FreqVIO> && timeout 60 git pull --ff-only
#   nohup bash script/diag/run_round3a.sh >/dev/null 2>&1 &
#   tail -n 8 logs/runs/round3a_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
ARMS="default no_tf32_override"
K="${K:-10}"
# pre-selected units; frozen failures per cell (revision/data_frozen/, K=5) in the comment below
UNITS="vector/corner_slow1@keep:0.1 vector/corner_slow1@keep:0.3 vector/corner_slow1@patch:16 vector/corner_slow1@patch:64
       fpv/indoor_forward_3@patch:64
       rpgdavis/boxes_6dof@patch:32
       rpgdavis/poster_6dof@patch:16 rpgdavis/poster_6dof@patch:32
       rpgdavis/shapes_6dof@patch:16
       vector/corner_slow1@keep:1.0 fpv/indoor_forward_3@keep:1.0 rpgdavis/boxes_6dof@keep:1.0
       rpgdavis/poster_6dof@keep:1.0 rpgdavis/shapes_6dof@keep:1.0"
#   corner keep:0.1 3/5, keep:0.3 3/5, patch:16 2/5, patch:64 2/5 | fpv patch:64 3/5 | boxes patch:32 2/5
#   poster patch:16 2/5, patch:32 2/5 | shapes patch:16 2/5 | clean (keep:1.0 = tq:0 = patch:96) controls
TAG=3a
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/round${TAG}_${STAMP}.log"
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
ln -sf "$(basename "$LOG")" "logs/runs/round${TAG}_latest.log"
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-<unset>} (the default arm inherits it)"
[ "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-}" = "1" ] \
  || fail "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is not 1 on this instance: the default arm would not be the TF32-on environment"
[ "${NVIDIA_TF32_OVERRIDE:-}" != "0" ] || fail "NVIDIA_TF32_OVERRIDE=0 disables TF32 globally: the default arm would not be TF32-on"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here: running setup (build + tests) first; progress in logs/runs/pilot_all_latest.log"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
fi
python tests/test_run_ledger.py > "logs/runs/test_round3_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_round3_${STAMP}.log"; fail "unit tests failed"; }
python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351 > "logs/runs/env_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_${STAMP}.log"; fail "Python environment differs from the earlier RTX 4090 rounds (see logs/runs/env_${STAMP}.log)"; }
python tests/test_event_slicing.py >> "logs/runs/test_round3_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_round3_${STAMP}.log"; fail "slicing tests failed"; }
echo "  DAVIS event selection (binary search vs the original mask, real data, CPU, ~1-2 min):"
python script/diag/check_davis_slicing.py | sed 's/^/    /' || fail "DAVIS event selection differs from the original"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))
R="logs/ledger/round${TAG}_${STAMP}"
echo "  GPUs: $GPUS | arms: $ARMS | runs per unit and arm: $K"

step "1/3 plan"
PLAN="logs/runs/round${TAG}_plan_${STAMP}.txt"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARMS \
  --k "$K" --seeds 1000 1001 1002 1003 1004 --cholinfo --ledger "$R" --dry-run > "$PLAN" || fail "plan failed (see $PLAN)"
head -n 2 "$PLAN"

step "2/3 run (one job per GPU at a time; progress lines [n/N])"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARMS \
  --k "$K" --seeds 1000 1001 1002 1003 1004 --cholinfo --ledger "$R" || fail "stopped (see $R)"

step "3/3 details"
python script/diag/pilot_details.py --ledger "$R" --out "revision/analysis/round${TAG}_details_${STAMP}.csv" | tail -n 12
echo
echo "Round 3A finished. Ledger: $R"
