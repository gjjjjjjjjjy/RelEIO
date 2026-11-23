#!/usr/bin/env bash
# Round 2C: which TF32 path matters (VECtor/desk, clean, seed 1000, 16 runs per arm, all GPUs).
#   default           container defaults: AMP on, TF32 on for cuBLAS matmul (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1) and cuDNN
#   no_tf32_override  the container's override variable removed, nothing else changed (= stock PyTorch defaults)
#   matmul_tf32_off   cuBLAS matmul TF32 off in code, cuDNN TF32 on
#   cudnn_tf32_off    cuDNN TF32 off in code, matmul TF32 on
#   ba_matmul_fp32    everything as default, except no TF32 for cuBLAS matmuls inside the visual BA call
#
#   cd <FreqVIO> && git pull --ff-only
#   nohup bash script/diag/run_tf32_split.sh >/dev/null 2>&1 &
#   tail -f logs/runs/tf32_split_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
K="${K:-16}"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/tf32_split_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/2 environment"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
# the *_latest link moves only after the guard, so a refused second launch leaves it on the running one
ln -sf "$(basename "$LOG")" logs/runs/tf32_split_latest.log
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-<unset>} (the default arm inherits it)"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here: running setup (build + tests) first; progress in logs/runs/pilot_all_latest.log"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
fi
python tests/test_run_ledger.py > "logs/runs/test_tf32_${STAMP}.log" 2>&1 || { tail -20 "logs/runs/test_tf32_${STAMP}.log"; fail "unit tests failed"; }
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))

step "1/2 VECtor/desk, five TF32 variants, $K each"
C="logs/ledger/round2C_${STAMP}"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --scenes vector/desk_normal1 \
  --arms default no_tf32_override matmul_tf32_off cudnn_tf32_off ba_matmul_fp32 --k "$K" --seeds 1000 \
  --cholinfo --ledger "$C" || fail "stopped (see $C)"

step "2/2 details"
python script/diag/pilot_details.py --ledger "$C" --out "revision/analysis/round2C_details_${STAMP}.csv" | tail -12
echo
echo "Round 2C finished. Ledger: $C"
