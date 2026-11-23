#!/usr/bin/env bash
# Cross-hardware check (reviewer R2-11, runbook section 8) on whatever NVIDIA GPUs this instance has.
# Same design as round 3A on RTX 4090, for its strongest cell and its clean control:
#   VECtor/corner_slow1 keep:0.1 and keep:1.0, default (container TF32 override on) vs no_tf32_override,
#   K=10 per unit and arm (seeds 1000-1004, each twice, the same arrangement as round 3A) = 40 runs.
# First a smoke stage (one keep:0.1 run per arm): recorded precision settings are checked per run, and the
# input fingerprint is compared with the RTX 4090 runs of round 3A; any difference stops the script.
#   nohup bash script/diag/run_hardware.sh >/dev/null 2>&1 &
#
#   cd <FreqVIO>      (no git pull while other instances are running from this directory)
#   nohup bash script/diag/run_hardware.sh >/dev/null 2>&1 &
#   tail -f logs/runs/hardware_latest_<model>.log     (model: h200, h100, l40s, rtx ...)
#
# The compiled extension must contain machine code for this GPU (sm_86 runs on Ada cards such as
# RTX 4090 and L40S; Hopper H100/H200 needs sm_90). On a fresh Python environment the script
# builds for "8.6 9.0" (covers all four card types). If the installed probe build lacks this
# GPU's code, the script STOPS instead of rebuilding, because the Python environment might be
# shared with another instance that is using the module.
set -euo pipefail
cd "$(dirname "$0")/../.."
K="${K:-10}"
UNITS="vector/corner_slow1@keep:0.1 vector/corner_slow1@keep:1.0"
ARMS="default no_tf32_override"
REF_4090="logs/ledger/round3a_20261120_062351"
STAMP=$(date +%Y%m%d_%H%M%S)
# hostnames can be identical across instances; GPU 0's UUID makes the id unique per machine
INST="$(hostname)-$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 2>/dev/null | cut -c5-12)"
MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | tr '[:upper:]' '[:lower:]' \
        | sed 's/nvidia //; s/geforce //' | awk '{print $1}')
mkdir -p logs/runs logs/ledger
LOG="logs/runs/hardware_${MODEL:-gpu}_${INST}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/4 environment"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
[ "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-}" = "1" ] \
  || fail "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is not 1 on this instance: the default arm would not be the TF32-on environment"
[ "${NVIDIA_TF32_OVERRIDE:-}" != "0" ] || fail "NVIDIA_TF32_OVERRIDE=0 disables TF32 globally: the default arm would not be TF32-on"
. script/diag/guard.sh
guard_idle
# the *_latest link moves only after the guard, so a refused second launch leaves it on the running one
ln -sf "$(basename "$LOG")" "logs/runs/hardware_latest_${MODEL:-gpu}.log"
python -c "import torch; print('torch OK', torch.__version__, torch.version.cuda, torch.cuda.device_count(), 'GPUs')" \
  || fail "import torch failed"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
read -r GPUNAME CC < <(python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name.replace(' ', '_'), '%d.%d' % (p.major, p.minor))")
case "$CC" in
  8.6|8.9) NEED=sm_86 ;;
  9.0)     NEED=sm_90 ;;
  *)       fail "compute capability $CC ($GPUNAME) not covered by this script" ;;
esac
TAG=$(echo "$GPUNAME" | tr '[:upper:]' '[:lower:]' | sed 's/nvidia_//; s/geforce_//; s/[^a-z0-9]/_/g')
echo "  GPU: $GPUNAME (compute capability $CC) -> needs $NEED | tag $TAG"

echo "  Python environment (packages the evaluation imports; library versions against the RTX 4090 runs):"
python script/diag/check_env.py --before-build --reference "$REF_4090" | sed 's/^/  /' \
  || fail "this instance's Python environment differs from the RTX 4090 runs: use the same image (or the same package versions) before comparing platforms"

step "1/4 extension module for this GPU"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
CUOBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  no probe build installed in this Python environment: building for 8.6 + 9.0 (about 12 min)"
  ARCH="8.6 9.0" SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 \
    || fail "build failed, see logs/runs/pilot_all_latest.log"
  SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)")
fi
[ -x "$CUOBJ" ] || fail "cuobjdump not found; cannot check that $SO contains $NEED"
ARCHS=$("$CUOBJ" --list-elf "$SO" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ' || true)
echo "  $SO"
echo "  machine code in cuda_ba: ${ARCHS:-none}"
echo "$ARCHS" | grep -qw "$NEED" || fail "installed cuda_ba has no $NEED code. If no other instance uses this Python environment, rebuild with: ARCH=\"8.6 9.0\" SETUP_ONLY=1 bash script/diag/run_pilot_all.sh"
sha256sum "$SO"
python tests/test_run_ledger.py > "logs/runs/test_hw_${INST}_${STAMP}.log" 2>&1 || { tail -20 "logs/runs/test_hw_${INST}_${STAMP}.log"; fail "unit tests failed"; }
python script/diag/check_env.py --reference "$REF_4090" > "logs/runs/env_hw_${INST}_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_hw_${INST}_${STAMP}.log"; fail "evaluation scripts do not import after the build"; }

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))
H="logs/ledger/hardware_${TAG}_${INST}_${STAMP}"
S="${H}_smoke"

step "2/4 smoke: one corner keep:0.1 run per arm; settings and input against the RTX 4090 runs"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units vector/corner_slow1@keep:0.1 --arms $ARMS \
  --k 1 --seeds 1000 --cholinfo --ledger "$S" || fail "smoke stopped (see $S)"
python script/diag/precision_matrix.py --summary-only --ledger "$S" > "${S}_summary.txt"
grep -E "^   (default|no_tf32_override) " "${S}_summary.txt" | sed 's/^/    recorded: /' || true
# positive evidence required: a missing reference ledger, missing fingerprints or no shared checkpoint stop the run
# (inside "if", set -e does not end the script before the message; pipefail passes compare_inputs' exit code)
if ! python script/diag/compare_inputs.py --ledgers "$S" "$REF_4090" --scene vector/corner_slow1 --cell keep:0.1 | sed 's/^/    /'; then
  fail "input not shown to match the RTX 4090 runs of round 3A ($REF_4090)"
fi

step "3/4 corner keep:0.1 and clean, $ARMS, $K each, on all GPUs of this instance"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARMS \
  --k "$K" --seeds 1000 1001 1002 1003 1004 --cholinfo --ledger "$H" \
  || fail "stopped (see $H)"

echo "  input against the RTX 4090 runs, both conditions (keep:0.1 and the clean control):"
INPUT_OK=1
for cell in keep:0.1 keep:1.0; do
  python script/diag/compare_inputs.py --ledgers "$H" "$REF_4090" --scene vector/corner_slow1 --cell "$cell" | sed 's/^/    /' \
    || INPUT_OK=0
done

step "4/4 summary"
python script/diag/pilot_details.py --ledger "$H" --out "revision/analysis/hardware_${TAG}_${INST}_details_${STAMP}.csv" | tail -12
echo
[ "$INPUT_OK" = 1 ] || fail "runs recorded in $H, but the input was not shown to match the RTX 4090 runs: do not compare the platforms"
echo "Hardware run finished on $GPUNAME. Ledger: $H"
