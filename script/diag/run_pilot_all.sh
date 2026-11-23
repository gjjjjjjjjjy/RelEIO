#!/usr/bin/env bash
# One command for a fresh GPU instance: environment fix, rebuild of the CUDA
# extensions (with the Cholesky probe), unit tests, smoke test, and, only if the
# smoke test passes its control checks, the precision pilot on every visible GPU
# in the background.
#
#   cd <FreqVIO> && git pull
#   nohup bash script/diag/run_pilot_all.sh >/dev/null 2>&1 &
#   tail -f logs/runs/pilot_all_latest.log
#
# SETUP_ONLY=1 stops after the unit tests (used by run_round2.sh on a fresh instance).
# ARCH: CUDA architecture to compile for. 8.6 for RTX 4090 (the 4090 runs the
# sm_86 code of the image's default list "7.5 8.0 8.6 9.0 10.0 12.0+PTX"), 9.0 for H200.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCH="${ARCH:-8.6}"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/pilot_all_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0/6 environment (libucc), git state"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
# the *_latest link moves only after the guard, so a refused second launch leaves it on the running one
ln -sf "pilot_all_${STAMP}.log" logs/runs/pilot_all_latest.log
python -c "import torch; print('torch OK', torch.__version__, torch.version.cuda, torch.cuda.device_count(), 'GPUs')" \
  || fail "import torch failed"
echo "  python: $(command -v python) | $(python -m pip --version 2>/dev/null | cut -d" " -f1-2,4-)"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
[ "$NGPU" -ge 1 ] || fail "no GPU visible"
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv,noheader

step "1/6 back up the installed extension modules"
BK="backups/so_backup_${STAMP}"   # inside FreqVIO; backups/ is git-ignored
mkdir -p "$BK"
for m in cuda_ba cuda_corr lietorch_backends; do
  if f=$(python -c "import $m; print($m.__file__)" 2>/dev/null); then
    cp -p "$f" "$BK/" && echo "  $m -> $f  (backed up)"
  else
    echo "  $m not installed yet"
  fi
done

step "2/6 build (TORCH_CUDA_ARCH_LIST=$ARCH; progress every 30 s)"
# Instances can share the project directory AND the same hostname, so the build folder is keyed
# by hostname plus the UUID of GPU 0, which is unique per machine.
INST="$(hostname)-$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 2>/dev/null | cut -c5-12)"
BLOG="logs/runs/build_${INST}_${STAMP}.log"
# Build from a per-instance copy (build_hosts/<instance>/src, git-ignored) so that several instances
# sharing this project directory never compile into the same build/ folder at the same time.
BSRC="build_hosts/${INST}/src"
rm -rf "$BSRC"
mkdir -p "$BSRC/thirdparty"
cp -a setup.py dpvo "$BSRC/"
cp -a thirdparty/eigen-3.4.0 "$BSRC/thirdparty/"
(cd "$BSRC" && TORCH_CUDA_ARCH_LIST="$ARCH" python -m pip install . -v --no-build-isolation --break-system-packages) > "$BLOG" 2>&1 &
BPID=$!
while kill -0 "$BPID" 2>/dev/null; do
  sleep 30
  echo "  building... $(grep -oE '\[[0-9]+/[0-9]+\]' "$BLOG" | tail -1 || true)"
done
wait "$BPID" || { tail -30 "$BLOG"; fail "build failed (full output: $BLOG)"; }
grep -E "Successfully installed" "$BLOG" | tail -1 || true

step "3/6 check that cuda_ba is the freshly built FreqVIO version"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)")
echo "  $SO"
[ "$(grep -ac 'cholinfo-1' "$SO")" -ge 1 ] || fail "cuda_ba has no Cholesky probe marker (not the FreqVIO build)"
sha256sum "$SO"

step "4/6 unit tests"
for t in "tests/test_numtrace.py" "tests/test_run_ledger.py" "-m unittest tests.test_sensor_stress"; do
  TLOG="logs/runs/test_${STAMP}.log"
  # shellcheck disable=SC2086
  if python $t > "$TLOG" 2>&1; then
    echo "  OK   $t: $(grep -E '^Ran [0-9]+ tests' "$TLOG" | tail -1)"
  else
    tail -20 "$TLOG"; fail "unit test failed: $t"
  fi
done

if [ "${SETUP_ONLY:-0}" = "1" ]; then
  step "setup finished (SETUP_ONLY=1: no smoke test, no pilot)"
  exit 0
fi

step "5/6 smoke test on GPU 0 (3 runs, stops at the first failed control check)"
SMOKE="logs/ledger/smoke_${STAMP}"
python script/diag/precision_pilot.py --gpu 0 --n 1 --n-untraced 1 --cholinfo --ledger "$SMOKE" \
  || fail "smoke test stopped (see $SMOKE)"

step "6/6 precision pilot on $NGPU GPU(s), background"
N=$(( (20 + NGPU - 1) / NGPU ))                    # traced runs per arm per GPU
U=$(( (10 + NGPU / 2) / NGPU )); [ "$U" -ge 1 ] || U=1   # untraced runs per GPU
PILOT="logs/ledger/precision_pilot_${STAMP}"
echo "  per GPU: $N default + $N fp32_strict + $U untraced; total $((NGPU * (2 * N + U))) runs -> $PILOT"
for g in $(seq 0 $((NGPU - 1))); do
  nohup python script/diag/precision_pilot.py --gpu "$g" --n "$N" --n-untraced "$U" --cholinfo \
    --ledger "$PILOT" > "logs/ledger/pilot_gpu${g}_${STAMP}.out" 2>&1 &
done
sleep 5
echo
echo "Pilot started. Progress (runs finished):  wc -l < $PILOT/ledger.jsonl"
echo "Summary at any time:  python script/diag/precision_pilot.py --summary-only --ledger $PILOT"
echo "Per-GPU output:       tail logs/ledger/pilot_gpu0_${STAMP}.out"
