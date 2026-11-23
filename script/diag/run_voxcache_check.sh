#!/usr/bin/env bash
# Check that reusing CPU-built voxel caches gives bit-identical estimator input, and how much time it saves.
# GPUs 0-3: no cache (2 runs per scene); GPUs 4-7: cache in a fresh namespace tied to the voxel code
# (4 runs per scene; the first builds the cache, the others wait for it and load it). Same ledger, so the
# summary compares the input fingerprints of all runs of a scene. Needs 8 GPUs.
#   cd <FreqVIO> && git pull --ff-only
#   nohup bash script/diag/run_voxcache_check.sh >/dev/null 2>&1 &
#   tail -f logs/runs/voxcache_check_latest.log
set -euo pipefail
cd "$(dirname "$0")/../.."
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/voxcache_check_${STAMP}.log"
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
ln -sf "$(basename "$LOG")" logs/runs/voxcache_check_latest.log
NGPU=$(python -c "import torch; print(torch.cuda.device_count())") || fail "import torch failed"
[ "$NGPU" -ge 8 ] || fail "needs 8 GPUs, found $NGPU"
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  building first (progress in logs/runs/pilot_all_latest.log)"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed"
fi
V="logs/ledger/voxcache_check_${STAMP}"
SCENES="rpgdavis/boxes_6dof vector/desk_normal1"

step "1/2 no cache on GPUs 0-3 and cache on GPUs 4-7, at the same time"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus 0 1 2 3 --scenes $SCENES --arms default --k 2 --seeds 1000 \
  --cholinfo --ledger "$V" > "logs/runs/voxcache_check_nocache_${STAMP}.log" 2>&1 &
P1=$!
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus 4 5 6 7 --scenes $SCENES --arms default --k 4 --seeds 1000 \
  --cholinfo --voxcache --ledger "$V" > "logs/runs/voxcache_check_cache_${STAMP}.log" 2>&1 &
P2=$!
S1=0; S2=0
wait $P1 || S1=$?
wait $P2 || S2=$?
grep -E '^\[|^RESULT|^STOP' "logs/runs/voxcache_check_nocache_${STAMP}.log" "logs/runs/voxcache_check_cache_${STAMP}.log" || true
[ "$S1" = 0 ] && [ "$S2" = 0 ] || fail "one of the two parts stopped (exit $S1 / $S2)"

step "2/2 summary (look at 'Input identity' and the wall-time line)"
python script/diag/precision_matrix.py --summary-only --ledger "$V"
ls -la data/RPGDAVIS_pp/boxes_6dof/_voxcache_*_ns* data/VECtor/desk_normal1/_voxcache_*_ns* 2>/dev/null || true
echo
echo "Voxel-cache check finished. Ledger: $V"
