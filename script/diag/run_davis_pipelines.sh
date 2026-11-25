#!/usr/bin/env bash
# Comparison of DAVIS240C data pipelines: DEIO's own preprocessing, entry
# script and configuration (dataset label rpgdavis_up) against the pipeline of the main matrix (rpgdavis), on
# boxes_6dof and poster_6dof, clean input, override removed, seeds 2000-2009: 2 x 2 x 10 = 40 runs.
#
#   MODE=smoke bash script/diag/run_davis_pipelines.sh   # one run per sequence and pipeline (seed 2000), then checks
#   MODE=main  bash script/diag/run_davis_pipelines.sh   # the 40 runs; refuses to start without a passed smoke
#
# Prerequisite: bash script/diag/prep_davis_upstream.sh boxes_6dof poster_6dof (writes data/DAVIS240C_upstream).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${MODE:-smoke}"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger
LOG="logs/runs/davis_pipelines_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

UNITS="rpgdavis_up/boxes_6dof@clean rpgdavis/boxes_6dof@clean rpgdavis_up/poster_6dof@clean rpgdavis/poster_6dof@clean"
ARM=no_tf32_override
SEEDS="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009"

step "0 environment and data (mode $MODE)"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
ln -sf "$(basename "$LOG")" "logs/runs/davis_pipelines_latest.log"
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
# the probe-enabled cuda_ba of the main matrix: an instance whose container was restarted has the image's build, which
# lacks the Cholesky probe (same check and setup as run_round4.sh; the build is the FreqVIO source, ARCH 8.6 as there)
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARK=0
if [ -n "$SO" ]; then MARK=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARK=0; fi
if [ "${MARK:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here (${SO:-not importable}): running setup (build + tests) first; progress in logs/runs/pilot_all_latest.log"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh >/dev/null 2>&1 || fail "setup failed, see logs/runs/pilot_all_latest.log"
  SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)")
  [ "$(grep -ac 'cholinfo-1' "$SO")" -ge 1 ] || fail "cuda_ba still has no Cholesky probe marker after the setup ($SO)"
fi
echo "  cuda_ba: $SO (probe marker present) sha256 $(sha256sum "$SO" | cut -c1-16)"
python tests/test_run_ledger.py > "logs/runs/test_davis_${STAMP}.log" 2>&1 || { tail -n 20 "logs/runs/test_davis_${STAMP}.log"; fail "unit tests failed"; }
python script/diag/check_davis_slicing.py | sed 's/^/    /' || fail "DAVIS event selection differs from the original"
for s in boxes_6dof poster_6dof; do
  [ -s "data/DAVIS240C_upstream/$s/evs_left.h5" ] || fail "data/DAVIS240C_upstream/$s missing: run script/diag/prep_davis_upstream.sh first"
  [ -s "data/RPGDAVIS_pp/$s/mocap_data.txt" ] || fail "data/RPGDAVIS_pp/$s missing (data of the main matrix)"
done
python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351 > "logs/runs/env_davis_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_davis_${STAMP}.log"; fail "Python environment differs from the RTX 4090 rounds"; }
python script/diag/check_davis_pipelines.py || fail "the ground truth of the two pipelines does not hold the same poses"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))

if [ "$MODE" = smoke ]; then
  L="logs/ledger/davis_pipelines_smoke_${STAMP}"
  step "1 smoke: one run per sequence and pipeline, seed 2000"
  # shellcheck disable=SC2086
  python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARM --k 1 --seeds 2000 --cholinfo \
    --ledger "$L" || fail "smoke stopped (see $L)"
  step "2 checks of the smoke runs (seed, stress banner, trajectory saved, matching coverage)"
  python script/diag/check_davis_pipelines.py --mode smoke --ledgers "$L" || fail "smoke checks failed"
  python script/diag/deio_se3_eval.py --ledgers "$L" | tail -n 12
  echo "$L" > logs/ledger/davis_pipelines_smoke_PASSED
  echo; echo "Smoke passed ($L). Next: MODE=main bash script/diag/run_davis_pipelines.sh"
  exit 0
fi

[ "$MODE" = main ] || fail "MODE must be smoke or main"
[ -s logs/ledger/davis_pipelines_smoke_PASSED ] || fail "no passed smoke: run MODE=smoke first"
L="logs/ledger/davis_pipelines_main_${STAMP}"
step "1 main: 4 units x 10 seeds ($SEEDS), $ARM, on all GPUs"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARM --k 10 --seeds $SEEDS --cholinfo \
  --ledger "$L" || fail "stopped (see $L)"
step "2 checks and scores"
CHECK_OK=1
python script/diag/check_davis_pipelines.py --mode main --ledgers "$L" || CHECK_OK=0
python script/diag/deio_se3_eval.py --ledgers "$L" --out "records/davis_pipelines_scores_${STAMP}.csv" | tail -n 14
python script/diag/pilot_details.py --ledger "$L" --out "records/davis_pipelines_details_${STAMP}.csv" | tail -n 6
echo
[ "$CHECK_OK" = 1 ] || fail "runs recorded in $L, but some checks failed (see above): do not use the scores before resolving them"
echo "Finished. Ledger $L; scores records/davis_pipelines_scores_${STAMP}.csv"
echo "Commit and push: git add records/davis_pipelines_*_${STAMP}.csv && git commit -m 'DAVIS240C pipeline comparison' && git push"
