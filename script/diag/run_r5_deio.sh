#!/usr/bin/env bash
# DEIO runs of the pipeline comparison (round 5), FreqVIO image, all GPUs of the instance.
#   units  DEIO through our data-loading layer (r5ours) and through its own DAVIS240C pipeline (r5up), boxes_6dof and
#          poster_6dof, conditions clean / keep0.1 / tq10000us (12 units); with DIST=1 also r5up under the background
#          noise and the IMU clock offset (4 units). The perturbation is in the input files (script/diag/r5_prep_inputs.sh),
#          so every run carries the cell "clean" and the reader's own perturbation is off.
#   arm    no_tf32_override (the TF32 override removed), seeds 4000-4009
#
#   MODE=smoke bash script/diag/run_r5_deio.sh          # one run per unit (seed 4000), then the checks
#   MODE=main  bash script/diag/run_r5_deio.sh          # 10 runs per unit; refuses to start without a passed smoke
#   DIST=1 MODE=smoke|main bash script/diag/run_r5_deio.sh   # with the two disturbances (once approved)
#   PREP=1 DIST=1 MODE=all bash script/diag/run_r5_deio.sh     # fresh instance, nothing waits for a person:
#          the build (if cuda_ba is not installed here) runs in the background while script/diag/r5_prep_inputs.sh
#          prepares the inputs; then the smoke; if its checks pass, the main runs start at once in the same call
#
# Before any run: the regression tests, the environment against the RTX 4090 rounds (library versions), every input
# copy against its manifest (SHA-256 of events_left.h5) and every rebuilt DEIO input against its copy
# (script/diag/r5_check_upstream.py), these checks in parallel. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${MODE:-smoke}"
DIST="${DIST:-0}"
PREP="${PREP:-0}"
case "$MODE" in smoke|main|all) ;; *) echo "MODE must be smoke, main or all"; exit 1;; esac
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/runs logs/ledger records
LOG="logs/runs/r5_deio_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

ARM=no_tf32_override
SEEDS="4000 4001 4002 4003 4004 4005 4006 4007 4008 4009"
TAGS="clean keep0.1 tq10000us"
UNITS=""
for s in boxes_6dof poster_6dof; do
  for t in $TAGS; do UNITS="$UNITS r5up/${s}__$t@clean r5ours/${s}__$t@clean"; done
  if [ "$DIST" = 1 ]; then for t in noise5hz imu15000us; do UNITS="$UNITS r5up/${s}__$t@clean"; done; fi
done
DISTFLAG=""
if [ "$DIST" = 1 ]; then DISTFLAG="--dist"; fi
if [ "$DIST" = 1 ]; then MARK="logs/ledger/r5_deio_smoke_PASSED_dist"; else MARK="logs/ledger/r5_deio_smoke_PASSED"; fi

step "0 environment and inputs (mode $MODE, disturbances $DIST)"
git log --oneline -1
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true
. script/diag/guard.sh
guard_idle
ln -sf "$(basename "$LOG")" "logs/runs/r5_deio_latest.log"
nvidia-smi -L | sed 's/^/  /'
python -c "import torch; print('torch OK', torch.cuda.device_count(), 'GPUs')" || fail "import torch failed"
# the probe-enabled cuda_ba of the main matrix (same check and setup as run_round4.sh and run_davis_pipelines.sh)
SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)" 2>/dev/null || true)
MARKER=0
if [ -n "$SO" ]; then MARKER=$(grep -ac 'cholinfo-1' "$SO" 2>/dev/null) || MARKER=0; fi
SETUP_PID=""
if [ "${MARKER:-0}" -lt 1 ]; then
  echo "  probe-enabled cuda_ba not installed here (${SO:-not importable}): setup (build + tests) in the background"
  SETUP_ONLY=1 bash script/diag/run_pilot_all.sh > "logs/runs/r5_setup_${STAMP}.log" 2>&1 &
  SETUP_PID=$!
fi
if [ "$PREP" = 1 ]; then
  if [ "$DIST" = 1 ]; then PCONDS="clean keep:0.1 tq:10000 noise:5 imu:15000"; else PCONDS="clean keep:0.1 tq:10000"; fi
  step "0a input preparation (script/diag/r5_prep_inputs.sh, conditions $PCONDS)"
  CONDS="$PCONDS" bash script/diag/r5_prep_inputs.sh || fail "input preparation failed (logs/r5/prep_*.log)"
fi
if [ -n "$SETUP_PID" ]; then
  echo "  waiting for the setup (log logs/runs/r5_setup_${STAMP}.log)"
  wait "$SETUP_PID" || { tail -n 20 "logs/runs/r5_setup_${STAMP}.log"; fail "setup failed, see logs/runs/r5_setup_${STAMP}.log"; }
  SO=$(python -c "import cuda_ba; print(cuda_ba.__file__)")
  [ "$(grep -ac 'cholinfo-1' "$SO")" -ge 1 ] || fail "cuda_ba still has no Cholesky probe marker after the setup ($SO)"
fi
echo "  cuda_ba: $SO (probe marker present) sha256 $(sha256sum "$SO" | cut -c1-16)"
python tests/test_run_ledger.py > "logs/runs/test_r5_ledger_${STAMP}.log" 2>&1 \
  || { tail -n 20 "logs/runs/test_r5_ledger_${STAMP}.log"; fail "ledger unit tests failed"; }
python tests/test_r5_checks.py > "logs/runs/test_r5_checks_${STAMP}.log" 2>&1 \
  || { tail -n 20 "logs/runs/test_r5_checks_${STAMP}.log"; fail "round-5 regression tests failed"; }
python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351 > "logs/runs/env_r5_${STAMP}.log" 2>&1 \
  || { tail -n 15 "logs/runs/env_r5_${STAMP}.log"; fail "Python environment differs from the RTX 4090 rounds"; }
precheck() {   # unit -> one line; exit 0 only if its input passed
  local u="$1" ds rest scene seq L rc
  ds=${u%%/*}; rest=${u#*/}; scene=${rest%@*}; seq=${scene%%__*}
  L="logs/runs/r5_precheck_${ds}_${scene}_${STAMP}.log"
  if [ "$ds" = r5ours ]; then
    python - "$scene" > "$L" 2>&1 <<'EOF'
import hashlib, json, sys
d = "data/R5_ours/" + sys.argv[1]
h = hashlib.sha256()
with open(d + "/events_left.h5", "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        h.update(b)
ok = h.hexdigest() == json.load(open(d + "/r5_evidence.json"))["output"]["events_sha256"]
print("events_left.h5 %s its manifest" % ("matches" if ok else "DIFFERS FROM"))
sys.exit(0 if ok else 1)
EOF
  else
    python script/diag/r5_check_upstream.py "data/R5_ours/$scene" "data/R5_up/$scene" "data/RPGDAVIS_pp/$seq" > "$L" 2>&1
  fi
  rc=$?
  echo "  $ds/$scene: $(tail -n 1 "$L")$([ "$rc" = 0 ] || echo " -- FAILED, see $L")"
  return "$rc"
}
export -f precheck
export STAMP
MEM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 80)
NUNIT=$(echo $UNITS | wc -w | tr -d " ")
PJOBS=$(( MEM_GB / 20 )); [ "$PJOBS" -ge 1 ] || PJOBS=1; [ "$PJOBS" -le "$NUNIT" ] || PJOBS=$NUNIT
echo "  input checks of $NUNIT units, $PJOBS in parallel"
# shellcheck disable=SC2086
printf "%s\n" $UNITS | xargs -P "$PJOBS" -I{} bash -c 'precheck "$1"' _ {} \
  || fail "an input differs from its manifest or its copy (see the FAILED line above)"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPUS=$(seq -s " " 0 $((NGPU - 1)))

if [ "$MODE" = smoke ] || [ "$MODE" = all ]; then
  L="logs/ledger/r5_deio_smoke_${STAMP}"
  step "1 smoke: one run per unit, seed 4000, $ARM"
  # shellcheck disable=SC2086
  python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARM --k 1 --seeds 4000 --cholinfo \
    --ledger "$L" || fail "smoke stopped (see $L)"
  step "2 checks of the smoke runs"
  python script/diag/r5_check_runs.py --mode smoke $DISTFLAG --ledgers "$L" || fail "smoke checks failed"
  python script/diag/pilot_details.py --ledger "$L" | tail -n 6 || true
  echo "$L" > "$MARK"
  if [ "$MODE" = smoke ]; then
    echo; echo "Smoke passed ($L). Note the wall time per run above. Next: $([ "$DIST" = 1 ] && echo "DIST=1 ")MODE=main bash script/diag/run_r5_deio.sh"
    exit 0
  fi
  echo "  smoke passed ($L): main runs start now"
fi

[ -s "$MARK" ] || fail "no passed smoke for this plan: run MODE=smoke$([ "$DIST" = 1 ] && echo " with DIST=1") first"
L="logs/ledger/r5_deio_main_${STAMP}"
step "1 main: $(echo $UNITS | wc -w) units x 10 seeds ($SEEDS), $ARM, on all GPUs"
# shellcheck disable=SC2086
python script/diag/precision_matrix.py --gpus $GPUS --units $UNITS --arms $ARM --k 10 --seeds $SEEDS --cholinfo \
  --ledger "$L" || fail "stopped (see $L)"
step "2 checks and scores"
CHECK_OK=1
python script/diag/r5_check_runs.py --mode main $DISTFLAG --ledgers "$L" || CHECK_OK=0
python script/diag/deio_se3_eval.py --ledgers "$L" --out "records/r5_deio_scores_${STAMP}.csv" | tail -n 20
python script/diag/pilot_details.py --ledger "$L" --out "records/r5_deio_details_${STAMP}.csv" | tail -n 6
echo
[ "$CHECK_OK" = 1 ] || fail "runs recorded in $L, but some checks failed (see above): do not use the scores before resolving them"
echo "Finished. Ledger $L; scores records/r5_deio_scores_${STAMP}.csv"
echo "Commit and push: git add records/r5_deio_*_${STAMP}.csv && git commit -m 'Round 5: DEIO runs' && git push"
