#!/usr/bin/env bash
# One sequence and condition of the round-5 input preparation (called by script/diag/r5_prep_inputs.sh, which runs
# several in parallel): the perturbed copy, DEIO's own inputs rebuilt from it in a temporary folder, the read-back
# check of those inputs, and only then the final folder name. Exit 0 only if everything passed.
#   bash script/diag/r5_prep_one.sh <seq> <cond> <stamp>
set -euo pipefail
cd "$(dirname "$0")/../.."
s="$1"; c="$2"; STAMP="$3"
PYTHON="${PYTHON:-python}"
case "$c" in clean) t=clean;; keep:*) t="keep${c#keep:}";; tq:*) t="tq${c#tq:}us";;
  noise:*) t="noise${c#noise:}hz";; imu:*) t="imu${c#imu:}us";; *) echo "unknown condition $c"; exit 1;; esac
LOG="logs/r5/prep_${s}__${t}_${STAMP}.log"
set +e
(
  set -euo pipefail           # its own subshell: a failing step ends it (errexit is not suspended here)
  exec > "$LOG" 2>&1
  echo "=== $(date +%H:%M:%S) $s $c: perturbed copy"
  $PYTHON script/diag/r5_make_inputs.py --seq "$s" --cond "$c"
  echo "=== $(date +%H:%M:%S) $s $c: DEIO's own inputs rebuilt from the copy"
  WORK="data/R5_up/_tmp_${s}__${t}_$STAMP"
  $PYTHON script/pp_data/pp_davis240c_from_converted.py --seq "$s" --src "data/R5_ours/${s}__$t" --outdir "$WORK"
  echo "=== $(date +%H:%M:%S) $s $c: read-back check of the rebuilt inputs"
  CHK="logs/r5/check_upstream_${s}__${t}_$STAMP.log"
  rc=0
  $PYTHON script/diag/r5_check_upstream.py "data/R5_ours/${s}__$t" "$WORK" "data/RPGDAVIS_pp/$s" > "$CHK" 2>&1 || rc=$?
  tail -n 3 "$CHK"
  if [ "$rc" != 0 ]; then
    echo "=== $s $c: FAILED, DEIO's rebuilt inputs differ from the copy ($WORK kept for inspection)"
    exit 1
  fi
  mv "$WORK" "data/R5_up/${s}__$t"
  echo "=== $(date +%H:%M:%S) $s $c: PASSED"
)
RC=$?
set -e
if [ "$RC" != 0 ]; then
  echo "  $s $c: FAILED (log $LOG)"; tail -n 5 "$LOG" | sed 's/^/    /'; exit 1
fi
echo "  $s $c: passed ($LOG)"
