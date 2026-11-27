#!/usr/bin/env bash
# All SuperEIO bags of the pipeline comparison in parallel, SuperEIO image, CPU only: for every sequence the identity
# check of the clean recording and every perturbed bag (script/supereio/make_r5_bag.py; each run proves its bag against
# the h5 copies in data/R5_ours that script/diag/r5_prep_inputs.sh writes). It can start while the preparation is still
# running on the other instance: each job waits until the two copies it needs (clean and its condition) are complete
# (r5_make_inputs.py gives a copy its final name only when it is complete), for at most WAIT_MIN minutes (default 120).
# One log per job; fails if any job fails. A failed attempt never replaces a passed evidence file; an existing bag is
# never overwritten. Parallel jobs: JOBS (default: available memory / 16 GB, at most the number of jobs).
#
#   bash script/supereio/r5_make_bags.sh
#   CONDS="clean keep:0.1 tq:10000" bash script/supereio/r5_make_bags.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SEQS="${SEQS:-boxes_6dof poster_6dof}"
CONDS="${CONDS:-clean keep:0.1 tq:10000 noise:5 imu:15000}"
WAIT_MIN="${WAIT_MIN:-120}"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/r5
LOG="logs/r5/bags_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }
git log --oneline -1
python3 tests/test_r5_checks.py > "logs/r5/test_r5_checks_bags_${STAMP}.log" 2>&1 \
  || { tail -n 20 "logs/r5/test_r5_checks_bags_${STAMP}.log"; fail "round-5 regression tests failed"; }
JOBLIST="logs/r5/bag_jobs_${STAMP}.txt"
: > "$JOBLIST"
for s in $SEQS; do
  [ -s "data/rosbags/rpgdavis/$s.bag" ] || fail "data/rosbags/rpgdavis/$s.bag missing (script/supereio/fetch_bag.sh $s)"
  for c in $CONDS; do echo "$s $c" >> "$JOBLIST"; done
done
NJOB=$(grep -c . "$JOBLIST")
MEM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 64)
JOBS="${JOBS:-$(( MEM_GB / 16 ))}"
[ "$JOBS" -ge 1 ] || JOBS=1
[ "$JOBS" -le "$NJOB" ] || JOBS=$NJOB
echo "=== $(date +%H:%M:%S) $NJOB jobs, $JOBS in parallel (${MEM_GB} GB available)"
one() {   # seq cond -> log; prints one line
  local s="$1" c="$2" t L rc f ok waited=0
  case "$c" in clean) t=clean;; keep:*) t="keep${c#keep:}";; tq:*) t="tq${c#tq:}us";;
    noise:*) t="noise${c#noise:}hz";; imu:*) t="imu${c#imu:}us";; esac
  L="logs/r5/bag_${s}__${t}_${STAMP}.log"
  while :; do     # the copies this bag is checked against
    ok=1
    for f in "data/R5_ours/${s}__clean/r5_evidence.json" "data/R5_ours/${s}__$t/r5_evidence.json"; do
      [ -s "$f" ] || ok=0
    done
    [ "$ok" = 0 ] || break
    if [ "$waited" -ge $(( WAIT_MIN * 60 )) ]; then
      echo "  $s $c: data/R5_ours/${s}__clean$([ "$t" = clean ] || echo " or ${s}__$t") still not there after $WAIT_MIN min (exit 1)"
      return 1
    fi
    [ "$waited" -gt 0 ] || echo "  $s $c: waiting for data/R5_ours/${s}__clean$([ "$t" = clean ] || echo " and ${s}__$t")"
    sleep 20; waited=$(( waited + 20 ))
  done
  python3 script/supereio/make_r5_bag.py "$s" --cond "$c" > "$L" 2>&1
  rc=$?
  echo "  $s $c: $(tail -n 1 "$L") (exit $rc)"
  return "$rc"
}
export -f one
export STAMP WAIT_MIN
set +e
xargs -P "$JOBS" -L 1 bash -c 'one "$0" "$1"' < "$JOBLIST"
XRC=$?
set -e
[ "$XRC" = 0 ] || fail "at least one bag job failed (see the lines above and logs/r5/bag_*_${STAMP}.log)"
echo "=== $(date +%H:%M:%S) all $NJOB bag jobs passed; log $LOG"
