#!/usr/bin/env bash
# Inputs of the pipeline comparison, CPU only, FreqVIO image (run it inside the FreqVIO container: it writes
# /etc/ld.so.conf.d). Every sequence and condition is independent and runs in parallel (script/diag/r5_prep_one.sh):
#   1. data/R5_ours/<seq>__<tag>   perturbed copy of data/RPGDAVIS_pp/<seq> (script/diag/r5_make_inputs.py): the input of
#                                  our data-loading layer, perturbation applied once to the whole stream
#   2. data/R5_up/<seq>__<tag>     DEIO's own DAVIS240C inputs rebuilt from the copy
#                                  (script/pp_data/pp_davis240c_from_converted.py --src), the input of DEIO's pipeline
#   3. checks (script/diag/r5_check_upstream.py, before the rebuilt folder gets its final name): the events, frame
#      times, millisecond index, calibration, IMU and ground truth that DEIO's entry script reads, read back and
#      compared with the copy, the source and the trusted calibration
# then the window check of script/diag/r5_check_windows.py for each sequence. Fails if any job fails; never
# overwrites an existing folder. Parallel jobs: JOBS (default: available memory / 20 GB, at most the number of jobs).
#
#   bash script/diag/r5_prep_inputs.sh                                  # all 5 conditions of both sequences
#   CONDS="clean keep:0.1 tq:10000" bash script/diag/r5_prep_inputs.sh   # the core conditions only
set -euo pipefail
cd "$(dirname "$0")/../.."
SEQS="${SEQS:-boxes_6dof poster_6dof}"
CONDS="${CONDS:-clean keep:0.1 tq:10000 noise:5 imu:15000}"
PYTHON="${PYTHON:-python}"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/r5 data/R5_ours data/R5_up
LOG="logs/r5/prep_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf 2>/dev/null || true
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf 2>/dev/null || true
ldconfig >/dev/null 2>&1 || true
git log --oneline -1
tag_of() { case "$1" in clean) echo clean;; keep:*) echo "keep${1#keep:}";; tq:*) echo "tq${1#tq:}us";;
  noise:*) echo "noise${1#noise:}hz";; imu:*) echo "imu${1#imu:}us";; *) return 1;; esac; }

JOBLIST="logs/r5/prep_jobs_${STAMP}.txt"
: > "$JOBLIST"
for s in $SEQS; do
  [ -s "data/RPGDAVIS_pp/$s/events_left.h5" ] || fail "data/RPGDAVIS_pp/$s missing"
  [ -s "data/DAVIS240C_upstream/$s/rectify_map_left.h5" ] || fail "data/DAVIS240C_upstream/$s (trusted calibration) missing"
  for c in $CONDS; do
    t=$(tag_of "$c") || fail "unknown condition $c"
    [ ! -e "data/R5_ours/${s}__$t" ] || fail "data/R5_ours/${s}__$t exists; move it away first"
    [ ! -e "data/R5_up/${s}__$t" ] || fail "data/R5_up/${s}__$t exists; move it away first"
    echo "$s $c $STAMP" >> "$JOBLIST"
  done
done
NJOB=$(grep -c . "$JOBLIST")
MEM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 80)
JOBS="${JOBS:-$(( MEM_GB / 20 ))}"
[ "$JOBS" -ge 1 ] || JOBS=1
[ "$JOBS" -le "$NJOB" ] || JOBS=$NJOB
step "1 $NJOB jobs ($SEQS x $CONDS), $JOBS in parallel (${MEM_GB} GB available); one log per job in logs/r5/prep_<seq>__<tag>_$STAMP.log"
set +e
xargs -P "$JOBS" -L 1 bash script/diag/r5_prep_one.sh < "$JOBLIST"
XRC=$?
set -e
[ "$XRC" = 0 ] || fail "at least one job failed (see the FAILED lines above); passed folders are kept, failed ones are not created"

step "2 window check: old per-window perturbation against the whole-stream one (our reader's windows)"
set +e
WPIDS=()   # the PID of every check, saved at its start: a check that already ended still reports its exit code
for s in $SEQS; do $PYTHON script/diag/r5_check_windows.py "$s" > "logs/r5/check_windows_${s}_$STAMP.log" 2>&1 & WPIDS+=($!); done
WRC=0
for p in "${WPIDS[@]}"; do wait "$p" || WRC=1; done
set -e
for s in $SEQS; do cat "logs/r5/check_windows_${s}_$STAMP.log"; done
[ "$WRC" = 0 ] || fail "window check failed"
step "done; log $LOG"
ls -d data/R5_ours/* data/R5_up/*
