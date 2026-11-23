#!/usr/bin/env bash
# Cross-system IMU-bias check, SuperEIO side (revision/DIAG_PILOT_RUNBOOK.md section 13).
#   boxes_6dof and poster_6dof x {clean, gbias <GBIAS>} x K=10 runs, upstream configuration (VARIANT=orig, no
#   CONFIG_SET) = 40 runs, about 1.6 min each (about 65 min on one GPU). The bias bag carries (+d, -d, +d) deg/s in
#   rad/s on /dvs/imu from the first IMU message on (make_bias_bag.py, which stops unless the bag passes its checks),
#   the same vector DEIO gets for gbias:<d> (script/diag/run_imu_bias.sh). Conditions interleave round by round.
# Every attempt gets its own run folder (RUN_DIR), and the list file records folder and exit code for every attempt,
# so a failed attempt can never pick up an earlier run's output; cross_system_eval.py counts an attempt without
# summary.txt as undetermined. A smoke stage (one clean and one gbias boxes run, not counted) must pass first.
#   GBIAS=5 bash script/supereio/run_imu_bias.sh
# PARALLEL=n runs n attempts at once on one multi-GPU instance (one GPU, one ROS master port and its own block of
# nproc/n CPU cores per slot via taskset, thread pools capped to that block;
# attempts dealt round-robin so every slot mixes sequences and conditions). Because upstream SuperEIO drops event
# packets and stops solves on wall-clock time when the CPU is busy, the summary at the end compares the drop and
# time-limit counts of the clean runs with the single-run range (runbook section 13).
#   GBIAS=5 PARALLEL=8 bash script/supereio/run_imu_bias.sh
# Two single-GPU instances, one sequence each (each still runs one attempt at a time on its own machine):
#   GBIAS=5 SEQS=boxes_6dof bash script/supereio/run_imu_bias.sh     and on the other     GBIAS=5 SEQS=poster_6dof ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"   # the FreqVIO checkout: first directory with .git above this script
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
cd "$ROOT"
unset CONFIG_SET VARIANT BAG_SUFFIX RUN_DIR SUPEREIO_KEEP_ALL_EVENTS
GBIAS="${GBIAS:?set GBIAS (deg/s per axis) as agreed with the authors, e.g. GBIAS=5}"
K="${K:-10}"
PARALLEL="${PARALLEL:-1}"
SEQS="${SEQS:-boxes_6dof poster_6dof}"
SMOKE_SEQ="${SEQS%% *}"
TAG=$(printf "%g" "$GBIAS" | tr . p)
STAMP=$(date +%Y%m%d_%H%M%S)
LIST="outputs/supereio/imu_bias_runs_${STAMP}_$(echo "$SEQS" | tr " " "+").txt"
step() { echo "=== $(date +%H:%M:%S) $*"; }

attempt() {   # seq cond index tag-prefix [gpu port] -> one attempt in its own folder; prints folder and exit code
  local s="$1" c="$2" i="$3" p="$4" gpu="${5:-}" port="${6:-11311}" suf="" R rc
  [ "$c" = gbias ] && suf="__gbias$TAG"
  R="outputs/supereio/runs/${s}_${p}${c}${i}_${STAMP}"
  local pin=()
  if [ -n "$gpu" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu"
    # own block of CPU cores per slot: without it every process sizes its thread pools (OpenCV, OpenMP) to the
    # whole machine and the parallel runs crowd each other (first parallel batch: twice the time-limited solves)
    local per=$(( ${#CPUS[@]} / PARALLEL )) list
    list=$(IFS=,; echo "${CPUS[*]:$(( gpu * per )):$per}")   # whole physical cores, one socket where possible
    pin=(taskset -c "$list")
    export OMP_NUM_THREADS="$per" OPENBLAS_NUM_THREADS="$per" MKL_NUM_THREADS="$per"
  fi
  ROS_PORT="$port" PARALLEL_SLOTS="$PARALLEL" RUN_DIR="$ROOT/$R" BAG_SUFFIX="$suf" VARIANT=orig \
    ${pin[@]+"${pin[@]}"} bash script/supereio/run_seq.sh "$s" "${p}${c}${i}" > /dev/null 2>&1
  rc=$?
  echo "$R $rc"
}

step "bags: clean download, then the bias copy (stops unless axis check, read-back and sha256 pass)"
for s in $SEQS; do
  bash script/supereio/fetch_bag.sh "$s" | tail -n 1 || { echo "STOPPED: download of $s failed"; exit 1; }
  python3 script/supereio/make_bias_bag.py "$s" --gbias-deg-s "$GBIAS" || { echo "STOPPED: bias bag for $s"; exit 1; }
done

step "smoke: one clean and one gbias $SMOKE_SEQ run (not counted)"
for c in clean gbias; do
  read -r R rc < <(attempt "$SMOKE_SEQ" "$c" 0 smoke)
  [ "$rc" = 0 ] && [ -f "$R/summary.txt" ] || { echo "STOPPED: smoke $c failed (exit $rc, $R)"; exit 1; }
  grep -q "^bag .*$( [ "$c" = gbias ] && echo "__gbias$TAG" || echo "$SMOKE_SEQ.bag")" "$R/summary.txt" \
    || { echo "STOPPED: smoke $c played the wrong bag"; grep "^bag" "$R/summary.txt"; exit 1; }
  printf "  smoke %-6s " "$c"; python3 script/supereio/eval_traj.py "$R" 2>&1 | tail -n 2
done

# CPUs ordered by socket, then core, then hyperthread: consecutive blocks are whole cores on one socket
CPUS=()
while read -r c; do CPUS+=("$c"); done < <(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | grep -v '^#' \
  | sort -t, -k3,3n -k2,2n -k1,1n | cut -d, -f1)
[ "${#CPUS[@]}" -gt 0 ] || while read -r c; do CPUS+=("$c"); done < <(seq 0 $(( $(nproc --all) - 1 )))
step "runs: K=$K per sequence and condition, VARIANT=orig, $PARALLEL at a time; every attempt listed in $LIST"
ATTEMPTS=()
for i in $(seq 1 "$K"); do for s in $SEQS; do for c in clean gbias; do ATTEMPTS+=("$s $c $i"); done; done; done
shard() {   # slot k runs its attempts one after another on GPU k (ROS port 11311+10k); the deal rotates by one slot
  local k="$1" j s c i R rc         # per round, so every slot gets every sequence and condition
  for ((j = 0; j < ${#ATTEMPTS[@]}; j++)); do
    [ $(( (j + j / PARALLEL) % PARALLEL )) -eq "$k" ] || continue
    read -r s c i <<< "${ATTEMPTS[$j]}"
    if [ "$PARALLEL" -gt 1 ]; then read -r R rc < <(attempt "$s" "$c" "$i" ib "$k" $((11311 + 10 * k)))
    else read -r R rc < <(attempt "$s" "$c" "$i" ib); fi
    echo "$s $c $i $R rc=$rc slot=$k" >> "$LIST"
    echo "  done: $s $c $i (slot $k, exit $rc)"
  done
}
for ((k = 0; k < PARALLEL; k++)); do shard "$k" & done
wait
step "evaluation of every attempt"
while read -r s c i R rest; do
  printf "%-12s %-6s %2d %-12s " "$s" "$c" "$i" "$rest"
  if [ -f "$R/summary.txt" ]; then python3 script/supereio/eval_traj.py "$R" 2>&1 | tail -n 2
  else echo "no summary.txt: undetermined"; fi
done < <(sort -k1,1 -k2,2 -k3,3n "$LIST")
step "scheduling check (clean runs): event packets dropped and solves stopped by the time limit"
echo "  single-run range from earlier upstream runs on boxes_6dof (origc1-5): dropped 8-17 of 1794, time-limited 35-44 of 700"
while read -r s c i R rest; do
  [ "$c" = clean ] && [ -f "$R/summary.txt" ] || continue
  printf "  %-12s %2d  %s | %s\n" "$s" "$i" "$(grep -o 'dropped [0-9]*' "$R/summary.txt")" \
    "$(grep -o 'time limit [0-9]*' "$R/summary.txt")"
done < <(sort -k1,1 -k3,3n "$LIST")
step "done; $(wc -l < "$LIST") attempts listed in $LIST"
