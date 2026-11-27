#!/usr/bin/env bash
# SuperEIO runs of the pipeline comparison (round 5), SuperEIO image (Ubuntu 22.04), one instance with several GPUs.
#   boxes_6dof and poster_6dof x conditions clean / keep0.1 / tq10000us (DIST=1: also noise5hz / imu15000us) x K=10
#   attempts, the authors' configuration (VARIANT=orig). Clean plays the original bag; a perturbed condition plays
#   data/rosbags/rpgdavis/<seq>__r5_<tag>.bag, which script/supereio/make_r5_bag.py wrote and checked against the h5
#   copies the DEIO readers take. Every bag (and, for clean, the identity check) needs a passed evidence file whose
#   SHA-256 still matches; otherwise nothing starts.
# CPU allocation (the configuration of the valid batch of the bias experiment): each slot gets exactly 16 logical CPUs,
# whole physical cores taken from the CPUs this container may use (sched_getaffinity, not the host's count), one
# socket where possible, blocks disjoint; thread pools capped to 16; one GPU and one ROS port per slot; attempts dealt
# round-robin with a rotation per round. Slots = min(8, GPUs, complete blocks). After the runs every attempt's recorded
# CPU set (run_seq.sh writes it into summary.txt) must equal its planned block.
# TensorRT engines: the files in outputs/supereio/ws/.../event_detector/src/output are shared by every attempt. The
# smoke's first attempt (clean poster_6dof) runs alone, before any parallel one: it loads the engines on this GPU, or
# builds them if they are missing (one process, so no two attempts write the same file), and must complete with a
# scorable trajectory. Its engines.sha256 is the reference of the smoke and the main runs; the main runs start only if
# the files still match it, and every attempt must have recorded it. Whether they are the files of the bias
# experiment's runs is printed and recorded (a separate statement: equal files need not run at equal speed here).
# The clean boxes_6dof smoke attempt runs in the first parallel wave, so the smoke shows its time-limited solves at
# this concurrency (must be inside the single-run range 35-44).
# The plan is fixed: both sequences, K=10 in the main batch (K and SEQS from the shell are ignored).
# Acceptance (script/supereio/r5_accept.py; outcomes by the rule of the bias experiment, cross_system_eval.py):
#   smoke  every planned attempt present once, CPU set and engines exact, none undetermined (no summary, run_seq.sh or
#          rosbag play exit code not 0, a ROS process died), every clean attempt completed and scorable; an algorithm
#          failure under a perturbation is listed, not a reason to stop. Any miss fails the smoke (no marker written)
#   main   a machine-readable record of every attempt (algorithm failures and undetermined attempts stay in it,
#          counted apart) and of the batch's validity: CPU sets and engines exact, completed attempts scorable, and
#          the scheduling check of the bias experiment on all planned clean boxes_6dof attempts (each with its
#          checkpoint at 700 solves, e.g. 10/10; median of the solves stopped by the time limit in 35-44)
#
#   MODE=smoke bash script/supereio/run_r5_supereio.sh    # one attempt per sequence and condition (not counted)
#   MODE=main  bash script/supereio/run_r5_supereio.sh    # K=10 per sequence and condition; needs a passed smoke
#   DIST=1 MODE=... bash script/supereio/run_r5_supereio.sh
#   DIST=1 MODE=all bash script/supereio/run_r5_supereio.sh  # the smoke, then (only if it is accepted) the main runs
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
cd "$ROOT"
unset CONFIG_SET VARIANT BAG_SUFFIX RUN_DIR SUPEREIO_KEEP_ALL_EVENTS
MODE="${MODE:-smoke}"
if [ "$MODE" = all ]; then   # two separate calls, each with its own record; the main one only after an accepted smoke
  MODE=smoke bash "$ROOT/script/supereio/run_r5_supereio.sh" || exit 1
  MODE=main exec bash "$ROOT/script/supereio/run_r5_supereio.sh"
fi
case "$MODE" in smoke|main) ;; *) echo "MODE must be smoke, main or all"; exit 1;; esac
DIST="${DIST:-0}"
K=10                                # fixed: a K or SEQS left in the shell must not shrink the batch
SEQS="boxes_6dof poster_6dof"
TAGS="clean keep0.1 tq10000us"
if [ "$DIST" = 1 ]; then TAGS="$TAGS noise5hz imu15000us"; fi
if [ "$DIST" = 1 ]; then MARK="logs/r5/supereio_smoke_PASSED_dist"; else MARK="logs/r5/supereio_smoke_PASSED"; fi
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/supereio/runs logs/r5
LIST="outputs/supereio/r5_runs_${MODE}_${STAMP}.txt"
BLOCKS="outputs/supereio/r5_cpu_blocks_${MODE}_${STAMP}.txt"
LOG="logs/r5/supereio_${MODE}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! log: $LOG"; exit 1; }

step "0 machine, tests and inputs (mode $MODE, disturbances $DIST)"
git log --oneline -1
echo "  $(hostname | cut -c1-40): $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -n 1); $NGPU GPUs"
nvidia-smi -L | sed 's/^/  /'
python3 tests/test_r5_checks.py > "logs/r5/test_r5_checks_${STAMP}.log" 2>&1 \
  || { tail -n 20 "logs/r5/test_r5_checks_${STAMP}.log"; fail "round-5 regression tests failed"; }
for s in $SEQS; do
  for t in $TAGS; do
    python3 - "$s" "$t" <<'EOF' || fail "bag for $s $t not checked or changed since"
import hashlib, json, os, sys
s, t = sys.argv[1], sys.argv[2]
ev = "data/rosbags/rpgdavis/%s__r5_%s.json" % (s, t)
if not os.path.isfile(ev):
    print("  %s missing: run make_r5_bag.py %s --cond ... first" % (ev, s)); sys.exit(1)
e = json.load(open(ev))
bag = "data/rosbags/rpgdavis/%s.bag" % s if t == "clean" else "data/rosbags/rpgdavis/%s__r5_%s.bag" % (s, t)
want = e.get("source_sha256") if t == "clean" else e.get("output_sha256")
h = hashlib.sha256()
with open(bag, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        h.update(b)
ok = bool(e.get("passed")) and h.hexdigest() == want
print("  %s %s: %s (%s)" % (s, t, bag, "checked, unchanged" if ok else "NOT PASSED OR CHANGED"))
sys.exit(0 if ok else 1)
EOF
  done
done

SEXE="outputs/supereio/ws/devel_isolated/supereio_ba/lib/supereio_ba/supereio_ba"
MODEL="outputs/supereio/ws/src/SuperEIO/event_detector/src/output"
[ -x "$SEXE" ] || fail "SuperEIO is not built ($SEXE missing): bash script/supereio/build.sh"
echo "  SuperEIO built: $SEXE"
engines_now() { (cd "$MODEL" && sha256sum ./*.engine ./*.onnx 2>/dev/null) | sort; }
BIAS_ENG=$(ls -t outputs/supereio/runs/boxes_6dof_ibclean*/engines.sha256 2>/dev/null | head -n 1)
if [ "$MODE" = main ]; then
  [ -s "$MARK" ] || fail "no passed smoke for this plan: run MODE=smoke$([ "$DIST" = 1 ] && echo " with DIST=1") first"
  REF=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["engines_reference_file"])' "$(cat "$MARK")") \
    || fail "no engine reference in the smoke record $(cat "$MARK")"
  [ -s "$REF" ] || fail "engine reference $REF missing"
  diff <(engines_now) <(sort "$REF") > /dev/null || fail "the TensorRT engine files changed since the smoke ($REF)"
  echo "  TensorRT engines unchanged since the smoke ($REF)"
else
  REF="outputs/supereio/r5_engines_${STAMP}.sha256"
  if ls "$MODEL"/super_eventpoint.engine "$MODEL"/super_eventmatch.engine > /dev/null 2>&1; then
    echo "  TensorRT engines present; the first smoke attempt loads them alone before any parallel attempt"
  else
    echo "  TensorRT engines missing; the first smoke attempt builds them alone (minutes) before any parallel attempt"
  fi
fi

step "1 CPU blocks of 16 logical CPUs (whole cores) from the CPUs this container may use"
python3 script/supereio/r5_accept.py blocks --max-slots 8 --gpus "$NGPU" --out "$BLOCKS" || fail "no CPU allocation"
PARALLEL=$(grep -c . "$BLOCKS")
[ "$PARALLEL" -ge 1 ] || fail "no complete block of 16 logical CPUs"
mapfile -t BLOCK < "$BLOCKS"

attempt() {   # seq tag index slot port -> one attempt in its own folder; prints folder and exit code
  local s="$1" t="$2" i="$3" slot="$4" port="$5" suf="" R rc
  [ "$t" = clean ] || suf="__r5_$t"
  R="outputs/supereio/runs/${s}_r5${t}_${i}_${MODE}_${STAMP}"
  export CUDA_VISIBLE_DEVICES="$slot"
  export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
  ROS_PORT="$port" PARALLEL_SLOTS="$PARALLEL" RUN_DIR="$ROOT/$R" BAG_SUFFIX="$suf" VARIANT=orig \
    taskset -c "${BLOCK[$slot]}" bash script/supereio/run_seq.sh "$s" "r5${t}${i}" > /dev/null 2>&1
  rc=$?
  echo "$R $rc"
}

if [ "$MODE" = smoke ]; then KRUN=1; else KRUN="$K"; fi
if [ "$MODE" != smoke ] && [ ! -s "$MARK" ]; then fail "no passed smoke for this plan: run MODE=smoke first"; fi
ATTEMPTS=()
for i in $(seq 1 "$KRUN"); do for s in $SEQS; do for t in $TAGS; do ATTEMPTS+=("$s $t $i"); done; done; done
printf "%s\n" "${ATTEMPTS[@]}" > "$LIST.plan"
REST=("${ATTEMPTS[@]}")
if [ "$MODE" = smoke ]; then
  SOLO="poster_6dof clean 1"   # alone first; clean boxes_6dof then leads the first parallel wave
  REST=()
  for x in "${ATTEMPTS[@]}"; do [ "$x" = "$SOLO" ] || REST+=("$x"); done
  [ "${#REST[@]}" = $(( ${#ATTEMPTS[@]} - 1 )) ] || fail "the first attempt ($SOLO) is not in the plan"
  read -r s t i <<< "$SOLO"
  step "2a the first smoke attempt alone ($s $t $i, slot 0): TensorRT engines loaded or built by one process"
  read -r R rc < <(attempt "$s" "$t" "$i" 0 11311)
  echo "$s $t $i $R rc=$rc slot=0" >> "$LIST"
  echo "  done: $s $t $i (slot 0, exit $rc)"
  { [ "$rc" = 0 ] && [ -f "$R/summary.txt" ] && [ -s "$R/engines.sha256" ]; } \
    || fail "the first attempt failed (exit $rc, $R): see $R/launch.log and $R/engine_build_*.log"
  python3 script/supereio/eval_traj.py "$R" > "$R/eval.log" 2>&1; echo $? > "$R/eval.rc"
  python3 - "$R" "$rc" <<'EOF' || fail "the first attempt did not complete with a scorable trajectory ($R): no parallel attempt started"
import sys
sys.path.insert(0, "script/supereio")
import r5_accept as A
r = A.read_attempt(sys.argv[1], int(sys.argv[2]))
print("  first attempt: %s, %d poses, rule-D error %s %%" % (r["outcome"], r["poses"], r["se3_all"]))
sys.exit(0 if r["outcome"] == "completed" and r["se3_all"] is not None else 1)
EOF
  sort "$R/engines.sha256" > "$REF"
  diff <(engines_now) "$REF" > /dev/null || fail "engine files changed while the first attempt ran"
  grep -ihE "different models of devices|incompatible|deserializ|Error Code" "$R/launch.log" "$R"/engine_build_*.log \
    2>/dev/null | sort | uniq -c | head -n 5 | sed 's/^/  TensorRT message: /'
  if [ -z "$BIAS_ENG" ]; then echo "  engines of the bias experiment not found: cannot compare (recorded)"
  elif diff <(sort "$BIAS_ENG") "$REF" > /dev/null; then echo "  engines identical to the bias experiment's ($BIAS_ENG)"
  else echo "  engines DIFFER from the bias experiment's ($BIAS_ENG): recorded; tell the authors"; fi
  echo "  reference $REF"
fi
step "2 $MODE: ${#ATTEMPTS[@]} attempts (${#REST[@]} now), $PARALLEL at a time; every attempt listed in $LIST"
shard() {   # slot k runs its attempts one after another on GPU k; the deal rotates by one slot per round
  local k="$1" j s t i R rc
  for ((j = 0; j < ${#REST[@]}; j++)); do
    [ $(( (j + j / PARALLEL) % PARALLEL )) -eq "$k" ] || continue
    read -r s t i <<< "${REST[$j]}"
    read -r R rc < <(attempt "$s" "$t" "$i" "$k" $((11311 + 10 * k)))
    echo "$s $t $i $R rc=$rc slot=$k" >> "$LIST"
    echo "  done: $s $t $i (slot $k, exit $rc)"
  done
}
SPIDS=()
for ((k = 0; k < PARALLEL; k++)); do shard "$k" & SPIDS+=($!); done
wait "${SPIDS[@]}"   # the shards only: a bare wait also waits for the log's tee (exec > >(tee)) and never returns

step "3 scoring of every attempt (in parallel)"
for s in $SEQS; do   # the ground truth eval_traj.py falls back to, written once here rather than by parallel scorers
  [ -s "data/RPGDAVIS_pp/$s/raw_gt_stamped_ns_left.txt" ] || [ -s "outputs/supereio/gt/${s}_optitrack_davis_us.txt" ] \
    || python3 -c "import sys; sys.path.insert(0, 'script/supereio'); import eval_traj as E; \
E.gt_from_bag('data/rosbags/rpgdavis/%s.bag' % sys.argv[1], 'outputs/supereio/gt/%s_optitrack_davis_us.txt' % sys.argv[1])" "$s" \
    || fail "no ground truth for $s"
done
score() {   # run folder -> eval.log, eval.json and eval.rc in it
  [ -f "$1/summary.txt" ] || return 0
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 script/supereio/eval_traj.py "$1" > "$1/eval.log" 2>&1
  echo $? > "$1/eval.rc"
}
export -f score
awk '{print $4}' "$LIST" | xargs -P 16 -I{} bash -c 'score "$1"' _ {}
while read -r s t i R rest; do
  printf "%-12s %-11s %2d %-14s " "$s" "$t" "$i" "$rest"
  if [ -f "$R/summary.txt" ]; then echo "eval exit $(cat "$R/eval.rc" 2>/dev/null || echo '?') | $(tail -n 2 "$R/eval.log" | head -n 1 | cut -c1-120)"
  else echo "no summary.txt"; fi
done < <(sort -k1,1 -k2,2 -k3,3n "$LIST")

step "4 acceptance ($MODE)"
REC="outputs/supereio/r5_${MODE}_${STAMP}.json"
python3 script/supereio/r5_accept.py check --mode "$MODE" --plan "$LIST.plan" --list "$LIST" --blocks "$BLOCKS" \
  --engines "$REF" --bias-engines "$BIAS_ENG" --out "$REC"
ACC=$?
if [ "$MODE" = smoke ]; then
  [ "$ACC" = 0 ] || fail "smoke not accepted (see $REC): investigate before the main runs"
  echo "$REC" > "$MARK"
  echo; echo "Smoke passed ($REC). Next: $([ "$DIST" = 1 ] && echo "DIST=1 ")MODE=main bash script/supereio/run_r5_supereio.sh"
else
  echo; echo "Finished: record $REC (batch $([ "$ACC" = 0 ] && echo VALID || echo "NOT VALID: see the record"))."
  [ "$ACC" = 0 ] || exit 1
fi
