#!/usr/bin/env bash
# One SuperEIO run on a DAVIS240C rosbag, headless, the way upstream runs it (roslaunch + rosbag play at real-time
# rate), with upstream's algorithm and configuration file; only the output paths in the configuration are changed.
# Needs setup_env.sh and build.sh on this instance, and the bag (fetch_bag.sh).
#   bash script/supereio/run_seq.sh boxes_6dof [tag]
# Output: outputs/supereio/runs/<seq>_<tag>_<stamp>/ with config.yaml (+ config.diff against upstream), launch.log,
#   rosbag.log, evio_result_no_loop.csv (odometry: ns, xyz, qw qx qy qz, v; IMU frame), esvio_result_loop.txt
#   (loop-closed path), engines.sha256 and summary.txt. Then: python3 script/supereio/eval_traj.py <run folder>.
# The first run builds the TensorRT engines from the ONNX files (minutes), one node at a time so that two nodes never
# write the same engine file; they are kept in event_detector/src/output and reused, their sha256 recorded per run.
# VARIANT (environment, default orig):
#   orig  upstream behaviour: event packets are dropped when processing lags (only the newest unprocessed packet is
#         kept), the solver stops on a wall-clock limit (max_solver_time 0.04 s), loop closure runs and feeds
#         relocalisation back into the estimator, event_detector respawns. Only counted (build.sh patch, change 6).
#   ctrl  controlled runtime configuration (D8): four scheduling-dependent settings changed together, so a difference
#         to orig cannot be assigned to any single one of them, and equal repeated results would not prove the
#         pipeline deterministic in general. Every event packet is kept
#         (SUPEREIO_KEEP_ALL_EVENTS=1), max_solver_time 1000 s so the solver stops on max_num_iterations (8) only,
#         loop_closure node not started, no respawn. Algorithm and all other parameters unchanged.
#   orig_noloop  orig with only the loop_closure node not started (one factor: loop closure and the
#         relocalisation it feeds back), so that orig vs orig_noloop and orig_noloop vs ctrl separate it.
# CONFIG_SET="key=value ..." overrides single configuration entries on top of the variant (recorded in config.diff).
# After the bag ends, the run waits until the odometry file has stopped growing for 15 s (ctrl may lag behind).
# The summary counts restarts, "system reboot" (failure detection), dropped event packets, time-limited solves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"   # the FreqVIO checkout: first directory with .git above this script
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
SEQ="${1:?usage: [VARIANT=orig|orig_noloop|ctrl] $0 <sequence> [tag]}"
TAG="${2:-run}"
VARIANT="${VARIANT:-orig}"
case "$VARIANT" in orig|orig_noloop|ctrl) ;; *) echo "VARIANT must be orig, orig_noloop or ctrl"; exit 1 ;; esac
OUT="$ROOT/outputs/supereio"
WS="$OUT/ws"
SRC="$WS/src/SuperEIO"
MODEL="$SRC/event_detector/src/output"
LAUNCH="$ROOT/script/supereio/headless.launch"
BAG="$ROOT/data/rosbags/rpgdavis/$SEQ${BAG_SUFFIX:-}.bag"   # BAG_SUFFIX=__gbias5: bag from make_bias_bag.py
case "$SEQ" in
  boxes_6dof|poster_6dof|shapes_6dof) CFG="$SRC/config/240c/davis346.yaml"; LOOP_TOPIC=/event_detector/timesurface_map ;;
  *) echo "no SuperEIO configuration mapped for $SEQ"; exit 1 ;;
esac
[ -f "$BAG" ] || { echo "missing $BAG (fetch_bag.sh $SEQ; for a bias bag make_bias_bag.py)"; exit 1; }
[ -x "$WS/devel_isolated/supereio_ba/lib/supereio_ba/supereio_ba" ] || { echo "not built: run bash script/supereio/build.sh"; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN_DIR:-$OUT/runs/${SEQ}_${TAG}_${STAMP}}"   # RUN_DIR: the caller names the folder (it must not exist)
[ -e "$RUN" ] && { echo "run folder exists already: $RUN"; exit 1; }
mkdir -p "$RUN/pose_graph"
step() { echo "=== $(date +%H:%M:%S) $*"; }

sed -e "s#^output_path:.*#output_path: \"$RUN\"#" \
    -e "s#^pose_graph_save_path:.*#pose_graph_save_path: \"$RUN/pose_graph/\"#" "$CFG" > "$RUN/config.yaml"
LAUNCH_ARGS=()
if [ "$VARIANT" = ctrl ]; then
  sed -i -E 's/^max_solver_time:.*/max_solver_time: 1000.0  # FreqVIO ctrl: stop on max_num_iterations only/' "$RUN/config.yaml"
  export SUPEREIO_KEEP_ALL_EVENTS=1
  LAUNCH_ARGS=(loop:=false respawn_detector:=false)
else
  unset SUPEREIO_KEEP_ALL_EVENTS
  [ "$VARIANT" = orig_noloop ] && LAUNCH_ARGS=(loop:=false)
fi
for kv in ${CONFIG_SET:-}; do   # e.g. CONFIG_SET="estimate_td=0": one-factor overrides, recorded in config.diff
  k="${kv%%=*}"; v="${kv#*=}"
  grep -q "^$k:" "$RUN/config.yaml" || { echo "CONFIG_SET: no key $k in the configuration"; exit 1; }
  sed -i -E "s/^$k:.*/$k: $v  # FreqVIO override (CONFIG_SET)/" "$RUN/config.yaml"
done
diff "$CFG" "$RUN/config.yaml" > "$RUN/config.diff" || true

set +u; source "$WS/devel_isolated/setup.bash"; set -u
export ROS_LOG_DIR="$RUN/ros_logs" ROS_HOME="$RUN/ros_home"
PORT="${ROS_PORT:-11311}"
export ROS_MASTER_URI="http://localhost:$PORT"
PIDS=()
cleanup() {
  rosnode kill -a > /dev/null 2>&1 || true
  for p in "${PIDS[@]}"; do kill "$p" 2> /dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -9 "$p" 2> /dev/null || true; done
}
trap cleanup EXIT

subscribed() {  # topic node: true when node subscribes to topic
  rostopic info "$1" 2> /dev/null | sed -n '/^Subscribers:/,$p' | grep -q " $2 "
}
wait_for() {    # seconds description command...
  local n="$1" what="$2"; shift 2
  for _ in $(seq "$n"); do "$@" && return 0; sleep 1; done
  echo "  timed out after ${n} s waiting for $what"; return 1
}

LOAD0="$(cut -d" " -f1-3 /proc/loadavg | tr " " /)"
step "roscore on port $PORT"
roscore -p "$PORT" < /dev/null > "$RUN/roscore.log" 2>&1 &
PIDS+=($!)
wait_for 60 "the ROS master" bash -c "rostopic list > /dev/null 2>&1"

if [ ! -s "$MODEL/super_eventpoint.engine" ] || [ ! -s "$MODEL/super_eventmatch.engine" ]; then
  step "building the TensorRT engines from ONNX (first run only, minutes)"
  roslaunch "$LAUNCH" config_path:="$RUN/config.yaml" model_dir:="$MODEL" ba:=false loop:=false \
    respawn_detector:=false < /dev/null > "$RUN/engine_build_detector.log" 2>&1 &
  p=$!
  wait_for 1200 "event_detector to build its engine and subscribe" subscribed /dvs/events /event_detector \
    || { tail -n 20 "$RUN/engine_build_detector.log"; exit 1; }
  kill "$p"; wait "$p" 2> /dev/null || true
  roslaunch "$LAUNCH" config_path:="$RUN/config.yaml" model_dir:="$MODEL" detector:=false ba:=false \
    < /dev/null > "$RUN/engine_build_loop.log" 2>&1 &
  p=$!
  wait_for 1200 "loop_closure to build its engines and subscribe" subscribed "$LOOP_TOPIC" /loop_closure \
    || { tail -n 20 "$RUN/engine_build_loop.log"; exit 1; }
  kill "$p"; wait "$p" 2> /dev/null || true
  sleep 2
fi
(cd "$MODEL" && sha256sum ./*.engine ./*.onnx) > "$RUN/engines.sha256"

step "start the nodes (variant $VARIANT)"
roslaunch "$LAUNCH" config_path:="$RUN/config.yaml" model_dir:="$MODEL" "${LAUNCH_ARGS[@]}" \
  < /dev/null > "$RUN/launch.log" 2>&1 &
PIDS+=($!)
LOOP_CHECK="rostopic info $LOOP_TOPIC 2>/dev/null | sed -n '/^Subscribers:/,\$p' | grep -q ' /loop_closure '"
[ "$VARIANT" != orig ] && LOOP_CHECK=true
wait_for 300 "the nodes to subscribe" bash -c "
  rostopic info /dvs/events 2>/dev/null | sed -n '/^Subscribers:/,\$p' | grep -q ' /event_detector ' &&
  rostopic info /dvs/imu 2>/dev/null | sed -n '/^Subscribers:/,\$p' | grep -q ' /supereio_ba ' && $LOOP_CHECK" \
  || { tail -n 30 "$RUN/launch.log"; exit 1; }
sleep 3

step "rosbag play $SEQ.bag (real time)"
t0=$(date +%s)
BAG_RC=0
rosbag play --clock "$BAG" < /dev/null > "$RUN/rosbag.log" 2>&1 || BAG_RC=$?
[ "$BAG_RC" = 0 ] || echo "  rosbag play exited with $BAG_RC"
t1=$(date +%s)
last=-1; still=0   # wait until the odometry file has not grown for 15 s (at most 900 s)
for _ in $(seq 900); do
  now=$(stat -c %s "$RUN/evio_result_no_loop.csv" 2>/dev/null || echo 0)
  if [ "$now" = "$last" ]; then still=$((still + 1)); else still=0; last=$now; fi
  [ "$still" -ge 15 ] && break
  sleep 1
done
t2=$(date +%s)

step "stop"
cleanup
trap - EXIT

{
  echo "sequence $SEQ  tag $TAG  run $RUN"
  echo "variant $VARIANT"
  echo "bag $BAG"
  echo "machine $(hostname | cut -c1-40) nproc $(nproc) cpus $(taskset -cp $$ 2>/dev/null | sed 's/.*: //') load_at_start $LOAD0 gpu ${CUDA_VISIBLE_DEVICES:-all} ros_port $PORT parallel_slots ${PARALLEL_SLOTS:-1}"
  echo "cpu $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -n 1)"
  echo "SuperEIO $(git -C "$SRC" log -1 --format=%H)  FreqVIO $(git -C "$ROOT" log -1 --format=%h)"
  echo "bag play wall time: $((t1 - t0)) s, output settled $((t2 - t1)) s after the bag ended (incl. 15 s check)"
  echo "rosbag play exit code: $BAG_RC"
  echo "odometry poses (evio_result_no_loop.csv): $(grep -c . "$RUN/evio_result_no_loop.csv" 2>/dev/null || true)"
  echo "loop-closed poses (esvio_result_loop.txt, incl. header): $(grep -c . "$RUN/esvio_result_loop.txt" 2>/dev/null || true)"
  echo "event_detector restarts: $(grep -c 'restarting process' "$RUN/launch.log" || true)"
  echo "failure detections (system reboot!): $(grep -c 'system reboot' "$RUN/launch.log" || true)"
  echo "process deaths: $(grep -c 'process has died' "$RUN/launch.log" || true)"
  echo "event packets: $(grep '\[freqvio\].*event packets' "$RUN/launch.log" | tail -n 1 | sed 's/.*\[freqvio\] //')"
  echo "solver: $(grep '\[freqvio\] solves' "$RUN/launch.log" | tail -n 1 | sed 's/.*\[freqvio\] //')"
  echo "config changed against upstream:"; sed 's/^/  /' "$RUN/config.diff"
} | tee "$RUN/summary.txt"
echo "next: python3 script/supereio/eval_traj.py $RUN"
