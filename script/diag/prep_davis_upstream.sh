#!/usr/bin/env bash
# Inputs of DEIO's own DAVIS240C pipeline for the comparison of data pipelines (script/diag/run_davis_pipelines.sh).
# The files are those pp_davis240c.py writes, produced without ROS from the same recording that the
# main matrix uses (script/pp_data/pp_davis240c_from_converted.py, which follows every data-changing step of
# pp_davis240c.py). Works in a new scratch folder, checks every required file of every sequence, and only then moves
# the output to data/DAVIS240C_upstream/<seq>/. It never touches data/RPGDAVIS_pp.
#
#   bash script/diag/prep_davis_upstream.sh boxes_6dof poster_6dof
set -euo pipefail
cd "$(dirname "$0")/../.."
[ "$#" -gt 0 ] || { echo "usage: $0 <sequence> [...]"; exit 1; }
OUT=data/DAVIS240C_upstream
STAMP=$(date +%Y%m%d_%H%M%S)
WORK="data/_tmp_davis_upstream_$STAMP"
PYTHON="${PYTHON:-python}"
# the libucc fix this image needs before torch can be imported (same lines as the run scripts)
echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf
echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf
ldconfig >/dev/null 2>&1 || true

for s in "$@"; do
  [ -s "data/RPGDAVIS_pp/$s/events_left.h5" ] || { echo "STOPPED: data/RPGDAVIS_pp/$s missing"; exit 1; }
  [ ! -e "$OUT/$s" ] || { echo "STOPPED: $OUT/$s exists already; move it away first"; exit 1; }
done
mkdir -p "$WORK" "$OUT"
for s in "$@"; do
  echo "=== $(date +%H:%M:%S) $s"
  "$PYTHON" script/pp_data/pp_davis240c_from_converted.py --seq "$s" --outdir "$WORK/$s" 2>&1 | tee "$WORK/$s.log" \
    || { echo "STOPPED: conversion of $s failed (log $WORK/$s.log); nothing moved"; exit 1; }
done

REQ="calib_undist_left.txt rectify_map_left.h5 evs_left.h5 tss_imgs_us_left.txt raw_tss_imgs_ns_left.txt gt_stamped_left.txt imu_data.csv"
ok=1
for s in "$@"; do
  for f in $REQ; do
    if [ -s "$WORK/$s/$f" ]; then echo "  $s: $f ok"; else echo "  $s: $f MISSING or empty"; ok=0; fi
  done
done
[ "$ok" = 1 ] || { echo "STOPPED: incomplete; nothing moved (scratch folder $WORK)"; exit 1; }
for s in "$@"; do
  mv "$WORK/$s" "$OUT/$s"
  cp "$WORK/$s.log" "$OUT/$s/prep_$STAMP.log"
done
echo "done: $(printf '%s ' "$@")-> $OUT/. Next: the ground-truth check, python script/diag/check_davis_pipelines.py"
