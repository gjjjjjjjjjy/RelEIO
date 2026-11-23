#!/usr/bin/env bash
# Build SuperEIO (arclab-hku/SuperEIO, GPLv3) at a pinned commit in outputs/supereio/ws (shared disk, git-ignored).
# Run script/supereio/setup_env.sh on the instance first. The build links only against the system packages that
# setup_env.sh pins, so a workspace built once works on any instance that ran setup_env.sh.
#   bash script/supereio/build.sh
#
# Changes to the upstream tree, none of them in the algorithm (recorded in outputs/supereio/build_changes_<stamp>.txt):
#   1. dependency/ is not built (CATKIN_IGNORE): its packages download and build Ceres, glog etc., which apt
#      provides; the DVS message headers SuperEIO uses are vendored in event_detector/src/dvs_msgs.
#   2. the TensorRT engines shipped in the repository (built for another GPU with TensorRT 8.4) and the prebuilt
#      libTensorRTBuffer.so are deleted, only while they are still the upstream files: the engines are rebuilt from
#      the ONNX files on the first run (and then kept), the library by CMake.
#   3. -std=c++11 -> -std=c++17 in the four package CMakeLists: Ubuntu 22.04's ROS headers include log4cxx 0.12,
#      which uses std::shared_mutex (C++17), and Ceres 2.0 needs C++14 or later. libstdc++ still provides the
#      std::bind1st that SuperEIO uses (deprecated in C++11, removed from the C++17 standard).
#   4. `#include <opencv2/cvconfig.h>` is removed from four loop_closure files (VocabularyBinary.hpp/.cpp,
#      DVision/BRIEF.h/.cpp): it is an OpenCV build-internal header that Ubuntu's OpenCV package does not
#      install, and nothing in these files uses its HAVE_* macros.
#   5. In loop_closure's copy of the TensorRT sample helpers (TensorRTBuffer/include/common.h, buffers.h) the CUDA
#      status macro CHECK(status) is renamed TRT_CHECK: it replaced glog's CHECK, which the Ceres 2.0 headers
#      included later in loop_closure.h use as `CHECK(cond) << msg`. Same behaviour, 16 uses, no other change.
#   6. script/supereio/patches/freqvio_counters_keep_all_events.patch (event_detector_node.cpp, estimator.cpp):
#      counts the event packets upstream drops and the solves stopped by the wall-clock limit (printed as
#      "[freqvio] ..." lines); with SUPEREIO_KEEP_ALL_EVENTS=1 every event packet is kept in arrival order. Without
#      that variable the behaviour is upstream's (run_seq.sh VARIANT=orig); VARIANT=ctrl sets it.
# Built with catkin_make_isolated (one CMake project per package, as upstream's `catkin build`): event_detector and
# loop_closure both define a target called TensorRTBuffer, which a single catkin_make project rejects.
# Binaries: outputs/supereio/ws/devel_isolated/<pkg>/lib/<pkg>/<pkg>; source ws/devel_isolated/setup.bash to run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"   # the FreqVIO checkout: first directory with .git above this script
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
OUT="$ROOT/outputs/supereio"
WS="$OUT/ws"
SRC="$WS/src/SuperEIO"
COMMIT=39bc15235e25f92e0427f3914a084e806029e58a   # 2026-03-22, the commit read in revision/SECOND_SYSTEM_FEASIBILITY.md
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT/build_${STAMP}.log"
CHANGES="$OUT/build_changes_${STAMP}.txt"
step() { echo "=== $(date +%H:%M:%S) $*"; }

command -v catkin_make_isolated > /dev/null || { echo "catkin_make_isolated not found: run script/supereio/setup_env.sh first"; exit 1; }
mkdir -p "$WS/src"

step "1/3 source at $COMMIT"
if [ ! -d "$SRC/.git" ]; then
  if [ -e "$SRC" ]; then mv "$SRC" "$SRC.incomplete_$STAMP"; echo "  moved an incomplete checkout aside"; fi
  git clone -q https://github.com/arclab-hku/SuperEIO.git "$SRC"
fi
cd "$SRC"
git cat-file -e "${COMMIT}^{commit}" 2>/dev/null || git fetch -q origin
if [ "$(git rev-parse HEAD)" != "$COMMIT" ]; then git reset -q --hard && git checkout -q "$COMMIT"; fi
echo "  HEAD $(git log -1 --format='%H %ad' --date=short)"

step "2/3 build changes"
touch dependency/CATKIN_IGNORE
git ls-files -z '*.engine' '*libTensorRTBuffer.so' | while IFS= read -r -d '' f; do
  if [ -e "$f" ] && git diff --quiet -- "$f"; then rm -f "$f"; echo "  deleted upstream binary $f"; fi
done
sed -i -E 's/-std=c\+\+1[14]"/-std=c++17"/' camera_model/CMakeLists.txt event_detector/CMakeLists.txt \
  supereio_ba/CMakeLists.txt loop_closure/CMakeLists.txt
sed -i '/#include <opencv2\/cvconfig.h>/d' loop_closure/src/ThirdParty/VocabularyBinary.hpp \
  loop_closure/src/ThirdParty/VocabularyBinary.cpp loop_closure/src/ThirdParty/DVision/BRIEF.h \
  loop_closure/src/ThirdParty/DVision/BRIEF.cpp
sed -i -E 's/\bCHECK\(/TRT_CHECK(/g' loop_closure/src/ThirdParty/TensorRTBuffer/include/common.h \
  loop_closure/src/ThirdParty/TensorRTBuffer/include/buffers.h
PATCH="$ROOT/script/supereio/patches/freqvio_counters_keep_all_events.patch"
if git apply --reverse --check "$PATCH" 2> /dev/null; then echo "  patch already applied: $(basename "$PATCH")"
else git apply "$PATCH" && echo "  applied $(basename "$PATCH")"; fi
{
  echo "SuperEIO $(git log -1 --format='%H %ad' --date=short)"
  echo "--- git status --short"; git status --short
  echo "--- text changes"; git diff -- '*.txt' '*.h' '*.hpp' '*.cpp'
  echo "--- ONNX models"; sha256sum event_detector/src/output/*.onnx
} > "$CHANGES"
git status --short | sed 's/^/  /'

step "3/3 catkin_make_isolated (log: $LOG)"
J=$(nproc); [ "$J" -gt 32 ] && J=32
cd "$WS"
rm -rf build devel src/CMakeLists.txt   # leftovers of a single-project catkin_make attempt
if catkin_make_isolated -DCMAKE_BUILD_TYPE=Release -j"$J" < /dev/null > "$LOG" 2>&1; then
  bad=0
  for p in event_detector supereio_ba loop_closure; do
    b="devel_isolated/$p/lib/$p/$p"
    if [ -x "$b" ]; then echo "  built $b"; else echo "  MISSING $b"; bad=$((bad + 1)); fi
  done
  [ "$bad" -eq 0 ] && echo "RESULT: build OK" || { echo "RESULT: build finished but $bad binary(ies) missing"; exit 1; }
else
  echo "RESULT: BUILD FAILED. First errors:"
  grep -n -m 12 -A 3 -E "error:|CMake Error|undefined reference" "$LOG" || tail -n 25 "$LOG"
  exit 1
fi
