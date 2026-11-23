#!/usr/bin/env bash
# System environment for SuperEIO (the second event-inertial system) on a fresh
# ngc-cuda:12.4.1-devel-ubuntu22.04 instance, as root:
#   bash script/supereio/setup_env.sh
#
# - Ubuntu sources -> the first host in MIRRORS whose apt-get update has no errors. Measured on the cluster
#   (2026-11-21, 17.7 MB index): us.archive.ubuntu.com (official, 91.189.91.82) 3.8 MB/s, USTC 7.9 MB/s,
#   security.ubuntu.com 4.5 MB/s; archive.ubuntu.com resolves to an internal address and gave
#   nothing in 60 s; cn.archive.ubuntu.com redirects to Tsinghua, which answers 403, as does Aliyun.
#   All four pockets, -security included, are taken from the same host. The original file is kept as
#   /etc/apt/sources.list.orig. This changes only this instance. Override with APT_MIRRORS="host1 host2".
# - ROS1 from Ubuntu 22.04's own packages (ros-core 1.16, roscpp 1.15.14, cv_bridge 1.16; not ros.org Noetic, which
#   has no 22.04 build), Ceres 2.0, OpenCV 4.5.4, Eigen 3.4.
# - TensorRT 8.6.1.6 (CUDA 12.0 build), pinned and held: SuperEIO uses TensorRT 8 interfaces that version 10 removed.
#   apt pulls cuDNN 8.9.7 as a dependency.
# - Downloaded .deb files stay on the shared disk (outputs/supereio/apt_cache, git-ignored), so another instance
#   installs the same files without downloading them again.
# - apt is never wrapped in `timeout`: timeout moves it into a background process group and the terminal stops it
#   (state T) when it touches the tty at the end. Its stdin is /dev/null for the same reason.
#
# Writes outputs/supereio/env_<host>_<stamp>.txt (key versions) and dpkg_versions_<host>_<stamp>.txt.
# Exit code 0 only if every check at the end passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"   # the repository checkout: first directory with .git above this script
while [ "$ROOT" != / ] && [ ! -d "$ROOT/.git" ]; do ROOT="$(dirname "$ROOT")"; done
[ -d "$ROOT/.git" ] || { echo "FreqVIO checkout not found above $0"; exit 1; }
OUT="$ROOT/outputs/supereio"
CACHE="$OUT/apt_cache"
STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname | cut -c1-40)"
LOG="$OUT/setup_env_${STAMP}.log"
MIRRORS="${APT_MIRRORS:-us.archive.ubuntu.com mirrors.ustc.edu.cn security.ubuntu.com archive.ubuntu.com}"
TRT_VER="8.6.1.6-1+cuda12.0"
TRT_PKGS="libnvinfer8 libnvinfer-dev libnvinfer-headers-dev libnvonnxparsers8 libnvonnxparsers-dev
libnvinfer-plugin8 libnvinfer-plugin-dev libnvinfer-headers-plugin-dev"
WANT="ros-core-dev ros-core rosbash python3-roslaunch python3-rosmaster python3-rosbag python3-rostopic python3-rosnode
libtf-dev libcv-bridge-dev libimage-transport-dev libroslib-dev librosbag-dev ros-message-generation ros-message-runtime
libnav-msgs-dev libsensor-msgs-dev libgeometry-msgs-dev libstd-msgs-dev catkin python3-rospkg
libceres-dev libopencv-dev libeigen3-dev libgoogle-glog-dev libgflags-dev libyaml-cpp-dev libboost-all-dev
build-essential cmake git curl ca-certificates openssh-client python3-numpy python3-yaml"

export DEBIAN_FRONTEND=noninteractive
APT=(apt-get -y -o "Dir::Cache::archives=$CACHE" -o APT::Sandbox::User=root)
step() { echo "=== $(date +%H:%M:%S) $*"; }
fail() { echo "FAILED: $*"; echo "--- last lines of $LOG"; tail -n 15 "$LOG"; exit 1; }

. /etc/os-release
[ "${VERSION_ID:-}" = "22.04" ] || { echo "needs Ubuntu 22.04, this is ${PRETTY_NAME:-unknown}"; exit 1; }
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
mkdir -p "$CACHE/partial"
echo "log: $LOG"

step "1/5 apt sources -> first mirror that works ($MIRRORS), update"
[ -f /etc/apt/sources.list.orig ] || cp /etc/apt/sources.list /etc/apt/sources.list.orig
used=""
for m in $MIRRORS; do
  printf 'deb http://%s/ubuntu/ jammy%s main restricted universe multiverse\n' \
    "$m" "" "$m" "-updates" "$m" "-backports" "$m" "-security" > /etc/apt/sources.list
  rc=0; out="$("${APT[@]}" update < /dev/null 2>&1)" || rc=$?
  echo "$out" >> "$LOG"
  if [ "$rc" -eq 0 ] && ! grep -qE "^(E:|Err:|W: Failed)" <<< "$out"; then used="$m"; break; fi
  echo "  mirror $m failed:"; grep -E "^(E:|Err:|W: Failed)" <<< "$out" | head -n 3 | sed 's/^/    /'
done
[ -n "$used" ] || fail "apt-get update failed with every mirror"
echo "  using $used"

step "2/5 ROS1 and build dependencies (several minutes the first time: about 400 MB)"
HAVE=""
for p in $WANT; do
  if apt-cache show "$p" > /dev/null 2>&1; then HAVE="$HAVE $p"; else echo "  not in the package index, skipped: $p"; fi
done
# shellcheck disable=SC2086
"${APT[@]}" install --no-install-recommends $HAVE < /dev/null >> "$LOG" 2>&1 || fail "ROS1 install"

step "3/5 TensorRT $TRT_VER (the first time about 1.9 GB with cuDNN)"
# shellcheck disable=SC2046,SC2086
"${APT[@]}" install --no-install-recommends $(for p in $TRT_PKGS; do printf "%s=%s " "$p" "$TRT_VER"; done) \
  < /dev/null >> "$LOG" 2>&1 || fail "TensorRT install"
# shellcheck disable=SC2086
apt-mark hold $TRT_PKGS > /dev/null

step "4/5 checks"
bad=0
for c in roscore roslaunch rosbag rosrun catkin_make cmake g++ git curl python3 nvcc nvidia-smi; do
  if command -v "$c" > /dev/null; then printf "  %-12s %s\n" "$c" "$(command -v "$c")"
  else printf "  %-12s MISSING\n" "$c"; bad=$((bad + 1)); fi
done
for h in NvInfer.h NvOnnxParser.h; do
  if [ -f "/usr/include/x86_64-linux-gnu/$h" ]; then echo "  header $h ok"; else echo "  header $h MISSING"; bad=$((bad + 1)); fi
done
v="$(dpkg-query -W -f='${Version}' libnvinfer8 2>/dev/null || true)"
if [ "$v" = "$TRT_VER" ]; then echo "  libnvinfer8 $v (held)"; else echo "  libnvinfer8 is '$v', expected $TRT_VER"; bad=$((bad + 1)); fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU /' || bad=$((bad + 1))

step "5/5 record versions"
{
  echo "host $HOST  date $(date -Is)"
  echo "os $PRETTY_NAME"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/gpu /'
  nvcc --version | tail -n 1 | sed 's/^/nvcc /'
  dpkg-query -W -f='${Package} ${Version}\n' libnvinfer8 libnvonnxparsers8 libcudnn8 libceres-dev libopencv-dev \
    libeigen3-dev libroscpp-dev ros-core-dev libcv-bridge-dev librosbag-dev catkin libgoogle-glog-dev libyaml-cpp-dev \
    2>/dev/null || true
} > "$OUT/env_${HOST}_${STAMP}.txt"
dpkg -l > "$OUT/dpkg_versions_${HOST}_${STAMP}.txt"
sed 's/^/  /' "$OUT/env_${HOST}_${STAMP}.txt"
echo "  cache: $(du -sh "$CACHE" | cut -f1) in $CACHE"

if [ "$bad" -eq 0 ]; then echo "RESULT: environment OK"; else echo "RESULT: $bad problem(s)"; exit 1; fi
