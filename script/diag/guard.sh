# Sourced by the launchers in script/diag/ after fail() is defined.
# guard_idle stops a launch at once if this instance is already running an experiment. Without it,
# a second launch goes through setup and tests and only then finds the GPUs busy (per-run
# exclusivity check). pgrep and nvidia-smi see only this instance's processes, so instances
# sharing the project directory do not block each other.
guard_idle() {
  local other busy
  other=$(pgrep -f 'script/diag/precision_(matrix|pilot)\.py --gpu' | tr '\n' ' ' || true)
  if [ -n "${other// /}" ]; then
    fail "an experiment is already running on this instance (PID ${other% }). Progress: ls -t logs/runs/*_latest.log | head -3, then tail -3 the newest one"
  fi
  busy=$(timeout 30 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${busy:-0}" -gt 0 ]; then
    fail "$busy process(es) already on this instance's GPUs (see nvidia-smi); is another experiment still running?"
  fi
}
