#!/usr/bin/env bash
# RelEIO --- run E2VID+TartanVO (non-DPVO: event-to-video reconstruction + VO).
# Two stages: (1) reconstruct frames from (degraded) events with E2VID;
# (2) run TartanVO on the frames. See docs/INTEGRATION.md.
set -euo pipefail
E2VID_ROOT=${E2VID_ROOT:?}; TARTANVO_ROOT=${TARTANVO_ROOT:?}; DATA_ROOT=${DATA_ROOT:?}
KEEPS=${KEEPS:-"1.0 0.3 0.1"}; REPS=${REPS:-"1 2 3"}
for scene in "$@"; do for k in $KEEPS; do
  # (1) events(h5) -> degraded txt -> E2VID frames
  python tools/h5_to_e2vid_txt.py --h5 "$DATA_ROOT/$scene/events.h5" \
    --keep "$k" --out "/tmp/${scene}_k${k}.txt" --W 240 --H 180
  python "$E2VID_ROOT/run_reconstruction.py" -c "$E2VID_ROOT/pretrained/E2VID_lightweight.pth.tar" \
    -i "/tmp/${scene}_k${k}.txt" --output_folder "/tmp/recon" --dataset_name "${scene}_k${k}" \
    --fixed_duration --window_duration 33 --auto_hdr
  # (2) TartanVO on reconstructed frames -> ATE
  for r in $REPS; do
    python "$TARTANVO_ROOT/run_vo.py" --test-dir "/tmp/recon/${scene}_k${k}" \
      --timestamps "/tmp/recon/${scene}_k${k}/timestamps.txt" \
      --gt-file "$DATA_ROOT/$scene/gt.txt" --out-traj "traj_${scene}_k${k}_r${r}.txt"
  done
done; done
