"""Convert a monocular event HDF5 stream into the plain-text format E2VID reads.

This is the input adapter for the E2VID + TartanVO (reconstruction-based) system.
It applies the RelEIO event-density stress operator on the raw event stream
*before* reconstruction, so the degradation is injected on the event channel
exactly as it is for the patch-tracking systems.

E2VID input format (one event per line):
  header line: "width height"
  body lines : "t x y pol"   (t in **seconds**, float; x y int; pol in {0,1})
fields are space-separated.

Several HDF5 layouts are auto-detected:
  - top-level x/y/t/p  (e.g. RPG-DAVIS events_left.h5)
  - x/y/t/p under an 'events' group  (RAMP-VO style)

Timestamp unit is auto-detected to seconds (E2VID expects seconds) from the
total span; polarity is mapped to {0,1} (E2VID expects 0/1, not -1/1).

Sanity gate: after conversion, visually confirm the first E2VID frame is a
plausible grayscale image (not noise). A bad frame usually means a wrong
timestamp unit or polarity convention; the detection results are printed below
for cross-checking.

Usage:
  python tools/h5_to_e2vid_txt.py \
    --h5 <DATA_ROOT>/shapes_6dof/events_left.h5 --W 240 --H 180 \
    --out <DATA_ROOT>/shapes_6dof/events_e2vid.txt --keep 0.3
"""
import argparse
import os
import sys
import numpy as np
import h5py

# Reuse the single shared event-density operator so E2VID degrades events with
# exactly the same thinning rule as the patch-tracking systems.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stress_operators"))
from event_stress import thin_events  # noqa: E402


def _find_xytp(f):
    """Locate the x/y/t/p datasets inside the HDF5 file (top level or events/)."""
    for prefix in ["", "events/"]:
        keys = {k: f"{prefix}{k}" for k in ["x", "y", "t", "p"]}
        try:
            if all(f.get(v) is not None for v in keys.values()):
                return keys
        except Exception:
            continue
    raise KeyError(f"x/y/t/p not found (top level or events/). actual keys: {list(f.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--max_events", type=int, default=0, help=">0 keeps only the first N events (debug)")
    ap.add_argument("--keep", type=float, default=1.0,
                    help="event-density stress: fraction of events to keep (e.g. 1.0/0.3/0.1)")
    a = ap.parse_args()

    f = h5py.File(a.h5, "r")
    kp = _find_xytp(f)
    n = a.max_events if a.max_events else None
    sl = slice(0, n)
    x = np.asarray(f[kp["x"]][sl])
    y = np.asarray(f[kp["y"]][sl])
    t = np.asarray(f[kp["t"]][sl]).astype(np.float64)
    p = np.asarray(f[kp["p"]][sl])
    f.close()

    # --- timestamp unit auto-detection -> seconds ---
    # A VO sequence spans tens of seconds: span > 1e12 => ns, > 1e4 => us, else s.
    span = t.max() - t.min()
    if span > 1e12:
        unit, t_s = "ns", t * 1e-9
    elif span > 1e4:
        unit, t_s = "us", t * 1e-6
    else:
        unit, t_s = "s", t.copy()
    t_s = t_s - t_s.min()  # zero-base the start
    # ensure monotonic timestamps (E2VID reads sequentially and needs order)
    if not np.all(np.diff(t_s) >= 0):
        order = np.argsort(t_s, kind="stable")
        x, y, t_s, p = x[order], y[order], t_s[order], p[order]

    # --- polarity -> {0,1} ---
    pu = np.unique(p)
    p01 = (p > 0).astype(int)  # -1/1 -> 0/1; 0/1 unchanged

    # --- event-density stress (keep) via the shared RelEIO operator ---
    # thin_events is seeded by the stream start time, so the thinned stream is
    # bit-identical across repeated runs at a fixed keep level.
    n_before = len(t_s)
    if a.keep < 1.0:
        t_us = (t_s * 1e6).astype(np.int64)  # operator is seeded from a us timestamp
        t_us, x, y, p01 = thin_events(t_us, x, y, p01, keep=a.keep, t0_us=int(t_us[0]))
        t_s = t_us.astype(np.float64) * 1e-6
    actual_keep = len(t_s) / n_before if n_before else 0.0

    print(f"[h5->txt] {a.h5}")
    print(f"  events={n_before} -> {len(t_s)} (req_keep={a.keep} actual_keep={actual_keep:.4f})")
    print(f"  resolution={a.W}x{a.H}")
    print(f"  raw t span={span:.3g} -> detected unit={unit} -> seconds span={t_s.max():.3f}s")
    print(f"  raw polarity unique={pu[:5]} -> mapped to 0/1 (positive fraction={p01.mean():.3f})")
    print(f"  x range[{x.min()},{x.max()}] y range[{y.min()},{y.max()}] (must be within resolution)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # vectorized write (per-line f.write is too slow): stack and savetxt
    arr = np.column_stack([t_s, x.astype(np.int64), y.astype(np.int64), p01])
    with open(a.out, "w") as o:
        o.write(f"{a.W} {a.H}\n")
        np.savetxt(o, arr, fmt=["%.9f", "%d", "%d", "%d"])
    print(f"[done] -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
