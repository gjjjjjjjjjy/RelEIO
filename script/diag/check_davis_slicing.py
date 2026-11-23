#!/usr/bin/env python3
"""Check on the real RPG-DAVIS sequences that the binary-search window selection now used by the
DAVIS loader (utils/event_slicing.py, load_utils.mvsec_converted_iterator) selects exactly the
events of the original per-frame mask (ev_t >= t0) & (ev_t < t1).

For each sequence: the events after the loader's undistortion filter, whether their timestamps are
ordered (if not, the loader keeps the original mask and there is nothing to check), and, for
--frames windows spread over the sequence, the selected indices and the selected x / y / t / p
arrays (values and dtypes) from both methods. CPU only; about a minute per sequence.
Exit code 0 only if every checked window is identical.

  python script/diag/check_davis_slicing.py                       # boxes, poster, shapes
  python script/diag/check_davis_slicing.py --scenes boxes_6dof --frames 2000
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utils import event_slicing as S  # noqa: E402

DATA = os.path.join(ROOT, "data", "RPGDAVIS_pp")


def check(scene, frames=200, H=180, W=240):
    import h5py
    scenedir = os.path.join(DATA, scene)
    with h5py.File(os.path.join(scenedir, "events_left.h5"), "r") as f:
        t = f["t"][:].astype(np.float64)
        x = f["x"][:].astype(np.int64)
        y = f["y"][:].astype(np.int64)
        p = f["p"][:].astype(np.int64)
    p = (p > 0).astype(np.int64)
    n_raw = len(t)
    try:
        from utils.load_utils import _mvsec_undistort_lut
        lut = _mvsec_undistort_lut(scenedir, H, W)
    except Exception as e:  # noqa: BLE001 - reported, the check then runs on the raw events
        lut, note = None, "undistortion LUT not loaded (%s: %s); checked on raw events" % (type(e).__name__, e)
    else:
        note = "undistortion applied" if lut is not None else "no undistortion file"
    if lut is not None:
        lut_x, lut_y, _ = lut
        rx, ry = lut_x[y, x], lut_y[y, x]
        valid = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
        x, y, t, p = rx[valid], ry[valid], t[valid], p[valid]
    ts_tab = np.loadtxt(os.path.join(scenedir, "image_timestamps.txt"))
    tss = np.asarray(sorted(ts_tab[:, 1]))
    dt = float(np.diff(tss).mean())
    ordered = S.is_time_ordered(t)
    res = {"scene": scene, "events_raw": n_raw, "events": len(t), "note": note, "ordered": ordered,
           "windows": len(tss), "checked": 0, "mismatches": [], "events_compared": 0}
    if not ordered:
        return res                        # the loader keeps the original mask: identical by construction
    for i in sorted(set(np.linspace(0, len(tss) - 1, min(frames, len(tss))).astype(int).tolist())):
        t0, t1 = tss[i], tss[i] + dt
        mask = (t >= t0) & (t < t1)
        sl = S.window_index(t, t0, t1, True)
        same_idx = np.array_equal(np.flatnonzero(mask), np.arange(sl.start, sl.stop))
        a = [arr[mask] for arr in (x, y, t, p)]
        b = S.take((x, y, t, p), sl)
        same_val = all(np.array_equal(u, v) and u.dtype == v.dtype for u, v in zip(a, b))
        res["checked"] += 1
        res["events_compared"] += int(mask.sum())
        if not (same_idx and same_val):
            res["mismatches"].append(int(i))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["boxes_6dof", "poster_6dof", "shapes_6dof"])
    ap.add_argument("--frames", type=int, default=200, help="windows checked per sequence (spread evenly)")
    a = ap.parse_args()
    ok = True
    with ProcessPoolExecutor(max_workers=len(a.scenes)) as ex:
        for r in ex.map(check, a.scenes, [a.frames] * len(a.scenes)):
            good = not r["mismatches"]
            ok &= good
            print("%-12s events %d (raw %d, %s) | timestamps ordered: %s | windows %d, checked %d, events compared %d | %s"
                  % (r["scene"], r["events"], r["events_raw"], r["note"], r["ordered"], r["windows"], r["checked"],
                     r["events_compared"], "IDENTICAL" if good else "MISMATCH in windows %s" % r["mismatches"][:10]))
    print("RESULT: %s" % ("binary-search selection identical to the original mask" if ok else "MISMATCH, do not run"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
