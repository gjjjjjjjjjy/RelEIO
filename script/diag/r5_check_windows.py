#!/usr/bin/env python3
"""CPU check before the pipeline comparison: how much the order of perturbation changes what our reader sees.

The main matrix perturbed events inside our data-loading layer (utils/load_utils.mvsec_converted_iterator): each
frame window [t_k, t_k + dt) is selected by the original timestamps, then thinned with numpy RandomState seeded by
int(t_k) mod 2^31 and quantized on the grid round(t / tq) * tq (utils/event_stress.apply_event_stress). The
comparison perturbs the whole stream once (utils/r5_ops.py) and lets every reader window the result. This script
replays both on data/RPGDAVIS_pp/<seq> with the windows our reader uses (image_timestamps.txt, second column, sorted;
dt = mean frame interval, as with dT_ms unset) and reports, per sequence:
  tq   per source event, the windows holding its original and its quantized time: crossed to another window, left
       all windows (into a gap or past the end), entered one
  keep events kept per window by the old per-window draw and by the new order-free draw (totals, median relative
       difference, correlation) and the share of old-kept events the new draw also keeps (about keep by design)
No GPU, no estimator. Writes logs/r5/check_windows_<seq>.json.

  python3 script/diag/r5_check_windows.py boxes_6dof poster_6dof
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import h5py                                   # noqa: E402
from utils import r5_ops as R                 # noqa: E402

KEEP, TQ = 0.1, 10000


def windows(scenedir):
    ts = np.loadtxt(os.path.join(scenedir, "image_timestamps.txt"))
    tss = np.asarray(sorted(ts[:, 1]))
    return tss, float(np.diff(tss).mean())


def members(t, tss, dt):
    """For each window, the slice [lo, hi) of the time-sorted array t inside [ts, ts + dt)."""
    lo = np.searchsorted(t, tss, side="left")
    hi = np.searchsorted(t, tss + dt, side="left")
    return lo, hi


def _membership(t_sorted, ids, tss, dt, n):
    """Per source event: number of windows [ts, ts + dt) containing it, and the smallest and largest window index."""
    cnt = np.zeros(n, dtype=np.int64)
    kmin = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    kmax = np.full(n, -1, dtype=np.int64)
    lo = np.searchsorted(t_sorted, tss, side="left")
    hi = np.searchsorted(t_sorted, tss + dt, side="left")
    for k in range(len(tss)):
        e = ids[lo[k]:hi[k]]
        cnt[e] += 1
        np.minimum.at(kmin, e, k)
        np.maximum.at(kmax, e, k)
    return cnt, kmin, kmax


def window_moves(t, t_new, tss, dt):
    """Compare, event by event, the windows holding its original time t and its new time t_new.

    crossed: in windows before and after, but not the same ones; left: in a window before, in none after (it fell
    into a gap or past the last window); entered: the reverse. Exact for events in at most two windows (overlaps);
    the largest overlap seen is reported."""
    t = np.asarray(t, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    n = len(t)
    o_old = np.argsort(t, kind="stable")
    o_new = np.argsort(t_new, kind="stable")
    c0, a0, b0 = _membership(t[o_old], o_old, tss, dt, n)
    c1, a1, b1 = _membership(t_new[o_new], o_new, tss, dt, n)
    inside0, inside1 = c0 > 0, c1 > 0
    same = (c0 == c1) & ((a0 == a1) | ~inside0) & ((b0 == b1) | ~inside0)
    crossed = inside0 & inside1 & ~same
    left = inside0 & ~inside1
    entered = ~inside0 & inside1
    in_old = int(np.count_nonzero(inside0))
    in_any = int(np.count_nonzero(inside0 | inside1))
    return {"events_in_windows_old": in_old, "events_in_windows_either": in_any,
            "crossed": int(np.count_nonzero(crossed)),
            "left": int(np.count_nonzero(left)), "entered": int(np.count_nonzero(entered)),
            "share_changed": float(np.count_nonzero(crossed | left | entered)) / max(in_any, 1),
            "max_windows_per_event": int(max(c0.max(initial=0), c1.max(initial=0)))}


def check(seq):
    from utils.event_stress import thin_events, quantize_timestamps   # the main matrix's operators (not public)
    scenedir = os.path.join(ROOT, "data", "RPGDAVIS_pp", seq)
    with h5py.File(os.path.join(scenedir, "events_left.h5"), "r") as f:
        t, x, y, p = f["t"][:].astype(np.float64), f["x"][:], f["y"][:], f["p"][:]
    if not np.all(np.diff(t) >= 0):
        sys.exit("STOPPED: %s events not time-ordered" % seq)
    tss, dt = windows(scenedir)
    lo, hi = members(t, tss, dt)
    res = {"seq": seq, "events": int(len(t)), "windows": int(len(tss)), "dt_us": dt, "keep": KEEP, "tq_us": TQ}

    # --- quantization: which events change window (by source event id) ---------------------------------------
    res["tq"] = window_moves(t, R.quantize_us(t, TQ), tss, dt)
    q_old = quantize_timestamps(t, TQ)
    res["tq"]["events_quantized_differently_from_event_stress"] = int(
        np.count_nonzero(np.rint(q_old).astype(np.int64) != R.quantize_us(t, TQ)))

    # --- thinning: per-window counts under the old and the new draw ----------------------------------------
    keep_new = R.keep_mask(t, x, y, p, KEEP, seq)
    c_old, c_new, overlap, kept_old = [], [], 0, 0
    for k in range(len(tss)):
        a, b = lo[k], hi[k]
        n = b - a
        if n == 0:
            c_old.append(0)
            c_new.append(0)
            continue
        rng = np.random.RandomState(int(tss[k]) % (2 ** 31))      # thin_events, utils/event_stress.py
        sel_old = rng.rand(n) < KEEP
        chk = thin_events(t[a:b], x[a:b], y[a:b], p[a:b], KEEP, tss[k])
        if len(chk[0]) != int(sel_old.sum()):
            sys.exit("STOPPED: replay of thin_events disagrees at window %d" % k)
        sel_new = keep_new[a:b]
        c_old.append(int(sel_old.sum()))
        c_new.append(int(sel_new.sum()))
        kept_old += int(sel_old.sum())
        overlap += int(np.count_nonzero(sel_old & sel_new))
    c_old, c_new = np.asarray(c_old, float), np.asarray(c_new, float)
    nz = c_old > 0
    res["keep"] = {"kept_old": int(c_old.sum()), "kept_new": int(c_new.sum()),
                   "median_rel_diff_per_window": float(np.median(np.abs(c_new[nz] - c_old[nz]) / c_old[nz])),
                   "corr_per_window": float(np.corrcoef(c_old, c_new)[0, 1]) if len(c_old) > 2 else None,
                   "share_old_kept_also_new": overlap / max(kept_old, 1)}
    return res


def main():
    seqs = sys.argv[1:] or ["boxes_6dof", "poster_6dof"]
    os.makedirs(os.path.join(ROOT, "logs", "r5"), exist_ok=True)
    for s in seqs:
        r = check(s)
        with open(os.path.join(ROOT, "logs", "r5", "check_windows_%s.json" % s), "w") as fh:
            json.dump(r, fh, indent=1)
        q, k = r["tq"], r["keep"]
        print("[%s] %d events, %d windows of %.0f us" % (s, r["events"], r["windows"], r["dt_us"]))
        print("  tq %d us: of %d events in a window before or after, %d crossed to another window, %d left all windows, %d entered "
              "one (%.2f%% changed); %d events quantized differently from event_stress (half-grid ties)" % (
                  TQ, q["events_in_windows_either"], q["crossed"], q["left"], q["entered"], 100 * q["share_changed"],
                  q["events_quantized_differently_from_event_stress"]))
        print("  keep %.1f: kept %d (old per-window draw) vs %d (order-free draw); median per-window difference "
              "%.1f%%, correlation %.3f; %.1f%% of old-kept events also kept (about %.0f%% expected)" % (
                  KEEP, k["kept_old"], k["kept_new"], 100 * k["median_rel_diff_per_window"],
                  k["corr_per_window"], 100 * k["share_old_kept_also_new"], 100 * KEEP))


if __name__ == "__main__":
    main()
