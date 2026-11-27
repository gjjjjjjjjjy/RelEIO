#!/usr/bin/env python3
"""Perturbed copies of a DAVIS240C recording for the pipeline comparison.

Reads the recording the main matrix uses, data/RPGDAVIS_pp/<seq> (events_left.h5 from the text release, times in
microseconds), applies one condition with the order-free operators of utils/r5_ops.py to the whole event stream once,
and writes a complete scene folder that every DEIO reader takes as input:
  data/R5_ours/<seq>__<tag>/   all files of data/RPGDAVIS_pp/<seq> (voxel caches excluded), events_left.h5 replaced
                               by the perturbed stream (times in whole microseconds, sorted), r5_evidence.json
Our data-loading layer reads this folder with its own perturbation switched off (FREQVIO_EVENT_KEEP and
FREQVIO_TIME_QUANT_US unset); DEIO's own pipeline is rebuilt from it by pp_davis240c_from_converted.py --src; the
SuperEIO bag writer (script/supereio/make_r5_bag.py) applies the same operators to the rosbag and must reproduce the
per-second digest stored here. The clean condition is written the same way (times rounded to whole microseconds), so
all three readers start from identical sensor data.

  python3 script/diag/r5_make_inputs.py --seq boxes_6dof --cond clean
  python3 script/diag/r5_make_inputs.py --seq boxes_6dof --cond keep:0.1
  python3 script/diag/r5_make_inputs.py --seq boxes_6dof --cond tq:10000
  python3 script/diag/r5_make_inputs.py --seq boxes_6dof --cond noise:5        # background activity, Hz per pixel
  python3 script/diag/r5_make_inputs.py --seq boxes_6dof --cond imu:15000      # IMU times shifted by +15 ms
Refuses to overwrite an existing folder; move it away first.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import h5py                     # noqa: E402
from utils import r5_ops as R   # noqa: E402

SRC_ROOT = os.path.join(ROOT, "data", "RPGDAVIS_pp")
OUT_ROOT = os.path.join(ROOT, "data", "R5_ours")


def tag_of(cond):
    """Folder suffix of a condition: clean, keep0.1, tq10000us."""
    if cond == "clean":
        return "clean"
    if cond.startswith("keep:"):
        return "keep" + cond[5:]
    if cond.startswith("tq:"):
        return "tq%sus" % cond[3:]
    if cond.startswith("noise:"):
        return "noise%shz" % cond[6:]
    if cond.startswith("imu:"):
        return "imu%sus" % cond[4:]
    raise ValueError(cond)


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--cond", required=True,
                    help="clean | keep:<rho> | tq:<microseconds> | noise:<Hz per pixel> | imu:<microseconds>")
    a = ap.parse_args()
    src = os.path.join(SRC_ROOT, a.seq)
    ev_src = os.path.join(src, "events_left.h5")
    if not os.path.isfile(ev_src):
        sys.exit("STOPPED: %s missing" % ev_src)
    out = os.path.join(OUT_ROOT, "%s__%s" % (a.seq, tag_of(a.cond)))
    if os.path.exists(out):
        sys.exit("STOPPED: %s exists; move it away first" % out)
    work = out + ".tmp%d" % os.getpid()
    os.makedirs(work)
    t0 = time.time()
    copied = []
    for name in sorted(os.listdir(src)):
        if name == "events_left.h5" or name.startswith("_voxcache") or name.startswith("."):
            continue
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(work, name))
            copied.append(name)
    with h5py.File(ev_src, "r") as f:
        t, x, y, p = f["t"][:], f["x"][:], f["y"][:], f["p"][:]
        dtypes = {k: str(f[k].dtype) for k in ("t", "x", "y", "p")}
    p01 = (np.asarray(p) > 0).astype(np.int64)          # the text release stores 0/1; other files -1/1
    ti, xi, yi, pi = R.as_event_arrays(t, x, y, p01)
    frac = np.abs(np.asarray(t, dtype=np.float64) - ti)
    if not np.all(np.diff(ti) >= 0):
        shutil.rmtree(work)
        sys.exit("STOPPED: source events not time-ordered")
    n_first = min(2000, len(ti))                     # extended to the end of the timestamp group at the cut
    if n_first:
        n_first = int(np.searchsorted(ti, ti[n_first - 1], side="right"))
    src_digest = R.per_second_digest(ti, xi, yi, pi)
    tt, xx, yy, pp, rec = R.apply_condition(ti, xi, yi, pi, a.cond, a.seq)
    with h5py.File(os.path.join(work, "events_left.h5"), "w") as f:
        f.create_dataset("t", data=tt.astype(np.float64))
        f.create_dataset("x", data=xx.astype(np.float32))
        f.create_dataset("y", data=yy.astype(np.float32))
        f.create_dataset("p", data=pp.astype(np.int64))
    out_digest = R.per_second_digest(tt, xx, yy, pp)
    try:
        commit = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "?"
    ev = {
        "seq": a.seq, "cond": a.cond, "tag": tag_of(a.cond), "ops_version": R.VERSION, "freqvio": commit,
        "source": {"folder": os.path.relpath(src, ROOT), "events_sha256": sha256(ev_src), "dtypes": dtypes,
                   "polarity_values": sorted(int(v) for v in np.unique(p)[:4]),
                   "time_rounding_max_us": float(frac.max()) if len(frac) else 0.0,
                   "first_t_us": int(ti[0]), "last_t_us": int(ti[-1]),
                   "first_events": [[int(ti[i]), int(xi[i]), int(yi[i]), int(pi[i])] for i in range(n_first)],
                   "first_events_complete_groups": True,
                   "digest": R.digest_summary(src_digest), "per_second": {str(k): v for k, v in src_digest.items()}},
        "output": {"events_sha256": sha256(os.path.join(work, "events_left.h5")),
                   "digest": R.digest_summary(out_digest), "per_second": {str(k): v for k, v in out_digest.items()},
                   "time_ordered": bool(np.all(np.diff(tt) >= 0))},
        "record": rec, "copied_files": copied, "seconds": round(time.time() - t0, 1),
    }
    if not ev["output"]["time_ordered"]:
        shutil.rmtree(work)
        sys.exit("STOPPED: output not time-ordered")
    if a.cond == "clean" and ev["output"]["digest"]["sha256"] != ev["source"]["digest"]["sha256"]:
        shutil.rmtree(work)
        sys.exit("STOPPED: clean output differs from the source")
    off = R.imu_offset_us(a.cond)
    if off:
        imu = np.loadtxt(os.path.join(src, "imu_data.txt"))
        imu[:, 0] += off
        np.savetxt(os.path.join(work, "imu_data.txt"), imu, header="timestamp[us] wx wy wz ax ay az")
        ev["imu"] = {"offset_us": off, "rows": int(len(imu)), "sha256": sha256(os.path.join(work, "imu_data.txt"))}
    with open(os.path.join(work, "r5_evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=1)
    os.replace(work, out)
    print("[r5-inputs] %s %s: %d -> %d events, digest %s, %s" % (
        a.seq, a.cond, rec["n_in"], rec["n_out"], ev["output"]["digest"]["sha256"][:12], os.path.relpath(out, ROOT)))
    print("[r5-inputs] source time rounding <= %.3f us, polarity values %s" % (
        ev["source"]["time_rounding_max_us"], ev["source"]["polarity_values"]))


if __name__ == "__main__":
    main()
