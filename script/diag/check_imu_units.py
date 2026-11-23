#!/usr/bin/env python3
"""Are the gyroscope units of the preprocessed VECtor / HKU IMU files what DEIO expects (deg/s)? CPU only.

Two independent checks per sequence:
1. Conversion (VECtor, where the raw imu.txt is kept next to the output): row by row,
   processed gyro x pi/180 must equal the raw gyro (rad/s according to the dataset files), and the
   timestamps must be the raw seconds x 1e6. HKU's raw IMU is inside the rosbag: not re-read here.
2. Physics (all): the angular speed |w| from the gyroscope, converted from deg/s to rad/s, against the
   angular speed of the ground-truth orientation (rotation angle between poses 50 ms apart / time).
   |w| does not depend on the camera-IMU extrinsics (a fixed rotation keeps the norm). A unit error
   shows as a ratio near 57 or 1/57; with correct units it is near 1 (gyro noise and bias raise it
   somewhat on slow motion). Compared on time windows when IMU and ground truth share a time base,
   otherwise as whole-sequence means.
Accelerometer: mean |a| should be about 9.8 m/s^2.

  python script/diag/check_imu_units.py            # data/VECtor/{desk_normal1,corner_slow1}, data/HKU/vicon_dark1
Exit code 0 only if every check passes.
"""
import argparse
import os
import sys

import numpy as np

SEQS = (("vector", "data/VECtor/desk_normal1", "poses_evs_left.txt"),
        ("vector", "data/VECtor/corner_slow1", "poses_evs_left.txt"),
        ("hku", "data/HKU/vicon_dark1", "gt_stamped_davis346.txt"))
WIN_US = 50_000


def gt_angular_speed(t_us, q_xyzw, win_us=WIN_US):
    """(t_start, t_end, rad/s) for windows of about win_us between ground-truth orientations."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    j = np.searchsorted(t_us, t_us + win_us)
    ok = j < len(t_us)
    i, j = np.nonzero(ok)[0], j[ok]
    dot = np.clip(np.abs(np.sum(q[i] * q[j], axis=1)), 0.0, 1.0)
    dt = (t_us[j] - t_us[i]) / 1e6
    keep = dt > 0
    return t_us[i][keep], t_us[j][keep], (2.0 * np.arccos(dot[keep]) / dt[keep])


def physics(imu_t_us, gyro_deg, acc, gt_t_us, gt_q, min_speed=0.3):
    w = np.linalg.norm(gyro_deg, axis=1) * np.pi / 180.0
    t0, t1, s = gt_angular_speed(gt_t_us, gt_q)
    out = {"imu_mean_rad_s": float(w.mean()), "gt_mean_rad_s": float(s.mean()),
           "mean_ratio": float(w.mean() / s.mean()) if s.mean() > 0 else float("nan"),
           "acc_mean": float(np.linalg.norm(acc, axis=1).mean())}
    overlap = max(0.0, min(imu_t_us[-1], gt_t_us[-1]) - max(imu_t_us[0], gt_t_us[0]))
    out["overlap_s"] = overlap / 1e6
    sel = (s > min_speed) & (t0 >= imu_t_us[0]) & (t1 <= imu_t_us[-1])
    if overlap > 5e6 and sel.sum() >= 20:
        cs = np.concatenate([[0.0], np.cumsum(w)])
        a, b = np.searchsorted(imu_t_us, t0[sel]), np.searchsorted(imu_t_us, t1[sel])
        good = b > a
        mean_w = (cs[b[good]] - cs[a[good]]) / (b[good] - a[good])
        r = mean_w / s[sel][good]
        out.update(windows=int(good.sum()), window_ratio_median=float(np.median(r)),
                   window_ratio_iqr=(float(np.percentile(r, 25)), float(np.percentile(r, 75))))
    return out


def conversion(proc, raw):
    if len(proc) != len(raw):
        return {"rows": (len(proc), len(raw)), "ok": False, "why": "row counts differ"}
    dt = np.abs(proc[:, 0] - raw[:, 0] * 1e6).max()
    dg = np.abs(proc[:, 1:4] * np.pi / 180.0 - raw[:, 1:4]).max()
    da = np.abs(proc[:, 4:7] - raw[:, 4:7]).max()
    moving = np.linalg.norm(raw[:, 1:4], axis=1) > 0.1
    return {"rows": len(proc), "max_dt_us": float(dt), "max_gyro_diff_rad_s": float(dg), "max_acc_diff": float(da),
            "moving_rows": int(moving.sum()), "ok": bool(dt < 2.0 and dg < 1e-6 and da < 1e-6)}


def verdict(ph):
    r = ph.get("window_ratio_median", ph["mean_ratio"])
    lo, hi = (0.7, 1.4) if "window_ratio_median" in ph else (0.5, 2.5)
    return lo < r < hi and 8.5 < ph["acc_mean"] < 11.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    bad = 0
    for kind, d, gtname in SEQS:
        d = os.path.join(a.root, d)
        print("\n%s" % d)
        p_imu, p_gt = os.path.join(d, "imu_data.txt"), os.path.join(d, gtname)
        if not (os.path.exists(p_imu) and os.path.exists(p_gt)):
            print("  MISSING %s or %s" % (p_imu, p_gt))
            bad += 1
            continue
        proc = np.loadtxt(p_imu)
        gt = np.loadtxt(p_gt)
        if kind == "vector" and os.path.exists(os.path.join(d, "imu.txt")):
            raw = np.loadtxt(os.path.join(d, "imu.txt"))
            c = conversion(proc, raw)
            print("  1. conversion vs raw imu.txt: %s  %s" % ("OK" if c["ok"] else "FAILED", c))
            bad += not c["ok"]
        else:
            print("  1. conversion: raw IMU not available here (%s)" % ("rosbag" if kind == "hku" else "no imu.txt"))
        order = np.argsort(proc[:, 0])
        ph = physics(proc[order, 0], proc[order, 1:4], proc[order, 4:7], gt[:, 0], gt[:, 4:8])
        ok = verdict(ph)
        print("  2. gyro |w| (deg/s -> rad/s) vs ground-truth angular speed: %s" % ("OK" if ok else "FAILED"))
        for k, v in ph.items():
            print("       %-20s %s" % (k, v))
        bad += not ok
    print("\nRESULT: %s" % ("IMU units OK (gyro deg/s, acc m/s^2)" if not bad else "%d problem(s)" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
