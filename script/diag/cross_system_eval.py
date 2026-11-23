#!/usr/bin/env python3
"""Cross-system IMU-bias check: DEIO and SuperEIO under one evaluation rule. Read-only, numpy.

One ground truth for both systems: /optitrack/davis from the bag (outputs/supereio/gt/<seq>_optitrack_davis_us.txt,
made by script/supereio/eval_traj.py). DEIO's trajectories (runs/<id>/traj_est_tum.txt: t_us x y z qx qy qz qw, in
the time base of data/RPGDAVIS_pp/<seq>/mocap_data.txt) are moved to the bag's absolute time with the constant offset
between the two ground-truth files, after checking that they hold the same poses. SuperEIO's IMU poses are moved to
the camera with the run's extrinsics. Matching: nearest ground truth within 20 ms.
Primary metric: SE(3) without scale, fitted on all matched poses inside the common window W, error inside W,
MPE = mean error / ground-truth path length inside W x 100. W per sequence: from the latest first output to the
earliest last output over all clean runs of both systems. Also: SE(3) fitted on the first 5 s of W.
Outcomes per attempt: completed; failed (DEIO numerical failure; SuperEIO fewer than 10 poses, a process restart or
"system reboot"); undetermined, to be checked (DEIO timeout or other non-determinate outcome; SuperEIO attempt
without summary.txt, or whose record contradicts itself: run_seq.sh exit code not 0, rosbag play exit code not 0 or
missing, or a ROS process that died) - never counted as completed or as an algorithm failure, always shown in the
denominator; excluded (DEIO run whose
controls do not verify, incl. the [imu-stress] check). Errors are summarised over completed runs only.
Reading rule: a system is described as affected on a sequence only if the bias and clean MPE ranges do not overlap.

  python3 script/diag/cross_system_eval.py --deio-ledger logs/ledger/imu_bias_deio_<stamp> \\
      --supereio-list outputs/supereio/imu_bias_runs_<stamp>.txt --out records/imu_bias_cross_system.csv
"""
import argparse
import collections
import csv
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import precision_pilot as P  # noqa: E402
import run_ledger  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_traj", os.path.join(ROOT, "script", "supereio", "eval_traj.py"))
ET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ET)
MAX_DIFF_S = 0.02


def gt_offset_us(bag_gt, deio_gt):
    """Constant time offset deio -> bag, found by the position of DEIO's first ground-truth pose, then checked on
    every DEIO ground-truth pose (median position difference must be below 1 mm)."""
    d = np.linalg.norm(bag_gt[:, 1:4] - deio_gt[0, 1:4], axis=1)
    k = int(np.argmin(d))
    off = bag_gt[k, 0] - deio_gt[0, 0]
    j, ok = ET.match(bag_gt[:, 0], deio_gt[:, 0] + off, 0.002)
    diff = np.linalg.norm(bag_gt[j[ok], 1:4] - deio_gt[ok, 1:4], axis=1)
    med = float(np.median(diff)) if ok.any() else float("inf")
    if med > 1e-3 or ok.mean() < 0.95:
        sys.exit("DEIO and bag ground truth do not line up (median %.4f m, matched %.0f%%)" % (med, 100 * ok.mean()))
    return float(off), med


def camera_positions(est, R_ic, t_ic):
    R_wi = ET.quat_wxyz_to_R(est[:, 4:8])
    return est[:, 0] / 1e3, est[:, 1:4] + np.einsum("nij,j->ni", R_wi, t_ic)


def window_metric(t_us, p, gt, w0, w1, first_s=5.0):
    t_gt, p_gt = gt[:, 0], gt[:, 1:4]
    j, ok = ET.match(t_gt, t_us, MAX_DIFF_S)
    ok &= (t_us >= w0) & (t_us <= w1)
    if ok.sum() < 10:
        return None
    A, B = p[ok], p_gt[j[ok]]
    ing = (t_gt >= w0) & (t_gt <= w1)
    path = float(np.linalg.norm(np.diff(p_gt[ing], axis=0), axis=1).sum())
    out = {"matched": int(ok.sum()), "path_m": path}
    tA = t_us[ok]
    kk = np.searchsorted(tA, tA + 1e6)                 # local 1 s distance ratio, no alignment (1 = metric scale)
    ii = np.nonzero(kk < len(tA))[0]
    kk = kk[kk < len(tA)]
    dg = np.linalg.norm(B[kk] - B[ii], axis=1)
    de = np.linalg.norm(A[kk] - A[ii], axis=1)
    out["dist_ratio_1s"] = float(np.median(de[dg > 0.05] / dg[dg > 0.05])) if (dg > 0.05).any() else None
    s3, R3, t3 = ET.umeyama(A, B, True)   # secondary: scale the Sim(3) fit would need (1 = metric)
    out["sim3_W_scale"] = float(s3)
    out["sim3_W"] = float(np.linalg.norm(B - s3 * (R3 @ A.T).T - t3, axis=1).mean() / path * 100)
    for name, fit in (("se3_W", None), ("se3_W_first5s", (t_us[ok] - w0) / 1e6 <= first_s)):
        if fit is not None and fit.sum() < 3:
            out[name] = None
            continue
        _, R, t = ET.umeyama(A, B, False) if fit is None else ET.umeyama(A[fit], B[fit], False)
        e = np.linalg.norm(B - (R @ A.T).T - t, axis=1)
        out[name] = float(e.mean() / path * 100)
    return out


def load_deio(ledger, gts, deio_gts):
    runs = []
    for r in run_ledger.load_ledger(ledger):
        lab = r["labels"]
        seq = str(lab.get("scene")).split("/")[-1]
        cat = run_ledger.outcome_category(r["outcome_hint"])
        x = {"system": "DEIO", "seq": seq, "cond": "clean" if lab.get("cell") == "clean" else "gbias",
             "rep": lab.get("seed"), "run": r["run_id"], "status": cat, "t": None, "p": None}
        f = P._load_facts(ledger, r["run_id"])
        probs = ["facts.json missing"] if f is None else P.verify_controls(
            f, lab.get("arm"), bool(lab.get("cholinfo")), cell=lab.get("cell"))
        if cat == "not_run":
            continue
        if probs:
            x["status"] = "excluded: " + probs[0][:70]
        elif cat == "numerical_failure":
            x["status"] = "failed: numerical failure"
        elif cat != "completed":
            x["status"] = "undetermined: " + cat
        if x["status"] == "completed":
            f = os.path.join(ledger, "runs", r["run_id"], "traj_est_tum.txt")
            if os.path.exists(f):
                tum = np.loadtxt(f)
                off = deio_gts[seq][0]
                x["t"], x["p"] = tum[:, 0] + off, tum[:, 1:4]
            else:
                x["status"] = "no trajectory file"
        runs.append(x)
    return runs


def load_supereio(listfile):
    runs = []
    with open(listfile) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        seq, cond, rep, rdir = parts[:4]
        rc = (parts[4].split("=", 1)[1] if len(parts) > 4 and parts[4].startswith("rc=") else "?")
        if not os.path.isabs(rdir):
            rdir = os.path.join(ROOT, rdir)
        x = {"system": "SuperEIO", "seq": seq, "cond": cond, "rep": int(rep), "run": os.path.basename(rdir),
             "status": "completed", "t": None, "p": None}
        sp = os.path.join(rdir, "summary.txt")
        if not os.path.exists(sp):
            x["status"] = "undetermined: no summary.txt (exit %s)" % rc
            runs.append(x)
            continue
        with open(sp) as fh:
            summ = fh.read()
        n = lambda key: int((re.search(re.escape(key) + r"\s*(\d+)", summ) or [0, 0])[1])  # noqa: E731
        est = ET.load_estimate(os.path.join(rdir, "evio_result_no_loop.csv")) \
            if os.path.exists(os.path.join(rdir, "evio_result_no_loop.csv")) else np.zeros((0, 8))
        bag_rc = re.search(r"rosbag play exit code:\s*(\d+)", summ)
        if rc != "0":
            x["status"] = "undetermined: run_seq exit %s (check)" % rc
        elif not bag_rc or bag_rc.group(1) != "0":
            x["status"] = "undetermined: rosbag play exit %s (check)" % (bag_rc.group(1) if bag_rc else "not recorded")
        elif n("process deaths:"):
            x["status"] = "undetermined: a ROS process died (check)"
        elif len(est) < 10:
            x["status"] = "failed: no trajectory"
        elif n("event_detector restarts:") or n("failure detections (system reboot!):"):
            x["status"] = "failed: restart or reboot"
        if x["status"] == "completed":
            with open(os.path.join(rdir, "config.yaml")) as fh:
                cfg = fh.read()
            R_ic = ET.opencv_matrix(cfg, "extrinsicRotation").reshape(3, 3)
            t_ic = ET.opencv_matrix(cfg, "extrinsicTranslation")
            x["t"], x["p"] = camera_positions(est, R_ic, t_ic)
        runs.append(x)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deio-ledger", required=True)
    ap.add_argument("--supereio-list", required=True, nargs="+", help="one or more run lists (split instances)")
    ap.add_argument("--out")
    a = ap.parse_args()
    seqs = ("boxes_6dof", "poster_6dof")
    gts, deio_gts = {}, {}
    for s in seqs:
        p = os.path.join(ROOT, "outputs", "supereio", "gt", "%s_optitrack_davis_us.txt" % s)
        if not os.path.exists(p):
            sys.exit("missing %s: run script/supereio/eval_traj.py on one %s run first" % (p, s))
        gts[s] = np.loadtxt(p)
        dg = np.loadtxt(os.path.join(ROOT, "data", "RPGDAVIS_pp", s, "mocap_data.txt"))
        dg = dg[dg[:, 0] > 0]
        deio_gts[s] = gt_offset_us(gts[s], dg)
        print("ground truth %s: DEIO time + %.0f us = bag time (median pose difference %.2e m)" % (s, *deio_gts[s]))
    runs = load_deio(a.deio_ledger, gts, deio_gts) + [x for lf in a.supereio_list for x in load_supereio(lf)]
    runs = [x for x in runs if x["seq"] in seqs]
    win = {}
    for s in seqs:
        clean = [x for x in runs if x["seq"] == s and x["cond"] == "clean" and x["status"] == "completed"]
        if not clean:
            sys.exit("no completed clean run for %s: no window" % s)
        win[s] = (max(x["t"][0] for x in clean), min(x["t"][-1] for x in clean))
        print("window W %s: %.1f s (from %.2f s after the ground-truth start)"
              % (s, (win[s][1] - win[s][0]) / 1e6, (win[s][0] - gts[s][0, 0]) / 1e6))
    rows = []
    for x in runs:
        m = window_metric(x["t"], x["p"], gts[x["seq"]], *win[x["seq"]]) if x["t"] is not None else None
        rows.append({k: x[k] for k in ("system", "seq", "cond", "rep", "run", "status")} |
                    {"se3_W": m and m["se3_W"], "se3_W_first5s": m and m["se3_W_first5s"],
                     "sim3_W": m and m["sim3_W"], "sim3_W_scale": m and m["sim3_W_scale"],
                     "dist_ratio_1s": m and m["dist_ratio_1s"],
                     "matched": m and m["matched"]})
    print("\n%-9s %-12s %-6s %-5s %-10s %-7s %-7s %-8s %-34s %-12s %-30s %-20s" % (
        "system", "sequence", "cond", "runs", "completed", "failed", "undet.", "excluded",
        "SE(3) MPE in W: median [min, max] %", "first-5 s", "Sim(3) in W: MPE / scale", "1 s dist est/gt"))
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["system"], r["seq"], r["cond"])].append(r)
    ranges = {}
    for k in sorted(groups):
        g = groups[k]
        v = sorted(r["se3_W"] for r in g if r["status"] == "completed" and r["se3_W"] is not None)
        f5 = sorted(r["se3_W_first5s"] for r in g if r["status"] == "completed" and r["se3_W_first5s"] is not None)
        ranges[k] = (v[0], v[-1]) if v else None
        cnt = lambda p: sum(r["status"].startswith(p) for r in g)  # noqa: E731
        sm = [r["sim3_W"] for r in g if r["status"] == "completed" and r["sim3_W"] is not None]
        sc = sorted(r["sim3_W_scale"] for r in g if r["status"] == "completed" and r["sim3_W_scale"] is not None)
        dr = sorted(r["dist_ratio_1s"] for r in g if r["status"] == "completed" and r["dist_ratio_1s"] is not None)
        print("%-9s %-12s %-6s %-5d %-10s %-7d %-7d %-8d %-34s %-12s %-30s %-20s" % (
            k[0], k[1], k[2], len(g), "%d/%d" % (cnt("completed"), len(g)), cnt("failed"), cnt("undetermined"),
            cnt("excluded"), "%.2f [%.2f, %.2f]" % (float(np.median(v)), v[0], v[-1]) if v else "-",
            "%.2f" % float(np.median(f5)) if f5 else "-",
            "%.2f / %.3f [%.3f, %.3f]" % (float(np.median(sm)), float(np.median(sc)), sc[0], sc[-1]) if sc else "-",
            "%.2f [%.2f, %.2f]" % (float(np.median(dr)), dr[0], dr[-1]) if dr else "-"))
        other = collections.Counter(r["status"] for r in g if r["status"] != "completed")
        if other:
            print("          not counted: %s" % dict(other))
    print("\nREADING (bias vs clean, per system and sequence; ranges that do not overlap = affected)")
    for sysname in ("DEIO", "SuperEIO"):
        for s in seqs:
            c, b = ranges.get((sysname, s, "clean")), ranges.get((sysname, s, "gbias"))
            if not c or not b:
                print("  %-9s %-12s cannot compare (missing completed runs)" % (sysname, s))
                continue
            sep = b[0] > c[1] or b[1] < c[0]
            print("  %-9s %-12s %s" % (sysname, s, "affected: ranges do not overlap" if sep
                                        else "no difference seen in 10 runs (ranges overlap)"))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.splitext(a.out)[0] + "_window.json", "w") as fh:
            json.dump({s: {"w0_us": win[s][0], "w1_us": win[s][1]} for s in seqs}, fh, indent=2)
        print("\nper-run table:", a.out)


if __name__ == "__main__":
    main()
