#!/usr/bin/env python3
"""MPE of a SuperEIO run against the DEIO preprocessing's ground truth, under several alignment rules.

DEIO (utils/eval_utils.py log_results, called with _n_to_align=1000 for DAVIS240C via script/eval_deio/mvsec.py, and
for VECtor, HKU, TUM-VIE; -1 = all poses only for UZH-FPV) reports MPE = mean translation APE / ground-truth path
length x 100, after evo associates the trajectories within 1 s and fits Umeyama WITH scale on the first 1000
associated poses; the error is taken over all associated poses, the path length over the whole ground-truth file.
This script repeats that with numpy only (no evo on the SuperEIO instance). Because the fit uses the first 1000
poses, the rule depends on the output rate of the system (DEIO and SuperEIO output at different rates).

SuperEIO writes the IMU (body) pose; the ground truth (/optitrack/davis, T_marker_cam0 = I in pp_davis240c.py) is the
camera pose. Each estimate is moved to the camera with the run's extrinsics (imu^R_cam, imu^T_cam):
p_wc = p_wi + R_wi t_ic. Estimates are matched to the nearest ground-truth time within --max-diff (DEIO's evo call
uses max_diff = 1 s; the number matched at 1 s is reported too).

Alignments, errors always over all matched poses, MPE = mean error / ground-truth path length x 100:
  deio         with scale, fitted on the first 1000 matched poses (the rule of DEIO's main table; for a run with
               fewer than 1000 poses it is the same as sim3_all)
  sim3_all     with scale, all matched poses
  se3_all      without scale, all matched poses
  se3_first5s  without scale, fitted only on the poses in the first --first-s seconds (5) of the estimate: the
               protocol SuperEIO's paper states for DAVIS240C and Mono HKU (arXiv 2503.22963v1, Sec. IV-B: "alignment
               based on poses from the initial 5-second interval"). The paper gives no MPE formula, so its
               denominator here is DEIO's (path length); it does not say whether Table II had loop closure on.
With scale, a poor estimate can be shrunk towards one point (scale near 0): the error then approaches the spread of
the ground truth around its mean, whatever the estimate. Scales outside 0.5-2 are flagged "degenerate".
Diagnostic: the time shift of the estimate (-0.5 to 0.5 s, 10 ms steps) that minimises the se3_all error; a best
shift far from 0 would point to a time-base problem rather than to the estimator.
Local tracking check, free of any alignment and of the unknown marker-camera rotation: for pose pairs 1 s apart,
the rotation angle and the distance travelled, estimate vs ground truth (the angle of a relative rotation does not
change under a constant hand-eye rotation). A working VIO gives angle differences of about a degree and distance
ratios near 1; a ratio far from 1 means the metric scale is wrong.

  python3 script/supereio/eval_traj.py outputs/supereio/runs/boxes_6dof_pilot_<stamp> [--gt <file>] [--loop]
Default ground truth: data/RPGDAVIS_pp/<seq>/raw_gt_stamped_ns_left.txt (absolute time in microseconds despite the
name; columns t x y z qx qy qz qw). If it is not there, the same file is made from the bag the run played
(/optitrack/davis, header stamps, pose unchanged: what pp_davis240c.py does with T_marker_cam0 = I) and kept as
outputs/supereio/gt/<seq>_optitrack_davis_us.txt. Writes eval.json into the run folder.
"""
import argparse
import json
import os
import re
import sys

import numpy as np


def quat_wxyz_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def opencv_matrix(text, name):
    m = re.search(name + r":\s*!!opencv-matrix.*?data:\s*\[([^\]]*)\]", text, re.S)
    if not m:
        raise ValueError("%s not found in the configuration" % name)
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()])


def umeyama(src, dst, with_scale):
    """s, R, t minimising |dst - (s R src + t)|^2 (Umeyama 1991), src/dst N x 3."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def load_estimate(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            vals = [v for v in line.strip().split(",") if v.strip()]
            if len(vals) >= 8:
                try:
                    rows.append([float(v) for v in vals[:8]])
                except ValueError:
                    continue   # header line of the loop file
    return np.array(rows)


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def local_tracking(t_us, p, R, t_gt_us, p_gt, R_gt, dt_s=1.0):
    """Rotation angle and distance between poses dt_s apart, estimate vs ground truth (no alignment needed)."""
    k = np.searchsorted(t_us, t_us + dt_s * 1e6)
    ok = k < len(t_us)
    i, k = np.nonzero(ok)[0], k[ok]
    if len(i) < 10:
        return None
    a_est = rot_angle_deg(np.einsum("nji,njk->nik", R[i], R[k]))
    a_gt = rot_angle_deg(np.einsum("nji,njk->nik", R_gt[i], R_gt[k]))
    d_est = np.linalg.norm(p[k] - p[i], axis=1)
    d_gt = np.linalg.norm(p_gt[k] - p_gt[i], axis=1)
    sel = d_gt > 0.05
    return {"pairs": int(len(i)), "angle_diff_deg_median": float(np.median(np.abs(a_est - a_gt))),
            "angle_gt_deg_median": float(np.median(a_gt)),
            "distance_ratio_median": float(np.median(d_est[sel] / d_gt[sel])) if sel.any() else None,
            "distance_gt_m_median": float(np.median(d_gt))}


def match(t_gt_us, t_est_us, max_diff_s):
    j = np.clip(np.searchsorted(t_gt_us, t_est_us), 1, len(t_gt_us) - 1)
    j = np.where(np.abs(t_gt_us[j - 1] - t_est_us) < np.abs(t_gt_us[j] - t_est_us), j - 1, j)
    return j, np.abs(t_gt_us[j] - t_est_us) <= max_diff_s * 1e6


def evaluate(est, gt, R_ic, t_ic, max_diff_s, first_s=5.0, n_deio=1000):
    t_est_us = est[:, 0] / 1e3                     # ns -> us
    p_wi, R_wi = est[:, 1:4], quat_wxyz_to_R(est[:, 4:8])
    p_wc = p_wi + np.einsum("nij,j->ni", R_wi, t_ic)
    t_gt_us, p_gt = gt[:, 0], gt[:, 1:4]
    order = np.argsort(t_gt_us)
    t_gt_us, p_gt = t_gt_us[order], p_gt[order]
    path_len = float(np.linalg.norm(np.diff(p_gt, axis=0), axis=1).sum())
    j, ok = match(t_gt_us, t_est_us, max_diff_s)
    A, B = p_wc[ok], p_gt[j[ok]]
    tA = (t_est_us[ok] - t_est_us[0]) / 1e6
    out = {"estimates": int(len(est)), "matched": int(ok.sum()), "gt_poses": int(len(gt)),
           "matched_at_1s": int((np.abs(t_gt_us[j] - t_est_us) <= 1e6).sum()),
           "gt_path_length_m": path_len,
           "est_time_span_s": float((t_est_us[-1] - t_est_us[0]) / 1e6) if len(est) else 0.0,
           "gt_time_span_s": float((t_gt_us[-1] - t_gt_us[0]) / 1e6),
           "first_estimate_after_gt_start_s": float((t_est_us[0] - t_gt_us[0]) / 1e6) if len(est) else None}
    if ok.sum() < 10:
        out["error"] = "fewer than 10 matched poses"
        return out
    first_n = np.arange(len(A)) < n_deio
    for name, scale, fit in (("deio", True, first_n), ("sim3_all", True, None), ("se3_all", False, None),
                             ("se3_first5s", False, tA <= first_s)):
        if fit is not None and fit.sum() < 3:
            out[name] = {"error": "fewer than 3 poses in the first %.1f s" % first_s}
            continue
        s, R, t = umeyama(A, B, scale) if fit is None else umeyama(A[fit], B[fit], scale)
        e = np.linalg.norm(B - (s * (R @ A.T).T + t), axis=1)
        out[name] = {"scale": float(s), "ape_mean_m": float(e.mean()), "ate_rmse_m": float(np.sqrt((e ** 2).mean())),
                     "mpe_percent": float(e.mean() / path_len * 100),
                     "fitted_on": int(len(A) if fit is None else fit.sum()),
                     "degenerate_scale": bool(scale and not 0.5 <= s <= 2.0)}
    out["gt_spread_mpe_percent"] = float(np.linalg.norm(B - B.mean(0), axis=1).mean() / path_len * 100)
    out["local_1s"] = local_tracking(t_est_us[ok], p_wc[ok], R_wi[ok] @ R_ic, t_gt_us[j[ok]], p_gt[j[ok]],
                                     quat_wxyz_to_R(gt[order][j[ok]][:, [7, 4, 5, 6]]))
    best = None
    for dt in np.arange(-0.5, 0.5001, 0.01):
        jj, kk = match(t_gt_us, t_est_us + dt * 1e6, max_diff_s)
        if kk.sum() < 10:
            continue
        s, R, t = umeyama(p_wc[kk], p_gt[jj[kk]], False)
        e = np.linalg.norm(p_gt[jj[kk]] - (R @ p_wc[kk].T).T - t, axis=1).mean() / path_len * 100
        if best is None or e < best[1]:
            best = (float(round(dt, 3)), float(e))
    out["time_shift_scan_se3_all"] = {"best_shift_s": best[0], "mpe_percent_at_best": best[1]} if best else None
    return out


def gt_from_bag(bag_path, out_path, topic="/optitrack/davis"):
    import rosbag   # python3-rosbag, installed by setup_env.sh
    rows = []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            p = msg.pose.pose if msg._type == "nav_msgs/Odometry" else msg.pose
            rows.append([msg.header.stamp.to_nsec() / 1e3, p.position.x, p.position.y, p.position.z,
                         p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    if not rows:
        sys.exit("no %s messages in %s" % (topic, bag_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, np.array(rows), fmt="%.3f %.9f %.9f %.9f %.9f %.9f %.9f %.9f")
    print("ground truth: %d poses from %s %s -> %s" % (len(rows), bag_path, topic, out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--gt")
    ap.add_argument("--loop", action="store_true", help="evaluate esvio_result_loop.txt instead of the odometry")
    ap.add_argument("--max-diff", type=float, default=0.02, help="seconds (default 0.02)")
    ap.add_argument("--first-s", type=float, default=5.0, help="window for se3_first5s, seconds (default 5)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    summ = os.path.join(a.run, "summary.txt")
    m = re.match(r"sequence (\S+)", open(summ).readline()) if os.path.exists(summ) else None
    seq = m.group(1) if m else re.sub(r"_[^_]+_\d{8}_\d{6}$", "", os.path.basename(a.run.rstrip("/")))
    gt_path = a.gt or os.path.join(root, "data", "RPGDAVIS_pp", seq, "raw_gt_stamped_ns_left.txt")
    if not a.gt and not os.path.exists(gt_path):
        gt_path = os.path.join(root, "outputs", "supereio", "gt", "%s_optitrack_davis_us.txt" % seq)
        if not os.path.exists(gt_path):
            gt_from_bag(os.path.join(root, "data", "rosbags", "rpgdavis", seq + ".bag"), gt_path)
    est_path = os.path.join(a.run, "esvio_result_loop.txt" if a.loop else "evio_result_no_loop.csv")
    for p in (gt_path, est_path):
        if not os.path.exists(p):
            sys.exit("missing %s" % p)
    cfg = open(os.path.join(a.run, "config.yaml")).read()
    if re.search(r"^T_camera_imu:\s*1", cfg, re.M):
        sys.exit("T_camera_imu: 1 (extrinsics given as IMU->camera) is not handled")
    R_ic = opencv_matrix(cfg, "extrinsicRotation").reshape(3, 3)
    t_ic = opencv_matrix(cfg, "extrinsicTranslation")
    est = load_estimate(est_path)
    if len(est) == 0:
        sys.exit("no poses in %s" % est_path)
    gt = np.loadtxt(gt_path)
    res = dict(sequence=seq, estimate_file=os.path.basename(est_path), gt_file=gt_path, max_diff_s=a.max_diff,
               **evaluate(est, gt, R_ic, t_ic, a.max_diff, a.first_s))
    print(json.dumps(res, indent=2))
    if "deio" in res:
        f5 = res["se3_first5s"].get("mpe_percent", float("nan"))
        d, ts = res["deio"], res["time_shift_scan_se3_all"] or {}
        print("MPE deio-rule %.3f %% (scale %.3f%s) | se3 all %.3f %% | se3 first %gs %.3f %% | gt spread %.3f %% | "
              "best shift %+.2f s (%.3f %%) | matched %d/%d | est %.1f of %.1f s"
              % (d["mpe_percent"], d["scale"], ", DEGENERATE" if d["degenerate_scale"] else "",
                 res["se3_all"]["mpe_percent"], a.first_s, f5, res["gt_spread_mpe_percent"],
                 ts.get("best_shift_s", float("nan")), ts.get("mpe_percent_at_best", float("nan")),
                 res["matched"], res["estimates"], res["est_time_span_s"], res["gt_time_span_s"]))
        lt = res.get("local_1s") or {}
        if lt:
            print("local 1 s: rotation angle |est-gt| median %.2f deg (gt median %.1f deg) | distance ratio est/gt "
                  "median %.3f (gt median %.3f m)" % (lt["angle_diff_deg_median"], lt["angle_gt_deg_median"],
                                                     lt["distance_ratio_median"] or float("nan"),
                                                     lt["distance_gt_m_median"]))
    with open(os.path.join(a.run, "eval_loop.json" if a.loop else "eval.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
