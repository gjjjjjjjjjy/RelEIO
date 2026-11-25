#!/usr/bin/env python3
"""Checks of the scoring rules: known faults and the correct rate of OpenLORIS. CPU only, numpy only.

faults  known faults injected into the ground truth of every sequence of the main matrix, scored by every rule.
        The "estimate" is the ground truth sampled at 20 Hz, then: rigid (yaw 30 deg, shift 1 m; not a failure),
        scale 0.5 and 2 (about the first pose), drift (from mid-time, growing linearly to 2% or 20% of the path length
        at the end), diverge (from mid-time, growing exponentially to 200% of the path length), stationary (the first
        pose throughout), freeze (the pose at mid-time from then on). Rules: A (similarity on the first 1000 matched
        poses, 1 s matching; all poses on UZH-FPV), C (similarity, all), D (rigid, all), first5s (rigid on the first
        5 s), and the correct rate CR of OpenLORIS at epsilon = 1% and 10% of the path length. Each reading of A, C, D
        and first5s is classified at 1%, 10% and 100%: flagged (above the tolerance), missed (within it), or not
        computable (non-finite, e.g. a stationary estimate under a fit with scale).
crt     the correct rate (CR, Eq. 2 of OpenLORIS, arXiv 1911.05603) and CR-T (Eq. 3) of every completed run of the
        given ledgers, next to rule D. A pose is correct if, after one rigid alignment of the whole trajectory, its
        position error is at most epsilon (1% or 10% of the ground-truth path length); the orientation criterion of
        OpenLORIS (30 deg) is not applied (no calibration between the estimated body frame and
        the marker frame of the ground truth); each correct pose counts for at most delta = 1 s; CR divides by the ground-truth time span, CR-T
        by the span from the first estimate. Estimates without a ground-truth pose within 20 ms count as incorrect.
        Crashed runs have no trajectory; they are listed with CR "not computable", never dropped or set to zero.
        Orientation, observation only: the median orientation error of every run is reported with the estimate
        quaternion read as xyzw (the file format) and as wxyz. The orientation conventions were not verified
        independently, so no orientation criterion is applied and only position-based correct rates are reported.

  python3 script/diag/scoring_checks.py faults --out records/scoring_faults_<date>.csv
  python3 script/diag/scoring_checks.py crt --ledgers logs/ledger/round4_core_main_all_20261121_023641 \\
      --out records/crt_round4_<date>.csv
"""
import argparse
import collections
import csv
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deio_se3_eval as D   # noqa: E402
import run_ledger            # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEQUENCES = [("rpgdavis", "boxes_6dof"), ("rpgdavis", "poster_6dof"), ("rpgdavis", "shapes_6dof"),
             ("vector", "desk_normal1"), ("vector", "corner_slow1"), ("hku", "vicon_dark1"),
             ("tumvie", "mocap-3d-trans"), ("fpv", "indoor_forward_3")]
TOLS = (1.0, 10.0, 100.0)
CR_EPS = (1.0, 10.0)          # % of the path length
PHI_DEG, DELTA_S, MATCH_S = 30.0, 1.0, 0.02


# ------------------------------------------------------------------ rotations (quaternions x y z w, as in the files)
def quat_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def R_to_quat(R):
    w = np.sqrt(np.clip(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 1e-12, None)) / 2
    return np.stack([(R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
                     (R[:, 1, 0] - R[:, 0, 1]) / (4 * w), w], -1)


def angle_deg(Ra, Rb):
    c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2          # trace(Ra^T Rb) = sum of elementwise products
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ------------------------------------------------------------------ rules
def readings(est, gt, n_align):
    """Rules A, C, D, first5s (% of the path length) of deio_se3_eval.evaluate, non-finite -> None."""
    m = D.evaluate(est, gt, n_align)
    out = {}
    for k, name in (("deio_rule", "A"), ("sim3_all", "C"), ("se3_all", "D"), ("se3_first5s", "first5s")):
        v = m.get(k)
        out[name] = v if v is not None and np.isfinite(v) else None
    return out


def correct_rate(est, gt, eps_pct, quat_order="xyzw", use_orientation=False):
    """CR (Eq. 2) and CR-T (Eq. 3) of OpenLORIS; the median orientation error (deg) and how far the per-pose rotation
    offset between estimate and ground truth spreads around its mean (deg), reported as observations only: the
    orientation conventions of estimate and ground truth were not verified independently. The orientation criterion is
    applied only with use_orientation (off for every reported correct rate, which uses positions only)."""
    o = np.argsort(gt[:, 0])
    g = gt[o]
    path = float(np.linalg.norm(np.diff(g[:, 1:4], axis=0), axis=1).sum())
    t_e = est[:, 0]
    j, ok = D.ET.match(g[:, 0], t_e, MATCH_S)
    if ok.sum() < 3:
        return None
    s, R, t = D.ET.umeyama(est[ok, 1:4], g[j[ok], 1:4], False)
    q = est[:, 4:8] if quat_order == "xyzw" else est[:, [5, 6, 7, 4]]
    Re = quat_to_R(q)
    ate = np.full(len(est), np.inf)
    aoe = np.full(len(est), np.inf)
    ate[ok] = np.linalg.norm(g[j[ok], 1:4] - (est[ok, 1:4] @ R.T + t), axis=1)
    aoe[ok] = angle_deg(quat_to_R(g[j[ok], 4:8]), R @ Re[ok])
    t_next = np.r_[t_e[1:], g[-1, 0]]
    dt = np.minimum(np.clip(t_next - t_e, 0, None), DELTA_S * 1e6)
    off = quat_to_R(g[j[ok], 4:8]).transpose(0, 2, 1) @ (R @ Re[ok])          # per-pose rotation offset
    U, _, Vt = np.linalg.svd(off.sum(0))
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt                # chordal mean of the offsets
    out = {"aoe_median_deg": float(np.median(aoe[ok])),
           "offset_spread_deg": float(np.median(angle_deg(off, np.broadcast_to(Rm, off.shape))))}
    for e in eps_pct:
        good = (ate <= e / 100 * path) & ((aoe <= PHI_DEG) if use_orientation else True)
        num = float((dt * good).sum())
        out["CR_%g" % e] = num / (g[-1, 0] - g[0, 0])
        out["CRT_%g" % e] = num / (g[-1, 0] - t_e[0]) if g[-1, 0] > t_e[0] else None
    out["unmatched"] = int((~ok).sum())
    return out


# ------------------------------------------------------------------ known faults
def inject(gt, fault):
    """Estimate (t x y z qx qy qz qw) from the ground truth sampled at 20 Hz, with one known fault."""
    g = gt[np.argsort(gt[:, 0])]
    grid = np.arange(g[0, 0], g[-1, 0], 5e4)
    g = g[np.unique(np.searchsorted(g[:, 0], grid).clip(0, len(g) - 1))]
    t, p, q = g[:, 0], g[:, 1:4].copy(), g[:, 4:8].copy()
    L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    u = np.clip((t - (t[0] + t[-1]) / 2) / ((t[-1] - t[0]) / 2), 0, None)   # 0 until mid-time, 1 at the end
    mid = int(np.searchsorted(t, (t[0] + t[-1]) / 2))
    ex = np.array([1.0, 0, 0])
    if fault == "rigid":
        Ry = yaw(30)
        p = p @ Ry.T + np.array([1.0, 0, 0])
        q = R_to_quat(Ry[None] @ quat_to_R(q))
    elif fault.startswith("scale"):
        s = float(fault[5:])
        p = p[0] + s * (p - p[0])
    elif fault.startswith("drift"):
        p = p + (float(fault[5:]) / 100 * L * u)[:, None] * ex
    elif fault == "diverge200":
        p = p + (2.0 * L * (np.exp(5 * u) - 1) / (math.e ** 5 - 1))[:, None] * ex
    elif fault == "stationary":
        p, q = np.tile(p[0], (len(p), 1)), np.tile(q[0], (len(q), 1))
    elif fault == "freeze":
        p[mid:], q[mid:] = p[mid], q[mid]
    elif fault != "none":
        raise ValueError(fault)
    return np.c_[t, p, q]


FAULTS = ["none", "rigid", "scale0.5", "scale2", "drift2", "drift20", "diverge200", "stationary", "freeze"]


def run_faults(out_path):
    rows = []
    for ds, sc in SEQUENCES:
        gt = D.load_gt(ds, sc)
        for f in FAULTS:
            est = inject(gt, f)
            r = readings(est, gt, D.N_ALIGN.get(ds, 1000))
            cr = correct_rate(est, gt, CR_EPS) or {}
            rows.append(dict(sequence="%s/%s" % (ds, sc), fault=f, **r,
                             **{k: v for k, v in cr.items() if k.startswith("CR_")}))
    keys = list(rows[0].keys())
    if out_path:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("known faults, %d sequences. Per fault and rule: median reading (%% of the path length) and, at 1/10/100%%,"
          " flagged/missed/not computable over the sequences; CR: median correct rate." % len(SEQUENCES))
    print("%-11s %-8s %9s %-26s" % ("fault", "rule", "median", "1% | 10% | 100% (f/m/n)"))
    for f in FAULTS:
        rs = [r for r in rows if r["fault"] == f]
        for rule in ("A", "C", "D", "first5s"):
            v = [r[rule] for r in rs]
            fin = sorted(x for x in v if x is not None)
            med = "%.3g" % float(np.median(fin)) if fin else "-"
            cls = []
            for tol in TOLS:
                n_f = sum(x is not None and x > tol for x in v)
                n_m = sum(x is not None and x <= tol for x in v)
                cls.append("%d/%d/%d" % (n_f, n_m, len(v) - n_f - n_m))
            print("%-11s %-8s %9s %s" % (f, rule, med, " | ".join(cls)))
        crs = " ".join("CR(%g%%)=%.2f" % (e, float(np.median([r.get("CR_%g" % e, np.nan) for r in rs]))) for e in CR_EPS)
        print("%-11s %-8s %s" % (f, "CR", crs))
    if out_path:
        print("per-sequence table:", out_path)


# ------------------------------------------------------------------ correct rate
def run_crt(ledgers, out_path):
    rows, gts = [], {}
    for ld in ledgers:
        if not os.path.isfile(os.path.join(ld, "ledger.jsonl")):
            sys.exit("ERROR: no ledger at %s" % ld)
        for r in run_ledger.load_ledger(ld):
            lab = r["labels"]
            ds, sc = lab.get("dataset"), str(lab.get("scene")).split("/")[-1]
            if ds not in D.GT_FILES:
                continue
            cat = run_ledger.outcome_category(r["outcome_hint"])
            row = dict(ledger=os.path.basename(ld.rstrip("/")), run=r["run_id"], scene="%s/%s" % (ds, sc),
                       cell=lab.get("cell"), arm=lab.get("arm"), seed=lab.get("seed"), outcome=cat)
            f = os.path.join(ld, "runs", r["run_id"], "traj_est_tum.txt")
            if cat != "completed" or not os.path.isfile(f):
                row["crt_status"] = "not computable (%s)" % ("no trajectory" if cat == "completed" else cat)
                rows.append(row)
                continue
            if (ds, sc) not in gts:
                gts[(ds, sc)] = D.load_gt(ds, sc)
            est, gt = np.loadtxt(f, ndmin=2), gts[(ds, sc)]
            m = readings(est, gt, D.N_ALIGN.get(ds, 1000))
            row.update(rule_A=m["A"], rule_D=m["D"])
            for order in ("xyzw", "wxyz"):
                c = correct_rate(est, gt, CR_EPS, order)          # position criterion only
                if c is None:
                    row["crt_status"] = "not computable (fewer than 3 matched poses)"
                    break
                row["aoe_median_deg_%s" % order] = c["aoe_median_deg"]
                row["offset_spread_deg_%s" % order] = c["offset_spread_deg"]
                if order == "xyzw":
                    row.update({k: v for k, v in c.items() if k != "aoe_median_deg"}, crt_status="ok")
            rows.append(row)
    if not rows:
        sys.exit("ERROR: no runs in %s" % " ".join(ledgers))
    keys = sorted({k for r in rows for k in r})
    if out_path:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    ok = [r for r in rows if r.get("crt_status") == "ok"]
    print("correct rate: %d runs, %d with a correct rate, %d not computable" % (len(rows), len(ok), len(rows) - len(ok)))
    acc = [r for r in ok if (r.get("rule_D") or 1e9) <= 10]
    med = lambda k: float(np.median([r[k] for r in acc] or [np.nan]))   # noqa: E731
    print("orientation diagnostic, runs with rule D <= 10%% (%d): median orientation error xyzw %.2f deg, wxyz %.2f deg;"
          " spread of the rotation offset around its mean xyzw %.2f deg, wxyz %.2f deg. CR below uses the position"
          " criterion only." % (len(acc), med("aoe_median_deg_xyzw"), med("aoe_median_deg_wxyz"),
                                                     med("offset_spread_deg_xyzw"), med("offset_spread_deg_wxyz")))
    for arm in sorted({r["arm"] for r in rows}):
        g = [r for r in rows if r["arm"] == arm]
        go = [r for r in g if r.get("crt_status") == "ok"]
        print("  %-17s runs %3d | with CR %3d | median CR(1%%) %.2f CR(10%%) %.2f | not computable: %s" % (
            arm, len(g), len(go), float(np.median([r["CR_1"] for r in go] or [np.nan])),
            float(np.median([r["CR_10"] for r in go] or [np.nan])),
            dict(collections.Counter(r["crt_status"] for r in g if r.get("crt_status") != "ok"))))
    if out_path:
        print("per-run table:", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("faults")
    f.add_argument("--out")
    c = sub.add_parser("crt")
    c.add_argument("--ledgers", nargs="+", required=True)
    c.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "faults":
        run_faults(a.out)
    else:
        run_crt(a.ledgers, a.out)


if __name__ == "__main__":
    main()
