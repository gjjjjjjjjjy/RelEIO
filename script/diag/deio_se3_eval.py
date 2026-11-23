#!/usr/bin/env python3
"""Re-evaluate saved DEIO trajectories (ledger-4 runs keep runs/<id>/traj_est_tum.txt) without the scale correction.

DEIO's own MPE (utils/eval_utils.py, _n_to_align=1000) fits Sim(3) on the first 1000 associated poses. On short
hand-held sequences a diverging trajectory can then be shrunk towards one point (scale near 0), and the error
approaches the spread of the ground truth around its mean, whatever the estimate. This script repeats
DEIO's rule and adds rules that cannot hide a wrong scale, per completed run:
Changing one thing at a time, from DEIO's rule to the scale-free rule (error over all matched poses / full path):
  A deio_rule      Sim(3) fitted on the first 1000 poses, matching within 1 s (evo, as DEIO)
  B sim3_1000_20ms Sim(3) fitted on the first 1000 poses, matching within 20 ms      (A->B: matching only)
  C sim3_all       Sim(3) fitted on all poses, matching within 20 ms                 (B->C: fit window only)
  D se3_all        SE(3) fitted on all poses, matching within 20 ms                  (C->D: scale only)
Every completed run is listed: one without a trajectory file, or with fewer than 10 matched poses, is counted as
"no score" in its group and in the total, never as a pass.
  sim3_scale  scale of a Sim(3) fit on all matched poses (near 0 = collapsed)
  dist_ratio  median of |dp_est| / |dp_gt| over pose pairs 1 s apart (no alignment; 1 = metric scale)
  gt_spread   mean distance of the matched ground truth from its mean / path length (what a collapsed fit returns)
Ground truth: the file DEIO's evaluation script reads for the run, loaded the same way, so it shares the time base
of the trajectory (rows t_us x y z qx qy qz qw):
  rpgdavis data/RPGDAVIS_pp/<s>/mocap_data.txt (t > 0)        vector data/VECtor/<s>/poses_evs_left.txt
  hku      data/HKU/<s>/gt_stamped_davis346.txt               tumvie data/TUM-VIE/<s>/mocap_data.txt (unit by magnitude, t > 0)
  fpv      data/UZHFPV/<s>/stamped_groundtruth_us.txt
DEIO's rule fits on the first 1000 poses for every dataset except UZH-FPV, where it uses all (_n_to_align=-1).

  python3 script/diag/deio_se3_eval.py --ledgers logs/ledger/round4_core_main_all_20261121_023641 \\
      [--scenes rpgdavis/boxes_6dof vector/desk_normal1 ...] [--out records/deio_se3_round4.csv]
"""
import argparse
import collections
import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_ledger  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_traj", os.path.join(ROOT, "script", "supereio", "eval_traj.py"))
ET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ET)


GT_FILES = {"rpgdavis": ("data/RPGDAVIS_pp", "mocap_data.txt"), "vector": ("data/VECtor", "poses_evs_left.txt"),
            "hku": ("data/HKU", "gt_stamped_davis346.txt"), "tumvie": ("data/TUM-VIE", "mocap_data.txt"),
            "fpv": ("data/UZHFPV", "stamped_groundtruth_us.txt")}
N_ALIGN = {"fpv": None}   # None = all poses


def load_gt(dataset, scene):
    base, name = GT_FILES[dataset]
    # scene folder as the evaluation scripts resolve it (tumvie.py _resolve_scene_dir drops "-trans"/"-calibB")
    cands = [scene, scene.replace("-trans", ""), scene.replace("-calibB", ""), scene.replace("-trans-calibB", "")]
    paths = [os.path.join(ROOT, base, c, name) for c in dict.fromkeys(cands)]
    path = next((p for p in paths if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError("ground truth for %s/%s: tried %s" % (dataset, scene, paths))
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                v = [float(x) for x in line.replace(",", " ").split()[:8]]
            except ValueError:
                continue
            if len(v) == 8:
                rows.append(v)
    g = np.array(rows)
    if dataset == "tumvie":      # as _to_us in script/eval_deio/tumvie.py
        t = g[:, 0]
        mx, span = np.abs(t).max(), t.max() - t.min()
        g[:, 0] = t / 1e3 if mx > 1e12 else (t if (mx > 1e5 or span > 1e5) else t * 1e6)
    if dataset in ("rpgdavis", "tumvie"):
        g = g[g[:, 0] > 0]
    return g


def evaluate(traj, gt, n_align=1000):
    t_e, p_e = traj[:, 0], traj[:, 1:4]
    o = np.argsort(gt[:, 0])
    t_g, p_g = gt[o, 0], gt[o, 1:4]
    path = float(np.linalg.norm(np.diff(p_g, axis=0), axis=1).sum())
    out = {}
    j1, ok1 = ET.match(t_g, t_e, 1.0)
    A1, B1 = p_e[ok1], p_g[j1[ok1]]
    if len(A1) >= 10:
        n = len(A1) if n_align is None else min(n_align, len(A1))
        s, R, t = ET.umeyama(A1[:n], B1[:n], True)
        out["deio_rule"] = float(np.linalg.norm(B1 - s * (R @ A1.T).T - t, axis=1).mean() / path * 100)
    j, ok = ET.match(t_g, t_e, 0.02)
    A, B = p_e[ok], p_g[j[ok]]
    if len(A) < 10:
        return out

    def err(s, R, t):
        return float(np.linalg.norm(B - s * (R @ A.T).T - t, axis=1).mean() / path * 100)
    nb = len(A) if n_align is None else min(n_align, len(A))
    out["sim3_1000_20ms"] = err(*ET.umeyama(A[:nb], B[:nb], True))
    s3, R3, t3 = ET.umeyama(A, B, True)
    out["sim3_all"], out["sim3_scale"] = err(s3, R3, t3), float(s3)
    out["se3_all"] = err(*ET.umeyama(A, B, False))
    out["gt_spread"] = float(np.linalg.norm(B - B.mean(0), axis=1).mean() / path * 100)
    te = t_e[ok]
    k = np.searchsorted(te, te + 1e6)
    i = np.nonzero(k < len(te))[0]
    k = k[k < len(te)]
    dg, de = np.linalg.norm(B[k] - B[i], axis=1), np.linalg.norm(A[k] - A[i], axis=1)
    out["dist_ratio"] = float(np.median(de[dg > 0.05] / dg[dg > 0.05])) if (dg > 0.05).any() else None
    out["est_extent_m"] = float(np.linalg.norm(A - A[0], axis=1).max())
    out["gt_extent_m"] = float(np.linalg.norm(B - B[0], axis=1).max())
    return out


def med(v):
    v = [x for x in v if x is not None]
    return "%.2f" % float(np.median(v)) if v else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+", help="dataset/scene (default: every scene in the ledgers)")
    ap.add_argument("--out")
    a = ap.parse_args()
    gts, rows = {}, []
    for ld in a.ledgers:
        for r in run_ledger.load_ledger(ld):
            lab = r["labels"]
            ds, sc = lab.get("dataset"), str(lab.get("scene")).split("/")[-1]
            if ds not in GT_FILES or (a.scenes and "%s/%s" % (ds, sc) not in a.scenes):
                continue
            if run_ledger.outcome_category(r["outcome_hint"]) != "completed":
                continue
            f = os.path.join(ld, "runs", r["run_id"], "traj_est_tum.txt")
            if (ds, sc) not in gts:
                gts[(ds, sc)] = load_gt(ds, sc)
            m = evaluate(np.loadtxt(f), gts[(ds, sc)], N_ALIGN.get(ds, 1000)) if os.path.exists(f) else {}
            m["score"] = "missing trajectory" if not os.path.exists(f) else (
                "ok" if np.isfinite(m.get("se3_all", np.nan)) else "no score")
            rows.append(dict(ledger=os.path.basename(ld.rstrip("/")), scene="%s/%s" % (ds, sc), cell=lab.get("cell"),
                             arm=lab.get("arm"), seed=lab.get("seed"), run=r["run_id"], ledger_mpe=r.get("mpe"), **m))
    groups = collections.defaultdict(list)
    for x in rows:
        groups[(x["ledger"][:28], x["scene"], x["cell"], x["arm"])].append(x)
    print("medians of the MPE (%) under rules A-D (see --help); counts of runs above 100% under C and D")
    print("%-24s %-23s %-9s %-17s %4s %5s %-8s %-8s %-8s %-8s %-9s %5s %5s %6s %-8s %-7s" % (
        "ledger", "scene", "cell", "arm", "n", "noscr", "ledger", "A deio", "B 20ms", "C all", "D SE(3)",
        "C>100", "D>100", "sc<.5", "dist 1s", "spread"))
    for k in sorted(groups, key=str):
        g = groups[k]
        print("%-24s %-23s %-9s %-17s %4d %5d %-8s %-8s %-8s %-8s %-9s %5d %5d %6d %-8s %-7s" % (
            k[0][:24], k[1], k[2], k[3], len(g), sum(x["score"] != "ok" for x in g),
            med([x["ledger_mpe"] for x in g]), med([x.get("deio_rule") for x in g]),
            med([x.get("sim3_1000_20ms") for x in g]), med([x.get("sim3_all") for x in g]),
            med([x.get("se3_all") for x in g]), sum((x.get("sim3_all") or 0) > 100 for x in g),
            sum((x.get("se3_all") or 0) > 100 for x in g), sum((x.get("sim3_scale") or 1) < 0.5 for x in g),
            med([x.get("dist_ratio") for x in g]), med([x.get("gt_spread") for x in g])))
    print("\nTOTAL completed runs %d | scored %d | missing trajectory %d | no score %d" % (
        len(rows), sum(x["score"] == "ok" for x in rows), sum(x["score"] == "missing trajectory" for x in rows),
        sum(x["score"] == "no score" for x in rows)))
    for arm in sorted({x["arm"] for x in rows}):
        g = [x for x in rows if x["arm"] == arm]
        print("  %-17s completed %3d | D (SE(3)) <= 100%%: %3d | > 100%%: %3d | no score: %d | C (Sim(3) all) > 100%%: %d"
              % (arm, len(g), sum(x["score"] == "ok" and x["se3_all"] <= 100 for x in g),
                 sum(x["score"] == "ok" and x["se3_all"] > 100 for x in g), sum(x["score"] != "ok" for x in g),
                 sum(x["score"] == "ok" and x["sim3_all"] > 100 for x in g)))
    if a.out and rows:
        keys = sorted({k for x in rows for k in x})
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("per-run table:", a.out)


if __name__ == "__main__":
    main()
