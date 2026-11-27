#!/usr/bin/env python3
"""Readouts of the pipeline comparison (round 5), as fixed before the runs. Read-only, CPU, numpy.

Readers: DEIO through our data-loading layer (deio_ours: dataset r5ours), DEIO through its own DAVIS240C pipeline
(deio_own: r5up), SuperEIO (supereio: its rosbag). One ground truth and one time base for all, as in the bias
experiment (script/diag/cross_system_eval.py, whose functions are used here): /optitrack/davis from the bag
(outputs/supereio/gt/<seq>_optitrack_davis_us.txt). DEIO's trajectories are moved to the bag time with the offset
between the reader's ground-truth file and the bag's, checked pose by pose (gt_offset_us); SuperEIO's IMU poses are
moved to the camera with the run's extrinsics. Matching: nearest ground truth within 20 ms.
Common window W per sequence: from the latest first pose to the earliest last pose over all completed clean runs of
the three readers (the runs that set the two ends are printed).
Rules, per trajectory:
  D  SE(3) without scale fitted on all matched poses in W; MPE = mean error / ground-truth path length in W x 100
  C  Sim(3) on the same poses (only the scale differs from D)
  A  DEIO only: DEIO's published evaluation (Sim(3) on the first 1000 poses, matching within 1 s), on the whole
     trajectory as DEIO computes it (script/diag/deio_se3_eval.py)
Outcomes: DEIO completed / failed (numerical failure) / undetermined / excluded (the configuration and input checks
of r5_check_runs.py or the controls of precision_pilot.py not met); SuperEIO by the bias experiment's rule
(script/supereio/r5_accept.py), excluded if its recorded identity, CPU set or engines differ from the plan. A
SuperEIO batch whose acceptance failed is not read unless --allow-invalid. A SuperEIO attempt that is undetermined
but left a trajectory (at least 10 poses, e.g. a ROS process that died at shutdown or was respawned) keeps its
outcome: it is not counted in readouts 2-4 and does not set the window; its scores in W are listed apart, as a
sensitivity reading only.
Readouts per reader, sequence and condition (systems are not ranked):
  1 completion: completed / failed / undetermined; excluded runs listed apart
  2 within 10% under D: count over all configuration-verified runs (all but excluded); a run without a D score
    (not completed, or under 10 matched poses in W) counts as not within
  3 flips on the same trajectory, both directions, over the runs scored under both rules: C vs D for every
    reader, A vs D for the two DEIO readers ("C<=10<D": within under C, not under D)
  4 D above 100% (mean error larger than the ground-truth path length in W), denominator as in 2

  python3 script/diag/r5_eval.py --deio-ledger logs/ledger/r5_deio_main_<stamp> \\
      --supereio-record outputs/supereio/r5_main_<stamp>.json --out records/r5_eval_<stamp>.csv
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "script", "supereio"))
import cross_system_eval as CS   # noqa: E402  (ground-truth offset, camera positions, window metric)
import deio_se3_eval as D        # noqa: E402  (ground truth of the DEIO readers, rule A)
import precision_pilot as P      # noqa: E402  (controls of a DEIO run)
import r5_accept as RA           # noqa: E402  (SuperEIO run folders)
import r5_check_runs as CR       # noqa: E402  (configuration and input checks of a DEIO run)
import run_ledger                # noqa: E402

SEQS = ("boxes_6dof", "poster_6dof")
CONDS = ("clean", "keep0.1", "tq10000us", "noise5hz", "imu15000us")
READERS = ("deio_ours", "deio_own", "supereio")
LABEL = {"deio_ours": "DEIO, our loader", "deio_own": "DEIO, own pipeline", "supereio": "SuperEIO"}
TOL, PATH = 10.0, 100.0


# ---- loading ---------------------------------------------------------------------------------------------------
def load_deio(ledger, gts):
    """DEIO runs of both readers in one ledger, trajectories on the bag time base. Returns (runs, offsets)."""
    runs, offs = [], {}
    for r in run_ledger.load_ledger(ledger):
        lab = r["labels"]
        ds, scene = lab.get("dataset"), str(lab.get("scene")).split("/")[-1]
        if ds not in ("r5ours", "r5up") or "__" not in scene:
            continue
        cat = run_ledger.outcome_category(r["outcome_hint"])
        if cat == "not_run":
            continue
        seq, cond = scene.split("__", 1)
        x = {"reader": "deio_ours" if ds == "r5ours" else "deio_own", "seq": seq, "cond": cond,
             "rep": lab.get("seed"), "run": r["run_id"], "status": "completed", "t": None, "p": None, "A": None,
             "D_full": None}
        rdir = os.path.join(ledger, "runs", r["run_id"])
        facts = P._load_facts(ledger, r["run_id"])
        probs = ["facts.json missing"] if facts is None else P.verify_controls(
            facts, lab.get("arm"), bool(lab.get("cholinfo")), cell=lab.get("cell"))
        probs += CR.run_problems((ds, scene), lab, CR.run_log(rdir))
        if probs:
            x["status"] = "excluded: " + probs[0][:80]
        elif cat == "numerical_failure":
            x["status"] = "failed: numerical failure"
        elif cat != "completed":
            x["status"] = "undetermined: " + cat
        else:
            f = os.path.join(rdir, "traj_est_tum.txt")
            if not os.path.isfile(f):
                x["status"] = "undetermined: completed without a saved trajectory"
            else:
                tum = np.loadtxt(f, ndmin=2)
                g = D.load_gt(ds, scene)
                if (ds, scene) not in offs:
                    offs[(ds, scene)] = CS.gt_offset_us(gts[seq], g)[0]
                x["t"], x["p"] = tum[:, 0] + offs[(ds, scene)], tum[:, 1:4]
                m = D.evaluate(tum, g, 1000) if len(tum) >= 10 else {}
                x["A"], x["D_full"] = m.get("deio_rule"), m.get("se3_all")
        runs.append(x)
    return runs, offs


def planned(a):
    """Whether the attempt's summary.txt names its planned sequence, tag, folder, variant and bag: the record's
    as_planned if it has one, else checked again from summary.txt (records written before r5_accept.py stored it)."""
    if a.get("as_planned") is not None:
        return bool(a["as_planned"])
    R = RA.absrun(a["run"])
    idn = RA.read_attempt(R, a.get("exit_code", 0)).get("identity")
    return RA.identity_as_planned(idn, a["seq"], a["cond"], a["index"], R)


def load_supereio(record_path, allow_invalid=False):
    """SuperEIO attempts of a main batch, from its acceptance record (one entry per planned attempt)."""
    rec = json.load(open(record_path))
    if rec.get("mode") != "main":
        sys.exit("%s is not the record of a main batch" % record_path)
    if not rec.get("valid") and not allow_invalid:
        sys.exit("SuperEIO batch %s did not pass its acceptance (%s); --allow-invalid reads it anyway"
                 % (record_path, "; ".join(rec.get("problems", [])[:3])))
    runs = []
    for a in rec["attempts"]:
        x = {"reader": "supereio", "seq": a["seq"], "cond": a["cond"], "rep": a["index"],
             "run": os.path.basename(str(a.get("run", ""))), "status": a["outcome"], "t": None, "p": None,
             "A": None, "D_full": a.get("se3_all")}
        if a["outcome"] == "missing":
            x["status"] = "undetermined: not run"
        elif a.get("summary") and not (planned(a) and a.get("cpus_exact") and a.get("engines_exact")):
            x["status"] = "excluded: as planned %s, CPU set %s, engines %s" % (
                planned(a), a.get("cpus_exact"), a.get("engines_exact"))
        elif a["outcome"] == "completed" or (a["outcome"].startswith("undetermined") and a.get("summary")
                                             and (a.get("poses") or 0) >= 10):
            R = RA.absrun(a["run"])
            est = CS.ET.load_estimate(os.path.join(R, "evio_result_no_loop.csv"))
            cfg = open(os.path.join(R, "config.yaml")).read()
            R_ic = CS.ET.opencv_matrix(cfg, "extrinsicRotation").reshape(3, 3)
            t_ic = CS.ET.opencv_matrix(cfg, "extrinsicTranslation")
            x["t"], x["p"] = CS.camera_positions(est, R_ic, t_ic)
        runs.append(x)
    return runs, rec


# ---- window and readouts ---------------------------------------------------------------------------------------
def common_window(runs, seqs=SEQS):
    """{seq: (w0, w1)} over the completed clean runs of every reader, and which runs set the two ends."""
    win, ends = {}, {}
    for s in seqs:
        clean = [x for x in runs if x["seq"] == s and x["cond"] == "clean" and x["status"] == "completed"
                 and x["t"] is not None and len(x["t"])]
        if not clean:
            sys.exit("no completed clean run of %s: no common window" % s)
        a = max(clean, key=lambda x: float(np.min(x["t"])))
        b = min(clean, key=lambda x: float(np.max(x["t"])))
        win[s] = (float(np.min(a["t"])), float(np.max(b["t"])))
        ends[s] = {"start_set_by": [a["reader"], a["run"]], "end_set_by": [b["reader"], b["run"]],
                   "readers": sorted({x["reader"] for x in clean})}
        if win[s][1] <= win[s][0]:
            sys.exit("empty common window for %s" % s)
    return win, ends


def score(runs, gts, win):
    for x in runs:
        m = CS.window_metric(x["t"], x["p"], gts[x["seq"]], *win[x["seq"]]) if x["t"] is not None else None
        x["D"], x["C"] = (m["se3_W"], m["sim3_W"]) if m else (None, None)
        x["scale"], x["matched_W"], x["path_W_m"] = (m["sim3_W_scale"], m["matched"], m["path_m"]) if m else (None,) * 3
    return runs


def readouts(rows):
    """Readouts 1-4 per (reader, seq, cond) from rows with reader, seq, cond, status, D, C, A."""
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["reader"], r["seq"], r["cond"])].append(r)
    for k, g in groups.items():
        st = lambda p: sum(str(r["status"]).startswith(p) for r in g)  # noqa: E731
        den = [r for r in g if not str(r["status"]).startswith("excluded")]
        done = [r for r in den if r["status"] == "completed"]
        dv = sorted(r["D"] for r in done if r["D"] is not None)
        o = {"runs": len(g), "completed": st("completed"), "failed": st("failed"), "undetermined": st("undetermined"),
             "excluded": st("excluded"), "denominator": len(den),
             "within_10_D": sum(v <= TOL for v in dv), "above_100_D": sum(v > PATH for v in dv),
             "D_median": float(np.median(dv)) if dv else None, "D_min": dv[0] if dv else None,
             "D_max": dv[-1] if dv else None}
        both = [r for r in done if r["C"] is not None and r["D"] is not None]
        o.update(CD_scored=len(both), C_within_D_not=sum(r["C"] <= TOL < r["D"] for r in both),
                 D_within_C_not=sum(r["D"] <= TOL < r["C"] for r in both))
        sc = sorted(r["scale"] for r in done if r.get("scale") is not None)
        o["scale_median"] = float(np.median(sc)) if sc else None
        if k[0] != "supereio":
            ba = [r for r in done if r["A"] is not None and r["D"] is not None]
            o.update(AD_scored=len(ba), A_within_D_not=sum(r["A"] <= TOL < r["D"] for r in ba),
                     D_within_A_not=sum(r["D"] <= TOL < r["A"] for r in ba))
        out[k] = o
    return out


def print_table(ro):
    f = lambda v: "-" if v is None else ("%.3g" % v)  # noqa: E731
    print("\n%-19s %-11s %-10s %4s %-15s %-8s %-8s %-9s %-9s %-9s %-9s %-24s %-6s" % (
        "reader", "sequence", "condition", "runs", "done/fail/und/ex", "D<=10", "D>100", "C<=10<D", "D<=10<C",
        "A<=10<D", "D<=10<A", "D median [min, max] %", "scale"))
    for rd in READERS:
        for s in SEQS:
            for c in CONDS:
                o = ro.get((rd, s, c))
                if not o:
                    continue
                n = o["denominator"]
                ad = ("%d/%d" % (o["A_within_D_not"], o["AD_scored"]), "%d/%d" % (o["D_within_A_not"], o["AD_scored"])) \
                    if "AD_scored" in o else ("", "")
                print("%-19s %-11s %-10s %4d %-15s %-8s %-8s %-9s %-9s %-9s %-9s %-24s %-6s" % (
                    LABEL[rd], s, c, o["runs"], "%d/%d/%d/%d" % (o["completed"], o["failed"], o["undetermined"],
                                                                 o["excluded"]),
                    "%d/%d" % (o["within_10_D"], n), "%d/%d" % (o["above_100_D"], n),
                    "%d/%d" % (o["C_within_D_not"], o["CD_scored"]), "%d/%d" % (o["D_within_C_not"], o["CD_scored"]),
                    ad[0], ad[1], "%s [%s, %s]" % (f(o["D_median"]), f(o["D_min"]), f(o["D_max"])),
                    f(o["scale_median"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deio-ledger", required=True)
    ap.add_argument("--supereio-record", help="outputs/supereio/r5_main_<stamp>.json (without it W is DEIO's only)")
    ap.add_argument("--allow-invalid", action="store_true")
    ap.add_argument("--out", help="per-run CSV; <out minus .csv>_readouts.json gets the readouts and the window")
    a = ap.parse_args()
    gts = {}
    for s in SEQS:
        p = os.path.join(ROOT, "outputs", "supereio", "gt", "%s_optitrack_davis_us.txt" % s)
        if not os.path.exists(p):
            sys.exit("missing %s: run script/supereio/eval_traj.py on one %s run first" % (p, s))
        gts[s] = np.loadtxt(p)
    runs, offs = load_deio(a.deio_ledger, gts)
    for (ds, scene), off in sorted(offs.items()):
        print("ground truth %s/%s: reader time + %.0f us = bag time" % (ds, scene, off))
    rec = None
    if a.supereio_record:
        sr, rec = load_supereio(a.supereio_record, a.allow_invalid)
        runs += sr
    else:
        print("WARNING: no SuperEIO record; the window is set by the DEIO readers only")
    runs = [x for x in runs if x["seq"] in SEQS]
    win, ends = common_window(runs)
    for s in SEQS:
        print("window W %s: %.2f s, from %.2f s after the ground-truth start; start set by %s %s, end by %s %s "
              "(readers %s)" % (s, (win[s][1] - win[s][0]) / 1e6, (win[s][0] - gts[s][0, 0]) / 1e6,
                                *ends[s]["start_set_by"], *ends[s]["end_set_by"], ", ".join(ends[s]["readers"])))
    score(runs, gts, win)
    ro = readouts(runs)
    print_table(ro)
    other = collections.Counter((x["reader"], x["status"]) for x in runs if x["status"] != "completed")
    for (rd, stt), n in sorted(other.items()):
        print("  %-19s %s: %d" % (LABEL[rd], stt, n))
    und = [x for x in runs if str(x["status"]).startswith("undetermined") and x.get("D") is not None]
    if und:
        print("\nundetermined attempts with a trajectory: scores in W, NOT counted in readouts 2-4 (sensitivity only)")
        for x in sorted(und, key=lambda x: (x["reader"], x["seq"], x["cond"], str(x["rep"]))):
            print("  %-19s %-11s %-10s %-4s D %.3g  C %.3g  scale %.3g  (%s)" % (
                LABEL[x["reader"]], x["seq"], x["cond"], x["rep"], x["D"], x["C"], x["scale"], x["status"]))
    if a.out:
        keys = ["reader", "seq", "cond", "rep", "run", "status", "D", "C", "A", "scale", "matched_W", "path_W_m",
                "D_full"]
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for x in runs:
                w.writerow({k: x.get(k) for k in keys})
        side = {"deio_ledger": a.deio_ledger, "supereio_record": a.supereio_record,
                "supereio_batch_valid": rec.get("valid") if rec else None,
                "window": {s: {"w0_us": win[s][0], "w1_us": win[s][1], **ends[s]} for s in SEQS},
                "deio_gt_offsets_us": {"%s/%s" % k: v for k, v in offs.items()},
                "readouts": {"%s|%s|%s" % k: v for k, v in sorted(ro.items())}}
        with open(os.path.splitext(a.out)[0] + "_readouts.json", "w") as fh:
            json.dump(side, fh, indent=1)
        print("\nper-run table %s; readouts %s_readouts.json" % (a.out, os.path.splitext(a.out)[0]))


if __name__ == "__main__":
    main()
