#!/usr/bin/env python3
"""Checks for the comparison of DAVIS240C data pipelines (DEIO's own against ours), CPU only.

1. Ground truth, in both directions. DEIO's own preprocessing (data/DAVIS240C_upstream/<s>/gt_stamped_left.txt, time
   relative to its own origin) and ours (data/RPGDAVIS_pp/<s>/mocap_data.txt) must hold the same poses: the constant
   time offset is found from the first upstream pose; then at least 99% of the poses of EACH file must match a pose of
   the other within 2 ms, the matched positions must agree within 1 mm (median and 99th percentile), the two files must
   start and end within 50 ms of each other, and their path lengths (the denominator of the MPE) must agree within 0.1%.
   A file that is a truncated copy of the other therefore fails.
2. Runs. Every planned unit (pipeline x sequence) must be present with the planned number of runs and the planned
   seeds; each run must print its seed ("[seed] trial 0 ... seed=N") and the [event-stress] banner with keep 1.0 and
   tq 0; a completed run must have its trajectory, with at least 90% of its poses matched to the ground truth within
   20 ms. Every completed run is scored (deio_se3_eval.evaluate) and its readings are printed: matched poses, matched
   poses in the first 5 s, and the errors under rules A, C, D and the first-5-s rigid rule. --mode smoke additionally
   requires that every unit COMPLETED and that all four readings were computed: the smoke has to show that each
   pipeline yields a scored trajectory. A large error is no smoke failure; a missing trajectory or a reading that
   cannot be computed is, and a crash in the smoke stops the comparison for investigation (it is not repeated with
   another seed). In --mode main, crashes and unscorable readings are results: they are listed, not filtered.
An empty or missing ledger fails. Exit status 1 if any check fails.
  python3 script/diag/check_davis_pipelines.py                                   # ground truth only
  python3 script/diag/check_davis_pipelines.py --mode smoke --ledgers logs/ledger/davis_pipelines_smoke_<stamp>
  python3 script/diag/check_davis_pipelines.py --mode main  --ledgers logs/ledger/davis_pipelines_main_<stamp>
"""
import argparse
import collections
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import deio_se3_eval as D   # noqa: E402
import run_ledger            # noqa: E402

SEQS = ("boxes_6dof", "poster_6dof")
UNITS = [(ds, s) for s in SEQS for ds in ("rpgdavis_up", "rpgdavis")]
PLAN = {"smoke": [2000], "main": list(range(2000, 2010))}


def path_length(g):
    return float(np.linalg.norm(np.diff(g[:, 1:4], axis=0), axis=1).sum())


def compare_ground_truth(up, ours, label=""):
    """True if the two ground-truth arrays (rows t_us x y z ...) hold the same poses; prints the evidence."""
    up, ours = up[np.argsort(up[:, 0])], ours[np.argsort(ours[:, 0])]
    k = int(np.argmin(np.linalg.norm(ours[:, 1:4] - up[0, 1:4], axis=1)))
    off = ours[k, 0] - up[0, 0]
    j_uo, ok_uo = D.ET.match(ours[:, 0], up[:, 0] + off, 0.002)      # each upstream pose -> ours
    j_ou, ok_ou = D.ET.match(up[:, 0] + off, ours[:, 0], 0.002)      # each of our poses -> upstream
    d = np.linalg.norm(ours[j_uo[ok_uo], 1:4] - up[ok_uo, 1:4], axis=1) if ok_uo.any() else np.array([np.inf])
    start = abs(ours[0, 0] - (up[0, 0] + off)) / 1e3
    end = abs(ours[-1, 0] - (up[-1, 0] + off)) / 1e3
    lu, lo = path_length(up), path_length(ours)
    checks = {"coverage upstream->ours >= 99%": ok_uo.mean() >= 0.99,
              "coverage ours->upstream >= 99%": ok_ou.mean() >= 0.99,
              "positions within 1 mm": bool(np.median(d) < 1e-3 and np.percentile(d, 99) < 1e-3),
              "start and end within 50 ms": start <= 50 and end <= 50,
              "path lengths within 0.1%": abs(lu - lo) <= 1e-3 * max(lu, lo)}
    good = all(checks.values())
    print("  %-12s upstream %6d poses %.3f m | ours %6d poses %.3f m | coverage %.1f%% / %.1f%% | start/end %.0f/%.0f ms"
          " | position difference median %.2g m, 99th pct %.2g m -> %s"
          % (label, len(up), lu, len(ours), lo, 100 * ok_uo.mean(), 100 * ok_ou.mean(), start, end, np.median(d),
             np.percentile(d, 99), "ok" if good else "FAILED: " + ", ".join(k for k, v in checks.items() if not v)))
    return good


def run_log(rdir):
    for name in ("run.log", "log.txt", "stdout.log"):
        p = os.path.join(rdir, name)
        if os.path.isfile(p):
            return open(p, errors="replace").read()
    return ""


def check_runs(ledger, mode):
    """Number of problems in the ledger for the planned units and seeds of `mode`."""
    recs = run_ledger.load_ledger(ledger)
    problems = []
    if not recs:
        problems.append("ledger %s is empty or missing" % ledger)
    by_unit = collections.defaultdict(list)
    for r in recs:
        lab = r["labels"]
        by_unit[(lab.get("dataset"), str(lab.get("scene")).split("/")[-1])].append(r)
    extra = sorted(set(by_unit) - set(UNITS))
    if extra:
        problems.append("units outside the plan: %s" % extra)
    for unit in UNITS:
        rs = by_unit.get(unit, [])
        seeds = sorted(int(r["labels"].get("seed")) for r in rs)
        if seeds != PLAN[mode]:
            problems.append("%s/%s: seeds %s, planned %s" % (unit[0], unit[1], seeds, PLAN[mode]))
        for r in rs:
            lab, cat = r["labels"], run_ledger.outcome_category(r["outcome_hint"])
            rdir = os.path.join(ledger, "runs", r["run_id"])
            log = run_log(rdir)
            p = []
            printed = [int(s) for s in re.findall(r"\[seed\] trial 0 .*?seed=(\d+)", log)]
            if printed != [int(lab.get("seed"))]:
                p.append("seed printed %s, label %s" % (printed or "none", lab.get("seed")))
            stress = re.findall(r"\[event-stress\][^\n]*keep=([0-9.]+) tq_us=(\d+)", log)
            if not stress or any(abs(float(k) - 1.0) > 1e-9 or int(q) != 0 for k, q in stress):
                p.append("[event-stress] %s, expected keep 1.0 tq 0" % (stress or "missing"))
            if lab.get("cell") != "clean":
                p.append("cell %r, planned clean" % lab.get("cell"))
            cover, reading = None, ""
            if cat == "completed":
                f = os.path.join(rdir, "traj_est_tum.txt")
                if not os.path.isfile(f):
                    p.append("completed but no trajectory saved")
                else:
                    est = np.loadtxt(f, ndmin=2)
                    gt = D.load_gt(*unit)
                    _, ok = D.ET.match(np.sort(gt[:, 0]), est[:, 0], 0.02)
                    cover = float(ok.mean()) if len(est) else 0.0
                    if cover < 0.9:
                        p.append("only %.0f%% of the poses match the ground truth within 20 ms" % (100 * cover))
                    m = D.evaluate(est, gt, D.N_ALIGN.get(unit[0], 1000)) if len(est) >= 10 else {}
                    vals = {k: m.get(k) for k in ("deio_rule", "sim3_all", "se3_all", "se3_first5s")}
                    fmt = lambda v: "-" if v is None or not np.isfinite(v) else "%.3g" % v   # noqa: E731
                    reading = "n %s, in 5 s %s | A %s C %s D %s first5s %s | " % (
                        m.get("matched_20ms", 0), m.get("matched_first5s", 0), *[fmt(v) for v in vals.values()])
                    missing = [k for k, v in vals.items() if v is None or not np.isfinite(v)]
                    if missing and mode == "smoke":
                        p.append("no score for %s" % ", ".join(missing))
                    elif missing:
                        reading += "not scorable: %s | " % ", ".join(missing)
            elif mode == "smoke":
                p.append("smoke run did not complete (%s): stop and investigate, do not reseed" % cat)
            print("  %-12s %-12s seed %-5s %-18s %s%s%s" % (unit[0], unit[1], lab.get("seed"), cat,
                  "" if cover is None else "matched %.0f%% | " % (100 * cover), reading, "; ".join(p) or "ok"))
            problems += ["%s/%s seed %s: %s" % (unit[0], unit[1], lab.get("seed"), x) for x in p]
    for x in problems:
        if "seed " not in x.split(":")[0]:
            print("  " + x)
    return len(problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", nargs="*", default=[])
    ap.add_argument("--mode", choices=("smoke", "main"), default="main")
    a = ap.parse_args()
    print("ground truth, DEIO's preprocessing against ours (both directions):")
    ok = all([compare_ground_truth(D.load_gt("rpgdavis_up", s), D.load_gt("rpgdavis", s), s) for s in SEQS])
    nbad = 0
    for ld in a.ledgers:
        print("runs of %s (mode %s):" % (ld, a.mode))
        nbad += check_runs(ld, a.mode)
    print("RESULT: %s" % ("all checks passed" if ok and nbad == 0 else
                          "FAILED (ground truth %s, %d problem(s) in the runs)" % ("ok" if ok else "different", nbad)))
    sys.exit(0 if ok and nbad == 0 else 1)


if __name__ == "__main__":
    main()
