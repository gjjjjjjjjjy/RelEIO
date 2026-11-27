#!/usr/bin/env python3
"""Checks of the DEIO runs of the pipeline comparison (round 5), CPU only.

Units: DEIO through our data-loading layer (dataset r5ours, data/R5_ours/<seq>__<tag>) and through its own DAVIS240C
pipeline (r5up, data/R5_up/<seq>__<tag>), boxes_6dof and poster_6dof, conditions clean, keep0.1, tq10000us; with
--dist also r5up under noise5hz and imu15000us. Every run must:
  - print its seed ("[seed] trial 0 ... seed=N") and the [event-stress] banner with keep 1.0 and tq 0: the
    perturbation is in the input files, so the reader's own perturbation must be off
  - carry the cell "clean" (nothing perturbed inside the reader)
  - r5ours: read exactly the events of its copy ("[event-slicing] ... events=N" equals the copy's manifest)
  - r5up: read the folder of its unit ("Preloaded ... on data/R5_up/<seq>__<tag>")
  - if completed: have its trajectory with at least 90% of its poses matched to the ground truth within 20 ms;
    its readings under rules A, C, D and the first-5-s rigid rule are printed
--mode smoke (seed 4000) additionally requires every unit to complete with all four readings computed; a smoke run
that does not complete stops the comparison for investigation (it is not repeated with another seed). In --mode main
(seeds 4000-4009) crashes and unscorable readings are results: listed, never filtered.
Exit status 1 if any check fails.

  python3 script/diag/r5_check_runs.py --mode smoke --ledgers logs/ledger/r5_deio_smoke_<stamp>
  python3 script/diag/r5_check_runs.py --mode main --dist --ledgers logs/ledger/r5_deio_main_<stamp>
"""
import argparse
import collections
import json
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
CORE = ("clean", "keep0.1", "tq10000us")
DIST = ("noise5hz", "imu15000us")
PLAN = {"smoke": [4000], "main": list(range(4000, 4010))}


def planned_units(dist):
    u = [(ds, "%s__%s" % (s, t)) for s in SEQS for t in CORE for ds in ("r5up", "r5ours")]
    if dist:
        u += [("r5up", "%s__%s" % (s, t)) for s in SEQS for t in DIST]
    return u


def run_log(rdir):
    for name in ("run.log", "log.txt", "stdout.log"):
        p = os.path.join(rdir, name)
        if os.path.isfile(p):
            return open(p, errors="replace").read()
    return ""


def input_problems(unit, log):
    """Checks that the reader took the planned input (see the module docstring)."""
    ds, scene = unit
    p = []
    if ds == "r5ours":
        ev = json.load(open(os.path.join(ROOT, "data", "R5_ours", scene, "r5_evidence.json")))
        want = ev["output"]["digest"]["events"]
        got = [int(n) for n in re.findall(r"\[event-slicing\] \w+ events=(\d+)", log)]
        if got != [want]:
            p.append("our reader read events %s, the copy holds %d" % (got or "none", want))
    else:
        folders = re.findall(r"Preloaded \d+ davis240c voxels.* on (\S+)", log)
        if not folders or any(os.path.normpath(f).split(os.sep)[-2:] != ["R5_up", scene] for f in folders):
            p.append("DEIO's reader took %s, planned data/R5_up/%s" % (folders or "nothing", scene))
    return p


def run_problems(unit, lab, log):
    """Configuration and input checks of one run (no trajectory): seed printed, reader perturbation off, cell clean,
    planned input taken."""
    p = []
    printed = [int(s) for s in re.findall(r"\[seed\] trial 0 .*?seed=(\d+)", log)]
    if printed != [int(lab.get("seed"))]:
        p.append("seed printed %s, label %s" % (printed or "none", lab.get("seed")))
    stress = re.findall(r"\[event-stress\][^\n]*keep=([0-9.]+) tq_us=(\d+)", log)
    if not stress or any(abs(float(k) - 1.0) > 1e-9 or int(q) != 0 for k, q in stress):
        p.append("[event-stress] %s, expected keep 1.0 tq 0" % (stress or "missing"))
    if lab.get("cell") != "clean":
        p.append("cell %r, planned clean" % lab.get("cell"))
    return p + input_problems(unit, log)


def check_runs(ledger, mode, dist):
    recs = run_ledger.load_ledger(ledger)
    problems = []
    if not recs:
        problems.append("ledger %s is empty or missing" % ledger)
    by_unit = collections.defaultdict(list)
    for r in recs:
        lab = r["labels"]
        by_unit[(lab.get("dataset"), str(lab.get("scene")).split("/")[-1])].append(r)
    units = planned_units(dist)
    extra = sorted(set(by_unit) - set(units))
    if extra:
        problems.append("units outside the plan: %s" % extra)
    for unit in units:
        rs = by_unit.get(unit, [])
        seeds = sorted(int(r["labels"].get("seed")) for r in rs)
        if seeds != PLAN[mode]:
            problems.append("%s/%s: seeds %s, planned %s" % (unit[0], unit[1], seeds, PLAN[mode]))
        for r in rs:
            lab, cat = r["labels"], run_ledger.outcome_category(r["outcome_hint"])
            rdir = os.path.join(ledger, "runs", r["run_id"])
            log = run_log(rdir)
            p = run_problems(unit, lab, log)
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
                    reading = "n %s | A %s C %s D %s first5s %s | " % (m.get("matched_20ms", 0),
                                                                      *[fmt(v) for v in vals.values()])
                    missing = [k for k, v in vals.items() if v is None or not np.isfinite(v)]
                    if missing and mode == "smoke":
                        p.append("no score for %s" % ", ".join(missing))
                    elif missing:
                        reading += "not scorable: %s | " % ", ".join(missing)
            elif mode == "smoke":
                p.append("smoke run did not complete (%s): stop and investigate, do not reseed" % cat)
            print("  %-7s %-24s seed %-5s %-14s %s%s%s" % (unit[0], unit[1], lab.get("seed"), cat,
                  "" if cover is None else "matched %.0f%% | " % (100 * cover), reading, "; ".join(p) or "ok"))
            problems += ["%s/%s seed %s: %s" % (unit[0], unit[1], lab.get("seed"), x) for x in p]
    for x in problems:
        if "seed " not in x.split(":")[0]:
            print("  " + x)
    return len(problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", nargs="+", required=True)
    ap.add_argument("--mode", choices=("smoke", "main"), default="main")
    ap.add_argument("--dist", action="store_true", help="the plan includes the noise and IMU-offset units")
    a = ap.parse_args()
    nbad = 0
    for ld in a.ledgers:
        print("runs of %s (mode %s%s):" % (ld, a.mode, ", with disturbances" if a.dist else ""))
        nbad += check_runs(ld, a.mode, a.dist)
    print("RESULT: %s" % ("all checks passed" if nbad == 0 else "FAILED (%d problem(s))" % nbad))
    sys.exit(0 if nbad == 0 else 1)


if __name__ == "__main__":
    main()
