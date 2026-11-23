#!/usr/bin/env python3
"""Round 4 main table and reliability curve, with crashes and trajectory error reported together. Read-only.

Inputs (committed from the cluster): every run with its outcome (round4_all_attribution_*.csv, 640 runs) and the
re-scored error of every completed run (deio_se3_round4_steps_*.csv, from script/diag/deio_se3_eval.py).
For each setting the denominator is all 320 runs; a numerical crash counts as "not within any tolerance".
  R_D(eps) = share of the 320 runs that completed with SE(3) MPE <= eps   (no scale correction: the main rule)
  R_A(eps) = the same with DEIO's own rule A (Sim(3), first 1000 poses; all poses on UZH-FPV)   (contrast)
Also written: per condition (sequence x cell x setting) runs, crashes, completed, completed with SE(3) MPE <= 100%,
median SE(3) MPE of completed runs, median under rule A, and runs with a small fitted Sim(3) scale (< 0.5; accompanies a collapsed fit, does not establish one).
The 100% line is a reference for display, not a navigation safety bound.

  python3 script/diag/round4_reliability.py \\
      --outcomes revision/analysis/round4_all_attribution_20261121_023641.csv \\
      --scores revision/analysis/deio_se3_round4_steps_20261121.csv --outdir revision/analysis
"""
import argparse
import collections
import csv
import math
import os
import sys

import numpy as np

ARMS = ("no_tf32_override", "default")
LABEL = {"no_tf32_override": "override removed", "default": "container default"}
CELLS = ("clean", "keep:0.1", "tq:10000", "patch:16")


def load(outcomes, scores):
    runs = {}
    with open(outcomes) as fh:
        for r in csv.DictReader(fh):
            runs[r["run_id"]] = {"arm": r["arm"], "cell": r["cell"], "scene": r["scene"], "category": r["category"]}
    with open(scores) as fh:
        for r in csv.DictReader(fh):
            x = runs.get(r["run"])
            if x is None:
                sys.exit("scored run %s is not in the outcome table" % r["run"])
            x.update(score=r["score"], D=float(r["se3_all"]) if r["se3_all"] else math.nan,
                     A=float(r["deio_rule"]) if r["deio_rule"] else math.nan,
                     scale=float(r["sim3_scale"]) if r["sim3_scale"] else math.nan)
    bad = [k for k, x in runs.items() if x["category"] == "completed" and x.get("score") != "ok"]
    if bad:
        sys.exit("%d completed runs without a score, e.g. %s: they would be counted as passes" % (len(bad), bad[0]))
    return runs


def reliability(runs, arm, rule, eps):
    g = [x for x in runs.values() if x["arm"] == arm]
    v = np.array([x[rule] if x["category"] == "completed" else np.inf for x in g])
    return np.array([(v <= e).sum() for e in eps]) / len(g), len(g)


def table(runs, path):
    groups = collections.defaultdict(list)
    for x in runs.values():
        groups[(x["scene"], x["cell"], x["arm"])].append(x)
    rows = []
    for (sc, cell, arm), g in sorted(groups.items(), key=lambda kv: (kv[0][0], CELLS.index(kv[0][1]), kv[0][2])):
        done = [x for x in g if x["category"] == "completed"]
        rows.append({"sequence": sc, "cell": cell, "setting": arm, "runs": len(g),
                     "crashes": sum(x["category"] == "numerical_failure" for x in g), "completed": len(done),
                     "completed_se3_le_100": sum(x["D"] <= 100 for x in done),
                     "median_se3_mpe": round(float(np.median([x["D"] for x in done])), 3) if done else "",
                     "median_rule_a_mpe": round(float(np.median([x["A"] for x in done])), 3) if done else "",
                     "sim3_scale_below_0p5": sum(x["scale"] < 0.5 for x in done)})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def figure(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import nature_style as ns
    ns.apply_nature_style()
    eps = np.logspace(-1, 4, 400)
    color = {"no_tf32_override": ns.NAT_BLUE, "default": ns.NAT_ORANGE}
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    at10 = {}

    def wilson(k, n, z=1.96):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h

    for arm in ARMS:
        for rule, ls, lw in (("D", "-", 1.6), ("A", (0, (3, 2)), 1.1)):
            r, n = reliability(runs, arm, rule, eps)
            if rule == "D":      # pointwise Wilson 95% band for the main rule
                lo, hi = np.array([wilson(round(v * n), n) for v in r]).T
                ax.fill_between(eps, lo, hi, color=color[arm], alpha=0.16, lw=0, zorder=2)
            ax.plot(eps, r, ls=ls, lw=lw, color=color[arm], solid_capstyle="round", zorder=3)
            at10[(arm, rule)] = (reliability(runs, arm, rule, [10.0])[0][0], n)
    ax.axvline(10.0, color=ns.INK, lw=0.6, ls=":", zorder=1)
    # counts at the 10% tolerance: a dot on each curve and one small table in the empty part of the axes
    for (arm, rule), (y, n) in at10.items():
        ax.plot([10.0], [y], marker="o", ms=3.2, color=color[arm], mec="white", mew=0.5, zorder=5)
    txt = ("within 10%%  rule A / D\n"
           "removed:     %d / %d\n"
           "default:     %d / %d") % (
        round(at10[("no_tf32_override", "A")][0] * 320), round(at10[("no_tf32_override", "D")][0] * 320),
        round(at10[("default", "A")][0] * 320), round(at10[("default", "D")][0] * 320))
    ax.text(30, 0.47, txt, fontsize=6.3, color=ns.INK, ha="left", va="center", linespacing=1.3, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(eps[0], eps[-1])
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("tolerance on MPE (%)")
    ax.set_ylabel(r"$\bar{R}(\varepsilon)$: share of the 320 runs" "\ncompleted within tolerance")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color["no_tf32_override"], lw=1.6, label=LABEL["no_tf32_override"]),
               Line2D([], [], color=color["default"], lw=1.6, label=LABEL["default"]),
               Line2D([], [], color=ns.INK, lw=1.6, label="rule D (SE(3), 95% band)"),
               Line2D([], [], color=ns.INK, lw=1.1, ls=(0, (3, 2)), label="rule A (DEIO's evaluation)")]
    ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=False, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--outdir", default="revision/analysis")
    a = ap.parse_args()
    runs = load(a.outcomes, a.scores)
    rows = table(runs, os.path.join(a.outdir, "round4_main_table_se3.csv"))
    for arm in ARMS:
        g = [x for x in runs.values() if x["arm"] == arm]
        done = [x for x in g if x["category"] == "completed"]
        print("%-17s runs %d | crashes %d | completed %d | completed with SE(3) <= 100%%: %d | with rule A <= 100%%: %d"
              % (arm, len(g), sum(x["category"] == "numerical_failure" for x in g), len(done),
                 sum(x["D"] <= 100 for x in done), sum(x["A"] <= 100 for x in done)))
        for e in (1, 10, 100):
            rd, _ = reliability(runs, arm, "D", [e])
            ra, _ = reliability(runs, arm, "A", [e])
            print("    within %4g%%: SE(3) %.3f | rule A %.3f" % (e, rd[0], ra[0]))
    os.makedirs(os.path.join(a.outdir, "figs"), exist_ok=True)
    figure(runs, os.path.join(a.outdir, "figs", "round4_reliability.pdf"))
    print("table: %s (%d rows); figure: %s" % (os.path.join(a.outdir, "round4_main_table_se3.csv"), len(rows),
                                             os.path.join(a.outdir, "figs", "round4_reliability.pdf")))


if __name__ == "__main__":
    main()
