#!/usr/bin/env python3
"""Round 4 main table, by the rules fixed before the run (revision/DIAG_PILOT_RUNBOOK.md section 12). Read-only.

Counted: runs whose controls verify (the same checks as the matrix summary) in the main ledger. Determinate = completed
or numerical failure; other outcomes are listed, never counted as successes. Per condition (sequence x cell) and arm:
  numerical failures / determinate runs with the Wilson 95% interval; for 0 failures "not observed in n" and the upper
  bound (no "0%"); completed runs with MPE above the threshold (100%, sensitivity 50 and 200), reported separately;
  the difference default - no_tf32_override with Newcombe's hybrid score interval (Newcombe 1998, method 10);
  MPE paired by seed, only pairs where both runs completed.
Interaction, per sequence and axis (8 x 3 = 24 groups, all reported, exploratory): the difference between arms at the
strongest level of the axis minus the difference on clean input, 95% interval by MOVER (Zou & Donner 2008). Only a
group whose interval excludes 0 may be described as "a larger difference on this sequence and axis".
Pooled rows give equal weight to every condition (each has the same K); they are not a population estimate.
Check of the paired design: both arms share sequence, cell and seed, so the difference is also given with Newcombe's
interval for paired proportions (Newcombe 1998b, method 10). When one arm has no failures its correlation term is
0 and the paired interval equals the unpaired one; the script reports how many conditions differ.
Medians and quartiles: linear interpolation (the median of an even count is the mean of the two middle values).
Detection: with independent runs and a per-run failure probability of 10%, K runs show at least one failure with
probability 1 - 0.9^K (K = 10: 65.1%). This is a detection statement, not the precision of a rate.

  python script/diag/round4_analysis.py --ledger logs/ledger/round4_core_main_all_20261121_023641 \\
      --out revision/analysis/round4_main_table.csv
MPE in the ledger is DEIO's own evaluation (Sim(3) fitted on the first 1000 associated poses, error over all of them);
see script/supereio/eval_traj.py for why that rule is not used across systems.
"""
import argparse
import collections
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_pilot as P  # noqa: E402
import run_ledger  # noqa: E402

ARMS = ("default", "no_tf32_override")
THETA = 100.0
THETAS = (50.0, 100.0, 200.0)
AXES = (("keep", "keep:0.1"), ("tq", "tq:10000"), ("patch", "patch:16"))


def newcombe(k1, n1, k2, n2):
    """p1 - p2 with Newcombe's hybrid score interval (method 10), from the two Wilson intervals."""
    if not n1 or not n2:
        return (float("nan"),) * 3
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = P.wilson(k1, n1)
    l2, u2 = P.wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def newcombe_paired(a, b, c, d):
    """Paired difference p1 - p2 = (b - c) / n from the 2x2 table a (both fail), b (only arm 1 fails), c (only arm 2
    fails), d (neither); Newcombe (1998b) method 10: Wilson intervals of the margins with the phi correction.
    With phi = 0 (e.g. one arm without failures) it equals newcombe() for n1 = n2 = n (tested); the phi != 0 branch
    has not been checked against a published example."""
    n = a + b + c + d
    if not n:
        return (float("nan"),) * 3
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = P.wilson(a + b, n)
    l2, u2 = P.wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    d_ = p1 - p2
    return d_, d_ - dl, d_ + du


def mover(a, b):
    """theta_a - theta_b for independent estimates a = (est, lo, hi), b = (est, lo, hi) (Zou & Donner 2008)."""
    (t1, l1, u1), (t2, l2, u2) = a, b
    d = t1 - t2
    return d, d - math.sqrt((t1 - l1) ** 2 + (u2 - t2) ** 2), d + math.sqrt((u1 - t1) ** 2 + (t2 - l2) ** 2)


def collect(ledger):
    rows, excluded = [], collections.Counter()
    for r in run_ledger.load_ledger(ledger):
        lab = r["labels"]
        arm, cat = lab.get("arm"), run_ledger.outcome_category(r["outcome_hint"])
        if arm not in ARMS:
            excluded["arm %s" % arm] += 1
            continue
        if cat == "not_run":
            excluded["not run"] += 1
            continue
        f = P._load_facts(ledger, r["run_id"])
        probs = ["facts.json missing"] if f is None else P.verify_controls(
            f, arm, bool(lab.get("cholinfo")), cell=lab.get("cell"))
        if probs:
            excluded["control check failed: " + probs[0][:60]] += 1
            continue
        rows.append({"run_id": r["run_id"], "arm": arm, "seed": lab.get("seed"), "cell": lab.get("cell"),
                     "scene": "%s/%s" % (lab.get("dataset"), lab.get("scene")), "category": cat,
                     "mpe": r.get("mpe"), "wall_s": r.get("wall_s")})
    return rows, excluded


def stats(g, theta=THETA):
    det = [x for x in g if x["category"] in ("completed", "numerical_failure")]
    fail = sum(x["category"] == "numerical_failure" for x in det)
    done = [x for x in det if x["category"] == "completed"]
    above = sum(1 for x in done if x["mpe"] is not None and x["mpe"] > theta)
    mpes = [x["mpe"] for x in done if x["mpe"] is not None]
    return {"n": len(det), "fail": fail, "above": above, "other": len(g) - len(det),
            "mpe_med": q(mpes, 0.5) if mpes else None}


def rate(k, n):
    if not n:
        return "-"
    lo, hi = P.wilson(k, n)
    if k == 0:
        return "0/%d (not observed; upper %.0f%%)" % (n, 100 * hi)
    return "%d/%d [%.0f, %.0f]%%" % (k, n, 100 * lo, 100 * hi)


def paired(rows, key):
    by = collections.defaultdict(dict)
    for x in rows:
        if x["category"] == "completed" and x["mpe"] is not None:
            by[key(x) + (x["seed"],)][x["arm"]] = x["mpe"]
    return [v["no_tf32_override"] - v["default"] for v in by.values() if len(v) == 2]


def q(v, p):
    """Quantile with linear interpolation between order statistics (numpy's default)."""
    v = sorted(v)
    if not v:
        return float("nan")
    h = p * (len(v) - 1)
    i = int(math.floor(h))
    return v[i] if i + 1 >= len(v) else v[i] + (h - i) * (v[i + 1] - v[i])


def paired_table(g):
    """a, b, c, d over seeds where both arms are determinate (arm 1 = default, arm 2 = no_tf32_override)."""
    by = collections.defaultdict(dict)
    for x in g:
        if x["category"] in ("completed", "numerical_failure"):
            by[(x["scene"], x["cell"], x["seed"])][x["arm"]] = x["category"] == "numerical_failure"
    t = collections.Counter()
    for v in by.values():
        if len(v) == 2:
            t[(v["default"], v["no_tf32_override"])] += 1
    return t[(True, True)], t[(True, False)], t[(False, True)], t[(False, False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", help="per-condition CSV")
    a = ap.parse_args()
    recs = run_ledger.load_ledger(a.ledger)
    rows, excluded = collect(a.ledger)
    print("ledger %s | records %d | counted %d | GPUs %s | code %s" % (
        a.ledger, len(recs), len(rows), sorted({r["gpu_name"] for r in recs}),
        sorted({(r["git_commit"] or "")[:10] for r in recs})))
    for k, n in sorted(excluded.items()):
        print("  excluded %4d  %s" % (n, k))
    others = collections.Counter(x["category"] for x in rows if x["category"] not in ("completed", "numerical_failure"))
    if others:
        print("  not determinate (kept out of every denominator):", dict(others))
    seqs = sorted({x["scene"] for x in rows})
    cells = ["clean"] + [c for _, c in AXES]
    grp = collections.defaultdict(list)
    for x in rows:
        grp[(x["scene"], x["cell"], x["arm"])].append(x)
    k_per = collections.Counter(len(v) for v in grp.values())
    print("runs per condition and arm:", dict(k_per), "| detection 1-0.9^K:",
          {k: "%.1f%%" % (100 * (1 - 0.9 ** k)) for k in sorted(k_per)})

    def block(title, key_rows):
        print("\n" + title)
        print("  %-34s %-30s %-30s %-24s %-9s %-9s %-15s" % (
            "condition", "failures default", "failures no_tf32_override", "difference [95% Newcombe]",
            "above def", "above off", "MPE med def/off"))
        for label, g_def, g_off in key_rows:
            s1, s2 = stats(g_def), stats(g_off)
            d, lo, hi = newcombe(s1["fail"], s1["n"], s2["fail"], s2["n"])
            med = "%s/%s" % tuple("-" if s["mpe_med"] is None else "%.2f" % s["mpe_med"] for s in (s1, s2))
            print("  %-34s %-30s %-30s %+5.0f%% [%+4.0f, %+4.0f]%%      %-9d %-9d %-15s" % (
                label, rate(s1["fail"], s1["n"]), rate(s2["fail"], s2["n"]), 100 * d, 100 * lo, 100 * hi,
                s1["above"], s2["above"], med))
            yield label, s1, s2, (d, lo, hi)

    out_rows = []
    total = [("all conditions", [x for x in rows if x["arm"] == "default"],
              [x for x in rows if x["arm"] == "no_tf32_override"])]
    list(block("POOLED (equal weight per condition; 'above' = completed with MPE > %g%%)" % THETA, total))
    list(block("BY CELL, pooled over sequences",
               [(c, [x for x in rows if x["cell"] == c and x["arm"] == "default"],
                 [x for x in rows if x["cell"] == c and x["arm"] == "no_tf32_override"]) for c in cells]))
    diffs = {}
    for label, s1, s2, d in block("BY SEQUENCE AND CELL", [("%s %s" % (s, c), grp[(s, c, "default")],
                                                             grp[(s, c, "no_tf32_override")])
                                                            for s in seqs for c in cells]):
        s, c = label.split(" ")
        diffs[(s, c)] = d
        out_rows.append({"scene": s, "cell": c, "n_default": s1["n"], "fail_default": s1["fail"],
                         "above_default": s1["above"], "n_off": s2["n"], "fail_off": s2["fail"],
                         "above_off": s2["above"], "diff": d[0], "diff_lo": d[1], "diff_hi": d[2],
                         "mpe_med_default": s1["mpe_med"], "mpe_med_off": s2["mpe_med"]})

    print("\nPAIRED-DESIGN CHECK (pairs by sequence, cell and seed; Newcombe 1998b method 10)")
    groups = [("all conditions", rows)] + [(c, [x for x in rows if x["cell"] == c]) for c in cells]
    for label, g in groups:
        a_, b_, c_, d_ = paired_table(g)
        pd, plo, phi_ = newcombe_paired(a_, b_, c_, d_)
        print("  %-16s pairs %3d (both fail %d, only default %d, only off %d) | paired %+5.1f%% [%+5.1f, %+5.1f]%%" % (
            label, a_ + b_ + c_ + d_, a_, b_, c_, 100 * pd, 100 * plo, 100 * phi_))
    same, worst = 0, 0.0
    for s_ in seqs:
        for c in cells:
            a_, b_, c_, d_ = paired_table(grp[(s_, c, "default")] + grp[(s_, c, "no_tf32_override")])
            _, plo, phi_ = newcombe_paired(a_, b_, c_, d_)
            _, ulo, uhi = diffs[(s_, c)]
            dev = max(abs(plo - ulo), abs(phi_ - uhi))
            same += dev < 1e-12
            worst = max(worst, dev)
    print("  per condition: paired interval equal to the unpaired one in %d of %d; largest difference %.1f points"
          % (same, len(seqs) * len(cells), 100 * worst))

    print("\nINTERACTION (exploratory, all 24 groups): [difference at the strongest level] - [difference on clean], MOVER 95%")
    print("  %-30s %-6s %-22s %-22s %-26s" % ("sequence", "axis", "difference strongest", "difference clean",
                                               "interaction [95%]"))
    for s in seqs:
        for axis, c in AXES:
            if (s, c) not in diffs or (s, "clean") not in diffs:
                continue
            dd, dc = diffs[(s, c)], diffs[(s, "clean")]
            i, lo, hi = mover(dd, dc)
            flag = "  interval excludes 0" if (lo > 0 or hi < 0) else ""
            print("  %-30s %-6s %+5.0f%% [%+4.0f,%+4.0f]     %+5.0f%% [%+4.0f,%+4.0f]     %+5.0f%% [%+4.0f, %+4.0f]%s" % (
                s, axis, 100 * dd[0], 100 * dd[1], 100 * dd[2], 100 * dc[0], 100 * dc[1], 100 * dc[2],
                100 * i, 100 * lo, 100 * hi, flag))

    print("\nCOMPLETED RUNS ABOVE THE MPE THRESHOLD (counted separately from numerical failures)")
    done = [x for x in rows if x["category"] == "completed" and x["mpe"] is not None]
    for t in THETAS:
        print("  MPE > %g%%: default %d, no_tf32_override %d (of %d / %d completed)" % (
            t, sum(x["mpe"] > t for x in done if x["arm"] == "default"),
            sum(x["mpe"] > t for x in done if x["arm"] == "no_tf32_override"),
            sum(x["arm"] == "default" for x in done), sum(x["arm"] == "no_tf32_override" for x in done)))
    for x in sorted((x for x in done if x["mpe"] > min(THETAS)), key=lambda x: (x["scene"], x["cell"], x["arm"])):
        print("    %-30s %-10s %-17s seed %s  MPE %.2f%%" % (x["scene"], x["cell"], x["arm"], x["seed"], x["mpe"]))

    print("\nMPE PAIRED BY SEED (no_tf32_override - default, pairs where both completed)")
    for c in ["all"] + cells:
        v = paired([x for x in rows if c == "all" or x["cell"] == c], lambda x: (x["scene"], x["cell"]))
        if v:
            print("  %-10s pairs %3d | median %+.3f | IQR [%+.3f, %+.3f] | equal at the logged precision %d" % (
                c, len(v), q(v, 0.5), q(v, 0.25), q(v, 0.75), sum(abs(d) < 1e-9 for d in v)))
    walls = {arm: sorted(x["wall_s"] for x in done if x["arm"] == arm and x["wall_s"]) for arm in ARMS}
    print("\nWALL TIME of completed runs: " + " | ".join(
        "%s median %.0f s (%.0f-%.0f)" % (arm, q(w, 0.5), w[0], w[-1]) for arm, w in walls.items() if w))
    if a.out and out_rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("\nper-condition table written to", a.out)


if __name__ == "__main__":
    main()
