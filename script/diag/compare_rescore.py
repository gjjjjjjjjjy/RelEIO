#!/usr/bin/env python3
"""Reconcile a rescoring of the main matrix with the frozen per-run table (rescoring with the first-5-s rigid rule).

The rescoring (deio_se3_eval.py on logs/ledger/round4_core_main_all_20261121_023641) must reproduce the frozen table
records/deio_se3_round4_steps_20261121.csv run by run before its new column is used:
  - the same set of run ids (592), each once, with the same scene, cell, arm and seed;
  - 320 runs of the override-removed arm and 272 of the container default;
  - rules A-D (deio_rule, sim3_1000_20ms, sim3_all, se3_all) and the scale equal to 1e-9 (same code, same data).
Then it reports the new column se3_first5s: scorable and not scorable per arm and dataset, and the medians on clean
input with the override removed of rule A, the first-5-s rigid rule, and rules C and D, per sequence (for the
comparison with DEIO's published values). Exit status 1 if the reconciliation fails.

  python3 script/diag/compare_rescore.py --new records/deio_round4_scores_paper_rule_20261125.csv
"""
import argparse
import collections
import csv
import math
import sys

FROZEN = "records/deio_se3_round4_steps_20261121.csv"
RULES = ("deio_rule", "sim3_1000_20ms", "sim3_all", "se3_all", "sim3_scale")
KEYS = ("scene", "cell", "arm", "seed")


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def median(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--frozen", default=FROZEN)
    a = ap.parse_args()
    new, old = list(csv.DictReader(open(a.new))), list(csv.DictReader(open(a.frozen)))
    problems = []
    ids_new = collections.Counter(r["run"] for r in new)
    ids_old = collections.Counter(r["run"] for r in old)
    dup = [k for k, c in ids_new.items() if c > 1]
    if dup:
        problems.append("%d run ids appear more than once in the new table" % len(dup))
    if set(ids_new) != set(ids_old):
        problems.append("run ids differ: %d only new, %d only frozen" % (len(set(ids_new) - set(ids_old)),
                                                                        len(set(ids_old) - set(ids_new))))
    arms = collections.Counter(r["arm"] for r in new)
    if arms != collections.Counter({"no_tf32_override": 320, "default": 272}):
        problems.append("runs per arm %s, expected 320 override removed and 272 default" % dict(arms))
    old_by = {r["run"]: r for r in old}
    diff_keys, diff_vals = 0, collections.Counter()
    for r in new:
        o = old_by.get(r["run"])
        if o is None:
            continue
        if any(str(r[k]) != str(o[k]) for k in KEYS):
            diff_keys += 1
        for k in RULES:
            x, y = num(r.get(k)), num(o.get(k))
            if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9 * max(1.0, abs(y))):
                diff_vals[k] += 1
    if diff_keys:
        problems.append("%d runs with a different scene, cell, arm or seed" % diff_keys)
    for k, c in diff_vals.items():
        problems.append("%d runs with a different %s" % (c, k))
    print("reconciliation with %s: %d runs new, %d frozen, arms %s" % (a.frozen, len(new), len(old), dict(arms)))
    print("  rules A-D and the scale, run by run: %s" % ("identical" if not diff_vals else dict(diff_vals)))

    print("\nnew column se3_first5s (rigid alignment on the first 5 s of the estimate):")
    tally = collections.Counter((r["arm"], r["scene"].split("/")[0], num(r.get("se3_first5s")) is not None) for r in new)
    for arm in ("no_tf32_override", "default"):
        for ds in sorted({r["scene"].split("/")[0] for r in new}):
            s, u = tally[(arm, ds, True)], tally[(arm, ds, False)]
            if s or u:
                print("  %-17s %-9s scorable %3d, not scorable %3d" % (arm, ds, s, u))
    print("\nclean input, override removed: medians (%% of the path length) per sequence")
    print("  %-26s %3s %8s %10s %8s %8s" % ("sequence", "n", "rule A", "first 5 s", "rule C", "rule D"))
    groups = collections.defaultdict(list)
    for r in new:
        if r["arm"] == "no_tf32_override" and r["cell"] == "clean":
            groups[r["scene"]].append(r)
    for sc in sorted(groups):
        g = groups[sc]
        cols = [median([num(r.get(k)) for r in g]) for k in ("deio_rule", "se3_first5s", "sim3_all", "se3_all")]
        print("  %-26s %3d %s" % (sc, len(g), " ".join("%8s" % ("-" if v is None else "%.3f" % v) for v in cols)))
    print("\nRESULT: %s" % ("reconciliation passed" if not problems else "FAILED: " + "; ".join(problems)))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
