#!/usr/bin/env python3
"""Recompute the headline numbers of the paper from the per-run records and compare them with the paper.

Standard library only. Reads the records directory (see script/_layout.py):
  round4_all_attribution_*.csv         every run of the main matrix (640): arm, input condition, sequence, outcome
  deio_se3_round4_steps_*.csv          every completed run (592) scored under rules A-D
  hardware_*_details_*.csv             the replication runs on three further GPU models (40 runs each)
  all_rounds_attribution_*.csv         the DEIO runs of every round (1412), with ledger and input condition
  imu_bias_cross_system_pinned_*.csv   the common-gyroscope-bias runs of DEIO and SuperEIO (80)

Arms: "default" is the container default (TF32 override set), "no_tf32_override" is the override removed.
Rules: A = DEIO's evaluation code (deio_rule), B = A with 20 ms matching (sim3_1000_20ms), C = B fitted on all
poses (sim3_all), D = C without scale, the main rule (se3_all). Errors are MPE in percent of the path length.
A crashed run counts as outside every tolerance.

  python3 script/headline_numbers.py        # prints every quantity next to the value in the paper; exit 1 on a mismatch
"""
import collections
import csv
import glob
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _layout  # noqa: E402

REC = _layout.RECORDS
ARMS = ("default", "no_tf32_override")
CELLS = ("clean", "keep:0.1", "tq:10000", "patch:16")
RULES = collections.OrderedDict([("A", "deio_rule"), ("B", "sim3_1000_20ms"), ("C", "sim3_all"), ("D", "se3_all")])
SEEDS = [str(s) for s in range(2000, 2010)]
rows_checked, mismatches = 0, []


def one(pattern):
    f = sorted(glob.glob(os.path.join(REC, pattern)))
    assert len(f) == 1, "expected one file for %s in %s, found %s" % (pattern, REC, f)
    return list(csv.DictReader(open(f[0])))


def check(label, got, paper):
    """Print one quantity; `paper` is the value stated in the manuscript."""
    global rows_checked
    rows_checked += 1
    ok = got == paper
    if not ok:
        mismatches.append(label)
    print("  %-72s %-22s paper: %-22s %s" % (label, got, paper, "ok" if ok else "MISMATCH"))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def newcombe(k1, n1, k2, n2):
    """Newcombe hybrid score interval for p2 - p1 (method 10), in percentage points."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    return 100 * d, 100 * (d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)), 100 * (d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def newcombe_paired(e, f, g, h):
    """Newcombe interval for the difference of two paired proportions (1998, method 10), in percentage points.
    e: both, f: first only, g: second only, h: neither; the difference is first minus second."""
    n = e + f + g + h
    p1, p2 = (e + f) / n, (e + g) / n
    l1, u1 = wilson(e + f, n)
    l2, u2 = wilson(e + g, n)
    a = (e + f) * (g + h) * (e + g) * (f + h)
    b = e * h - f * g
    phi = 0.0 if a == 0 else (b - n / 2 if b > n / 2 else 0.0 if b >= 0 else b) / math.sqrt(a)
    lo = math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return 100 * (p1 - p2), 100 * (p1 - p2 - lo), 100 * (p1 - p2 + hi)


def pct(lo, hi):
    return "[%d, %d]" % (round(100 * lo), round(100 * hi))


att = one("round4_all_attribution_*.csv")
steps = one("deio_se3_round4_steps_*.csv")
crashed = lambda r: r["outcome"].startswith("numerical_failure")   # noqa: E731

print("Main matrix: outcomes (%s)" % _layout.rel(REC))
check("runs", len(att), 640)
check("distinct run ids", len({r["run_id"] for r in att}), 640)
check("outcomes other than completed / numerical failure", sum(not (crashed(r) or r["outcome"] == "completed") for r in att), 0)
for arm, paper in zip(ARMS, (48, 0)):
    n = sum(r["arm"] == arm for r in att)
    k = sum(r["arm"] == arm and crashed(r) for r in att)
    check("crashes, %s" % arm, "%d/%d" % (k, n), "%d/320" % paper)
lo, hi = wilson(48, 320)
check("  Wilson 95% CI of 48/320 (%)", pct(lo, hi), "[12, 19]")
check("  upper bound of 0/320 (%)", "%.1f" % (100 * wilson(0, 320)[1]), "1.2")
d, dl, du = newcombe(0, 320, 48, 320)
check("  difference between arms (pp), Newcombe 95% CI", "%d [%d, %d]" % (round(d), round(dl), round(du)), "15 [11, 19]")
for cell, paper in zip(CELLS, (10, 19, 6, 13)):
    k = sum(r["arm"] == "default" and r["cell"] == cell and crashed(r) for r in att)
    check("crashes, default, %s" % cell, "%d/80" % k, "%d/80" % paper)
check("sequences with a crash (default)", len({r["scene"] for r in att if r["arm"] == "default" and crashed(r)}), 8)
check("completed runs with a score", len(steps), 592)
check("visual-inertial alignment completed: completed runs / crashed runs",
      (sum(r["vi_initialized"] == "True" for r in att if r["outcome"] == "completed"), sum(r["vi_initialized"] == "True" for r in att if crashed(r))),
      (592, 0))
check("  all of them are completed runs of the matrix", {r["run"] for r in steps} == {r["run_id"] for r in att if r["outcome"] == "completed"}, True)

print("\nReliability: runs within a tolerance, of 320 per arm")
by_arm = {a: [r for r in steps if r["arm"] == a] for a in ARMS}
for arm, rule, tol, paper in [("no_tf32_override", "D", 10, 227), ("no_tf32_override", "D", 1, 136), ("no_tf32_override", "D", 100, 251),
                              ("no_tf32_override", "A", 10, 313), ("no_tf32_override", "A", 1, 246), ("no_tf32_override", "A", 100, 317),
                              ("default", "D", 10, 195), ("default", "A", 10, 267)]:
    k = sum(float(r[RULES[rule]]) <= tol for r in by_arm[arm])
    check("%s, rule %s, within %d%%" % (arm, rule, tol), k, paper)
k = sum(float(r["se3_all"]) <= 10 for r in by_arm["default"])
check("default, rule D, within 10%: share of the completed runs / of all 320 runs (%)",
      "%.1f / %.1f" % (100.0 * k / len(by_arm["default"]), 100.0 * k / 320), "71.7 / 60.9")
off = by_arm["no_tf32_override"]
check("override removed: pass rule A and fail rule D at 10%", sum(float(r["deio_rule"]) <= 10 < float(r["se3_all"]) for r in off), 86)
check("override removed: pass rule D and fail rule A at 10%", sum(float(r["se3_all"]) <= 10 < float(r["deio_rule"]) for r in off), 0)
for rule, paper in zip(RULES, (3, 3, 0, 69)):
    check("override removed: runs above the path length, rule %s" % rule, sum(float(r[RULES[rule]]) > 100 for r in off), paper)
check("override removed: rule D error below the stationary baseline", sum(float(r["se3_all"]) < float(r["gt_spread"]) for r in off), 173)
spread = [float(r["gt_spread"]) for r in steps]
check("ground-truth spread of the completed runs: smallest / largest (% of the path length) / largest at most 4.05",
      ("%.2f" % min(spread), "%.2f" % max(spread), max(spread) <= 4.05), ("0.37", "4.05", True))
hidden = collections.Counter((r["scene"].split("/")[1], r["cell"]) for r in off if float(r["se3_all"]) > 100 and float(r["deio_rule"]) <= 10)
check("override removed: above the path length under rule D and within 10% under rule A, boxes_6dof / all sequences",
      (sum(v for (q, _), v in hidden.items() if q == "boxes_6dof"), sum(hidden.values())), (37, 62))
check("  on poster_6dof and shapes_6dof / of these under timestamp quantization",
      (sum(v for (q, _), v in hidden.items() if q in ("poster_6dof", "shapes_6dof")),
       sum(v for (q, c), v in hidden.items() if q in ("poster_6dof", "shapes_6dof") and c == "tq:10000")), (22, 13))
check("  on desk_normal1 (rule A does not report the ground-truth spread there)", sum(v for (q, _), v in hidden.items() if q == "desk_normal1"), 0)
rule_a = [float(r["deio_rule"]) for r in off if r["scene"].startswith("rpgdavis/") and float(r["se3_all"]) > 100]
check("override removed, DAVIS240C runs above the path length under rule D: runs / rule A smallest, largest (%) / at most 1%",
      (len(rule_a), "%.2f" % min(rule_a), "%.2f" % max(rule_a), sum(x <= 1 for x in rule_a)), (59, "0.48", "1.92", 58))
same = lambda scene: sum((float(r["deio_rule"]) <= 10) == (float(r["se3_all"]) <= 10) for r in off if r["scene"] == scene)   # noqa: E731
check("override removed: runs with the same verdict of rules A and D at 10%, TUM-VIE / HKU (of 40)",
      (same("tumvie/mocap-3d-trans"), same("hku/vicon_dark1")), (40, 39))
box = {arm: [statistics.median(float(r["deio_rule"]) for r in by_arm[arm] if r["scene"] == "rpgdavis/boxes_6dof" and r["cell"] == c) for c in CELLS]
       for arm in ARMS}
check("boxes_6dof, rule A medians over the four input conditions (%): override removed / both arms",
      ("%.2f-%.2f" % (min(box["no_tf32_override"]), max(box["no_tf32_override"])),
       "%.2f-%.2f" % (min(box["default"] + box["no_tf32_override"]), max(box["default"] + box["no_tf32_override"]))), ("0.73-0.80", "0.73-0.82"))

print("\nTable IV: runs within 10% per input condition, of 80 (removed D / A; default D / A / crashes)")
table4 = {"clean": (67, 79, 63, 70, 10), "keep:0.1": (47, 75, 26, 57, 19), "tq:10000": (56, 80, 55, 74, 6), "patch:16": (57, 79, 51, 66, 13)}
for cell in CELLS:
    got = tuple(sum(float(r[RULES[rule]]) <= 10 for r in by_arm[arm] if r["cell"] == cell)
                for arm, rule in [("no_tf32_override", "D"), ("no_tf32_override", "A"), ("default", "D"), ("default", "A")])
    got += (sum(r["arm"] == "default" and r["cell"] == cell and crashed(r) for r in att),)
    check(cell, got, table4[cell])
d, dl, du = newcombe(67, 80, 47, 80)
check("  reduced event density, override removed, rule D (pp)", "%d [%d, %d]" % (round(d), round(dl), round(du)), "-25 [-38, -11]")
d, dl, du = newcombe(67, 80, 57, 80)
check("  reduced patch budget, override removed, rule D (pp)", "%.1f [%.0f, %+.1f]" % (d, dl, du), "-12.5 [-25, +0.5]")
# sensitivity check, not pre-specified: the same comparison on pairs of runs with the same sequence and seed
within = {(r["scene"], r["cell"], r["seed"]): float(r["se3_all"]) <= 10 for r in off}
pairs = sorted({(s, sd) for s, _, sd in within})
for cell, paper in [("keep:0.1", (80, 20, 0, "0.000")), ("tq:10000", (80, 16, 5, "0.027")), ("patch:16", (80, 13, 3, "0.021"))]:
    b = sum(within[(s, "clean", sd)] and not within[(s, cell, sd)] for s, sd in pairs)
    c = sum(within[(s, cell, sd)] and not within[(s, "clean", sd)] for s, sd in pairs)
    p = min(1.0, 2.0 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))   # exact McNemar test
    check("  paired, %s: pairs / within 10%% on clean only / under the condition only / p" % cell, (len(pairs), b, c, "%.3f" % p), paper)

print("\nPaired by sequence, input condition and seed")
key = lambda r: (r["scene"], r["cell"], r["seed"])   # noqa: E731
score = {arm: {key(r): float(r["se3_all"]) for r in by_arm[arm]} for arm in ARMS}
both = sorted(set(score["default"]) & set(score["no_tf32_override"]))
check("pairs completed in both arms", len(both), 272)
check("  within 10% with the override, not without it", sum(score["default"][k] <= 10 < score["no_tf32_override"][k] for k in both), 14)
check("  within 10% without the override, not with it", sum(score["no_tf32_override"][k] <= 10 < score["default"][k] for k in both), 11)
check("  median paired difference of the rule D error (pp)",
      "%.3f" % statistics.median(score["no_tf32_override"][k] - score["default"][k] for k in both), "-0.001")
lost = [k for k in score["no_tf32_override"] if k not in score["default"]]        # the seeds whose default run crashed
check("crashed with the override", len(lost), 48)
keys = set(score["no_tf32_override"]) | set(score["default"])                    # every sequence, input condition and seed
cd, cr = keys - set(score["default"]), keys - set(score["no_tf32_override"])     # crashed with / without the override
d, dl, du = newcombe_paired(len(cd & cr), len(cd - cr), len(cr - cd), len(keys - cd - cr))
check("  crash difference between arms over %d pairs (pp), Newcombe interval for paired proportions" % len(keys),
      "%d [%d, %d]" % (round(d), round(dl), round(du)), "15 [11, 19]")
check("  of these, within 10% without the override", sum(score["no_tf32_override"][k] <= 10 for k in lost), 35)
check("every condition has ten seeds per arm among completed + crashed runs",
      all({r["seed"] for r in off if (r["scene"], r["cell"]) == sc} == set(SEEDS) for sc in {(r["scene"], r["cell"]) for r in off}), True)

print("\nReplication on three further GPU models (corner_slow1, two input conditions, 20 runs per arm and GPU)")
for tag, name, paper in [("l40s", "L40S", 10), ("h200", "H200", 10), ("h100", "H100", 11)]:
    hw = one("hardware_%s_*_details_*.csv" % tag)
    k = collections.Counter(r["arm"] for r in hw if crashed(r))
    n = collections.Counter(r["arm"] for r in hw)
    check("%s: crashes default / override removed" % name,
          "%d/%d, %d/%d" % (k["default"], n["default"], k["no_tf32_override"], n["no_tf32_override"]), "%d/20, 0/20" % paper)

print("\nAll rounds (%d runs in all_rounds_attribution_*.csv)" % len(one("all_rounds_attribution_*.csv")))
allr = one("all_rounds_attribution_*.csv")
check("runs / distinct run ids", (len(allr), len({r["run_id"] for r in allr})), (1412, 1412))
check("outcomes other than completed / numerical failure (all rounds)", sum(not (crashed(r) or r["outcome"] == "completed") for r in allr), 0)
check("main-matrix rows identical to round4_all_attribution_*.csv",
      {r["run_id"]: r for r in allr if r["ledger"].startswith("round4_")} == {r["run_id"]: r for r in att}, True)


def count(ledger, arm, scene=None, cell=None):
    g = [r for r in allr if r["ledger"].startswith(ledger) and r["arm"] == arm
         and (scene is None or r["scene"] == scene) and (cell is None or r["cell"] == cell)]
    return "%d/%d" % (sum(crashed(r) for r in g), len(g))


print("  Table III, precision switches (round 2C, desk_normal1, clean)")
for arm, label, paper in [("default", "container default", "8/16"), ("cudnn_tf32_off", "cuDNN TF32 off", "8/16"),
                          ("no_tf32_override", "override removed", "0/16"), ("matmul_tf32_off", "matrix-multiplication TF32 off", "0/16"),
                          ("ba_matmul_fp32", "TF32 off in the BA call only", "0/16")]:
    check("  %s" % label, count("round2C_", arm), paper)
seen = collections.defaultdict(set)   # outcomes per GPU of the machine, container default (one seed, identical input)
for r in allr:
    if r["ledger"].startswith("round2C_") and r["arm"] == "default":
        seen[r["gpu"]].add(crashed(r))
check("  container default: GPUs used / GPUs with both a completed and a crashed run", (len(seen), sum(len(v) == 2 for v in seen.values())), (8, 5))
print("  Table S1, diagnostic rounds")
for ledger, arm, label, paper in [("precision_pilot_", "default", "round 1, default", "10/24"), ("precision_pilot_", "fp32_strict", "round 1, full FP32", "0/24"),
                                  ("round2A_", "default", "round 2A, default", "5/16"), ("round2A_", "amp_off", "round 2A, mixed precision off", "1/16"),
                                  ("round2A_", "tf32_off", "round 2A, TF32 off", "0/16"), ("round2A_", "fp32_strict", "round 2A, full FP32", "0/16"),
                                  ("round2B_", "default", "round 2B, default", "7/70"), ("round2B_", "fp32_strict", "round 2B, full FP32", "0/70"),
                                  ("round3a_", "default", "round 3A, default", "41/140"), ("round3a_", "no_tf32_override", "round 3A, override removed", "0/140")]:
    check("  %s" % label, count(ledger, arm), paper)
g = [r for r in allr if r["ledger"].startswith("round3a_") and r["arm"] == "default"]
check("  round 3A, default: degraded / clean",
      "%d/%d, %d/%d" % (sum(crashed(r) for r in g if r["cell"] != "keep:1.0"), sum(r["cell"] != "keep:1.0" for r in g),
                        sum(crashed(r) for r in g if r["cell"] == "keep:1.0"), sum(r["cell"] == "keep:1.0" for r in g)), "35/90, 6/50")
print("  Table S2, round 3A per sequence and condition (default; override removed is 0/10 in every cell)")
table_s2 = [("vector/corner_slow1", "keep:0.1", 8), ("vector/corner_slow1", "keep:0.3", 6), ("vector/corner_slow1", "patch:16", 5),
            ("vector/corner_slow1", "patch:64", 2), ("vector/corner_slow1", "keep:1.0", 1), ("fpv/indoor_forward_3", "patch:64", 4),
            ("fpv/indoor_forward_3", "keep:1.0", 2), ("rpgdavis/boxes_6dof", "patch:32", 2), ("rpgdavis/boxes_6dof", "keep:1.0", 2),
            ("rpgdavis/poster_6dof", "patch:16", 6), ("rpgdavis/poster_6dof", "patch:32", 2), ("rpgdavis/poster_6dof", "keep:1.0", 1),
            ("rpgdavis/shapes_6dof", "patch:16", 0), ("rpgdavis/shapes_6dof", "keep:1.0", 0)]
check("  cells in round 3A", len({(r["scene"], r["cell"]) for r in allr if r["ledger"].startswith("round3a_")}), 14)
for scene, cell, paper in table_s2:
    check("  %s, %s" % (scene, cell), (count("round3a_", "default", scene, cell), count("round3a_", "no_tf32_override", scene, cell)),
          ("%d/10" % paper, "0/10"))
print("  Table S3, four GPU models (corner_slow1; reduced event density, clean: default / override removed)")
tot = collections.Counter()
for ledger, name, pd, pc in [("round3a_", "RTX 4090", 8, 1), ("hardware_l40s_", "L40S", 9, 1), ("hardware_h200_", "H200", 7, 3), ("hardware_h100_", "H100", 9, 2)]:
    got = tuple(count(ledger, arm, "vector/corner_slow1", cell) for cell in ("keep:0.1", "keep:1.0") for arm in ("default", "no_tf32_override"))
    check("  %s" % name, got, ("%d/10" % pd, "0/10", "%d/10" % pc, "0/10"))
    tot["d"] += int(got[0].split("/")[0])
    tot["c"] += int(got[2].split("/")[0])
check("  total with the override: reduced event density, clean", "%d/40, %d/40" % (tot["d"], tot["c"]), "33/40, 7/40")
mem = lambda arm: statistics.median(float(r["peak_reserved_mb"]) for r in allr   # noqa: E731
                                    if r["ledger"].startswith("precision_pilot_") and r["arm"] == arm and r["outcome"] == "completed")
check("  reserved GPU memory, default / full FP32 (MiB, round 1)", "%.0f / %.0f" % (mem("default"), mem("fp32_strict")), "2114 / 4744")

print("\nCommon gyroscope bias (rule D on the common time window)")
bias = one("imu_bias_cross_system_pinned_*.csv")
check("runs, all completed", (len(bias), all(r["status"] == "completed" for r in bias)), (80, True))
med = lambda system, seq, cond: statistics.median(float(r["se3_W"]) for r in bias   # noqa: E731
                                                  if (r["system"], r["seq"], r["cond"]) == (system, seq, cond))
check("SuperEIO boxes_6dof median, biased / clean (%)",
      "%.2f / %.2f" % (med("SuperEIO", "boxes_6dof", "gbias"), med("SuperEIO", "boxes_6dof", "clean")), "5.37 / 4.24")

print("\n%d quantities checked, %d mismatch%s%s" % (rows_checked, len(mismatches), "" if len(mismatches) == 1 else "es",
                                                   (": " + "; ".join(mismatches)) if mismatches else ""))
sys.exit(1 if mismatches else 0)
