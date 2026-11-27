#!/usr/bin/env python3
"""Recompute the headline numbers of the paper from the per-run records and compare them with the paper.

Standard library only. Reads the records directory (see script/_layout.py):
  round4_all_attribution_*.csv         every run of the main matrix (640): arm, input condition, sequence, outcome
  deio_se3_round4_steps_*.csv          every completed run (592) scored under rules A-D
  hardware_*_details_*.csv             the replication runs on three further GPU models (40 runs each)
  all_rounds_attribution_*.csv         the DEIO runs of every round (1412), with ledger and input condition
  imu_bias_cross_system_pinned_*.csv   the common-gyroscope-bias runs of DEIO and SuperEIO (80)
  davis_pipelines_scores_*.csv         DAVIS240C read by DEIO's own pipeline and by our data-loading layer (40)
  deio_round4_scores_paper_rule_*.csv  the completed runs (592) with the rigid first-5-s alignment added
  scoring_faults_*.csv                 known faults injected into the ground truth of the eight sequences
  crt_round4_*.csv                     the correct rate (OpenLORIS) of every run of the main matrix (640)
  r5_eval_*.csv, r5_eval_*_readouts.json  round 5: three readers of the same perturbed recordings (260), readouts
  r5_supereio_main_*.json              round 5: acceptance record of the SuperEIO batch (100 attempts)
  r5_deio_scores_*.csv                 round 5: DEIO runs scored, RTX 4090 batch and L40S batch (160 each)
  r5_evidence/check_windows_*.log      round 5: per-window against whole-stream perturbation, windows of our layer

Arms: "default" is the container default (TF32 override set), "no_tf32_override" is the override removed.
Rules: A = DEIO's evaluation code (deio_rule), B = A with 20 ms matching (sim3_1000_20ms), C = B fitted on all
poses (sim3_all), D = C without scale, the main rule (se3_all). Errors are MPE in percent of the path length.
A crashed run counts as outside every tolerance.

  python3 script/headline_numbers.py        # prints every quantity next to the value in the paper; exit 1 on a mismatch
"""
import collections
import csv
import glob
import json
import math
import os
import re
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
for label, sub, paper in (("DAVIS240C", [r for r in off if r["scene"].startswith("rpgdavis/")], "67 / 120"),
                          ("other datasets", [r for r in off if not r["scene"].startswith("rpgdavis/")], "19 / 200")):
    check("override removed: pass rule A and fail rule D at 10%%, %s (runs / of)" % label,
          "%d / %d" % (sum(float(r["deio_rule"]) <= 10 < float(r["se3_all"]) for r in sub), len(sub)), paper)
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
sim_only = lambda system, tol: [r for r in bias if r["system"] == system   # noqa: E731
                                and float(r["sim3_W"]) <= tol < float(r["se3_W"])]
check("rescored with a similarity alignment: within 10% under it but not under the rigid one, DEIO / SuperEIO",
      (len(sim_only("DEIO", 10)), len(sim_only("SuperEIO", 10))), (17, 2))
check("  the reverse (within 10% under the rigid alignment only), both systems",
      sum(float(r["se3_W"]) <= 10 < float(r["sim3_W"]) for r in bias), 0)
check("  the two SuperEIO runs: rigid -> similarity error (%)",
      ", ".join("%.1f -> %.2f" % (float(r["se3_W"]), float(r["sim3_W"]))
                for r in sorted(sim_only("SuperEIO", 10), key=lambda r: float(r["se3_W"]))), "40.1 -> 0.75, 91.6 -> 0.50")
check("  largest similarity-alignment error of the 80 runs (%)", "%.2f" % max(float(r["sim3_W"]) for r in bias), "0.75")

print("\nScoring limit (similarity alignment fitted on all scored poses)")
check("every rule C error of the 592 completed runs at most 50% of the path length",
      all(float(r["sim3_all"]) <= 50 for r in steps), True)

print("\nDAVIS240C: DEIO's own pipeline against our data-loading layer (clean, override removed, seeds 2000-2009)")
pipe = one("davis_pipelines_scores_*.csv")
check("runs, all scored", (len(pipe), all(r["score"] == "ok" for r in pipe)), (40, True))
unit = lambda ds, seq: [r for r in pipe if r["scene"] == "%s/%s" % (ds, seq)]   # noqa: E731
check("rule D above the path length: boxes DEIO's own / ours, poster DEIO's own / ours",
      tuple(sum(float(r["se3_all"]) > 100 for r in unit(ds, seq))
            for seq in ("boxes_6dof", "poster_6dof") for ds in ("rpgdavis_up", "rpgdavis")), (0, 10, 0, 0))
pmed = lambda ds, seq, k: "%.2f" % statistics.median(float(r[k]) for r in unit(ds, seq))   # noqa: E731
check("  rule A median with DEIO's own pipeline, boxes / poster (%)",
      (pmed("rpgdavis_up", "boxes_6dof", "deio_rule"), pmed("rpgdavis_up", "poster_6dof", "deio_rule")), ("0.08", "0.08"))
check("  rigid first-5-s alignment, DEIO's own pipeline, boxes / poster (%)",
      (pmed("rpgdavis_up", "boxes_6dof", "se3_first5s"), pmed("rpgdavis_up", "poster_6dof", "se3_first5s")),
      ("1.51", "0.47"))
check("  main matrix, same seeds, our layer: boxes runs above the path length under rule D",
      sum(float(r["se3_all"]) > 100 for r in steps if r["arm"] == "no_tf32_override" and r["cell"] == "clean"
          and r["scene"] == "rpgdavis/boxes_6dof" and r["seed"] in SEEDS), 9)
pr = [r for r in one("deio_round4_scores_paper_rule_*.csv") if r["arm"] == "no_tf32_override" and r["cell"] == "clean"
      and r["scene"] == "hku/vicon_dark1"]
check("  HKU, our layer: rule A / rigid first-5-s median (%)",
      ("%.2f" % statistics.median(float(r["deio_rule"]) for r in pr),
       "%.2f" % statistics.median(float(r["se3_first5s"]) for r in pr)), ("0.06", "0.50"))

print("\nKnown faults injected into the ground truth (8 sequences)")
flt = one("scoring_faults_*.csv")
fnum = lambda x: float(x) if x not in ("", "nan", "None") else float("nan")   # noqa: E731
check("readings above 10% for stationary, frozen, and rescaled estimates, any rule",
      sum(fnum(r[k]) > 10 for r in flt if r["fault"] in ("stationary", "freeze", "scale0.5", "scale2")
          for k in ("A", "C", "D", "first5s")), 0)
check("  exponential divergence flagged at 10%: rule D / rule A (of 8 sequences)",
      tuple(sum(fnum(r[k]) > 10 for r in flt if r["fault"] == "diverge200") for k in ("D", "A")), (8, 2))
check("  late divergence within 10%: rule C / rule D (of 8 sequences)",
      tuple(sum(fnum(r[k]) <= 10 for r in flt if r["fault"] == "diverge200") for k in ("C", "D")), (8, 0))
check("  stationary estimate: within 10% under rule D, not computable under rule A / rule C",
      (sum(fnum(r["D"]) <= 10 for r in flt if r["fault"] == "stationary"),
       sum(fnum(r["A"]) != fnum(r["A"]) for r in flt if r["fault"] == "stationary"),
       sum(fnum(r["C"]) != fnum(r["C"]) for r in flt if r["fault"] == "stationary")), (8, 8, 8))
check("  readings above the path length, any fault and rule",
      sum(fnum(r[k]) > 100 for r in flt for k in ("A", "C", "D", "first5s")), 0)

print("\nCorrect rate (OpenLORIS, position criterion, one rigid alignment) on the runs of the main matrix")
crt_all = one("crt_round4_*.csv")
check("runs with a correct rate / without a trajectory (crashed)",
      (sum(r["crt_status"] == "ok" for r in crt_all), sum(r["crt_status"] != "ok" and r["outcome"] != "completed"
                                                       for r in crt_all)), (592, 48))
crt = [r for r in crt_all if r["arm"] == "no_tf32_override"]
ab = [float(r["CR_10"]) for r in crt if float(r["rule_A"]) <= 10 < float(r["rule_D"])]
win = [float(r["CR_10"]) for r in crt if float(r["rule_D"]) <= 10]
check("override removed, 10%: runs within under A but beyond under D, their median CR, max",
      (len(ab), "%.2f" % statistics.median(ab), "%.2f" % max(ab)), (86, "0.00", "0.51"))
check("  median CR of the runs within 10% under rule D", "%.2f" % statistics.median(win), "0.89")
ori = [r for r in one("crt_round4_*.csv") if r["crt_status"] == "ok" and float(r["rule_D"]) <= 10]
check("  orientation check (both arms, rule D within 10%): runs, median error, median offset spread (deg)",
      (len(ori), "%.1f" % statistics.median(float(r["aoe_median_deg_xyzw"]) for r in ori),
       "%.1f" % statistics.median(float(r["offset_spread_deg_xyzw"]) for r in ori)), (422, "76.5", "4.3"))

print("\nRound 5: three readers of the same perturbed recordings (boxes_6dof, poster_6dof), RTX 4090")
r5 = one("r5_eval_*.csv")
fn = lambda v: float(v) if v not in ("", "None") else float("nan")  # noqa: E731
check("runs / distinct run ids / per reader (DEIO own, DEIO ours, SuperEIO)",
      (len(r5), len({r["run"] for r in r5}), tuple(sum(r["reader"] == k for r in r5) for k in ("deio_own", "deio_ours", "supereio"))),
      (260, 260, (100, 60, 100)))
check("excluded runs", sum(r["status"].startswith("excluded") for r in r5), 0)
own = [r for r in r5 if r["reader"] == "deio_own"]
check("DEIO own pipeline: completed / clean runs within 10% under rule D",
      (sum(r["status"] == "completed" for r in own), "%d/%d" % (sum(fn(r["D"]) <= 10 for r in own if r["cond"] == "clean"),
                                                                sum(r["cond"] == "clean" for r in own))), (100, "20/20"))
st = [r for r in own if r["cond"] != "clean"]
ac = [r for r in st if fn(r["A"]) <= 10 < fn(r["D"]) and fn(r["C"]) <= 10 < fn(r["D"])]
check("  under stress: within 10% only under rules A and C / of the runs under stress / of them beyond the path length",
      (len(ac), len(st), sum(fn(r["D"]) > 100 for r in ac)), (27, 80, 14))
check("  under stress: within only under A but not C, or only under C but not A", sum((fn(r["A"]) <= 10) != (fn(r["C"]) <= 10) for r in st if fn(r["D"]) > 10), 0)
check("  within 10% under rule D: IMU offset boxes / poster, noise boxes / poster",
      tuple(sum(fn(r["D"]) <= 10 for r in own if r["cond"] == c and r["seq"] == q)
            for c in ("imu15000us", "noise5hz") for q in ("boxes_6dof", "poster_6dof")), (2, 3, 7, 10))
se = [r for r in r5 if r["reader"] == "supereio"]
sdone = [r for r in se if r["status"] == "completed"]
check("SuperEIO: completed / failed / undetermined", (len(sdone), sum(r["status"].startswith("failed") for r in se),
                                                       sum(r["status"].startswith("undetermined") for r in se)), (76, 19, 5))
check("  within 10% under rule C but not rule D, of the completed runs",
      "%d/%d" % (sum(fn(r["C"]) <= 10 < fn(r["D"]) for r in sdone), len(sdone)), "15/76")
check("  reduced event density: completed attempts of 20", sum(r["status"] == "completed" for r in se if r["cond"] == "keep0.1"), 0)
check("  undetermined attempts all on poster_6dof", {r["seq"] for r in se if r["status"].startswith("undetermined")}, {"poster_6dof"})
check("  undetermined with a trajectory, rule D in W (sensitivity only)",
      sorted(("%.2f" if fn(r["D"]) < 10 else "%.1f") % fn(r["D"]) for r in se
             if r["status"].startswith("undetermined") and fn(r["D"]) == fn(r["D"])), sorted(["0.62", "0.67", "1.79", "89.1"]))
check("  within 10% under rule D: IMU offset boxes / poster, noise boxes / poster",
      tuple(sum(fn(r["D"]) <= 10 for r in se if r["cond"] == c and r["seq"] == q and r["status"] == "completed")
            for c in ("imu15000us", "noise5hz") for q in ("boxes_6dof", "poster_6dof")), (10, 7, 10, 2))
worst = max((r for r in sdone if r["seq"] == "boxes_6dof" and r["cond"] == "clean"), key=lambda r: fn(r["D"]))
check("  clean boxes_6dof run with the largest rule D error: rule C / rule D (%)", ("%.2f" % fn(worst["C"]), "%d" % round(fn(worst["D"]))), ("0.75", "265"))
done = [r for r in r5 if r["status"] == "completed"]
check("all readers: completed trajectories / within under C but not D / the reverse (C and, for DEIO, A)",
      (len(done), sum(fn(r["C"]) <= 10 < fn(r["D"]) for r in done),
       sum(fn(r["D"]) <= 10 < fn(r["C"]) for r in done) + sum(fn(r["D"]) <= 10 < fn(r["A"]) for r in done if r["reader"] != "supereio")),
      (236, 75, 0))
ours_tq = [r for r in r5 if r["reader"] == "deio_ours" and r["seq"] == "poster_6dof" and r["cond"] == "tq10000us"]
mm_tq = [r for r in steps if r["arm"] == "no_tf32_override" and r["cell"] == "tq:10000" and r["scene"] == "rpgdavis/poster_6dof"]
check("poster_6dof, timestamp quantization, our layer, beyond the path length: round 5 / main matrix",
      ("%d/%d" % (sum(fn(r["D"]) > 100 for r in ours_tq), len(ours_tq)), "%d/%d" % (sum(float(r["se3_all"]) > 100 for r in mm_tq), len(mm_tq))),
      ("1/10", "9/10"))
side = json.load(open(sorted(glob.glob(os.path.join(REC, "r5_eval_*_readouts.json")))[0]))
check("common window boxes / poster (s), readers that set it",
      tuple("%.1f" % ((w["w1_us"] - w["w0_us"]) / 1e6) for w in (side["window"]["boxes_6dof"], side["window"]["poster_6dof"]))
      + (sorted(side["window"]["boxes_6dof"]["readers"]),), ("57.7", "57.7", ["deio_ours", "deio_own", "supereio"]))
acc = json.load(open(sorted(glob.glob(os.path.join(REC, "r5_supereio_main_*.json")))[0]))
sc = acc["scheduling_check"]
check("SuperEIO batch valid; scheduling check: checkpoints / median time-limited solves",
      (acc["valid"], "%d/%d" % (sc["with_checkpoint_700"], sc["planned"]), sc["median"]), (True, "10/10", 38.5))
check("  engines identical to the bias experiment's / distinct run folders",
      (acc["engines_vs_bias_experiment"]["identical"], acc["distinct_run_folders"]), (True, 100))
l40 = one("r5_deio_scores_*_132849.csv")
r4090 = one("r5_deio_scores_*_152156.csv")
check("DEIO runs scored: RTX 4090 batch / L40S batch (kept as a record)", (len(r4090), len(l40)), (160, 160))
check("  RTX 4090 batch: the same runs as in r5_eval_*.csv", {r["run"] for r in r4090} == {r["run"] for r in r5 if r["reader"] != "supereio"}, True)
pipes = one("davis_pipelines_scores_*.csv")
bias = one("imu_bias_cross_system_pinned_*.csv")
check("released records: DEIO runs / SuperEIO runs",
      (len(allr) + len(pipes) + len(r4090) + len(l40), sum(r["system"] == "SuperEIO" for r in bias) + len(se)), (1772, 140))

for seq, paper in (("boxes_6dof", "5.45"), ("poster_6dof", "5.72")):
    log = open(sorted(glob.glob(os.path.join(REC, "r5_evidence", "check_windows_%s_*.log" % seq)))[0]).read()
    m = re.search(r"tq 10000 us: of (\d+) events in a window before or after, (\d+) crossed to another window, "
                  r"(\d+) left all windows, (\d+) entered one", log)
    n, moved = int(m.group(1)), int(m.group(2)) + int(m.group(3)) + int(m.group(4))
    check("%s: changed window membership, per-window vs whole-stream quantization (%% of events in a window)" % seq,
          "%.2f" % (100.0 * moved / n), paper)

print("\n%d quantities checked, %d mismatch%s%s" % (rows_checked, len(mismatches), "" if len(mismatches) == 1 else "es",
                                                   (": " + "; ".join(mismatches)) if mismatches else ""))
sys.exit(1 if mismatches else 0)
