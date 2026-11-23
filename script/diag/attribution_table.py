#!/usr/bin/env python3
"""Failure-attribution counts over several ledgers, computed from the run records (read-only).

Each run is counted once (unique run_id over all given ledgers; a duplicate is reported and skipped).
Counted: traced runs (the untraced reference arm has no tracer and no Cholesky probe) that are
control-valid (the same checks as the matrix summary) and have a determinate outcome (completed or
numerical failure). Smoke ledgers are not passed in. Everything excluded is listed with its reason.

Per ledger and in total, split by whether TF32 matrix multiplication was on inside the visual BA:
  runs, numerical failures, failures with a Cholesky flag, completed runs with a flag,
  first detection site of each failure, IMU in use at first detection,
  wall time and GPU memory of completed runs (median, min, max): the PyTorch allocator peaks recorded
  by numtrace (all ledgers), and from ledger-4 on the device memory sampled by nvidia-smi every 30 s
  (includes the CUDA context; the largest sample, a lower bound of the peak).
Wall times are of traced runs: the tracer synchronises the GPU at its checkpoints.

  python script/diag/attribution_table.py --ledgers logs/ledger/precision_pilot_20261119_144734 \\
      logs/ledger/round2A_20261120_022120 ... --out revision/analysis/attribution_table.csv
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_details as D  # noqa: E402
import precision_pilot as P  # noqa: E402
import run_ledger  # noqa: E402

# TF32 matrix multiplication inside the visual BA call, per arm (container override variable set)
BA_TF32 = {"default": True, "amp_off": True, "cudnn_tf32_off": True,
           "fp32_strict": False, "tf32_off": False, "matmul_tf32_off": False,
           "ba_matmul_fp32": False, "no_tf32_override": False}


def _stats(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return "-"
    return "%.0f (%.0f-%.0f)" % (v[len(v) // 2], v[0], v[-1])


def collect(ledgers):
    seen, rows, excluded = set(), [], collections.Counter()
    for ld in ledgers:
        recs = {r["run_id"]: r for r in run_ledger.load_ledger(ld)}
        if not recs:
            excluded["%s: no ledger.jsonl or no runs" % ld] += 1
            continue
        for d in D.details(ld):
            rid, r = d["run_id"], recs[d["run_id"]]
            name = os.path.basename(ld.rstrip("/"))
            if rid in seen:
                excluded["duplicate run_id (already counted from another ledger)"] += 1
                continue
            seen.add(rid)
            cat = run_ledger.outcome_category(r["outcome_hint"])
            arm = r["labels"].get("arm")
            if arm == P.UNTRACED:
                excluded["%s: untraced reference arm" % name] += 1
                continue
            if arm not in BA_TF32:
                excluded["%s: arm %s not classified" % (name, arm)] += 1
                continue
            if cat not in ("completed", "numerical_failure"):
                excluded["%s: outcome %s" % (name, cat)] += 1
                continue
            f = P._load_facts(ld, rid)
            probs = ["facts.json missing"] if f is None else P.verify_controls(
                f, arm, bool(r["labels"].get("cholinfo")), cell=r["labels"].get("cell"))
            if probs:
                excluded["%s: control check failed" % name] += 1
                continue
            rs = ((f or {}).get("numtrace") or {}).get("run_summary") or {}
            d.update(ledger=name, ba_tf32=BA_TF32[arm], category=cat, cell=r["labels"].get("cell") or "base",
                     scene="%s/%s" % (r["labels"].get("dataset"), r["labels"].get("scene")),
                     peak_alloc_mb=rs.get("peak_mem_allocated_mb"), peak_reserved_mb=rs.get("peak_mem_reserved_mb"),
                     device_mem_mib=((f or {}).get("gpu_watch") or {}).get("mem_used_mib_max"))
            rows.append(d)
    return rows, excluded


def table(rows, key):
    groups = collections.defaultdict(list)
    for x in rows:
        groups[key(x)].append(x)
    print("  %-44s %6s %6s %10s %10s  %-26s %-14s %-16s %-16s %-16s %-16s" % (
        "group", "runs", "fail", "fail+flag", "done+flag", "first detection (failures)", "IMU at first",
        "wall s done", "peak alloc MB", "peak reserved MB", "device MiB"))
    for k in sorted(groups, key=str):
        g = groups[k]
        fail = [x for x in g if x["category"] == "numerical_failure"]
        done = [x for x in g if x["category"] == "completed"]
        where = collections.Counter(x["first_where"] for x in fail)
        imu = collections.Counter(x["first_imu_enabled"] for x in fail)
        print("  %-44s %6d %6d %10d %10d  %-26s %-14s %-16s %-16s %-16s %-16s" % (
            k, len(g), len(fail), sum(1 for x in fail if x["chol_flags"]), sum(1 for x in done if x["chol_flags"]),
            " ".join("%s:%d" % kv for kv in sorted(where.items(), key=str)),
            " ".join("%s:%d" % kv for kv in sorted(imu.items(), key=str)),
            _stats(x["wall_s"] for x in done), _stats(x["peak_alloc_mb"] for x in done),
            _stats(x["peak_reserved_mb"] for x in done), _stats(x["device_mem_mib"] for x in done)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, excluded = collect(a.ledgers)
    print("counted runs: %d (unique run_id, traced, control-valid, completed or numerical failure)" % len(rows))
    for k, n in sorted(excluded.items()):
        print("  excluded %4d  %s" % (n, k))
    tf = lambda x: "BA TF32 on " if x["ba_tf32"] else "BA TF32 off"  # noqa: E731
    print("\nPER LEDGER:")
    table(rows, lambda x: "%s | %s" % (x["ledger"][:30], tf(x)))
    print("\nPER LEDGER AND CELL TYPE (clean = keep 1.0, tq 0, 96 patches):")
    def clean(x):
        c = P.parse_cell(x["cell"])
        return "clean" if (c["keep"], c["tq_us"], c["patches"]) == (1.0, 0, P.CLEAN_PATCHES) else "degraded"
    table(rows, lambda x: "%s | %s | %s" % (x["ledger"][:22], clean(x), tf(x)))
    print("\nTOTAL:")
    table(rows, tf)
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nper-run table written to", a.out)


if __name__ == "__main__":
    main()
