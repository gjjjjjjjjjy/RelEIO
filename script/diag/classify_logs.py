#!/usr/bin/env python3
"""Classify existing run logs by failure site and reconcile them with a frozen CSV.

Read-only. Uses the same parsing rules as the new run ledger (run_ledger.parse_facts),
so historical and new runs are labelled identically.

Examples (from the repository root):
  # all logs under a directory, summary by outcome hint and exception site
  python3 script/diag/classify_logs.py --logs logs/sweep_v2
  # reconcile with the frozen per-run CSV (one row per dataset/scene/level/seed)
  python3 script/diag/classify_logs.py --logs logs/sweep_v2 --csv paper/data/keep_per_seed_v2_final.csv --axis keep
  python3 script/diag/classify_logs.py --logs logs/time_sweep_v2 --csv paper/data/time_per_seed_v2_final.csv --axis tq
  python3 script/diag/classify_logs.py --logs logs/patch_sweep_v2 --csv paper/data/patch_per_seed_v2_final.csv --axis patch
  # write the per-log table
  ... --out records/keep_log_facts.csv

Reconciliation reports, per CSV row: no log found / several candidate logs / log says
completed but CSV says crash (or the reverse) / MPE mismatch / non-numerical failure
(timeout, infrastructure, other exception) recorded as a crash. It does not edit anything.
"""
import argparse
import collections
import csv
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ledger import parse_facts, outcome_category, CATEGORIES, file_sha256  # noqa: E402

NAME = re.compile(r"^(?P<ds>[a-z0-9]+)_(?P<scene>.+)_(?P<axis>keep|tq|patch)(?P<level>[0-9.]+)_seed(?P<seed>\d+)"
                  r"(?P<rest>.*)\.log(?:\.gz)?$")
LOG_SUFFIXES = (".log", ".log.gz")
CSV_LEVEL_COL = {"keep": "keep", "tq": "tq_us", "patch": "patch"}


def scan(logdir):
    rows = []
    for root, _, files in os.walk(logdir):
        for fn in sorted(files):
            if not fn.endswith(LOG_SUFFIXES):
                continue
            path = os.path.join(root, fn)
            opener = gzip.open if fn.endswith(".gz") else open
            with opener(path, "rt", errors="ignore") as lf:
                f = parse_facts(lf.read())
            m = NAME.match(fn)
            exc = f.get("exception") or {}
            rows.append({
                "path": path, "ds": m["ds"] if m else None, "scene": m["scene"] if m else None,
                "axis": m["axis"] if m else None, "level": m["level"] if m else None,
                "seed": m["seed"] if m else None, "name_suffix": m["rest"] if m else None,
                "outcome_hint": f["outcome_hint"], "outcome_category": outcome_category(f["outcome_hint"]),
                "mpe": f["mpe"], "ape_rmse_m": f["ape_rmse_m"],
                "exc_type": exc.get("type"), "exc_site": exc.get("site"),
                "vi_init_success": f["counts"]["vi_init_success"],
                "nan_gravity_print": f["counts"]["nan_gravity_print"],
                "warning_ba_failed": f["counts"]["warning_ba_failed"],
                "mtime": int(os.path.getmtime(path)),
            })
    return rows


def _norm_level(v):
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return str(v)


def reconcile(rows, csv_path, axis, theta=100.0):
    col = CSV_LEVEL_COL[axis]
    index = collections.defaultdict(list)
    for r in rows:
        if r["axis"] == axis:
            index[(r["ds"], r["scene"], _norm_level(r["level"]), r["seed"])].append(r)
    issues = collections.Counter()
    details = []
    n = 0
    with open(csv_path) as fh:
        csv_rows = list(csv.DictReader(fh))
    for c in csv_rows:
        n += 1
        key = (c["dataset"], c["scene"], _norm_level(c[col]), c["seed"])
        cands = index.get(key, [])
        if not cands:
            issues["no_log_found"] += 1
            continue
        if len(cands) > 1:
            issues["several_candidate_logs"] += 1
            details.append(("several_candidate_logs", key, [x["path"] for x in cands]))
        log = cands[0]
        csv_crash = c["status"] != "ok"
        hint = log["outcome_hint"]
        if csv_crash and hint == "completed":
            issues["csv_crash_but_log_completed"] += 1
            details.append(("csv_crash_but_log_completed", key, log["path"]))
        elif not csv_crash and hint != "completed":
            issues["csv_ok_but_log_not_completed"] += 1
            details.append(("csv_ok_but_log_not_completed", key, log["path"], hint))
        elif csv_crash:
            issues["crash:" + hint] += 1
            if not hint.startswith("numerical_failure"):
                details.append(("crash_not_numerical", key, log["path"], hint))
        else:
            issues["ok:completed"] += 1
            if c["mpe"] and log["mpe"] is not None and abs(float(c["mpe"]) - log["mpe"]) > 0.006:
                issues["mpe_mismatch"] += 1
                details.append(("mpe_mismatch", key, c["mpe"], log["mpe"], log["path"]))
    return n, issues, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--axis", choices=sorted(CSV_LEVEL_COL))
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.path.isdir(a.logs):
        ap.error("not a directory: %s" % a.logs)
    rows = scan(a.logs)
    print("logs scanned: %d under %s (exit codes unknown for historical logs)" % (len(rows), a.logs))
    if not rows:
        other = collections.Counter(os.path.splitext(fn)[1] or fn for _, _, fs in os.walk(a.logs) for fn in fs)
        print("  no *.log / *.log.gz files here. Files present by extension: %s" % (dict(other.most_common(10)) or "none"))
    cats = collections.Counter(r["outcome_category"] for r in rows)
    print("  by category: " + "  ".join("%s=%d" % (c, cats.get(c, 0)) for c in CATEGORIES))
    by = collections.Counter((r["outcome_hint"], r["exc_site"]) for r in rows)
    for (hint, site), k in by.most_common():
        print("  %4d  %-48s %s" % (k, hint, site or ""))
    fail = [r for r in rows if r["outcome_hint"].startswith("numerical_failure")]
    if fail:
        early = sum(1 for r in fail if r["vi_init_success"] == 0)
        print("numerical failures: %d, of which %d before any successful VI initialisation" % (len(fail), early))
    if a.out and not rows:
        print("per-log table not written (no logs)")
    elif a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("per-log table written to", a.out)
    if a.csv:
        if not a.axis:
            ap.error("--axis is required with --csv")
        n, issues, details = reconcile(rows, a.csv, a.axis)
        print("\nreconciliation against %s (%d rows, sha256 %s):" % (a.csv, n, file_sha256(a.csv)))
        for k, v in sorted(issues.items()):
            print("  %-45s %d" % (k, v))
        for d in details[:40]:
            print("  ", d)
        if len(details) > 40:
            print("   ... %d more" % (len(details) - 40))


if __name__ == "__main__":
    main()
