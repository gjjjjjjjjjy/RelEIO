"""Cross-pipeline comparison table: DEIO (main) vs DEVO event-only vs loop-closure.

Reads the per_seed CSVs of three pipelines, aggregates success / crash% / median /
worst MPE per (dataset/scene, keep), and lists only the (sequence, keep) cells
covered by all three (or by the selected pipelines) for a fair within-cell comparison.

Conventions (followed in the paper):
- DEVO = IMU-off ablation (same front end, same weights), not a separate algorithm.
- No cross-pipeline absolute-accuracy ranking (different modalities/configs); only
  whether failure behaviour differs across pipelines within the same cell.
- Degradation and failure are *non-monotonic* (a mid level may be more stable than
  clean); do not describe it as a monotonic "degradation -> worse".

Usage:
  python scripts/make_crosspipeline_table.py            # default: read paper/data + cross-algorithm CSVs
  python scripts/make_crosspipeline_table.py --md out.md # also write a markdown table
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "paper", "data")

# pipeline display name -> CSV path (relative to paper/data). The main DEIO keep axis
# has no pipeline column, so it is treated as deio.
SOURCES = [
    ("DEIO",          "keep_per_seed_v2_final.csv",      "deio"),
    ("DEVO(evt-only)", "devo_eventonly_per_seed_v2.csv", "eventonly"),
    ("loop-closure",  "loopclosure_per_seed_v2_final.csv", "loopclosure"),
]


def load_csv(path, pipeline_label):
    """Read one per_seed CSV -> dict[(ds,scene,keep)] -> list of (status, mpe)."""
    rows = defaultdict(list)
    if not os.path.isfile(path):
        return rows, False
    with open(path) as f:
        for r in csv.DictReader(f):
            ds, scene, keep = r["dataset"], r["scene"], r["keep"]
            status = r.get("status", "")
            mpe = r.get("mpe", "")
            mpe = float(mpe) if mpe not in ("", None) else None
            rows[(ds, scene, keep)].append((status, mpe))
    return rows, True


def agg(seed_rows):
    """For one cell: (success/n, crash%, median, worst). crash = mpe is None or status==crash."""
    n = len(seed_rows)
    mpes = [m for (s, m) in seed_rows if m is not None and s != "crash"]
    nc = n - len(mpes)
    med = np.median(mpes) if mpes else float("nan")
    wst = max(mpes) if mpes else float("nan")
    return len(mpes), n, nc / n * 100 if n else float("nan"), med, wst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None, help="also write a markdown table to this path")
    a = ap.parse_args()

    loaded = []
    for label, fname, pl in SOURCES:
        path = os.path.join(DATA, fname)
        rows, ok = load_csv(path, pl)
        loaded.append((label, rows, ok, fname))
        print(f"[{'OK ' if ok else 'MISS'}] {label:16} <- {fname}"
              + ("" if ok else "  (file missing, this pipeline's columns left blank)"))

    # Compare only cells covered by at least 2 pipelines (preferring all three). key=(ds,scene,keep).
    all_keys = set()
    for _, rows, ok, _ in loaded:
        if ok:
            all_keys |= set(rows.keys())
    # Sort by dataset, scene, keep (descending, 1.0 -> 0.1)
    def keep_sort(k):
        try:
            return -float(k[2])
        except ValueError:
            return 0
    keys = sorted(all_keys, key=lambda k: (k[0], k[1], keep_sort(k)))

    headers = ["dataset/scene", "keep"]
    for label, _, _, _ in loaded:
        headers += [f"{label}\nsucc", "crash%", "med", "worst"]

    # Text table
    print("\n" + "=" * 110)
    print("Cross-pipeline comparison (success/n | crash% | median MPE | worst MPE)  -- within-cell comparison, not absolute ranking")
    print("=" * 110)
    line = f"{'dataset/scene':22} {'keep':5}"
    for label, _, _, _ in loaded:
        line += f" | {label:16}"
    print(line)
    print("-" * 110)

    md_rows = []
    for k in keys:
        ds, scene, keep = k
        cells = []
        line = f"{ds+'/'+scene:22} {keep:5}"
        for label, rows, ok, _ in loaded:
            if ok and k in rows:
                ns, n, cr, med, wst = agg(rows[k])
                cell = f"{ns}/{n} {cr:.0f}% {med:.2f}/{wst:.2f}"
            else:
                cell = "—"
            cells.append(cell)
            line += f" | {cell:16}"
        print(line)
        md_rows.append((f"{ds}/{scene}", keep, cells))

    print("-" * 110)
    print("Note: cell = success/n  crash%  median/worst MPE(%). '-' = this pipeline does not cover this cell.")
    print("DEVO = IMU-off ablation (not a separate algorithm). Degradation and failure are non-monotonic (a mid level may be more stable than clean).")

    if a.md:
        labels = [l for l, _, _, _ in loaded]
        with open(a.md, "w") as f:
            f.write("# Cross-pipeline comparison table (DEIO / DEVO event-only / loop-closure)\n\n")
            f.write("> Within-cell comparison (same sequence, same keep level), **not** an absolute-accuracy ranking (different modalities/configs).\n")
            f.write("> DEVO = IMU-off ablation (same front end, same weights), not a separate algorithm.\n")
            f.write("> cell = success/n - crash% - median/worst MPE(%). Degradation and failure are **non-monotonic**.\n\n")
            hdr = "| dataset/scene | keep | " + " | ".join(labels) + " |\n"
            sep = "|" + "---|" * (2 + len(labels)) + "\n"
            f.write(hdr); f.write(sep)
            for ds_scene, keep, cells in md_rows:
                f.write(f"| {ds_scene} | {keep} | " + " | ".join(cells) + " |\n")
        print(f"\n[md] wrote {a.md}")


if __name__ == "__main__":
    main()
