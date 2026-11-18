#!/usr/bin/env python3
"""Compare numerical crashes on common sequence/keep cells, with explicit counts."""
import argparse
from collections import defaultdict
from pathlib import Path

from analysis import OUTPUT, RECORDS, matched_systems, write_csv
from make_paper_tables import table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=OUTPUT/"tables")
    args = parser.parse_args()
    records = matched_systems(args.records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/"cross_system_cells.csv", records)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["comparison"], row["system"])].append(row)
    rows = []
    for (comparison, system), cells in grouped.items():
        n = sum(r["n"] for r in cells)
        crash = sum(r["crash"] for r in cells)
        scenes = len({(r["dataset"], r["scene"]) for r in cells})
        rows.append(f"{comparison} & {system} & {scenes} & {len(cells)} & {crash}/{n}")
    caption = (r"Numerical crashes on matched sequence/keep cells for each pair. "
               r"Counts are pooled descriptively over the listed conditions; per-cell intervals are provided separately. "
               r"Pair-specific scene sets and repeat counts differ. Extreme-completion thresholds are not shared "
               r"across MPE and ATE systems; this table supports no architecture-level causal ranking.")
    (args.out/"Table3.tex").write_text(table(caption, "tab:revised_matched", "llrrr",
        r"Comparison with DEIO & System & Scenes & Cells & Crashes/runs", rows, True), encoding="utf-8")
    for row in rows:
        print(row)
    print(f"Matched tables: {args.out.resolve()}")


if __name__ == "__main__":
    main()
