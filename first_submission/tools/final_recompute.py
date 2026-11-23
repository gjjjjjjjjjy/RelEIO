#!/usr/bin/env python3
"""Recompute descriptive results from records/, independent of working directory."""
import argparse
import json
from pathlib import Path

from analysis import AXES, OUTPUT, RECORDS, load_main, report, success_curve, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = report(args.records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(args.out/"conditions.csv", result["cells"])
    write_csv(args.out/"threshold_sensitivity.csv", result["threshold_sensitivity"])
    write_csv(args.out/"matched_systems.csv", result["matched_systems"])
    curves = []
    for axis, cells in load_main(args.records).items():
        for key, runs in cells.items():
            for row in success_curve(runs):
                curves.append({"axis": axis, "dataset": key[0], "scene": key[1], "level": key[2], **row})
    write_csv(args.out/"success_curves.csv", curves)
    print(json.dumps({key: result[key] for key in ["totals", "q1", "q3_first_degraded"]}, indent=2))
    for axis in AXES:
        print(axis, "Q2 clean-scene ordering:", result["axes"][axis]["q2"]["median_opposite_fraction"])
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
