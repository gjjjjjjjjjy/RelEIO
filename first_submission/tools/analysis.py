"""Descriptive analyses shared by the record, table and figure entry points.

Q2/Q3 reuse runs: pair counts are not independent experimental sample sizes.
"""
from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
OUTPUT = ROOT / "reports"
THRESHOLD = Decimal("100")
AXES = {
    "keep": ("deio_keep.csv", "keep", ("1.0", "0.5", "0.3", "0.1"), "Event density"),
    "patch": ("deio_patch.csv", "patch", ("96", "64", "32", "16"), "Patch budget"),
    "time": ("deio_time.csv", "tq_us", ("0", "1000", "5000", "10000"), "Timestamp quantization"),
}
SCENES = {
    ("vector", "desk_normal1"): "VECtor/desk",
    ("vector", "corner_slow1"): "VECtor/corner",
    ("hku", "vicon_dark1"): "HKU/dark",
    ("rpgdavis", "boxes_6dof"): "DAVIS/boxes",
    ("rpgdavis", "poster_6dof"): "DAVIS/poster",
    ("tumvie", "mocap-3d-trans"): "TUM-VIE/mocap",
    ("rpgdavis", "shapes_6dof"): "DAVIS/shapes",
    ("fpv", "indoor_forward_3"): "UZH-FPV/indoor",
}
SYSTEMS = {
    "DEVO (IMU-off)": ("devo.csv", "mpe", "seed"),
    "DEIO + loop closure": ("loopclosure.csv", "mpe", "seed"),
    "RAMP-VO": ("rampvo.csv", "ate", "seed"),
    "E2VID + TartanVO": ("e2vid_tartanvo.csv", "ate", "rep"),
}


@dataclass(frozen=True)
class Run:
    dataset: str
    scene: str
    level: str
    repeat: str
    status: str
    error: Decimal | None


def read_records(path, level_column, metric="mpe", repeat_column="seed"):
    cells = defaultdict(list)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        required = {"dataset", "scene", level_column, repeat_column, "status", metric}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[c] is None for c in required):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            key = (row["dataset"], row["scene"], row[level_column])
            identity = (*key, row[repeat_column])
            if any(value == "" for value in identity):
                raise ValueError(f"{path}:{line}: missing run identifier")
            if identity in seen:
                raise ValueError(f"{path}:{line}: duplicate run {identity}")
            seen.add(identity)
            status = row["status"]
            if status == "ok":
                if row[metric] == "":
                    raise ValueError(f"{path}:{line}: completed run is missing {metric}")
                error = Decimal(row[metric])
                if not error.is_finite() or error < 0:
                    raise ValueError(f"{path}:{line}: completed run needs finite nonnegative {metric}")
            elif status == "crash":
                if row[metric] != "":
                    raise ValueError(f"{path}:{line}: crash has a populated {metric}")
                error = None
            else:
                raise ValueError(f"{path}:{line}: unknown status {status!r}")
            cells[key].append(Run(*key, row[repeat_column], status, error))
    if not cells:
        raise ValueError(f"{path}: no runs")
    return dict(cells)


def load_main(records=RECORDS):
    axes = {}
    for axis, (filename, column, levels, _) in AXES.items():
        cells = read_records(Path(records) / filename, column)
        expected = {(*scene, level) for scene in SCENES for level in levels}
        if set(cells) != expected:
            raise ValueError(f"{filename}: missing cells {sorted(expected-set(cells))}; unexpected cells {sorted(set(cells)-expected)}")
        axes[axis] = cells
    return axes


def outcome(run, threshold=THRESHOLD):
    if run.status == "crash":
        return "crash"
    if run.error > threshold:
        return "extreme"
    return "within_threshold"


def wilson(failures, n):
    if n <= 0 or failures < 0 or failures > n:
        raise ValueError("Wilson interval needs 0 <= failures <= n and n > 0")
    z = 1.959963984540054
    p = failures / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def percentile(values, q):
    values = sorted(values)
    position = Decimal(str(q)) * (len(values)-1)
    left = int(position)
    if left == len(values)-1:
        return values[left]
    fraction = position-left
    return values[left]*(1-fraction) + values[left+1]*fraction


def summarize(runs, threshold=THRESHOLD):
    if len(runs) == 0:
        raise ValueError("Cannot summarize an empty condition")
    valid = [r.error for r in runs if outcome(r, threshold) == "within_threshold"]
    crash = sum(r.status == "crash" for r in runs)
    extreme = sum(outcome(r, threshold) == "extreme" for r in runs)
    low, high = wilson(crash+extreme, len(runs))
    return {
        "n": len(runs), "within_threshold": len(valid), "crash": crash,
        "extreme": extreme, "failure_rate": (crash+extreme)/len(runs),
        "failure_ci_low": low, "failure_ci_high": high,
        "median_mpe": float(st.median(valid)) if valid else None,
        "worst_mpe": float(max(valid)) if valid else None,
        "iqr_mpe": float(percentile(valid, .75)-percentile(valid, .25)) if len(valid) >= 2 else None,
    }


def q1(cells):
    partial = []
    for runs in cells.values():
        s = summarize(runs)
        if 0 < s["within_threshold"] < s["n"]:
            partial.append(s["within_threshold"]/s["n"])
    return {"partial_cells": len(partial), "cells": len(cells),
            "mean_miss_fraction": st.mean(partial) if partial else None}


def direction_counts(clean, degraded):
    # Same-pool median reference; completed-run accuracy, not total reliability.
    c = [r.error for r in clean if r.status == "ok"]
    d = [r.error for r in degraded if r.status == "ok"]
    counts = {"pairs": len(c)*len(d), "opposite": 0, "agree": 0, "ties": 0,
              "reference_unavailable": int(len(c) == 0 or len(d) == 0)}
    if counts["reference_unavailable"]:
        return counts
    reference = st.median(d)-st.median(c)
    for x in c:
        for y in d:
            delta = y-x
            if reference == 0 or delta == 0:
                counts["ties"] += 1
            elif reference*delta < 0:
                counts["opposite"] += 1
            else:
                counts["agree"] += 1
    return counts


def q3(cells, clean_level, degraded_level):
    rows = [direction_counts(cells[(*scene, clean_level)], cells[(*scene, degraded_level)]) for scene in SCENES]
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    directed = totals["agree"]+totals["opposite"]
    totals["directed_pairs"] = directed
    totals["opposite_fraction"] = totals["opposite"]/directed if directed else None
    totals["clean_level"], totals["degraded_level"] = clean_level, degraded_level
    return totals


def q2(cells, clean_level):
    clean = {scene: {r.repeat: r for r in cells[(*scene, clean_level)]} for scene in SCENES}
    repeats = sorted({repeat for scene in clean.values() for repeat in scene})
    rows = []
    for a, b in combinations(repeats, 2):
        counts = {"seed_a": a, "seed_b": b, "opposite": 0, "agree": 0, "ties": 0, "unavailable": 0}
        for x, y in combinations(SCENES, 2):
            if any(seed not in clean[scene] for seed in (a, b) for scene in (x, y)):
                counts["unavailable"] += 1
                continue
            runs = [clean[x][a], clean[y][a], clean[x][b], clean[y][b]]
            if any(run.status == "crash" for run in runs):
                counts["unavailable"] += 1
                continue
            da, db = runs[0].error-runs[1].error, runs[2].error-runs[3].error
            if da == 0 or db == 0:
                counts["ties"] += 1
            elif da*db < 0:
                counts["opposite"] += 1
            else:
                counts["agree"] += 1
        n = counts["agree"]+counts["opposite"]
        counts["opposite_fraction"] = counts["opposite"]/n if n else None
        rows.append(counts)
    rates = [row["opposite_fraction"] for row in rows if row["opposite_fraction"] is not None]
    return {"median_opposite_fraction": st.median(rates) if rates else None, "seed_pairs": rows}


def success_curve(runs):
    thresholds = sorted({Decimal(0), *(r.error for r in runs if r.status == "ok")})
    return [{"threshold_mpe": float(t), "successes": sum(r.status == "ok" and r.error <= t for r in runs),
             "n": len(runs)} for t in thresholds]


def matched_systems(records=RECORDS):
    main = read_records(Path(records)/"deio_keep.csv", "keep")
    pairs = []
    for name, (filename, metric, repeat) in SYSTEMS.items():
        other = read_records(Path(records)/filename, "keep", metric, repeat)
        keys = sorted(set(main).intersection(other))
        if not keys:
            raise ValueError(f"DEIO and {name} have no common condition cells")
        for key in keys:
            for system, runs, unit in [("DEIO", main[key], "MPE (%)"), (name, other[key], "MPE (%)" if metric == "mpe" else "ATE (m)")]:
                crash = sum(r.status == "crash" for r in runs)
                low, high = wilson(crash, len(runs))
                completed = [r.error for r in runs if r.status == "ok"]
                pairs.append({"comparison": name, "system": system, "dataset": key[0], "scene": key[1],
                              "keep": key[2], "n": len(runs), "crash": crash,
                              "crash_rate": crash/len(runs), "ci_low": low, "ci_high": high,
                              "metric": unit, "completed_median": float(st.median(completed)) if completed else None})
    return pairs


def report(records=RECORDS):
    axes = load_main(records)
    cells, thresholds = [], []
    summaries, pooled = {}, {}
    for axis, data in axes.items():
        levels = AXES[axis][2]
        summaries[axis] = {"q1": q1(data), "q2": q2(data, levels[0]),
                           "q3": [q3(data, levels[0], level) for level in levels[1:]]}
        for key, runs in data.items():
            pooled[(axis, *key)] = runs
            cells.append({"axis": axis, "dataset": key[0], "scene": key[1], "level": key[2], **summarize(runs)})
        for t in (50, 100, 200):
            outcomes = [summarize(runs, Decimal(t)) for runs in data.values()]
            thresholds.append({"axis": axis, "threshold_mpe": t,
                               "n": sum(s["n"] for s in outcomes), "crash": sum(s["crash"] for s in outcomes),
                               "extreme": sum(s["extreme"] for s in outcomes)})
    first = [summaries[axis]["q3"][0] for axis in AXES]
    combined = {k: sum(row[k] for row in first) for k in ("pairs", "opposite", "agree", "ties", "directed_pairs")}
    n = combined["directed_pairs"]
    combined["opposite_fraction"] = combined["opposite"]/n if n else None
    return {
        "definitions": {
            "outcome": "crash; completed MPE > 100%; completed MPE <= 100%",
            "q1": "Mean observed success fraction within observed partial-failure cells; conditional descriptive statistic",
            "q2": "Clean scene-error ordering reversals between seed pairs; completed runs; ties separate; ordinary median",
            "q3": "Strict opposite completed-run MPE direction relative to same-pool medians; ties separate; not independent validation",
            "q3_uncertainty": "Pairs reuse runs. No pair-level binomial or bootstrap confidence interval is reported.",
            "q1_uncertainty": "Former cell-bootstrap interval omitted within-cell uncertainty and shared-sequence dependence; not used as an inferential interval.",
            "cross_system": "Numerical crash only on common sequence/keep cells. Different error metrics remain separate.",
        },
        "totals": {"cells": len(cells), **{k: sum(s[k] for s in cells) for k in ("n", "within_threshold", "crash", "extreme")}},
        "q1": q1(pooled), "q3_first_degraded": combined, "axes": summaries,
        "cells": cells, "threshold_sensitivity": thresholds, "matched_systems": matched_systems(records),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"{path}: no rows to write")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
