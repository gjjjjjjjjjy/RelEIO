#!/usr/bin/env python3
"""Run an explicit adapter command per condition; preserve every process outcome."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import string
import subprocess
import time

LEVELS = {"keep": ("1.0", "0.5", "0.3", "0.1"),
          "time": ("0", "1000", "5000", "10000"),
          "patch": ("96", "64", "32", "16")}


def command_fields(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def execute(*, template, system, scenes, seeds, axes, output):
    tokens = shlex.split(template)
    if not tokens:
        raise ValueError("Empty adapter command")
    fields = set().union(*(command_fields(token) for token in tokens))
    required = {"scene", "seed"}
    required.update({"keep": "keep", "time": "tq_us", "patch": "patch"}[axis] for axis in axes)
    allowed = {"scene", "seed", "keep", "tq_us", "patch", "output"}
    if required-fields:
        raise ValueError(f"Command must explicitly consume {sorted(required-fields)}")
    if fields-allowed:
        raise ValueError(f"Unknown command fields: {sorted(fields-allowed)}")
    if len(set(scenes)) != len(scenes) or len(set(seeds)) != len(seeds):
        raise ValueError("Scenes and seeds must be unique")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    columns = ["job", "system", "axis", "scene", "seed", "keep", "tq_us", "patch",
               "started_utc", "wall_seconds", "exit_code"]
    failures = 0
    with (output/"execution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        job = 0
        for scene in scenes:
            for axis in axes:
                for level in LEVELS[axis]:
                    for seed in seeds:
                        params = {"scene": scene, "seed": str(seed), "keep": "1.0", "tq_us": "0", "patch": "96"}
                        params[{"keep": "keep", "time": "tq_us", "patch": "patch"}[axis]] = level
                        directory = output/f"{job:05d}"
                        directory.mkdir()
                        params["output"] = str(directory)
                        argv = [token.format(**params) for token in tokens]
                        if shutil.which(argv[0]) is None:
                            raise FileNotFoundError(f"Executable unavailable: {argv[0]}")
                        started = datetime.now(timezone.utc).isoformat()
                        (directory/"command.json").write_text(json.dumps(
                            {"system": system, "axis": axis, "parameters": params, "argv": argv},
                            indent=2)+"\n", encoding="utf-8")
                        before = time.monotonic()
                        with (directory/"stdout.log").open("w") as stdout, (directory/"stderr.log").open("w") as stderr:
                            process = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
                        seconds = time.monotonic()-before
                        writer.writerow({"job": job, "system": system, "axis": axis,
                                         **{k: params[k] for k in ["scene", "seed", "keep", "tq_us", "patch"]},
                                         "started_utc": started, "wall_seconds": f"{seconds:.6f}", "exit_code": process.returncode})
                        stream.flush()
                        if process.returncode != 0:
                            failures += 1
                            print(f"job {job}: exit {process.returncode}; evidence in {directory}", flush=True)
                        job += 1
    print(f"{job} processes; {failures} nonzero exits. Execution log: {output/'execution.csv'}")
    # An exit code needs diagnosis; it is not automatically a numerical crash.
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="argv template, no shell evaluation")
    parser.add_argument("--system", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--axes", nargs="+", choices=tuple(LEVELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(template=args.command, system=args.system, scenes=args.scenes,
                             seeds=args.seeds, axes=args.axes, output=args.out))


if __name__ == "__main__":
    main()
