#!/usr/bin/env python3
"""Per-run details of a precision-pilot ledger (read-only).

For every run: arm, GPU, outcome, wall time, where the first non-finite value was DETECTED
(checkpoint, tensor, frame, whether IMU/VI initialisation had happened), how many visual-BA
calls produced non-finite output from finite input, the Cholesky probe flags (site, info,
which of S / y / dX were finite) and whether the first Cholesky flag came before or after the
first detection in the log. Ends with cross-tabulations.

  python script/diag/pilot_details.py --ledger logs/ledger/precision_pilot_20261119_144734
  ... --out revision/analysis/pilot_details.csv
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ledger  # noqa: E402

RE_CHOL = re.compile(r"^\[cholinfo\] (\{.*\"kind\":\"flag\".*\})\s*$")
RE_FIRST = re.compile(r"^\[numtrace\] (\{.*\"kind\": \"first_detected_nonfinite\".*\})\s*$")
RE_NONFIN = re.compile(r"^\[numtrace\] (\{.*\"kind\": \"nonfinite\".*\})\s*$")


def _json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def details(ledger_dir):
    rows = []
    for r in run_ledger.load_ledger(ledger_dir):
        rdir = os.path.join(ledger_dir, "runs", r["run_id"])
        facts = _json(os.path.join(rdir, "facts.json")) or {}
        man = _json(os.path.join(rdir, "manifest.json")) or {}
        chol, first_line, chol_line, later = [], None, None, []
        logp = os.path.join(rdir, "run.log")
        if os.path.exists(logp):
            with open(logp, errors="ignore") as f:
                for i, line in enumerate(f):
                    m = RE_CHOL.match(line)
                    if m:
                        chol.append(json.loads(m.group(1)))
                        chol_line = i if chol_line is None else chol_line
                    elif first_line is None and RE_FIRST.match(line):
                        first_line = i
                    elif len(later) < 3:
                        m = RE_NONFIN.match(line)
                        if m:
                            ev = json.loads(m.group(1))
                            later.append("%s:%s@%s" % (ev.get("where"), ev.get("tensor"), ev.get("frame")))
        nt = facts.get("numtrace") or {}
        fd = nt.get("first_detected_nonfinite") or {}
        rs = nt.get("run_summary") or {}
        counts = rs.get("counts") or {}
        order = None
        if chol_line is not None and first_line is not None:
            order = "chol_first" if chol_line < first_line else "detection_first"
        c0 = chol[0] if chol else {}
        rows.append({
            "run_id": r["run_id"], "arm": r["labels"].get("arm"),
            "gpu": (man.get("gpu") or {}).get("index"), "outcome": r["outcome_hint"], "mpe": r["mpe"],
            "wall_s": r["wall_s"],
            "first_where": fd.get("where"), "first_tensor": fd.get("tensor"), "first_frame": fd.get("frame"),
            "first_imu_enabled": fd.get("imu_enabled"), "first_vi_initialized": fd.get("is_initialized"),  # is_initialized = visual init (8 frames), not VI
            "next_detections": " ".join(later),
            "ba_nonfinite_from_finite_inputs": counts.get("fastba.BA:nonfinite_after_finite_inputs", 0),
            "frames_completed": rs.get("frames_completed"), "keyframes": rs.get("keyframes_n"),
            "vi_align_calls": rs.get("vi_align_calls"), "vi_initialized": rs.get("vi_initialized"),
            "chol_flags": len(chol), "chol_first_where": c0.get("where"), "chol_first_call": c0.get("call"),
            "chol_first_info": c0.get("info"), "chol_first_S_finite": c0.get("S_finite"),
            "chol_first_y_finite": c0.get("y_finite"), "chol_first_dX_finite": c0.get("dX_finite"),
            "chol_vs_detection": order,
            "chol_info_values": " ".join(sorted({str(c.get("info")) for c in chol})),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = details(a.ledger)
    if not rows:
        print("no runs in", a.ledger)
        return
    fail = [x for x in rows if x["outcome"] != "completed"]
    print("%d runs, %d not completed\n" % (len(rows), len(fail)))
    hdr = ("arm", "gpu", "outcome", "wall_s", "first_where", "first_tensor", "first_frame",
           "first_vi_initialized", "ba_nonfinite_from_finite_inputs", "chol_flags", "chol_first_where",
           "chol_first_info", "chol_first_S_finite", "chol_first_dX_finite", "chol_vs_detection", "next_detections")
    print("NOT COMPLETED:")
    for x in fail:
        print("  " + " | ".join("%s=%s" % (k, x[k]) for k in hdr))
    flagged_ok = [x for x in rows if x["outcome"] == "completed" and x["chol_flags"]]
    print("\nCOMPLETED BUT WITH CHOLESKY FLAGS: %d" % len(flagged_ok))
    for x in flagged_ok:
        print("  " + " | ".join("%s=%s" % (k, x[k]) for k in hdr if k not in ("first_where", "first_tensor", "first_frame", "next_detections")))
    print("\nCROSS-TABS (traced arms):")
    tab = collections.Counter((x["arm"], x["outcome"] == "completed", x["chol_flags"] > 0)
                              for x in rows if x["arm"] != "default_untraced")
    for arm in sorted({x["arm"] for x in rows} - {"default_untraced"}):
        print("  %-16s failed&flag=%d failed&noflag=%d completed&flag=%d completed&noflag=%d" % (
            arm, tab[(arm, False, True)], tab[(arm, False, False)], tab[(arm, True, True)], tab[(arm, True, False)]))
    print("  first detection frames (failures):", sorted(x["first_frame"] for x in fail if x["first_frame"] is not None))
    print("  at first detection, IMU in use (VI initialised):",
          collections.Counter(x["first_imu_enabled"] for x in fail if x["first_where"]),
          "| visual initialisation done:", collections.Counter(x["first_vi_initialized"] for x in fail if x["first_where"]))
    print("  Cholesky info values in flagged runs:", collections.Counter(x["chol_info_values"] for x in rows if x["chol_flags"]))
    walls = collections.defaultdict(list)
    for x in rows:
        if x["outcome"] == "completed":
            walls[x["arm"]].append(x["wall_s"])
    for arm, w in sorted(walls.items()):
        w = sorted(w)
        print("  wall time of completed runs, %-16s median %.0f s (min %.0f, max %.0f)" % (arm, w[len(w) // 2], w[0], w[-1]))
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print("\nper-run table written to", a.out)


if __name__ == "__main__":
    main()
