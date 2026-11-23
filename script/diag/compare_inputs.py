#!/usr/bin/env python3
"""Do two ledgers (e.g. two GPU platforms) feed bit-identical input to the estimator for one condition?

Uses every traced run of the condition (dataset/scene, and keep / tq from the cell; the eval seed,
arm and patch count do not change the input) in both ledgers, and requires positive evidence:
  - each ledger has at least one run with input checkpoints;
  - the two ledgers share at least one checkpoint frame;
  - every checkpoint frame reached by two or more runs (from either ledger) has one value;
  - the runs that got furthest agree on the frame count and the whole-input fingerprint.
Exit code 0: identical, with the deepest frame compared across the two ledgers printed;
1: the input differs; 2: not enough evidence (missing ledger, no traced runs, no shared checkpoint).

  python script/diag/compare_inputs.py --ledgers logs/ledger/hw_x_smoke logs/ledger/round3a_20261120_062351 \\
      --scene vector/corner_slow1 --cell keep:0.1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_matrix as M  # noqa: E402
import precision_pilot as P  # noqa: E402
import run_ledger  # noqa: E402


def runs_for(ledger, scene, cell):
    want = P.parse_cell(cell)
    out = []
    if not os.path.isdir(ledger):
        return None
    for r in run_ledger.load_ledger(ledger):
        lb = r["labels"]
        if "%s/%s" % (lb.get("dataset"), lb.get("scene")) != scene:
            continue
        c = P.parse_cell(lb.get("cell"))
        if (c["keep"], c["tq_us"]) != (want["keep"], want["tq_us"]):
            continue
        f = P._load_facts(ledger, r["run_id"]) or {}
        rs = (f.get("numtrace") or {}).get("run_summary") or {}
        if rs.get("input_checkpoints"):
            out.append(dict(rs, _completed=run_ledger.outcome_category(r.get("outcome_hint") or "") == "completed"))
    return out


def compare(ledger_a, ledger_b, scene, cell):
    a, b = runs_for(ledger_a, scene, cell), runs_for(ledger_b, scene, cell)
    if a is None or b is None:
        return 2, "ledger not found: %s" % (ledger_a if a is None else ledger_b)
    if not a or not b:
        return 2, "no traced runs with input checkpoints for %s %s in %s" % (scene, cell, ledger_a if not a else ledger_b)
    keys_a = set().union(*(set(int(k) for k in x["input_checkpoints"]) for x in a))
    keys_b = set().union(*(set(int(k) for k in x["input_checkpoints"]) for x in b))
    cross = keys_a & keys_b
    c = M.input_consistency(a + b)
    # runs that completed on each platform: same frame count and a non-empty, equal whole-input fingerprint
    done = [x for x in a + b if x["_completed"]]
    done_a, done_b = [x for x in a if x["_completed"]], [x for x in b if x["_completed"]]
    frames = {x.get("input_frames") for x in done}
    fps = {x.get("input_fingerprint") for x in done}
    empty = sum(1 for x in done if not x.get("input_fingerprint"))
    text = "%s %s: %d + %d runs (completed %d + %d); deepest checkpoint compared across the two ledgers: frame %s" % (
        scene, cell, len(a), len(b), len(done_a), len(done_b), max(cross) if cross else None)
    if c["differs_at"]:
        return 1, text + "; DIFFERENT at frames %s" % c["differs_at"]
    if done and (len(frames) > 1 or len(fps - {None, ""}) > 1):
        return 1, text + "; completed runs differ (frames %s, %d fingerprints)" % (sorted(frames, key=str), len(fps))
    if not cross:
        return 2, text + "; no shared checkpoint, nothing compared"
    if empty:
        return 2, text + "; %d completed run(s) without a whole-input fingerprint" % empty
    if done_a and done_b:
        return 0, text + "; completed runs on both: %d frames, one fingerprint; identical" % frames.pop()
    return 0, text + "; no completed run in %s: only the common prefix up to frame %d is verified" % (
        "the first ledger" if not done_a else "the second ledger", max(cross))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", nargs=2, required=True)
    ap.add_argument("--scene", required=True, help="dataset/scene")
    ap.add_argument("--cell", default="clean")
    a = ap.parse_args()
    code, text = compare(a.ledgers[0], a.ledgers[1], a.scene, a.cell)
    print("INPUT %s: %s" % ({0: "MATCH", 1: "DIFFERS", 2: "NOT VERIFIED"}[code], text))
    return code


if __name__ == "__main__":
    sys.exit(main())
