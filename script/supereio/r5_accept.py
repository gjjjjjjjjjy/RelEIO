#!/usr/bin/env python3
"""CPU allocation and acceptance of the SuperEIO runs of the pipeline comparison (round 5).

  blocks  CPU blocks for the parallel slots: exactly 16 logical CPUs each, whole physical cores (all hyperthreads of a
          core), taken only from the CPUs this process may use (os.sched_getaffinity, i.e. the container's set, not the
          host's count), cores in socket/core order, a block never spanning two sockets, blocks disjoint. Writes one
          comma-separated block per line, at most min(--max-slots, --gpus) lines.
  check   reads the plan (one "seq tag index" per line), the list of finished attempts (run_r5_supereio.sh), the
          blocks and the reference TensorRT engine hashes, and records for every attempt: present, exit code,
          summary.txt, its outcome by the rule of the bias experiment (script/diag/cross_system_eval.py, same order):
            undetermined  no summary.txt; run_seq.sh exit code not 0; rosbag play exit code not 0 or not recorded; a
                          ROS process died (never counted as completed or as an algorithm failure)
            failed        fewer than 10 odometry poses; an event_detector restart or a "system reboot" (the
                          estimator's own failure detection)
            completed     otherwise
          what summary.txt says the attempt was (sequence, tag r5<cond><index>, run folder, variant orig and the bag
          it played, which must be the planned condition's bag), whether its run folder is its own (no folder may
          serve two attempts), the CPU set run_seq.sh recorded (must equal the planned block of its slot), the engine
          hashes it recorded (must equal the reference), eval.json with a finite rule-D (se3_all) error, and the solves
          stopped by the time limit. Writes a JSON record (--out), with the comparison of the reference engines with
          the bias experiment's (--bias-engines; recorded, not a condition: same engines within this batch and same
          engines as the bias experiment are separate statements, and equal files do not mean equal speed on another
          GPU model).
          smoke: accepted only if every planned attempt is present once, in its own folder, as what it was planned to
                 be, with its exact CPU block and the reference engines; none is undetermined; every clean attempt
                 completed with a scorable trajectory; every completed attempt is scorable; and the clean boxes_6dof
                 attempt (run among the parallel attempts, not alone) recorded its checkpoint at 700 solves with a
                 number of time-limited solves inside the single-run range 35-44. An algorithm failure under a
                 perturbation is a result: recorded and listed, not a reason to stop.
          main:  every attempt is recorded as it ended (algorithm failures and undetermined attempts stay in the record
                 and are counted separately); the batch is valid only if the plan is the full one (both sequences, the
                 three core conditions at least, indices 1-10 for every sequence and condition), every planned attempt
                 is present once, in its own folder and as planned, every recorded CPU set and engine set is exact,
                 every completed attempt is scorable, and the scheduling check of the bias experiment holds on all
                 planned clean boxes_6dof attempts: each recorded its checkpoint at 700 solves (all of them, 10/10,
                 not a subset) and the median number of solves stopped by the time limit lies in 35-44.
          Exit 0 if accepted (smoke) or valid (main), 1 otherwise.

  python3 script/supereio/r5_accept.py blocks --max-slots 8 --gpus 8 --out blocks.txt
  python3 script/supereio/r5_accept.py check --mode smoke --plan list.plan --list list.txt --blocks blocks.txt \
      --engines engines.sha256 --out r.json
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_traj as ET  # noqa: E402  (numpy only)

PER_SLOT = 16
SCHED = (35, 44)
K_MAIN = 10
SEQS = ("boxes_6dof", "poster_6dof")
CORE = ("clean", "keep0.1", "tq10000us")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bag_of(seq, tag, root=ROOT):
    """The bag run_r5_supereio.sh plays for a sequence and condition."""
    return os.path.join(root, "data", "rosbags", "rpgdavis", seq + ("" if tag == "clean" else "__r5_" + tag) + ".bag")


def absrun(run_dir, root=ROOT):
    return os.path.normpath(run_dir if os.path.isabs(run_dir) else os.path.join(root, run_dir))


# ---- blocks ------------------------------------------------------------------------------------------------------
def topology():
    """[(cpu, core, socket)] from lscpu."""
    out = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        c, core, sock = (int(v) for v in line.split(",")[:3])
        rows.append((c, core, sock))
    return rows


def make_blocks(topo, allowed, per=PER_SLOT):
    """Blocks of exactly `per` logical CPUs from whole cores whose every thread is allowed; one socket per block."""
    cores = {}
    for c, core, sock in topo:
        cores.setdefault((sock, core), []).append(c)
    blocks, cur, cur_sock = [], [], None
    for (sock, core) in sorted(cores):
        threads = sorted(cores[(sock, core)])
        if not all(t in allowed for t in threads):
            continue                                   # a core only partly ours is never used
        if sock != cur_sock:
            cur, cur_sock = [], sock                   # an incomplete block is dropped at a socket boundary
        if len(cur) + len(threads) > per:
            continue
        cur += threads
        if len(cur) == per:
            blocks.append(cur)
            cur = []
    return blocks


# ---- check -------------------------------------------------------------------------------------------------------
def parse_cpu_list(s):
    """'0-7,64-71' -> {0..7, 64..71}."""
    out = set()
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def count(summ, key):
    """The number after `key` in summary.txt, 0 if absent (as in cross_system_eval.py)."""
    m = re.search(re.escape(key) + r"\s*(\d+)", summ)
    return int(m.group(1)) if m else 0


def classify(summ, rc, n_poses):
    """Outcome of one attempt that left a summary, by the bias experiment's rule (cross_system_eval.py), same order."""
    bag = re.search(r"rosbag play exit code:\s*(\d+)", summ)
    if rc != 0:
        return "undetermined: run_seq exit %s" % rc
    if not bag or bag.group(1) != "0":
        return "undetermined: rosbag play exit %s" % (bag.group(1) if bag else "not recorded")
    if count(summ, "process deaths:"):
        return "undetermined: a ROS process died"
    if n_poses < 10:
        return "failed: no trajectory"
    if count(summ, "event_detector restarts:") or count(summ, "failure detections (system reboot!):"):
        return "failed: restart or reboot"
    return "completed"


def read_engines(path):
    """Sorted non-empty lines of an engines.sha256 file, None if absent."""
    if not os.path.isfile(path):
        return None
    return sorted(l.strip() for l in open(path) if l.strip())


def read_attempt(run_dir, rc):
    """What one attempt left behind: outcome, recorded CPU set and engines, solver counter, rule-D error."""
    rec = {"summary": False, "cpus": None, "solves": None, "time_limited": None, "se3_all": None, "poses": 0,
           "engines": read_engines(os.path.join(run_dir, "engines.sha256"))}
    sp = os.path.join(run_dir, "summary.txt")
    if not os.path.isfile(sp):
        rec["outcome"] = "undetermined: no summary.txt (exit %s)" % rc
        return rec
    txt = open(sp, errors="replace").read()
    rec["summary"] = True
    m = re.search(r"^sequence (\S+)\s+tag (\S+)\s+run (.+?)\s*$", txt, re.M)
    v = re.search(r"^variant (\S+)", txt, re.M)
    b = re.search(r"^bag (.+?)\s*$", txt, re.M)
    rec["identity"] = {"seq": m.group(1) if m else None, "tag": m.group(2) if m else None,
                       "run": os.path.normpath(m.group(3)) if m else None, "variant": v.group(1) if v else None,
                       "bag": os.path.normpath(b.group(1)) if b else None}
    m = re.search(r" cpus ([0-9,\-]+) ", txt)
    rec["cpus"] = sorted(parse_cpu_list(m.group(1))) if m else None
    m = re.search(r"solver: solves (\d+) stopped by the time limit (\d+)", txt)
    if m:
        rec["solves"], rec["time_limited"] = int(m.group(1)), int(m.group(2))
    est = os.path.join(run_dir, "evio_result_no_loop.csv")
    rec["poses"] = int(len(ET.load_estimate(est))) if os.path.isfile(est) else 0
    rec["outcome"] = classify(txt, rc, rec["poses"])
    ep = os.path.join(run_dir, "eval.json")
    if os.path.isfile(ep):
        try:
            v = json.load(open(ep)).get("se3_all", {}).get("mpe_percent")
            rec["se3_all"] = float(v) if v is not None and math.isfinite(float(v)) else None
        except (ValueError, TypeError, AttributeError):
            rec["se3_all"] = None
    return rec


def planned_identity(s, t, i, run_dir, root=ROOT):
    """What summary.txt must say for attempt (s, t, i) in run_dir."""
    return {"seq": s, "tag": "r5%s%s" % (t, i), "run": absrun(run_dir, root), "variant": "orig",
            "bag": os.path.normpath(bag_of(s, t, root))}


def identity_as_planned(identity, s, t, i, run_dir, root=ROOT):
    return all((identity or {}).get(k) == v for k, v in planned_identity(s, t, i, run_dir, root).items())


def plan_complete(plan):
    """The full main plan: both sequences, the core conditions at least, the same conditions for both sequences,
    indices 1..K_MAIN once each for every sequence and condition."""
    idx = {}
    for s, t, i in plan:
        idx.setdefault((s, t), []).append(int(i))
    tags = {t for _, t in idx}
    return ({s for s, _ in idx} == set(SEQS) and set(CORE) <= tags
            and all(sorted(idx.get((s, t), [])) == list(range(1, K_MAIN + 1)) for s in SEQS for t in tags))


def check(mode, plan, rows, blocks, engines_ref, root=ROOT):
    """(accepted_or_valid, record). plan: [(s, t, i)]; rows: [(s, t, i, run_dir, rc, slot)]; blocks: [[cpus]];
    engines_ref: sorted lines of the reference engines.sha256."""
    got, folders = {}, {}
    for s, t, i, R, rc, slot in rows:
        got.setdefault((s, t, int(i)), []).append((R, rc, slot))
        folders.setdefault(absrun(R, root), []).append("%s %s %s" % (s, t, i))
    attempts, problems = [], []
    for f, users in sorted(folders.items()):
        if len(users) > 1:
            problems.append("run folder %s listed for %d attempts (%s)" % (f, len(users), ", ".join(users[:3])))
    if mode == "main" and not plan_complete(plan):
        problems.append("the plan is not the full main plan (2 sequences x conditions x indices 1-%d)" % K_MAIN)
    for s, t, i in plan:
        key = (s, t, int(i))
        entry = {"seq": s, "cond": t, "index": int(i)}
        if key not in got:
            entry["outcome"] = "missing"
            problems.append("%s %s %s: not run" % key)
            attempts.append(entry)
            continue
        if len(got[key]) > 1:
            problems.append("%s %s %s: listed more than once" % key)
        R, rc, slot = got[key][0]
        a = read_attempt(R, rc)
        planned = sorted(blocks[slot]) if 0 <= slot < len(blocks) else None
        a.update(run=R, exit_code=rc, slot=slot, cpus_exact=bool(a["cpus"] is not None and a["cpus"] == planned),
                 engines_exact=bool(engines_ref and a["engines"] == engines_ref), scorable=a["se3_all"] is not None)
        if a["summary"]:
            a["as_planned"] = identity_as_planned(a["identity"], s, t, i, R, root)
            if not a["as_planned"]:
                want = planned_identity(s, t, i, R, root)
                problems.append("%s %s %s: summary.txt says %s" % (s, t, i, ", ".join(
                    "%s %s (planned %s)" % (k, a["identity"].get(k), v) for k, v in want.items()
                    if a["identity"].get(k) != v)))
        entry.update(a)                    # after every per-attempt field (as_planned included) is set
        attempts.append(entry)
        if a["summary"] and not a["cpus_exact"]:
            problems.append("%s %s %s: CPU set %s, planned %s" % (s, t, i, a["cpus"], planned))
        if a["summary"] and not a["engines_exact"]:
            problems.append("%s %s %s: TensorRT engines differ from the reference (or not recorded)" % (s, t, i))
        if a["outcome"] == "completed" and not a["scorable"]:
            problems.append("%s %s %s: completed but not scorable (eval_traj.py)" % (s, t, i))
        if mode == "smoke":
            if a["outcome"].startswith("undetermined"):
                problems.append("%s %s %s: smoke attempt %s" % (s, t, i, a["outcome"]))
            elif t == "clean" and a["outcome"] != "completed":
                problems.append("%s %s %s: clean smoke attempt %s" % (s, t, i, a["outcome"]))
    planned_cb = [a for a in attempts if a["seq"] == "boxes_6dof" and a["cond"] == "clean"]
    with_cp = [a for a in planned_cb if a.get("time_limited") is not None and a.get("solves") == 700]
    vals = [a["time_limited"] for a in with_cp]
    med = statistics.median(vals) if vals else None
    sched_ok = bool(planned_cb) and len(with_cp) == len(planned_cb) and SCHED[0] <= med <= SCHED[1]
    if mode == "smoke" and not sched_ok:
        problems.append("clean boxes_6dof smoke attempt: checkpoint at 700 solves in %d/%d, time-limited %s, "
                        "single-run range %d-%d: scheduling at this concurrency not shown to match the bias "
                        "experiment" % (len(with_cp), len(planned_cb), vals, SCHED[0], SCHED[1]))
    outcomes = {}
    for a in attempts:
        k = a["outcome"].split(":")[0]
        outcomes[k] = outcomes.get(k, 0) + 1
    record = {"mode": mode, "attempts": attempts, "problems": problems, "outcomes": outcomes,
              "engines_reference": engines_ref,
              "distinct_run_folders": len(folders), "listed_attempts": len(rows),
              "algorithm_failures": ["%s %s %s: %s" % (a["seq"], a["cond"], a["index"], a["outcome"])
                                     for a in attempts if a["outcome"].startswith("failed")],
              "undetermined": ["%s %s %s: %s" % (a["seq"], a["cond"], a["index"], a["outcome"])
                               for a in attempts if a["outcome"].startswith("undetermined")],
              "scheduling_check": {"planned": len(planned_cb), "with_checkpoint_700": len(with_cp),
                                   "time_limited": vals, "median": med, "range": list(SCHED), "passed": sched_ok}}
    if mode == "smoke":
        ok = not problems
        record["accepted"] = ok
    else:
        ok = not problems and sched_ok
        record["valid"] = ok
    return ok, record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blocks")
    b.add_argument("--max-slots", type=int, default=8)
    b.add_argument("--gpus", type=int, required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--mode", choices=("smoke", "main"), required=True)
    c.add_argument("--plan", required=True)
    c.add_argument("--list", required=True)
    c.add_argument("--blocks", required=True)
    c.add_argument("--engines", required=True, help="reference engines.sha256 (the single first smoke attempt's)")
    c.add_argument("--bias-engines", default="", help="engines.sha256 of a bias-experiment run (recorded only)")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "blocks":
        allowed = set(os.sched_getaffinity(0))
        blocks = make_blocks(topology(), allowed)[:max(0, min(a.max_slots, a.gpus))]
        with open(a.out, "w") as fh:
            for bl in blocks:
                fh.write(",".join(str(c) for c in bl) + "\n")
        print("  %d CPUs allowed in this container; %d block(s) of %d: %s" % (
            len(allowed), len(blocks), PER_SLOT, " | ".join("%d..%d" % (bl[0], bl[-1]) for bl in blocks)))
        sys.exit(0 if blocks else 1)
    plan = [tuple(l.split()[:3]) for l in open(a.plan) if l.strip()]
    rows = []
    if os.path.isfile(a.list):
        for l in open(a.list):
            f = l.split()
            if len(f) >= 6:
                rows.append((f[0], f[1], f[2], f[3], int(f[4].split("=")[1]), int(f[5].split("=")[1])))
    blocks = [[int(x) for x in l.strip().split(",")] for l in open(a.blocks) if l.strip()]
    ref = read_engines(a.engines)
    if not ref:
        sys.exit("reference engines %s missing or empty" % a.engines)
    ok, rec = check(a.mode, plan, rows, blocks, ref)
    rec["engines_reference_file"] = a.engines
    bias = read_engines(a.bias_engines) if a.bias_engines else None
    rec["engines_vs_bias_experiment"] = {"file": a.bias_engines or None,
                                         "identical": (bias == ref) if bias else None}
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    for p in rec["problems"]:
        print("  problem: " + p)
    for p in rec["undetermined"]:
        print("  undetermined (to be checked, never counted as completed or failed): " + p)
    for p in rec["algorithm_failures"]:
        print("  algorithm failure (a result): " + p)
    eb = rec["engines_vs_bias_experiment"]
    print("  engines: the reference in every attempt of this batch: %s; identical to the bias experiment's: %s" % (
        "yes" if not any("engines differ" in p for p in rec["problems"]) else "NO",
        {True: "yes", False: "NO", None: "not found"}[eb["identical"]]))
    print("  run folders: %d distinct for %d listed attempts (%d planned)" % (
        rec["distinct_run_folders"], rec["listed_attempts"], len(plan)))
    sc = rec["scheduling_check"]
    print("  scheduling check (clean boxes_6dof): checkpoint at 700 solves in %d/%d planned attempts, time-limited "
          "%s, median %s -> %s" % (sc["with_checkpoint_700"], sc["planned"], sc["time_limited"], sc["median"],
                                   "passed" if sc["passed"] else "NOT passed"))
    print("  outcomes %s; %s" % (rec["outcomes"], ("ACCEPTED" if ok else "NOT ACCEPTED") if a.mode == "smoke"
                                  else ("batch VALID" if ok else "batch NOT VALID")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
