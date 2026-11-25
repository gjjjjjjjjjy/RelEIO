#!/usr/bin/env python3
"""Precision-policy matrix over several sequences and GPUs (screening, not confirmatory).

Every job is one (dataset, scene, arm, seed, stress cell) run, executed as an independent process through
the append-only ledger (run_ledger.run) on one GPU, one job per GPU at a time. Jobs sit in one
queue in an order that rotates the arms within each round and interleaves the scenes, and each
GPU pulls the next job when it is free, so every arm is spread over all GPUs and over time.
The same control checks as the single-GPU pilot are applied after every run; the first failed
check, or a GPU found in use, stops all workers.

Examples (repository root, GPU node):
  python script/diag/precision_matrix.py --gpus 0 1 2 3 4 5 6 7 --scenes vector/desk_normal1 \\
      --arms default amp_off tf32_off fp32_strict --k 16 --seeds 1000 --cholinfo --ledger logs/ledger/round2A
  python script/diag/precision_matrix.py --summary-only --ledger logs/ledger/round2A
Round 3 (frozen main-matrix cells, see precision_pilot.FROZEN_CELLS):
  python script/diag/precision_matrix.py --gpus 0 1 2 3 4 5 6 7 --scenes vector/desk_normal1 \\
      --cells frozen --arms default no_tf32_override --k 5 --seeds 1000 1001 1002 1003 1004 --cholinfo \\
      --voxcache rpgdavis --ledger logs/ledger/round3_x
"""
import argparse
import collections
import json
import os
import queue
import sys
import tempfile
import threading
import time
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precision_pilot as P  # noqa: E402
import run_ledger  # noqa: E402


def build_jobs(scenes, arms, k, seeds, untraced_k=0, cells=("base",), units=None):
    """Jobs as (dataset, scene, arm, seed, cell). Units are (dataset, scene, cell): every scene with
    every cell, or the explicit `units` list. For every unit, round r uses seed seeds[r % len(seeds)]
    for every arm of that round, rotates the arm order, and the untraced reference runs are spread
    evenly over the rounds. Units are interleaved job by job."""
    arms = list(arms)
    per_scene = []
    if units is None:
        units = [(ds, sc, cell) for ds, sc in scenes for cell in cells]
    for ds, sc, cell in units:
        rounds = []
        for r in range(k):
            j = r % len(arms)
            seed = seeds[r % len(seeds)]
            rounds.append([(ds, sc, a, seed, cell) for a in arms[j:] + arms[:j]])
        for i in range(untraced_k):
            r = min(k - 1, int((i + 0.5) * k / untraced_k)) if k else 0
            rounds[r].append((ds, sc, P.UNTRACED, seeds[r % len(seeds)], cell))
        per_scene.append([job for rnd in rounds for job in rnd])
    out = []
    while any(per_scene):
        for lst in per_scene:
            if lst:
                out.append(lst.pop(0))
    return out


def front_load_inputs(jobs, cached):
    """Move one job per distinct input condition (dataset, scene, keep, tq) of the datasets for which
    cached(dataset) is true to the front, so each voxel cache is built once, early, and later runs of
    that condition load it instead of waiting on the build lock. The moved jobs alternate between the
    arms. The multiset of jobs is unchanged."""
    arms = list(dict.fromkeys(j[2] for j in jobs))
    first, conds = {}, []
    for i, j in enumerate(jobs):
        if not cached(j[0]):
            continue
        c = P.parse_cell(j[4] if len(j) > 4 else "base")
        key = (j[0], j[1], c["keep"], c["tq_us"])
        if key not in first:
            first[key] = {}
            conds.append(key)
        first[key].setdefault(j[2], i)
    picked = []
    for n, key in enumerate(conds):
        want = arms[n % len(arms)]
        picked.append(first[key].get(want, min(first[key].values())))
    chosen = set(picked)
    return [jobs[i] for i in picked] + [j for i, j in enumerate(jobs) if i not in chosen]


# Typical wall time of one traced run (s), for balancing parts only: round-2B medians; DAVIS after the
# binary-search loader is an estimate. Unknown scenes count as 200 s.
SEQ_COST_S = {"vector/desk_normal1": 191, "vector/corner_slow1": 72, "hku/vicon_dark1": 235,
              "tumvie/mocap-3d-trans": 355, "fpv/indoor_forward_3": 95, "rpgdavis/boxes_6dof": 90,
              "rpgdavis/poster_6dof": 90, "rpgdavis/shapes_6dof": 60,
              "rpgdavis_up/boxes_6dof": 90, "rpgdavis_up/poster_6dof": 90}


def part_of_scenes(jobs, n):
    """Assign whole sequences to n parts (largest expected cost first, each to the least loaded part).
    Whole sequences, so that every comparison within a sequence (levels, cells, arms, seeds) runs on
    one instance; a stress level is never tied to an instance. Returns {"ds/scene": part (1-based)}."""
    cost = collections.Counter()
    for j in jobs:
        key = "%s/%s" % (j[0], j[1])
        cost[key] += SEQ_COST_S.get(key, 200)
    load, out = [0.0] * n, {}
    for key in sorted(cost, key=lambda k: (-cost[k], k)):
        p = min(range(n), key=lambda i: (load[i], i))
        out[key] = p + 1
        load[p] += cost[key]
    return out


def select_part(jobs, i, n, reference_scenes=None):
    """Part i of n (1-based): the jobs of the sequences part_of_scenes puts in part i, in plan order.
    reference_scenes ("ds/scene" list): compute the assignment from these sequences (equal weight per
    sequence, cost from SEQ_COST_S) instead of from the jobs, so that several plans (e.g. the main
    block and the extra repeats of the key cells) put every sequence on the same instance."""
    if reference_scenes:
        where = part_of_scenes([tuple(x.split("/", 1)) for x in reference_scenes], n)
    else:
        where = part_of_scenes(jobs, n)
    return [j for j in jobs if where["%s/%s" % (j[0], j[1])] == i]


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact test for a/n1 vs b/n2 (sum of tables no more likely than observed)."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    K, N = a + b, n1 + n2

    def p(x):
        return comb(n1, x) * comb(n2, K - x) / comb(N, K)
    po = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, K - n2), min(K, n1) + 1) if p(x) <= po * (1 + 1e-9)))


def _worker(gpu, jobs, args, stop, lock, done, errors=None):
    """Thread body. Any exception is recorded in `errors` and stops all workers, so the run
    can never end as a silent partial success."""
    try:
        _worker_loop(gpu, jobs, args, stop, lock, done)
    except Exception as e:  # noqa: BLE001 - recorded and re-raised as a non-zero exit in main()
        import traceback
        with lock:
            if errors is not None:
                errors.append((gpu, "%s: %s" % (type(e).__name__, e)))
            print("STOP: worker on GPU %d crashed:\n%s" % (gpu, traceback.format_exc()), flush=True)
        stop.set()


def _worker_loop(gpu, jobs, args, stop, lock, done):
    while not stop.is_set():
        try:
            idx, job = jobs.get_nowait()
        except queue.Empty:
            return
        ds, sc, arm, seed = job[:4]
        cell = P.parse_cell(job[4] if len(job) > 4 else "base")
        split = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        split.write(sc + "\n")
        split.close()
        try:
            cmd, inputs = P.build_cmd(ds, sc, seed, split.name, opts=cell["opts"])
            labels = {"experiment": "precision_matrix", "arm": arm, "dataset": ds, "scene": sc, "seed": seed,
                      "job_index": idx, "cholinfo": bool(args.cholinfo and arm != P.UNTRACED),
                      "cell": cell["cell"], "axis": cell["axis"], "level": cell["level"]}
            labels["voxcache"] = voxcache_for(args, ds)
            env = P.arm_env(arm, args.cholinfo, labels["voxcache"], cell=cell["cell"])
            rec = run_ledger.run(cmd, env, gpu, args.ledger, labels, inputs=inputs,
                                 timeout=args.timeout, require_exclusive=True, gpu_poll_s=args.gpu_poll)
        finally:
            os.unlink(split.name)
        with lock:
            done.append(rec)
            print("[%3d/%d] gpu%-2d %-22s %-16s seed %d  %-42s mpe=%s  %ss" % (
                len(done), args.total, gpu, "%s/%s" % (ds, sc), arm, seed, rec["outcome_hint"], rec["mpe"],
                rec["wall_s"]), flush=True)
        if run_ledger.outcome_category(rec["outcome_hint"]) == "not_run":
            with lock:
                print("STOP: GPU %d is not exclusive (see manifest of %s)" % (gpu, rec["run_id"]), flush=True)
            stop.set()
            return
        with open(os.path.join(args.ledger, "runs", rec["run_id"], "facts.json")) as fh:
            facts = json.load(fh)
        problems = P.verify_controls(facts, arm, labels["cholinfo"], cell=labels["cell"])
        if problems:
            with open(os.path.join(args.ledger, "runs", rec["run_id"], "control_problems.json"), "w") as f:
                json.dump(problems, f, indent=1)
            with lock:
                print("STOP: control verification failed for %s (%s):" % (rec["run_id"], arm), flush=True)
                for pr in problems:
                    print("   -", pr, flush=True)
            stop.set()
            return


def voxcache_for(args, dataset):
    """--voxcache absent: no cache; --voxcache alone: every dataset; --voxcache rpgdavis ...: only those."""
    v = getattr(args, "voxcache", None)
    if v is None or v is False:
        return False
    if v is True or v == []:
        return True
    return dataset in v


def scene_label(labels):
    """'dataset/scene' for clean round-1/2 runs, 'dataset/scene cell' for stress cells."""
    base = "%s/%s" % (labels.get("dataset"), labels.get("scene"))
    cell = labels.get("cell")
    return base if cell in (None, "base") else "%s %s" % (base, cell)


def input_consistency(summaries):
    """Bit-identity of the estimator input over a set of traced runs (numtrace run_summary records).

    Every checkpoint frame reached by two or more runs is compared (not only the latest checkpoint
    that all runs share, which an early stop would cut short), and the runs that got furthest must
    agree on the frame count and the whole-input fingerprint. Returns a dict with ok, the frames
    where runs disagree, the deepest frame compared by at least two runs, and the fingerprint count."""
    lst = [x for x in summaries if x.get("input_checkpoints")]
    values = collections.defaultdict(set)
    reached = collections.Counter()
    for x in lst:
        for k, v in x["input_checkpoints"].items():
            values[int(k)].add(v)
            reached[int(k)] += 1
    differs = sorted(k for k, v in values.items() if len(v) > 1)
    shared = [k for k, n in reached.items() if n >= 2]
    top = max((x.get("input_frames") or 0) for x in lst) if lst else 0
    full = [x for x in lst if (x.get("input_frames") or 0) == top]
    n_fp = len({x.get("input_fingerprint") for x in full})
    empty_fp = sum(1 for x in full if not x.get("input_fingerprint"))     # a missing hash proves nothing
    return {"runs": len(lst), "differs_at": differs, "compared_to": max(shared) if shared else None,
            "complete": len(full), "complete_frames": top, "fingerprints": n_fp, "empty_fingerprints": empty_fp,
            "ok": bool(lst) and not differs and n_fp == 1 and empty_fp == 0}


def _input_condition(scene_label, labels):
    c = P.parse_cell(labels.get("cell"))
    return scene_label.split(" ")[0] + ("" if (c["keep"], c["tq_us"]) == (1.0, 0) else
                                        " keep=%g tq_us=%d" % (c["keep"], c["tq_us"]))


def input_identity(groups):
    """Per input condition (scene, keep, tq): do all traced runs see bit-identical input, over arms,
    seeds and patch levels? (Thinning is seeded by frame timestamps, so the eval seed does not change
    the input; PATCHES_PER_FRAME acts after the input.) Returns {condition: text}."""
    by = collections.defaultdict(list)
    for (sc, arm), g in groups.items():
        for r, f in g["valid"]:
            rs = ((f or {}).get("numtrace") or {}).get("run_summary") or {}
            if rs.get("input_checkpoints"):
                by[_input_condition(sc, r["labels"])].append((r["labels"].get("seed"), rs))
    out = {}
    for sc, pairs in sorted(by.items()):
        seed = ",".join(sorted({str(sd) for sd, _ in pairs}))
        c = input_consistency([x for _, x in pairs])
        out[sc] = "seeds %s: %d runs, %s (every checkpoint reached by 2+ runs compared, deepest frame %s); " \
                  "complete runs %d (%d frames) with %d distinct fingerprint(s)%s" % (
                      seed, c["runs"], "identical" if not c["differs_at"] else "DIFFERENT at frames %s" % c["differs_at"],
                      c["compared_to"], c["complete"], c["complete_frames"], c["fingerprints"],
                      "" if c["ok"] else ("  !! INPUT NOT VERIFIED (empty fingerprint)" if c["empty_fingerprints"]
                                          and not c["differs_at"] else "  !! INPUT DIFFERS"))
    return out


def summarize(ledger_dir, reference="default"):
    """ledger_dir: one ledger, or a list (e.g. the parts of a round run on several instances)."""
    dirs = [ledger_dir] if isinstance(ledger_dir, str) else list(ledger_dir)
    recs = []
    for d in dirs:
        for r in run_ledger.load_ledger(d):
            r["_ledger"] = d
            recs.append(r)
    if not recs:
        print("no runs in", " ".join(dirs))
        return {}
    print("ledger:", " ".join(dirs), "| runs:", len(recs), "| GPUs:", len({r["gpu_uuid"] for r in recs}),
          "| code versions:", sorted({(r["git_commit"] or "")[:10] + "/" + (r["git_diff_sha256"] or "")[:8] for r in recs}),
          "| cuda_ba:", sorted({(r.get("cuda_ba_sha256") or "")[:12] for r in recs}))
    print("Counts use control-valid runs only; invalid and not-run attempts are listed separately.\n")
    groups = collections.defaultdict(lambda: {"valid": [], "invalid": [], "not_run": []})
    for r in recs:
        key = (scene_label(r["labels"]), r["labels"].get("arm"))
        g = groups[key]
        if run_ledger.outcome_category(r["outcome_hint"]) == "not_run":
            g["not_run"].append(r)
            continue
        f = P._load_facts(r["_ledger"], r["run_id"])
        probs = ["facts.json missing"] if f is None else P.verify_controls(
            f, key[1], bool(r["labels"].get("cholinfo")), cell=r["labels"].get("cell"))
        (g["invalid"] if probs else g["valid"]).append((r, f, probs) if probs else (r, f))
    out = {}
    settings = collections.defaultdict(collections.Counter)
    for (sc, arm), g in groups.items():
        for r, f in g["valid"]:
            pol = ((f or {}).get("numtrace") or {}).get("precision_policy") or {}
            if pol:
                settings[arm][(pol.get("cfg_MIXED_PRECISION"), pol.get("update_autocast"), pol.get("matmul_allow_tf32"),
                               pol.get("cudnn_allow_tf32"), pol.get("float32_matmul_precision"),
                               pol.get("NVIDIA_TF32_OVERRIDE"))] += 1
    if settings:
        print("Precision settings actually recorded per arm "
              "(MIXED_PRECISION, update autocast, matmul TF32, cuDNN TF32, fp32 matmul precision, NVIDIA_TF32_OVERRIDE):")
        for arm, c in sorted(settings.items()):
            print("   %-16s %s" % (arm, dict(c)))
        print()
    scenes = sorted({k[0] for k in groups})
    ident = input_identity(groups)
    if ident:
        print("Input identity (fingerprint of every voxel and timestamp fed to the estimator, traced runs):")
        for sc in scenes:
            if sc in ident:
                print("   %-26s %s" % (sc, ident[sc]))
        for sc in sorted(set(ident) - set(scenes)):
            print("   %-26s %s" % (sc, ident[sc]))
        print()
    W = max([26] + [len(x) for x in scenes])
    print("%-*s %-16s %6s %8s %14s %10s %6s %9s %8s  %s" % (W, "scene", "arm", "valid", "fail", "Wilson95", "p_vs_ref",
                                                          "chol", "median_s", "mpe_med", "first detected (failures)"))
    for sc in scenes:
        ref = groups.get((sc, reference))
        ref_stats = P.arm_stats(ref["valid"]) if ref else None
        for (s2, arm), g in sorted(groups.items()):
            if s2 != sc:
                continue
            st = P.arm_stats(g["valid"])
            k, n = st["failures"], st["determinate"]
            ci = "[%3.0f%%,%3.0f%%]" % tuple(100 * x for x in st["wilson95"]) if n else "-"
            pv = ""
            if ref_stats and arm != reference and n and ref_stats["determinate"]:
                pv = "%.2g" % fisher_two_sided(k, n, ref_stats["failures"], ref_stats["determinate"])
            chol = sum(1 for _, f in g["valid"] if ((f or {}).get("cholinfo") or {}).get("flagged", 0) > 0)
            walls = sorted(r["wall_s"] for r, _ in g["valid"] if r["outcome_hint"] == "completed")
            mpes = sorted(r["mpe"] for r, _ in g["valid"] if r["outcome_hint"] == "completed" and r.get("mpe") is not None)
            firsts = collections.Counter()
            for r, f in g["valid"]:
                if r["outcome_hint"] != "completed":
                    fd = ((f or {}).get("numtrace") or {}).get("first_detected_nonfinite")
                    firsts["%s:%s@%s" % (fd.get("where"), fd.get("tensor"), fd.get("frame")) if fd else "untraced/none"] += 1
            print("%-*s %-16s %6d %8s %14s %10s %6d %9s %8s  %s" % (
                W, sc, arm, len(g["valid"]), "%d/%d" % (k, n), ci, pv, chol,
                "%.0f" % walls[len(walls) // 2] if walls else "-",
                "%.3g" % mpes[len(mpes) // 2] if mpes else "-", dict(firsts) if firsts else ""))
            if st["undetermined"]:
                print("%-*s %-16s   undetermined=%d (upper bound if all failed: %d/%d)" % (
                    W, "", "", st["undetermined"], *st["upper_if_undetermined_failed"]))
            for r, _, probs in g["invalid"]:
                print("%-*s %-16s   EXCLUDED %s %s: %s" % (W, "", "", r["run_id"], r["outcome_hint"], "; ".join(probs)))
            if g["not_run"]:
                print("%-*s %-16s   not run: %d" % (W, "", "", len(g["not_run"])))
            vc = collections.defaultdict(list)
            for r, _ in g["valid"]:
                if r["outcome_hint"] == "completed":
                    vc[bool(r["labels"].get("voxcache"))].append(r["wall_s"])
            if len(vc) == 2:
                med = {k: sorted(v)[len(v) // 2] for k, v in vc.items()}
                print("%-*s %-16s   completed-run median wall time: voxel cache %.0f s (n=%d), no cache %.0f s (n=%d)" % (
                    W, "", "", med[True], len(vc[True]), med[False], len(vc[False])))
            out[(sc, arm)] = dict(st, n_valid=len(g["valid"]), n_invalid=len(g["invalid"]), cholesky_flag_runs=chol,
                                  p_vs_reference=pv or None)
        print()
    if any(" " in sc for sc in scenes):
        out["aggregate"] = aggregate(groups, reference)
    return out


CLEAN_CELLS = P.CLEAN_CELLS


def aggregate(groups, reference="default"):
    """Round-3 tables over stress cells (control-valid runs only): per arm, per cell over sequences,
    per sequence over cells, clean vs degraded cells within each arm, and paired accuracy against
    the reference arm (same sequence, cell and seed, both completed)."""
    tot = collections.defaultdict(lambda: [0, 0])
    by_cell = collections.defaultdict(lambda: [0, 0])
    by_seq = collections.defaultdict(lambda: [0, 0])
    by_cd = collections.defaultdict(lambda: [0, 0])
    mpe = collections.defaultdict(list)      # (seq, cell, seed, arm) -> [(job_index, MPE or None)], every valid run
    arms, cells, seqs = set(), set(), set()
    for (sc, arm), g in groups.items():
        if " " not in sc:
            continue
        seq, cell = sc.split(" ", 1)
        arms.add(arm), cells.add(cell), seqs.add(seq)
        st = P.arm_stats(g["valid"])
        for d, key in ((tot, arm), (by_cell, (cell, arm)), (by_seq, (seq, arm)),
                       (by_cd, ("clean" if cell in CLEAN_CELLS else "degraded", arm))):
            d[key][0] += st["failures"]
            d[key][1] += st["determinate"]
        for r, _ in g["valid"]:
            done = r["outcome_hint"] == "completed" and r.get("mpe") is not None
            mpe[(seq, cell, r["labels"].get("seed"), arm)].append((r["labels"].get("job_index") or 0,
                                                                   r["mpe"] if done else None))
    arms = sorted(arms, key=lambda a: (a != reference, a))
    known = list(dict.fromkeys(("clean",) + P.FROZEN_CELLS + P.MAIN_CELLS))
    order = [c for c in known if c in cells] + sorted(cells - set(known))

    def fmt(kn):
        return "%d/%d" % tuple(kn) if kn[1] else "-"

    def pval(kn, ref):
        return "%.2g" % fisher_two_sided(kn[0], kn[1], ref[0], ref[1]) if kn[1] and ref[1] else ""
    res = {"total": {a: tuple(tot[a]) for a in arms}, "cell": {}, "sequence": {}, "clean_vs_degraded": {},
           "paired_mpe": {}}
    print("ROUND-3 AGGREGATES (control-valid runs; clean cells = %s)" % ", ".join(CLEAN_CELLS))
    print("  all cells:       " + "   ".join("%s %s" % (a, fmt(tot[a])) for a in arms)
          + ("   p(arm vs %s): " % reference + " ".join("%s=%s" % (a, pval(tot[a], tot[reference]))
                                                  for a in arms if a != reference) if reference in tot else ""))
    print("  per cell (failures/determinate runs over all sequences):")
    for c in order:
        print("    %-12s " % c + "   ".join("%s %-7s" % (a, fmt(by_cell[(c, a)])) for a in arms)
              + "   p=" + " ".join(pval(by_cell[(c, a)], by_cell[(c, reference)]) for a in arms if a != reference))
        res["cell"][c] = {a: tuple(by_cell[(c, a)]) for a in arms}
    print("  per sequence (over all cells):")
    for q in sorted(seqs):
        print("    %-26s " % q + "   ".join("%s %-7s" % (a, fmt(by_seq[(q, a)])) for a in arms))
        res["sequence"][q] = {a: tuple(by_seq[(q, a)]) for a in arms}
    print("  clean vs degraded cells, within each arm:")
    for a in arms:
        cl, dg = by_cd[("clean", a)], by_cd[("degraded", a)]
        print("    %-16s clean %-8s degraded %-8s p=%s" % (a, fmt(cl), fmt(dg), pval(dg, cl)))
        res["clean_vs_degraded"][a] = (tuple(cl), tuple(dg))
    print("  paired accuracy vs %s (same sequence, cell, seed and repeat number in plan order; pairs where both"
          " runs completed; MPE in %%; similar medians do not establish equivalence):" % reference)
    for a in arms:
        if a == reference:
            continue
        pairs = []
        for k, runs in mpe.items():
            ref = mpe.get(k[:3] + (reference,))
            if k[3] != a or not ref:
                continue
            # repeat r of this seed in one arm with repeat r in the other (order fixed by the plan), then
            # keep the pair only if both completed: a failure never shifts the pairing of later repeats
            pairs += [(rv, av) for (_, rv), (_, av) in zip(sorted(ref), sorted(runs)) if rv is not None and av is not None]
        if not pairs:
            continue
        d = sorted(v - r for r, v in pairs)
        ref_m = sorted(r for r, _ in pairs)[len(pairs) // 2]
        arm_m = sorted(v for _, v in pairs)[len(pairs) // 2]
        print("    %-16s pairs %d | median MPE %s %.3g, %s %.3g | median paired difference %+.3g (IQR %+.3g .. %+.3g)" % (
            a, len(pairs), reference, ref_m, a, arm_m, d[len(d) // 2], d[len(d) // 4], d[(3 * len(d)) // 4]))
        res["paired_mpe"][a] = {"pairs": len(pairs), "median_diff": d[len(d) // 2]}
    print()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, nargs="+")
    ap.add_argument("--scenes", nargs="+", help="dataset/scene, e.g. vector/desk_normal1")
    ap.add_argument("--units", nargs="+", metavar="DATASET/SCENE@CELL",
                    help="explicit (scene, cell) units instead of --scenes x --cells, e.g. vector/corner_slow1@keep:0.1")
    ap.add_argument("--arms", nargs="+", default=list(P.TRACED_PAIR))
    ap.add_argument("--k", type=int, default=10, help="runs per (scene, traced arm)")
    ap.add_argument("--untraced-k", type=int, default=0, help="untraced reference runs per scene")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000])
    ap.add_argument("--cholinfo", action="store_true")
    ap.add_argument("--voxcache", nargs="*", metavar="DATASET",
                    help="reuse CPU-built voxel caches (namespace tied to the voxel code; input identity is checked "
                         "per run); alone: every dataset, with names: only those datasets (e.g. rpgdavis)")
    ap.add_argument("--cells", nargs="+", default=["base"],
                    help="stress cells: base, clean, keep:<r>, tq:<us>, patch:<n>, 'frozen' for the 12 frozen main-matrix "
                         "cells, or 'main' for the 10 conditions of the new main matrix")
    ap.add_argument("--ledger", required=True, nargs="+",
                    help="ledger directory; with --summary-only several can be given and are summarised together")
    ap.add_argument("--part-scenes", nargs="+", metavar="DATASET/SCENE",
                    help="sequences that define the part assignment (default: those of this plan)")
    ap.add_argument("--plan-out", help="write the whole plan (all parts, in queue order) as JSON to this file")
    ap.add_argument("--plan-check", help="stop unless the whole plan equals the JSON plan in this file (frozen plan)")
    ap.add_argument("--part", default="1/1",
                    help="i/N: run only the sequences assigned to part i of N (whole sequences per instance)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--gpu-poll", type=float, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.chdir(P.ROOT)
    if a.summary_only:
        summarize(a.ledger)
        return 0
    if len(a.ledger) != 1:
        ap.error("give exactly one --ledger when running jobs")
    a.ledger = a.ledger[0]
    try:
        part_i, part_n = (int(x) for x in a.part.split("/"))
        if not 1 <= part_i <= part_n:
            raise ValueError
    except ValueError:
        ap.error("--part must be i/N with 1 <= i <= N, got %r" % a.part)
    if not a.gpus or not (a.scenes or a.units) or (a.scenes and a.units):
        ap.error("--gpus and exactly one of --scenes / --units are required")
    bad = [x for x in a.arms if x not in P.ARM_ENV or x == P.UNTRACED]
    if bad:
        ap.error("unknown or untraced arm in --arms: %s (traced arms: %s)" % (bad, sorted(set(P.ARM_ENV) - {P.UNTRACED})))
    from script.run_degradation_sweep import DATASETS
    def scene(s):
        ds, _, sc = s.partition("/")
        if ds not in DATASETS or sc not in DATASETS[ds][3]:
            ap.error("unknown scene %r (known: %s)" % (s, ["%s/%s" % (d, x) for d in DATASETS for x in DATASETS[d][3]]))
        return ds, sc
    scenes = [scene(s) for s in (a.scenes or [])]
    cells = []
    for c in a.cells:
        for x in {"frozen": P.FROZEN_CELLS, "main": P.MAIN_CELLS}.get(c, [c]):
            try:
                cells.append(P.parse_cell(x)["cell"])
            except ValueError as e:
                ap.error(str(e))
    if len(set(cells)) != len(cells):
        ap.error("duplicate cells: %s" % cells)
    units = None
    if a.units:
        units = []
        for u in a.units:
            sc, sep, cell = u.partition("@")
            try:
                units.append(scene(sc) + (P.parse_cell(cell if sep else "base")["cell"],))
            except ValueError as e:
                ap.error(str(e))
        if len(set(units)) != len(units):
            ap.error("duplicate units")
        cells = sorted({u[2] for u in units})
    if a.voxcache:
        unknown = [d for d in a.voxcache if d not in DATASETS]
        if unknown:
            ap.error("unknown dataset(s) in --voxcache: %s" % unknown)
    jobs = build_jobs(scenes, a.arms, a.k, a.seeds, a.untraced_k, cells=cells, units=units)
    jobs = front_load_inputs(jobs, lambda ds: voxcache_for(a, ds))
    n_plan = len(jobs)
    plan_json = json.dumps([list(j) for j in jobs], indent=0)
    if a.plan_out:
        with open(a.plan_out, "w") as fh:
            fh.write(plan_json + "\n")
    if a.plan_check:
        with open(a.plan_check) as fh:
            frozen = json.load(fh)
        if frozen != [list(j) for j in jobs]:
            print("STOP: the plan differs from the frozen plan %s (%d vs %d jobs)" % (a.plan_check, len(jobs), len(frozen)))
            return 4
        print("plan identical to the frozen plan %s" % a.plan_check)
    jobs = select_part(jobs, part_i, part_n, reference_scenes=a.part_scenes)
    a.total = len(jobs)
    per = collections.Counter(("%s/%s" % (j[0], j[1]), j[2]) for j in jobs)
    print("part %d/%d: %d of %d planned jobs | sequences in this part: %s" % (
        part_i, part_n, len(jobs), n_plan, sorted({"%s/%s" % (j[0], j[1]) for j in jobs})))
    print("plan: %d jobs on GPUs %s | arms %s | k=%d | seeds %s | untraced per unit %d | cells %s | voxcache %s" % (
        len(jobs), a.gpus, a.arms, a.k, a.seeds, a.untraced_k, cells,
        "off" if a.voxcache is None else (a.voxcache or "all datasets")))
    for (sc, arm), n in sorted(per.items()):
        print("   %-26s %-16s %d" % (sc, arm, n))
    for arm in a.arms + ([P.UNTRACED] if a.untraced_k else []):
        env = P.arm_env(arm, a.cholinfo, a.voxcache is not None)
        print("env[%s] (clean cell; stress cells add FREQVIO_EVENT_KEEP / FREQVIO_TIME_QUANT_US or --opts "
              "PATCHES_PER_FRAME): %s" % (arm, " ".join(("%s=%s" % kv) if kv[1] is not None else ("unset " + kv[0])
                                              for kv in sorted(env.items()) if kv[0].startswith(("FREQVIO_", "TORCH_ALLOW")))))
    if a.dry_run:
        return 0
    os.makedirs(a.ledger, exist_ok=True)
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    stop, lock, done, errors = threading.Event(), threading.Lock(), [], []
    threads = []
    for g in a.gpus:
        t = threading.Thread(target=_worker, args=(g, q, a, stop, lock, done, errors), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)  # stagger start-up (data loading is CPU heavy)
    for t in threads:
        t.join()
    status = completeness(jobs, done, errors, stopped=stop.is_set())
    summarize(a.ledger)
    return status


def completeness(jobs, done, errors, stopped):
    """0 only if every planned job produced a ledger record and nothing stopped or crashed."""
    want = collections.Counter((j[0], j[1], j[2], j[3], j[4] if len(j) > 4 else "base") for j in jobs)
    got = collections.Counter((r["labels"].get("dataset"), r["labels"].get("scene"), r["labels"].get("arm"),
                               r["labels"].get("seed"), r["labels"].get("cell") or "base") for r in done)
    missing = want - got
    print("\nfinished %d of %d planned jobs" % (len(done), len(jobs)))
    if errors:
        print("!! worker errors: %s" % errors)
    if stopped:
        print("!! stopped early (control failure, GPU in use or worker error)")
    if missing:
        print("!! missing runs (dataset, scene, arm, seed, cell): %s" % dict(missing))
    ok = not errors and not stopped and not missing
    print("RESULT: %s\n" % ("all planned runs recorded" if ok else "INCOMPLETE, do not treat as finished"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
