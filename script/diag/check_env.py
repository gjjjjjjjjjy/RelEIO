#!/usr/bin/env python3
"""Is this instance's Python environment able to run the evaluation, and is it the same as a reference?

1. Imports every DEIO evaluation script of script/run_degradation_sweep.DATASETS as a module
   (its main block is not run), so a missing package fails here, not 0.4 s into the first run.
2. Checks that gtsam is the patched 4.3.0 build (CombinedImuFactor.evaluateErrorCustom, used by the
   visual-inertial initialisation; the stock pip gtsam lacks it, docs/GTSAM_BUILD_NOTES.md).
3. With --reference LEDGER: compares the library versions recorded in that ledger's manifests
   (torch, CUDA, cuDNN, numpy, scipy, gtsam, evo) with this instance. For a cross-hardware comparison
   they must be equal; otherwise hardware and software differences cannot be separated.
Exit code 0 only if everything imports and (with --reference) every version is equal.

  python script/diag/check_env.py --reference logs/ledger/round3a_20261120_062351
"""
import argparse
import glob
import importlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ledger  # noqa: E402

COMPARED = ("torch", "torch_cuda", "cudnn", "numpy", "scipy", "gtsam", "evo")
EXTENSIONS = ("cuda_ba", "cuda_corr", "lietorch_backends", "lietorch")   # built by `pip install .` (run_pilot_all.sh)


def import_eval_scripts(before_build=False):
    from script.run_degradation_sweep import DATASETS
    problems = []
    for ds, (evalpy, *_rest) in sorted(DATASETS.items()):
        path = os.path.join(ROOT, evalpy)
        try:
            spec = importlib.util.spec_from_file_location("_check_%s" % ds, path)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            print("  import %-9s %-34s OK" % (ds, evalpy))
        except (Exception, SystemExit) as e:  # noqa: BLE001 - every failure is reported
            if before_build and isinstance(e, ModuleNotFoundError) and e.name in EXTENSIONS:
                print("  import %-9s %-34s not yet: extension %s is built in the next step" % (ds, evalpy, e.name))
                continue
            problems.append("%s (%s): %s: %s" % (ds, evalpy, type(e).__name__, e))
            print("  import %-9s %-34s FAILED  %s: %s" % (ds, evalpy, type(e).__name__, e))
    return problems


def gtsam_patched():
    try:
        import gtsam
    except Exception as e:  # noqa: BLE001
        return "gtsam not importable: %s: %s" % (type(e).__name__, e)
    ok = hasattr(gtsam.gtsam.CombinedImuFactor, "evaluateErrorCustom")
    print("  gtsam %s at %s | evaluateErrorCustom: %s" % (getattr(gtsam, "__version__", "?"), gtsam.__file__, ok))
    return None if ok else "gtsam %s lacks evaluateErrorCustom (stock build, not thirdparty/gtsam 4.3.0)" % (
        getattr(gtsam, "__version__", "?"))


def compare_with(ledger):
    mans = sorted(glob.glob(os.path.join(ledger, "runs", "*", "manifest.json")))
    if not mans:
        return ["reference ledger %s has no manifests" % ledger]
    with open(mans[0]) as f:
        ref = json.load(f).get("python_env") or {}
    here = run_ledger.python_env(sys.executable)
    problems = []
    print("  library versions (reference %s | this instance):" % os.path.basename(ledger.rstrip("/")))
    for k in COMPARED:
        same = ref.get(k) == here.get(k)
        print("    %-10s %-30s %-30s %s" % (k, ref.get(k), here.get(k), "" if same else "<-- differs"))
        if not same:
            problems.append("%s: reference %s, here %s" % (k, ref.get(k), here.get(k)))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", help="ledger whose recorded library versions this instance must match")
    ap.add_argument("--before-build", action="store_true",
                    help="the CUDA extensions are not built yet: their absence is not a problem")
    a = ap.parse_args()
    os.chdir(ROOT)
    problems = import_eval_scripts(a.before_build)
    g = gtsam_patched()
    if g:
        problems.append(g)
    if a.reference:
        problems += compare_with(a.reference)
    print("RESULT: %s" % ("environment OK" if not problems else "%d problem(s):\n  - %s" % (
        len(problems), "\n  - ".join(problems))))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
