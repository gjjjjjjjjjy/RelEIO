#!/usr/bin/env python3
"""Precision-policy diagnostic pilot on one GPU (screening, not a confirmatory study).

Arms (every FREQVIO_* switch is set explicitly; inherited FREQVIO_* variables are scrubbed):
  default            original precision (AMP on, hard-coded autocast on, TF32 defaults), traced
  fp32_strict        precision policy: AMP off + hard-coded autocast off + TF32 off, traced
  default_untraced   original precision, no tracer, no Cholesky probe (reference for the
                     possible effect of the tracer's GPU synchronisation)
A difference between `default` and `fp32_strict` is attributable to the precision policy as a
whole (AMP and TF32 change together), not to float16 alone.

Design: one physical GPU, one run at a time, each run an independent process, fixed clean input
and evaluation seed, arms interleaved (ABBA for the traced pair; the untraced arm inserted every
other round), all runs recorded in the append-only ledger (script/diag/run_ledger.py).
GPU sharing is checked before each run (the run is not started if another process is on the GPU),
sampled during it and checked after it. This is sampled evidence, not a guarantee of exclusive use;
request an exclusive GPU from the scheduler for the formal pilot.
After every run the controls are verified from its own log; the pilot STOPS at the first failed
verification (GPU shared, wrong precision policy, a dtype field missing or wrong, a checkpoint not
installed, tracer errors, Cholesky probe requested but not shown as enabled, tracer output in the
untraced arm). The summary re-verifies every run and leaves control-invalid runs out of all
counts and intervals; they stay in the ledger and are listed separately.

Examples (repository root, GPU node):
  python script/diag/precision_pilot.py --gpu 0 --dry-run
  python script/diag/precision_pilot.py --gpu 0 --n 1 --n-untraced 1 --cholinfo --ledger logs/ledger/smoke
  python script/diag/precision_pilot.py --gpu 0 --n 20 --n-untraced 10 --cholinfo --ledger logs/ledger/precision_pilot_4090
  python script/diag/precision_pilot.py --summary-only --ledger logs/ledger/precision_pilot_4090
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ledger  # noqa: E402

# Explicit values for every switch that can change the code path. Matches the frozen
# main-matrix runs at the clean level (keep=1.0, tq=0, FNO off, frequency weighting off).
# Voxelisation on the CPU: GPU voxelisation (index_add_ with atomics, not bit-reproducible)
# became the default only on 2026-06-24 (f0dd18f), after the frozen runs. Without this,
# every run here would rebuild its voxels on the GPU (FREQVIO_NO_VOXCACHE=1).
BASE_ENV = {
    "FREQVIO_USE_FNO": "0", "FREQVIO_FREQ_WEIGHT": "0", "FREQVIO_EVENT_KEEP": "1.0",
    "FREQVIO_TIME_QUANT_US": "0", "FREQVIO_NO_VOXCACHE": "1", "FREQVIO_STRICT_VI_INIT": "0",
    "FREQVIO_DEBUG_BA": "0", "FREQVIO_VOXEL_DEVICE": "cpu", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
}
ARM_ENV = {
    "default": {"FREQVIO_PRECISION": "default", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    "fp32_strict": {"FREQVIO_PRECISION": "fp32_strict", "FREQVIO_FORCE_FP32": "1", "FREQVIO_NUMTRACE": "1"},
    # split arms: which half of fp32_strict matters
    "amp_off": {"FREQVIO_PRECISION": "amp_off", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    "tf32_off": {"FREQVIO_PRECISION": "tf32_off", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    # TF32 split: cuBLAS matmul vs cuDNN, set in code
    "matmul_tf32_off": {"FREQVIO_PRECISION": "matmul_tf32_off", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    "cudnn_tf32_off": {"FREQVIO_PRECISION": "cudnn_tf32_off", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    # TF32 globally as in default, but not inside the visual BA call
    "ba_matmul_fp32": {"FREQVIO_PRECISION": "ba_matmul_fp32", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1"},
    # environment of a stock PyTorch install: the container's TF32 override variable removed, code unchanged
    "no_tf32_override": {"FREQVIO_PRECISION": "default", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "1",
                         "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None},
    "default_untraced": {"FREQVIO_PRECISION": "default", "FREQVIO_FORCE_FP32": "0", "FREQVIO_NUMTRACE": "0",
                         "FREQVIO_BA_CHOLINFO": "0"},
}
UNTRACED = "default_untraced"
TRACED_PAIR = ("default", "fp32_strict")
_TF32_OFF = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
EXPECTED_POLICY = {  # recorded precision_policy fields that must hold for each traced arm
    "default": {"arm": "default", "cfg_MIXED_PRECISION": True, "update_autocast": True},
    "fp32_strict": dict({"arm": "fp32_strict", "cfg_MIXED_PRECISION": False, "update_autocast": False}, **_TF32_OFF),
    "amp_off": {"arm": "amp_off", "cfg_MIXED_PRECISION": False, "update_autocast": False},
    "tf32_off": dict({"arm": "tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True}, **_TF32_OFF),
    "matmul_tf32_off": {"arm": "matmul_tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True,
                        "matmul_allow_tf32": False, "cudnn_allow_tf32": True},
    "cudnn_tf32_off": {"arm": "cudnn_tf32_off", "cfg_MIXED_PRECISION": True, "update_autocast": True,
                       "cudnn_allow_tf32": False},
    "ba_matmul_fp32": {"arm": "ba_matmul_fp32", "ba_local_matmul_fp32": True, "cfg_MIXED_PRECISION": True,
                       "update_autocast": True},
    # the code sees policy "default"; what differs is the environment, so the effective matmul TF32 must be off
    "no_tf32_override": {"arm": "default", "cfg_MIXED_PRECISION": True, "update_autocast": True,
                         "matmul_allow_tf32": False, "cudnn_allow_tf32": True},
}
# Every checkpoint utils/numtrace.py installs; all six must be present in a traced run.
REQUIRED_HOOKS = ("_wrap_reproject", "_wrap_corr", "_wrap_network_update",
                  "_wrap_fastba", "_wrap_run_dba", "_wrap_vi_alignment")
# Fields of the dtype_probe record; all must be present.
DTYPE_FIELDS = ("net", "delta", "weight", "corr_input", "imap_", "fmap1_", "update_params",
                "autocast_enabled_now")
_F32, _F16 = "torch.float32", "torch.float16"
EXPECTED_DTYPES = {
    # strict arm: every observed tensor float32, autocast off inside the update operator
    "fp32_strict": dict({k: _F32 for k in DTYPE_FIELDS if k != "autocast_enabled_now"},
                        autocast_enabled_now=False),
    # default arm: half-precision feature memory (cfg.MIXED_PRECISION), autocast on, float32
    # weights; net/delta/weight/corr dtypes under autocast are recorded but not prescribed
    "default": {"imap_": _F16, "fmap1_": _F16, "update_params": _F32, "autocast_enabled_now": True},
}
EXPECTED_DTYPES["amp_off"] = EXPECTED_DTYPES["fp32_strict"]    # AMP off: same tensors as strict
for _a in ("tf32_off", "matmul_tf32_off", "cudnn_tf32_off", "no_tf32_override", "ba_matmul_fp32"):
    EXPECTED_DTYPES[_a] = EXPECTED_DTYPES["default"]           # AMP still on: same tensors as default


def build_cmd(dataset, scene, seed, split_path, opts=()):
    """opts: config overrides passed as --opts KEY VALUE ... (the patch axis: PATCHES_PER_FRAME n)."""
    from script.run_degradation_sweep import DATASETS
    evalpy, config, inputdir, _, extra = DATASETS[dataset]
    cmd = ["python", evalpy, "--inputdir", inputdir, "--config", config, "--val_split", split_path,
           "--enable_event", "--network", "DEVO.pth", "--stride", "1", "--trials", "1",
           "--seed", str(seed)] + list(extra)
    if opts:
        cmd += ["--opts"] + [str(x) for x in opts]
    inputs = [config, "DEVO.pth"] + sorted(p for p in glob.glob(os.path.join(inputdir, scene, "*"))
                                           if os.path.isfile(p))
    return cmd, inputs


VOXEL_SOURCES = ("utils/event_utils.py", "utils/load_utils.py", "utils/event_stress.py")

# ----------------------------------------------------------------------------- stress cells
# A cell is one stress condition of the frozen main matrix:
#   keep:<ratio>   event thinning (FREQVIO_EVENT_KEEP), as script/run_degradation_sweep.py
#   tq:<us>        timestamp quantisation (FREQVIO_TIME_QUANT_US), as script/run_time_sweep.py
#   patch:<n>      PATCHES_PER_FRAME via --opts, as script/run_patch_sweep.py
#   base           the clean condition of rounds 1-2 (no stress overrides)
# keep:1.0, tq:0 and patch:96 are the same clean condition; the frozen matrix ran each of them as
# its own cell, and so does this.
CLEAN_PATCHES = 96   # PATCHES_PER_FRAME in config/{vector,hku,tumvie,mvsec,uzhfpv}.yaml
# gbias:<d> = clean input plus a constant gyro bias of d deg/s on each axis with these signs (x, y, z), passed in
# rad/s. Reference for the size: InvenSense MPU-6150 (the DAVIS240C IMU) initial zero-rate-output tolerance
# +-25 deg/s at 25 C (PS-MPU-6100A-00 Rev 1.1, section 6.1).
GBIAS_SIGNS = (1.0, -1.0, 1.0)
FROZEN_CELLS = ("keep:1.0", "keep:0.5", "keep:0.3", "keep:0.1", "tq:0", "tq:1000", "tq:5000", "tq:10000",
                "patch:96", "patch:64", "patch:32", "patch:16")
# New main matrix: the clean condition once ("clean" = keep 1.0, tq 0, 96 patches), shared
# by the three axes, plus the 9 degraded levels: 10 distinct conditions per sequence.
MAIN_CELLS = ("clean", "keep:0.5", "keep:0.3", "keep:0.1", "tq:1000", "tq:5000", "tq:10000",
              "patch:64", "patch:32", "patch:16")
CLEAN_CELLS = ("clean", "keep:1.0", "tq:0", "patch:96")


def parse_cell(spec):
    """'base', 'clean', 'keep:0.3', 'tq:5000' or 'patch:32' -> dict(cell, axis, level, keep, tq_us, patches, env, opts).
    'clean' is the clean condition of the new main matrix, set explicitly (keep 1.0, tq 0, 96 patches)."""
    spec = (spec or "base").strip()
    if spec == "base":
        return {"cell": "base", "axis": None, "level": None, "keep": 1.0, "tq_us": 0, "patches": CLEAN_PATCHES,
                "env": {}, "opts": []}
    if spec == "clean":
        return {"cell": "clean", "axis": "clean", "level": None, "keep": 1.0, "tq_us": 0, "patches": CLEAN_PATCHES,
                "env": {"FREQVIO_EVENT_KEEP": "1.0", "FREQVIO_TIME_QUANT_US": "0"},
                "opts": ["PATCHES_PER_FRAME", str(CLEAN_PATCHES)]}
    axis, sep, level = spec.partition(":")
    if axis == "gbias" and sep:
        try:
            d = float(level)
        except ValueError:
            d = float("nan")
        if not (d > 0 and math.isfinite(d)):
            raise ValueError("bad cell %r: gbias:<deg/s per axis, > 0>" % spec)
        vec = [g * d * math.pi / 180.0 for g in GBIAS_SIGNS]
        return {"cell": spec, "axis": "gbias", "level": level, "keep": 1.0, "tq_us": 0, "patches": CLEAN_PATCHES,
                "gyro_bias_rad_s": vec,
                "env": {"FREQVIO_EVENT_KEEP": "1.0", "FREQVIO_TIME_QUANT_US": "0",
                        "FREQVIO_IMU_GYRO_BIAS_RAD_S": ",".join("%.10g" % v for v in vec)},
                "opts": ["PATCHES_PER_FRAME", str(CLEAN_PATCHES)]}
    try:
        if not sep or axis not in ("keep", "tq", "patch"):
            raise ValueError
        if axis == "keep":
            v = float(level)
            if not 0 < v <= 1:
                raise ValueError
            return {"cell": spec, "axis": axis, "level": level, "keep": v, "tq_us": 0, "patches": CLEAN_PATCHES,
                    "env": {"FREQVIO_EVENT_KEEP": level}, "opts": []}
        v = int(level)
        if axis == "tq":
            if v < 0:
                raise ValueError
            return {"cell": spec, "axis": axis, "level": level, "keep": 1.0, "tq_us": v, "patches": CLEAN_PATCHES,
                    "env": {"FREQVIO_TIME_QUANT_US": str(v)}, "opts": []}
        if v <= 0:
            raise ValueError
        return {"cell": spec, "axis": axis, "level": level, "keep": 1.0, "tq_us": 0, "patches": v,
                "env": {}, "opts": ["PATCHES_PER_FRAME", str(v)]}
    except ValueError:
        raise ValueError("bad cell %r: use base, keep:<0-1], tq:<us >= 0> or patch:<n > 0>" % spec) from None


def stress_problems(facts, cell, traced=True):
    """Stress settings the run itself reported that differ from the requested cell (empty = OK).
    Checked: every [event-stress] banner (printed when an event iterator is built), the end-of-run
    [stress-evidence] line when present (a run that dies early may not reach it), and, in traced
    runs, PATCHES_PER_FRAME as recorded by the tracer."""
    c = parse_cell(cell)
    p = []
    st = facts.get("stress") or {}

    def num(d, key, cast):
        try:
            return cast(d.get(key))
        except (TypeError, ValueError):
            return None
    cfgs = st.get("config") or []
    if not cfgs:
        p.append("no [event-stress] banner (stress settings not confirmed by the estimator)")
    for d in cfgs:
        k, q = num(d, "keep", float), num(d, "tq_us", lambda x: int(float(x)))
        if k is None or q is None or abs(k - c["keep"]) > 1e-9 or q != c["tq_us"]:
            p.append("[event-stress] keep=%s tq_us=%s, expected keep=%s tq_us=%s"
                     % (d.get("keep"), d.get("tq_us"), c["keep"], c["tq_us"]))
            break
    ev = st.get("evidence")
    if ev:
        k, q = num(ev, "keep", float), num(ev, "tq_us", lambda x: int(float(x)))
        if k is None or q is None or abs(k - c["keep"]) > 1e-9 or q != c["tq_us"]:
            p.append("[stress-evidence] keep=%s tq_us=%s, expected keep=%s tq_us=%s"
                     % (ev.get("keep"), ev.get("tq_us"), c["keep"], c["tq_us"]))
        n_in = num(ev, "n_in", int) or 0
        if n_in > 0 and c["keep"] < 1.0:
            r = num(ev, "actual_keep", float)
            if r is None or abs(r - c["keep"]) > 0.15:
                p.append("actual kept fraction %s far from keep=%s" % (ev.get("actual_keep"), c["keep"]))
        if n_in > 0 and c["tq_us"] > 0:
            changed = num({"x": (ev.get("tq_changed") or "").split("/")[0]}, "x", int)
            if not changed:
                p.append("tq_us=%s but no timestamp changed (tq_changed=%s)" % (c["tq_us"], ev.get("tq_changed")))
    want, ims = c.get("gyro_bias_rad_s"), facts.get("imu_stress")
    if want:
        got = None
        try:
            got = [float(v) for v in str((ims or {}).get("gyro_bias_rad_s")).split(",")]
        except ValueError:
            pass
        if not ims:
            p.append("no [imu-stress] line (gyro bias not confirmed by the estimator)")
        elif not got or len(got) != 3 or any(abs(x - y) > 1e-9 for x, y in zip(got, want)):
            p.append("[imu-stress] gyro_bias_rad_s=%s, expected %s" % (ims.get("gyro_bias_rad_s"), want))
        elif ims.get("changed") != "1":
            p.append("gyro bias requested but the IMU array did not change")
    elif ims and ims.get("gyro_bias_rad_s") != "none":
        p.append("[imu-stress] gyro_bias_rad_s=%s in a cell without IMU stress" % ims.get("gyro_bias_rad_s"))
    if traced:
        pol = ((facts.get("numtrace") or {}).get("precision_policy") or {})
        if pol.get("cfg_PATCHES_PER_FRAME") != c["patches"]:
            p.append("PATCHES_PER_FRAME=%r recorded, expected %d" % (pol.get("cfg_PATCHES_PER_FRAME"), c["patches"]))
    return p



def voxcache_namespace():
    """Cache namespace tied to the voxelisation code and device, so a cached voxel file is only
    reused by runs whose voxels would have been computed by exactly the same code on the CPU."""
    import hashlib
    h = hashlib.sha256(BASE_ENV["FREQVIO_VOXEL_DEVICE"].encode())
    for rel in VOXEL_SOURCES:
        with open(os.path.join(ROOT, rel), "rb") as f:
            h.update(f.read())
    return "cpu" + h.hexdigest()[:10]


def arm_env(arm, cholinfo, voxcache=False, cell=None):
    """The exact overrides a run of this arm gets (the runner adds CUDA_VISIBLE_DEVICES and FREQVIO_RUN_ID).
    voxcache=True reuses CPU-built voxel caches in a namespace tied to the voxel code (identical inputs;
    checked per run by the tracer's input fingerprint; keep and tq are part of every cache file name).
    cell: stress condition (parse_cell); its environment overrides are applied on top of BASE_ENV."""
    env = dict(BASE_ENV, PYTHONPATH=ROOT, **ARM_ENV[arm])
    env.update(parse_cell(cell)["env"])
    if arm != UNTRACED:
        env["FREQVIO_BA_CHOLINFO"] = "1" if cholinfo else "0"
    if voxcache:
        env["FREQVIO_NO_VOXCACHE"] = "0"
        env["FREQVIO_VOXCACHE_NS"] = voxcache_namespace()
    return env


def schedule(n, n_untraced, arms=TRACED_PAIR):
    """Traced arms in rotated order each round (ABBA for two arms); one untraced run after odd
    rounds until n_untraced, the rest at the end."""
    arms = list(arms)
    order, u = [], 0
    for r in range(n):
        k = r % len(arms)
        order += arms[k:] + arms[:k]
        if r % 2 == 1 and u < n_untraced:
            order.append(UNTRACED)
            u += 1
    while u < n_untraced:
        order.append(UNTRACED)
        u += 1
    return order


def gpu_problems(facts):
    """GPU sharing seen during or after the run (sampled evidence, not a guarantee)."""
    w = facts.get("gpu_watch")
    if not w:
        return ["gpu_watch record missing (run not executed by ledger-2 or later)"]
    p = []
    if (w.get("max_pids_seen") or 0) >= 2:
        p.append("more than one process seen on the GPU during the run: %s" % w.get("other_pids_seen"))
    if w.get("exclusive_at_end") is False:
        p.append("process(es) on the GPU after the run ended: %s" % w.get("pids_at_end"))
    return p


def verify_controls(facts, arm, cholinfo, cell=None):
    """Problems that invalidate this run as a controlled observation (empty list = OK).
    cell: the stress condition the run was labelled with; None (ledgers before round 3) skips the
    stress checks, any other value (including 'base') requires the run's own stress reports to match."""
    p = gpu_problems(facts)
    nt = facts.get("numtrace")
    if cell is not None:
        p += stress_problems(facts, cell, traced=arm != UNTRACED)
        # round 3+: the default arm stands for the container environment, in which cuBLAS matmul TF32
        # is on (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1). If it is off, default and no_tf32_override
        # are the same treatment, so the run cannot serve as the TF32-on arm.
        pol = ((nt or {}).get("precision_policy") or {})
        if arm == "default" and nt and pol and pol.get("matmul_allow_tf32") is not True:
            p.append("default arm recorded matmul_allow_tf32=%r, expected True (container TF32 override "
                     "missing on this instance?)" % pol.get("matmul_allow_tf32"))
        # NVIDIA_TF32_OVERRIDE=0 disables TF32 globally, whatever torch reports (PyTorch CUDA environment variables)
        if arm == "default" and nt and pol and str(pol.get("NVIDIA_TF32_OVERRIDE") or "").strip() == "0":
            p.append("default arm ran with NVIDIA_TF32_OVERRIDE=0, which disables TF32 globally")
    if arm == UNTRACED:
        if nt:
            p.append("tracer output present in the untraced arm")
        if (facts.get("cholinfo") or {}).get("enabled_banner"):
            p.append("Cholesky probe active in the untraced arm")
        return p
    if not nt:
        return p + ["no [numtrace] records (tracer not active)"]
    ti = nt.get("trace_installed") or {}
    if not ti:
        p.append("trace_installed record missing")
    else:
        if ti.get("install_errors"):
            p.append("tracer install errors: %s" % ti["install_errors"])
        missing = [h for h in REQUIRED_HOOKS if h not in (ti.get("installed") or [])]
        if missing:
            p.append("checkpoints not installed: %s" % missing)
    pol = nt.get("precision_policy") or {}
    if not pol:
        p.append("precision_policy record missing")
    for k, v in EXPECTED_POLICY[arm].items():
        if pol and pol.get(k) != v:
            p.append("precision policy %s=%r, expected %r" % (k, pol.get(k), v))
    dp = nt.get("dtype_probe")
    if not dp:
        p.append("dtype_probe missing (update operator never observed)")
    else:
        absent = [k for k in DTYPE_FIELDS if dp.get(k) is None]
        if absent:
            p.append("dtype_probe fields missing: %s" % absent)
        wrong = {k: (dp.get(k), v) for k, v in EXPECTED_DTYPES[arm].items()
                 if k not in absent and dp.get(k) != v}
        if wrong:
            p.append("dtype_probe (observed, expected): %s" % wrong)
    rs = nt.get("run_summary")
    if not rs:
        p.append("run_summary missing")
    elif rs.get("tracer_errors"):
        p.append("tracer errors: %s" % {k: v for k, v in rs.get("counts", {}).items() if k.startswith("tracer_")})
    if cholinfo and not (facts.get("cholinfo") or {}).get("enabled_banner"):
        p.append("Cholesky probe requested but no enabled banner (cuda_ba not rebuilt, or BA path not reached)")
    return p


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _load_facts(ledger_dir, run_id):
    fp = os.path.join(ledger_dir, "runs", run_id, "facts.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        return json.load(fh)


def partition(ledger_dir):
    """Per arm: valid runs, control-invalid runs (with reasons) and not-run attempts.

    Controls are re-verified from each run's own facts, so the result does not depend on
    whether the pilot loop got as far as writing control_problems.json. Nothing is deleted;
    invalid runs are only kept out of the comparison.
    """
    out = collections.defaultdict(lambda: {"valid": [], "invalid": [], "not_run": []})
    for r in run_ledger.load_ledger(ledger_dir):
        arm = r["labels"].get("arm")
        g = out[arm]
        if run_ledger.outcome_category(r["outcome_hint"]) == "not_run":
            g["not_run"].append(r)
            continue
        f = _load_facts(ledger_dir, r["run_id"])
        if f is None:
            g["invalid"].append((r, ["facts.json missing"], None))
            continue
        if arm not in ARM_ENV:
            g["invalid"].append((r, ["unknown arm %r" % arm], f))
            continue
        problems = verify_controls(f, arm, bool(r["labels"].get("cholinfo")))
        if problems:
            g["invalid"].append((r, problems, f))
        else:
            g["valid"].append((r, f))
    return out


def arm_stats(valid):
    """Counts over control-valid runs only."""
    cats = collections.Counter(run_ledger.outcome_category(r["outcome_hint"]) for r, _ in valid)
    k, comp, und = cats.get("numerical_failure", 0), cats.get("completed", 0), cats.get("undetermined", 0)
    return {"categories": dict(cats), "failures": k, "determinate": k + comp, "undetermined": und,
            "wilson95": wilson(k, k + comp) if k + comp else None,
            "upper_if_undetermined_failed": (k + und, k + comp + und)}


def summarize(ledger_dir):
    recs = run_ledger.load_ledger(ledger_dir)
    if not recs:
        print("no runs in", ledger_dir)
        return {}
    print("ledger:", ledger_dir, "| runs:", len(recs),
          "| GPUs:", sorted({(r["gpu_name"], r["gpu_uuid"]) for r in recs}, key=str),
          "| code versions:", sorted({(r["git_commit"] or "")[:10] + "/" + (r["git_diff_sha256"] or "")[:8]
                                      for r in recs}),
          "| cuda_ba:", sorted({(r.get("cuda_ba_sha256") or "")[:12] for r in recs}))
    print("Counts, intervals and checkpoints below use control-valid runs only.")
    parts = partition(ledger_dir)
    result = {}
    for arm, g in sorted(parts.items(), key=lambda kv: str(kv[0])):
        valid, st = g["valid"], arm_stats(g["valid"])
        result[arm] = dict(st, n_valid=len(valid), n_invalid=len(g["invalid"]), n_not_run=len(g["not_run"]))
        print("\n[%s] valid=%d  control-invalid=%d  not-run=%d" % (arm, len(valid), len(g["invalid"]), len(g["not_run"])))
        print("    valid runs by category: " + "  ".join("%s=%d" % (c, st["categories"].get(c, 0))
                                                     for c in run_ledger.CATEGORIES if c != "not_run"))
        if st["determinate"]:
            lo, hi = st["wilson95"]
            print("    numerical failures among determinate valid runs: %d/%d (Wilson 95%% [%.0f%%, %.0f%%]); "
                  "if all %d undetermined runs were failures: %d/%d. Screening only."
                  % (st["failures"], st["determinate"], 100 * lo, 100 * hi, st["undetermined"],
                     *st["upper_if_undetermined_failed"]))
        for h, c in collections.Counter(r["outcome_hint"] for r, _ in valid).most_common():
            print("    %3d  %s" % (c, h))
        done = sorted(r["mpe"] for r, _ in valid if r["outcome_hint"] == "completed")
        if done:
            print("    completed MPE: min %.3f / median %.3f / max %.3f" % (done[0], done[len(done) // 2], done[-1]))
        firsts, chol_flag, chol_on, near, blind = collections.Counter(), 0, 0, 0, 0
        for _, f in valid:
            nt = f.get("numtrace") or {}
            fd = nt.get("first_detected_nonfinite")
            firsts["%s:%s" % (fd.get("where"), fd.get("tensor")) if fd else "none"] += 1
            ch = f.get("cholinfo") or {}
            chol_on += bool(ch.get("enabled_banner"))
            chol_flag += ch.get("flagged", 0) > 0
            near += bool((nt.get("run_summary") or {}).get("fp16_near_overflow"))
            w = f.get("gpu_watch") or {}
            blind += bool(w.get("n_samples")) and not w.get("child_pid_seen") and not w.get("max_pids_seen")
        if valid:
            print("    first DETECTED non-finite checkpoint (not necessarily the origin):", dict(firsts))
            print("    Cholesky probe enabled in %d run(s); runs with Cholesky flags: %d; runs with fp16 near-overflow: %d"
                  % (chol_on, chol_flag, near))
            if blind:
                print("    !! in %d run(s) nvidia-smi never listed any process on the GPU, not even the run itself: "
                      "the sharing check could not see processes (container?); rely on the scheduler allocation" % blind)
        if g["invalid"]:
            print("    EXCLUDED (control verification failed; kept in the ledger, not in any count above):")
            for r, probs, _ in g["invalid"]:
                print("      %s  %-40s %s" % (r["run_id"], r["outcome_hint"], "; ".join(probs)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--dataset", default="vector")
    ap.add_argument("--scene", default="desk_normal1")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n", type=int, default=20, help="runs per traced arm (default, fp32_strict)")
    ap.add_argument("--n-untraced", type=int, default=10, help="runs of the untraced default arm")
    ap.add_argument("--ledger", default="logs/ledger/precision_pilot")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--cholinfo", action="store_true", help="enable the Cholesky probe in traced arms (rebuilt cuda_ba)")
    ap.add_argument("--gpu-poll", type=float, default=30, help="seconds between GPU process samples during a run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    os.chdir(ROOT)
    if a.summary_only:
        summarize(a.ledger)
        return 0
    if a.gpu is None:
        ap.error("--gpu is required")
    split = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    split.write(a.scene + "\n")
    split.close()
    cmd, inputs = build_cmd(a.dataset, a.scene, a.seed, split.name)
    order = schedule(a.n, a.n_untraced)
    print("plan: %d runs on GPU %d, %s/%s seed %d" % (len(order), a.gpu, a.dataset, a.scene, a.seed))
    print("order:", " ".join({"default": "D", "fp32_strict": "F", "amp_off": "A", "tf32_off": "T",
                              "matmul_tf32_off": "M", "cudnn_tf32_off": "C", "no_tf32_override": "N", "ba_matmul_fp32": "B",
                              UNTRACED: "u"}[x] for x in order))
    print("cmd:", " ".join(cmd))
    for arm in ARM_ENV:
        env = arm_env(arm, a.cholinfo)
        print("env[%s]: %s" % (arm, " ".join("%s=%s" % kv for kv in sorted(env.items()) if kv[0].startswith("FREQVIO_"))))
    if a.dry_run:
        os.unlink(split.name)
        return 0
    os.makedirs(a.ledger, exist_ok=True)
    status = 0
    try:
        for i, arm in enumerate(order):
            env = arm_env(arm, a.cholinfo)
            labels = {"experiment": "precision_pilot", "arm": arm, "dataset": a.dataset, "scene": a.scene,
                      "seed": a.seed, "order_index": i, "cholinfo": a.cholinfo and arm != "default_untraced"}
            rec = run_ledger.run(cmd, env, a.gpu, a.ledger, labels, inputs=inputs, timeout=a.timeout,
                                 require_exclusive=True, gpu_poll_s=a.gpu_poll)
            print("[%2d/%d] %-16s %s  %-45s mpe=%s  %ss" % (i + 1, len(order), arm, rec["run_id"],
                                                            rec["outcome_hint"], rec["mpe"], rec["wall_s"]), flush=True)
            if run_ledger.outcome_category(rec["outcome_hint"]) == "not_run":
                print("STOP: GPU %d is not exclusive (see manifest). Free the card and rerun." % a.gpu)
                status = 2
                break
            with open(os.path.join(a.ledger, "runs", rec["run_id"], "facts.json")) as fh:
                facts = json.load(fh)
            problems = verify_controls(facts, arm, labels["cholinfo"])
            if problems:
                with open(os.path.join(a.ledger, "runs", rec["run_id"], "control_problems.json"), "w") as f:
                    json.dump(problems, f, indent=1)
                print("STOP: control verification failed for run %s:" % rec["run_id"])
                for pr in problems:
                    print("   -", pr)
                status = 3
                break
    finally:
        os.unlink(split.name)
    summarize(a.ledger)
    return status


if __name__ == "__main__":
    sys.exit(main())
