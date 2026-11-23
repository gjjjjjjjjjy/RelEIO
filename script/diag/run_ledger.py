#!/usr/bin/env python3
"""Append-only run ledger for new experiments.

Every run gets its own directory  <ledger>/runs/<run_id>/  that is never reused:
  manifest.json  what was run: command, env overrides, code state (git commit, dirty-diff
                 hash, hashes of key source files and of the compiled cuda_ba module),
                 Python/torch/gtsam versions, GPU name/UUID/driver, whether other
                 processes were on the GPU at the start, input files.
  run.log        full stdout+stderr.
  facts.json     what happened, parsed from the log by the explicit rules in parse_facts():
                 exit code, timeout, MPE, APE RMSE, exception type/message/site, counts of
                 known warnings, numtrace / cholinfo records, GPU processes sampled during
                 the run and checked after it (gpu_watch). No paper outcome class is
                 assigned here; `outcome_hint` is a convenience label only.
One JSON line per run is appended to <ledger>/ledger.jsonl (file-locked).
Retries are new runs with new ids; nothing is replaced.

Usage as a library: see script/diag/precision_pilot.py.
CLI:  python script/diag/run_ledger.py --summarize logs/ledger/precision_pilot
"""
import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid

LEDGER_VERSION = "ledger-4"
KEY_SOURCES = ["devo/dba.py", "utils/eval_utils.py", "utils/numtrace.py", "utils/event_stress.py",
               "dpvo/fastba/ba_cuda.cu", "dpvo/fastba/ba.cpp", "devo/enet.py", "devo/selector.py"]
_HASH_CACHE = {}


# ----------------------------------------------------------------------------- hashing
def file_sha256(path, chunk=1 << 22):
    """SHA-256 of a file, cached per (path, size, mtime) within this process."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_size, st.st_mtime)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def _run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


# ----------------------------------------------------------------------------- environment
def code_state(root="."):
    diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--binary"], capture_output=True)
    status = _run(["git", "-C", root, "status", "--porcelain"]) or ""
    return {
        "git_commit": _run(["git", "-C", root, "rev-parse", "HEAD"]),
        "git_dirty_files": [l for l in status.splitlines() if l.strip()],
        "git_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
        "source_sha256": {p: file_sha256(os.path.join(root, p)) for p in KEY_SOURCES},
    }


_PY_PROBE = r"""
import json, importlib
out = {}
for m in ("torch", "numpy", "scipy", "gtsam", "evo"):
    try:
        mod = importlib.import_module(m); out[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        out[m] = "unavailable: %s" % type(e).__name__
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version()
except Exception:
    pass
try:
    import cuda_ba
    out["cuda_ba_file"] = cuda_ba.__file__
except Exception as e:
    out["cuda_ba_file"] = "unavailable: %s" % type(e).__name__
print(json.dumps(out))
"""


def python_env(python="python"):
    """Versions as seen by the interpreter that will run the job (probed once per process)."""
    if "_pyenv" not in _HASH_CACHE:
        raw = _run([python, "-c", _PY_PROBE], timeout=300)
        info = json.loads(raw) if raw else {"probe": "failed"}
        so = info.get("cuda_ba_file")
        if so and not str(so).startswith("unavailable"):
            info["cuda_ba_sha256"] = file_sha256(so)
        _HASH_CACHE["_pyenv"] = info
    return _HASH_CACHE["_pyenv"]


def gpu_info(gpu):
    """Name/UUID/driver of the physical GPU and processes currently on it."""
    q = _run(["nvidia-smi", "-i", str(gpu), "--query-gpu=index,name,uuid,driver_version,memory.total",
              "--format=csv,noheader"])
    info = {"cuda_visible_devices": str(gpu), "nvidia_smi": q}
    if q:
        parts = [p.strip() for p in q.split(",")]
        if len(parts) >= 5:
            info.update(index=parts[0], name=parts[1], uuid=parts[2], driver=parts[3], memory=parts[4])
    apps = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
    others = [a for a in (apps or "").splitlines() if info.get("uuid") and info["uuid"] in a]
    info["other_processes_at_start"] = others
    info["exclusive_at_start"] = (apps is not None) and not others
    return info


def gpu_pids(uuid):
    """PIDs of compute processes on the GPU with this UUID; None if nvidia-smi fails."""
    apps = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
    if apps is None:
        return None
    pids = []
    for line in apps.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == uuid:
            pids.append(parts[1])
    return pids


def gpu_mem_used(uuid):
    """Device memory in use on the GPU with this UUID (MiB, nvidia-smi memory.used); None if unavailable.
    With the GPU exclusive to the run this is the run's footprint including the CUDA context, which the
    PyTorch allocator peaks recorded by numtrace do not include."""
    q = _run(["nvidia-smi", "-i", str(uuid), "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(q.strip().splitlines()[0])
    except (AttributeError, ValueError, IndexError):
        return None


def wait_and_watch(proc, uuid, timeout, poll_s):
    """Wait for the child and sample, every poll_s seconds, which processes are on its GPU.

    Sampling is evidence, not a guarantee: a process that comes and goes between two samples
    is missed, and inside a container nvidia-smi may not list other containers' processes
    (or may list the child under a different PID). `max_pids_seen >= 2` is the unambiguous
    signal; `child_pid_seen` false with samples taken means the check cannot see processes.
    After the child exits, any process still on the GPU is someone else's.
    """
    t0 = time.time()
    w = {"uuid": uuid, "poll_s": poll_s, "child_pid": proc.pid, "n_samples": 0, "failed_samples": 0,
         "max_pids_seen": 0, "child_pid_seen": False, "other_pids_seen": [],
         "mem_samples": 0, "mem_used_mib_max": None}
    timed_out = False
    while True:
        remaining = timeout - (time.time() - t0)
        if remaining <= 0:
            proc.kill()
            proc.wait()
            timed_out = True
            break
        try:
            proc.wait(timeout=min(poll_s, remaining))
            break
        except subprocess.TimeoutExpired:
            pass
        if not uuid:
            continue
        # sampled every poll_s: the caching allocator rarely returns memory, so usage grows nearly
        # monotonically and the largest sample is close to the peak (a lower bound, not the exact peak)
        m = gpu_mem_used(uuid)
        if m is not None:
            w["mem_samples"] += 1
            w["mem_used_mib_max"] = m if w["mem_used_mib_max"] is None else max(w["mem_used_mib_max"], m)
        pids = gpu_pids(uuid)
        if pids is None:
            w["failed_samples"] += 1
            continue
        w["n_samples"] += 1
        w["max_pids_seen"] = max(w["max_pids_seen"], len(set(pids)))
        w["child_pid_seen"] = w["child_pid_seen"] or str(proc.pid) in pids
        w["other_pids_seen"] += [x for x in pids if x != str(proc.pid) and x not in w["other_pids_seen"]]
    end = gpu_pids(uuid) if uuid else None
    if end:  # allow the driver a moment to release the child's context
        time.sleep(2)
        end = gpu_pids(uuid)
    w["pids_at_end"] = end
    w["mem_used_mib_at_end"] = gpu_mem_used(uuid) if uuid else None   # idle baseline once the child is gone
    w["exclusive_at_end"] = None if end is None else not end
    return (None if timed_out else proc.returncode), timed_out, w


# ----------------------------------------------------------------------------- log parsing
_RE_MPE = re.compile(r"MPE is\s+([-+0-9.eE]+|nan|inf)")
_RE_RMSE = re.compile(r"^\s*rmse\s+([-+0-9.eE]+)", re.M)
_RE_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')
_RE_EXC = re.compile(r"^(\w+(?:Error|Exception|Interrupt))(?::\s*(.*))?$", re.M)
_RE_TAGGED = re.compile(r"^\[(numtrace|cholinfo)\] (\{.*\})\s*$", re.M)
# not anchored at the line start: tqdm writes its first progress bar ("0it [00:00, ?it/s]", no newline)
# before a lazily started event iterator prints its banner, so the banner can follow it on the same line
# (as script/stress_evidence.py, which parsed the frozen logs, already allowed)
_RE_STRESS_CFG = re.compile(r"\[event-stress\] ([^\r\n]*)")
_RE_STRESS_EV = re.compile(r"\[stress-evidence\] ([^\r\n]*)")
_RE_IMU_STRESS = re.compile(r"\[imu-stress\] ([^\r\n]*)")
_RE_KV = re.compile(r"(\w+)=(\S+)")


def _short(path):
    for anchor in ("/devo/", "/dpvo/", "/utils/", "/script/", "/scripts/"):
        if anchor in path:
            return anchor.strip("/") + "/" + path.split(anchor, 1)[1]
    return os.path.basename(path)


def parse_facts(text):
    """Facts from a run log. Rules are explicit so that every label can be audited."""
    facts = {}
    mpes = _RE_MPE.findall(text)
    facts["mpe"] = float(mpes[-1]) if mpes else None
    rmses = _RE_RMSE.findall(text)
    facts["ape_rmse_m"] = float(rmses[-1]) if rmses else None
    facts["runner_timeout"] = "[runner-timeout]" in text
    exc = None
    if "Traceback (most recent call last)" in text:
        tail = text[text.rfind("Traceback (most recent call last)"):]
        frames = _RE_FRAME.findall(tail)
        excs = _RE_EXC.findall(tail)
        ours = [(f, ln, fn) for f, ln, fn in frames
                if any(a in f for a in ("/devo/", "/dpvo/", "/utils/", "/script/"))]
        exc = {"type": excs[-1][0] if excs else None,
               "message": (excs[-1][1] or "")[:200] if excs else None,
               # deepest frame in this repository's code (library frames such as scipy skipped)
               "site": ("%s:%s:%s" % (_short(ours[-1][0]), ours[-1][1], ours[-1][2])) if ours else
                       (("%s:%s:%s" % (_short(frames[-1][0]), frames[-1][1], frames[-1][2])) if frames else None),
               "estimator_frames": ["%s:%s" % (_short(f), fn) for f, _, fn in ours][-6:]}
    facts["exception"] = exc
    facts["counts"] = {
        "warning_ba_failed": text.count("Warning BA failed"),
        "vi_init_success": text.count("V-I successfully initialized"),
        "vi_init_rejected": text.count("V-I init REJECTED"),
        "imu_excitation_not_enough": text.count("IMU excitation not enough"),
        "nan_gravity_print": text.count("the calculate g [nan"),
    }
    tagged = {"numtrace": [], "cholinfo": []}
    for tag, payload in _RE_TAGGED.findall(text):
        try:
            tagged[tag].append(json.loads(payload))
        except ValueError:
            pass
    nt = tagged["numtrace"]

    def first(kind):
        return next((r for r in nt if r.get("kind") == kind), None)

    probes = [r for r in nt if r.get("kind") == "vi_alignment_probe"]
    facts["numtrace"] = {
        "trace_installed": first("trace_installed"),
        "precision_policy": first("precision_policy"),
        "dtype_probe": first("dtype_probe"),
        "first_detected_nonfinite": first("first_detected_nonfinite"),
        "vi_alignment_in": [r for r in nt if r.get("kind") == "vi_alignment_in"][-4:],
        "vi_alignment_exception": first("vi_alignment_exception"),
        # every probe of the alignment call that raised (or the last call), in order
        "vi_alignment_probes": [r for r in probes if r.get("call") == probes[-1].get("call")] if probes else [],
        "run_summary": first("run_summary"),
    } if nt else None
    ch = tagged["cholinfo"]
    flags = [r for r in ch if r.get("kind") != "enabled"]
    facts["cholinfo"] = {"enabled_banner": any(r.get("kind") == "enabled" for r in ch),
                         "flagged": len(flags), "first_flag": flags[0] if flags else None}
    # stress settings as the estimator reported them: one [event-stress] line per event iterator, and the
    # end-of-run [stress-evidence] line (also printed on the crash path)
    evs = _RE_STRESS_EV.findall(text)
    facts["stress"] = {"config": [dict(_RE_KV.findall(x)) for x in _RE_STRESS_CFG.findall(text)][:8],
                       "evidence": dict(_RE_KV.findall(evs[-1])) if evs else None}
    ims = _RE_IMU_STRESS.findall(text)   # gyro bias as run_DEIO2 applied it (absent in logs before 2026-11-21)
    facts["imu_stress"] = dict(_RE_KV.findall(ims[-1])) if ims else None
    facts["outcome_hint"] = outcome_hint(facts)
    return facts


def outcome_hint(f, exit_code=None):
    """Convenience label, not a paper outcome class. Rules are applied in this order.

    exit_code=None means "unknown" (historical logs); a known non-zero exit code is never
    labelled completed, even if an MPE line was printed.
    """
    if f.get("runner_timeout"):
        return "timeout"
    e = f.get("exception")
    if e:
        msg, site = e.get("message") or "", e.get("site") or ""
        if e.get("type") == "ValueError" and "zero norm quaternion" in msg and "FromTwoVectors" in site:
            return "numerical_failure:vi_alignment_nan_gravity"
        if e.get("type") in ("RuntimeError", "OSError", "FileNotFoundError", "MemoryError") and any(
                k in msg for k in ("PytorchStreamReader", "out of memory", "No such file",
                                   "CUDA error", "cuDNN error", "NCCL")):
            return "infrastructure:" + e["type"]
        return "undetermined:exception:%s@%s" % (e.get("type"), site)
    mpe = f.get("mpe")
    if mpe is not None and mpe != mpe:
        return "undetermined:mpe_nan"
    if mpe is not None and mpe in (float("inf"), float("-inf")):
        return "undetermined:mpe_inf"
    if mpe is not None:
        if exit_code not in (None, 0):
            return "undetermined:nonzero_exit_after_mpe"
        return "completed"
    if exit_code not in (None, 0):
        return "undetermined:nonzero_exit_no_exception"
    return "undetermined:no_mpe_no_exception"


CATEGORIES = ("completed", "numerical_failure", "infrastructure", "timeout", "undetermined", "not_run")


def outcome_category(hint):
    """Map a hint to one of CATEGORIES. Undetermined runs are never counted as successes."""
    hint = str(hint)
    for c in CATEGORIES:
        if hint == c or hint.startswith(c + ":"):
            return c
    return "undetermined"


# ----------------------------------------------------------------------------- running
def new_run_id():
    # sortable by start time (microseconds), unique by the random suffix
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]


def _append_jsonl(path, record):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


SCRUB_PREFIXES = ("FREQVIO_",)  # never inherited: every FREQVIO_* switch must be set explicitly
RECORD_PREFIXES = ("FREQVIO_", "CUDA_", "CUBLAS_", "CUDNN_", "NVIDIA_", "PYTORCH_", "TORCH_",
                   "OMP_", "MKL_", "OPENBLAS_", "PYTHONPATH", "PYTHONHASHSEED", "LD_LIBRARY_PATH",
                   "SLURM_JOB", "SLURM_STEP", "SLURM_GPUS", "PBS_JOBID")  # scheduler job/GPU allocation


def build_env(env_overrides, gpu, run_id):
    """Child environment: inherited env minus FREQVIO_*, plus the explicit overrides (value None = unset)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUB_PREFIXES)}
    for k, v in env_overrides.items():
        if v is None:          # None removes an inherited variable (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE)
            env.pop(k, None)
        else:
            env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FREQVIO_RUN_ID"] = run_id
    return env


def effective_env(env):
    """The subset of the child environment that can influence numerics or code paths."""
    return {k: env[k] for k in sorted(env) if k.startswith(RECORD_PREFIXES)}


def run(cmd, env_overrides, gpu, ledger_dir, labels, inputs=(), timeout=1800, python="python",
        require_exclusive=True, gpu_poll_s=30):
    """Run one job, record everything, return the ledger record. Never overwrites."""
    run_id = new_run_id()
    rdir = os.path.join(ledger_dir, "runs", run_id)
    os.makedirs(rdir, exist_ok=False)
    env = build_env(env_overrides, gpu, run_id)
    env["FREQVIO_RUN_DIR"] = os.path.abspath(rdir)   # the evaluation keeps its trajectory here (traj_est_tum.txt)
    scrubbed = sorted(k for k in os.environ if k.startswith(SCRUB_PREFIXES) and k not in env_overrides)
    g = gpu_info(gpu)
    manifest = {
        "ledger_version": LEDGER_VERSION, "run_id": run_id, "labels": labels,
        "cmd": cmd, "env_overrides": env_overrides, "effective_env": effective_env(env),
        "scrubbed_inherited_vars": scrubbed,
        "cwd": os.getcwd(), "host": socket.gethostname(),
        "gpu": g, "code": code_state(), "python_env": python_env(python),
        "inputs": {p: {"sha256": file_sha256(p), "bytes": os.path.getsize(p) if os.path.exists(p) else None}
                   for p in inputs},
        "timeout_s": timeout, "started_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(rdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True, default=str)
    watch = None
    if require_exclusive and not g.get("exclusive_at_start"):
        facts = {"skipped": "gpu_not_exclusive", "outcome_hint": "not_run:gpu_not_exclusive"}
        exit_code, wall = None, 0.0
    else:
        log = os.path.join(rdir, "run.log")
        t0 = time.time()
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            exit_code, timed_out, watch = wait_and_watch(proc, g.get("uuid"), timeout, gpu_poll_s)
        wall = time.time() - t0
        if timed_out:
            with open(log, "a") as lf:
                lf.write("\n[runner-timeout] killed after %.0fs\n" % timeout)
        with open(log, errors="ignore") as lf:
            facts = parse_facts(lf.read())
        facts["outcome_hint"] = outcome_hint(facts, exit_code=exit_code)  # now the exit code is known
    facts.update({"exit_code": exit_code, "wall_s": round(wall, 2), "gpu_watch": watch,
                  "outcome_category": outcome_category(facts["outcome_hint"]),
                  "finished_utc": _dt.datetime.utcnow().isoformat() + "Z"})
    with open(os.path.join(rdir, "facts.json"), "w") as f:
        json.dump(facts, f, indent=1, sort_keys=True, default=str)
    record = {"run_id": run_id, "labels": labels, "gpu_uuid": g.get("uuid"), "gpu_name": g.get("name"),
              "exclusive_at_start": g.get("exclusive_at_start"),
              "exclusive_at_end": (watch or {}).get("exclusive_at_end"),
              "gpu_max_pids_seen": (watch or {}).get("max_pids_seen"),
              "git_commit": manifest["code"]["git_commit"],
              "git_diff_sha256": manifest["code"]["git_diff_sha256"],
              "cuda_ba_sha256": manifest["python_env"].get("cuda_ba_sha256"),
              "outcome_hint": facts.get("outcome_hint"), "outcome_category": facts["outcome_category"],
              "mpe": facts.get("mpe"),
              "exit_code": exit_code, "wall_s": facts["wall_s"]}
    _append_jsonl(os.path.join(ledger_dir, "ledger.jsonl"), record)
    return record


# ----------------------------------------------------------------------------- summary
def load_ledger(ledger_dir):
    path = os.path.join(ledger_dir, "ledger.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", required=True, help="ledger directory")
    a = ap.parse_args()
    recs = load_ledger(a.summarize)
    print("%d runs in %s" % (len(recs), a.summarize))
    for r in recs:
        print("%s  %-28s  %-45s  mpe=%s  %ss" % (r["run_id"], json.dumps(r["labels"], sort_keys=True)[:28],
                                               r["outcome_hint"], r["mpe"], r["wall_s"]))


if __name__ == "__main__":
    sys.exit(main())
