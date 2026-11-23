"""Opt-in numerical diagnostics and precision policy for DEIO runs.

Nothing here changes estimator values. Everything is off unless an environment
variable is set, so default runs keep the original code path:

  FREQVIO_NUMTRACE=1      record where non-finite values are first DETECTED among the
                          instrumented checkpoints (in pipeline order: reprojection, correlation,
                          update-operator inputs, update-operator outputs, visual BA in/out, IMU BA
                          in/out, per-frame state, visual-inertial alignment inputs and internals),
                          dtype probes and run statistics. "First detected" is not proof of origin:
                          a value can become non-finite between two checkpoints.
  FREQVIO_FORCE_FP32=1    precision-policy arm: cfg.MIXED_PRECISION=False, the hard-coded autocast
                          in DBA.update disabled (see update_autocast_enabled), TF32 off for matmul
                          and cuDNN, float32 matmul precision "highest". This changes AMP and TF32
                          together, so a difference between arms is attributable to the precision
                          policy as a whole, not to float16 alone.
  FREQVIO_NUMTRACE_OUT    optional JSONL file that receives the same records.

Records are single lines "[numtrace] {json}" so the run log alone suffices for analysis
(script/diag/run_ledger.py parses them). Checks call .item() and therefore synchronize the GPU.
That does not modify tensors, but in a program that is already nondeterministic it may change
scheduling; keep an untraced reference arm (see script/diag/precision_pilot.py).
"""
import json
import os
import sys
import time
import traceback

ENV_TRACE = "FREQVIO_NUMTRACE"
ENV_FP32 = "FREQVIO_FORCE_FP32"
ENV_OUT = "FREQVIO_NUMTRACE_OUT"
VERSION = "numtrace-7"
MAX_EVENTS = 60          # cap on logged non-finite events per run (the first is always logged)
FP16_NEAR_MAX = 6.0e4    # float16 max is 65504; values above this are close to overflow

# Evaluated once at import; the environment is fixed before the process starts.
ACTIVE = os.environ.get(ENV_TRACE, "0") == "1"
_CURRENT = None          # the installed tracer (one estimator per process)


def trace_enabled():
    return os.environ.get(ENV_TRACE, "0") == "1"


ENV_POLICY = "FREQVIO_PRECISION"
# policy -> (mixed precision off, matmul TF32 off, cuDNN TF32 off). "default" leaves everything as in
# the original code and the environment (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 in NGC images).
POLICIES = {"default": (False, False, False), "amp_off": (True, False, False), "tf32_off": (False, True, True),
            "fp32_strict": (True, True, True), "matmul_tf32_off": (False, True, False),
            "cudnn_tf32_off": (False, False, True),
            # global settings as default; only inside the visual BA call (dpvo.fastba.BA) the cuBLAS
            # matmuls run without TF32 (targeted test of the Schur-complement products in cuda_ba)
            "ba_matmul_fp32": (False, False, False)}
BA_LOCAL_FP32 = {"ba_matmul_fp32"}


def precision_policy():
    """Active precision policy. FREQVIO_PRECISION wins; FREQVIO_FORCE_FP32=1 is the older
    spelling of fp32_strict. Contradictory settings raise instead of guessing."""
    p = os.environ.get(ENV_POLICY, "").strip()
    legacy = os.environ.get(ENV_FP32, "0") == "1"
    if not p:
        return "fp32_strict" if legacy else "default"
    if p not in POLICIES:
        raise ValueError("%s=%r, expected one of %s" % (ENV_POLICY, p, sorted(POLICIES)))
    if legacy and p != "fp32_strict":
        raise ValueError("%s=1 contradicts %s=%s" % (ENV_FP32, ENV_POLICY, p))
    return p


def amp_disabled():
    return POLICIES[precision_policy()][0]


def matmul_tf32_disabled():
    return POLICIES[precision_policy()][1]


def cudnn_tf32_disabled():
    return POLICIES[precision_policy()][2]


def tf32_disabled():
    """Both TF32 switches off (kept for the first callers)."""
    return matmul_tf32_disabled() and cudnn_tf32_disabled()


def force_fp32():
    """Kept for callers of the first version: true only for the full fp32_strict policy."""
    return precision_policy() == "fp32_strict"


def update_autocast_enabled():
    """Value for the autocast that DBA.update originally hard-codes to True."""
    return not amp_disabled()


def _emit(kind, payload):
    rec = {"kind": kind, "t_wall": round(time.time(), 3)}
    rec.update(payload)
    line = "[numtrace] " + json.dumps(rec, default=str, sort_keys=True)
    print(line, flush=True)
    out = os.environ.get(ENV_OUT)
    if out:
        try:
            with open(out, "a") as f:
                f.write(line[len("[numtrace] "):] + "\n")
        except OSError:
            pass


def _torch_settings():
    import torch
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def _install_ba_local_fp32(torch):
    """Wrap dpvo.fastba.BA so that cuBLAS matmuls inside it run without TF32; restore afterwards."""
    from dpvo import fastba
    if getattr(fastba.BA, "_ba_local_fp32", False):
        return
    orig = fastba.BA

    def ba_fp32(*args, **kwargs):
        prev_flag, prev_prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            return orig(*args, **kwargs)
        finally:
            torch.set_float32_matmul_precision(prev_prec)
            torch.backends.cuda.matmul.allow_tf32 = prev_flag

    ba_fp32._ba_local_fp32 = True
    fastba.BA = ba_fp32


def apply_precision_policy(cfg):
    """Call before constructing the estimator. Default runs without tracing: touches nothing.

    Policies (FREQVIO_PRECISION): amp_off disables mixed precision (cfg.MIXED_PRECISION and the
    hard-coded autocast in DBA.update); matmul_tf32_off / cudnn_tf32_off disable TF32 for cuBLAS
    matmuls / cuDNN convolutions; tf32_off does both; fp32_strict disables everything. When tracing or changing anything, the settings actually in force
    are recorded.
    """
    policy = precision_policy()
    if not (trace_enabled() or policy != "default"):
        return {"arm": "default", "recorded": False}
    import torch
    if amp_disabled():
        frozen = bool(getattr(cfg, "is_frozen", lambda: False)())
        if frozen:
            cfg.defrost()
        cfg.MIXED_PRECISION = False
        if frozen:
            cfg.freeze()
    if matmul_tf32_disabled():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    if cudnn_tf32_disabled():
        torch.backends.cudnn.allow_tf32 = False
    ba_local = policy in BA_LOCAL_FP32
    if ba_local:
        _install_ba_local_fp32(torch)
    report = {
        "arm": policy,
        "ba_local_matmul_fp32": ba_local,
        "cfg_MIXED_PRECISION": bool(getattr(cfg, "MIXED_PRECISION", None)),
        "cfg_PATCHES_PER_FRAME": getattr(cfg, "PATCHES_PER_FRAME", None),
        "cfg_LOOP_CLOSURE": getattr(cfg, "LOOP_CLOSURE", None),
        "update_autocast": update_autocast_enabled(),
        "ba_cholinfo_requested": os.environ.get("FREQVIO_BA_CHOLINFO", "0") == "1",
        "version": VERSION,
    }
    report.update(_torch_settings())
    _emit("precision_policy", report)
    return report


# ----------------------------------------------------------------------------- summaries
def _tensor_status(t):
    """(all_finite, n_nan, n_inf, max_abs_finite, dtype, shape) for a floating tensor."""
    import torch
    fin = torch.isfinite(t)
    all_fin = bool(fin.all().item())
    if all_fin:
        n_nan = n_inf = 0
        max_abs = float(t.detach().abs().max().item()) if t.numel() else 0.0
    else:
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        finite_vals = t.detach()[fin]
        max_abs = float(finite_vals.abs().max().item()) if finite_vals.numel() else None
    return all_fin, n_nan, n_inf, max_abs, str(t.dtype).replace("torch.", ""), list(t.shape)


def _np_summary(v):
    """Finite flag, shape, norm and (for square matrices) conditioning of a numpy value."""
    import numpy as np
    a = np.asarray(v, dtype=np.float64)
    s = {"shape": list(a.shape), "finite": bool(np.all(np.isfinite(a)))}
    if a.size == 0:
        return s
    if s["finite"]:
        s["norm"] = float(np.linalg.norm(a))
        s["max_abs"] = float(np.max(np.abs(a)))
        if a.ndim == 2 and a.shape[0] == a.shape[1]:
            try:
                sv = np.linalg.svd(a, compute_uv=False)
                s["sigma_max"] = float(sv[0])
                s["sigma_min"] = float(sv[-1])
                s["cond"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
            except np.linalg.LinAlgError as e:
                s["svd_error"] = str(e)[:80]
    else:
        s["n_nan"] = int(np.isnan(a).sum())
        s["n_inf"] = int(np.isinf(a).sum())
    return s


def align_probe(stage, **values):
    """Record intermediate quantities inside VisualIMUAlignment (read-only).

    Called from devo/dba.py only under `if _numtrace.ACTIVE:`. Scalars that are normalisation
    denominators are passed explicitly (e.g. g_norm, bb_norm) so a zero or non-finite denominator
    is visible even when every input was finite.
    """
    if not ACTIVE:
        return
    try:
        tr = _CURRENT
        rec = {"stage": stage, "call": tr.vi_align_calls if tr else None,
               "frame": tr.frame if tr else None}
        for k, v in values.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rec[k] = float(v)
            else:
                rec[k] = _np_summary(v)
        _emit("vi_alignment_probe", rec)
        if tr is not None:
            bad = [k for k, v in rec.items() if isinstance(v, dict) and v.get("finite") is False]
            bad += [k for k, v in rec.items() if isinstance(v, float) and v != v]  # NaN scalar
            # a zero normalisation denominator turns finite inputs into NaN on the next line
            bad += ["zero:" + k for k, v in rec.items() if k.endswith("_norm") and v == 0.0]
            if bad:
                tr._record_detection("vi_alignment:" + stage, ",".join(sorted(bad)), {})
    except Exception as e:  # never break the run
        if _CURRENT is not None:
            _CURRENT._count("tracer_error:align_probe:" + type(e).__name__)


def _cuda_autocast_enabled(torch):
    """CUDA autocast state; the device argument exists from torch 2.4 on."""
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


class NumTrace:
    """Wraps selected methods of one DBA instance. Never alters arguments or results."""

    def __init__(self, slam):
        self.slam = slam
        import hashlib
        self._in_hash = hashlib.sha256()   # running fingerprint of the input stream (voxels + timestamps)
        self.in_frames = 0
        self.in_checkpoints = {}            # frames seen -> fingerprint prefix, to compare runs that stop early
        self.frame = 0          # index of the input frame currently being processed
        self.events = 0
        self.first = None
        self.counts = {}
        self.dtype_logged = False
        self.fp16_near_overflow = 0
        self.t0 = time.time()
        self.vi_align_calls = 0
        self.finished = False
        self.installed = []

    # ---------- helpers ----------
    def _count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def _record_detection(self, where, name, extra):
        self._count("nonfinite@" + where + ":" + name)
        ev = {"frame": self.frame, "n": int(getattr(self.slam, "n", -1)), "where": where,
              "tensor": name, "imu_enabled": bool(getattr(self.slam, "imu_enabled", False)),
              "is_initialized": bool(getattr(self.slam, "is_initialized", False))}
        ev.update(extra)
        if self.first is None:
            self.first = ev
            _emit("first_detected_nonfinite", ev)
        elif self.events < MAX_EVENTS:
            _emit("nonfinite", ev)
        self.events += 1

    def _check(self, where, tensors):
        """Return True if all given floating tensors are finite; record detections."""
        import torch
        ok = True
        for name, t in tensors.items():
            if not isinstance(t, torch.Tensor) and isinstance(getattr(t, "data", None), torch.Tensor):
                t = t.data  # lietorch group objects keep their tensor in .data
            if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() == 0:
                continue
            all_fin, n_nan, n_inf, max_abs, dtype, shape = _tensor_status(t)
            if dtype == "float16" and max_abs is not None and max_abs > FP16_NEAR_MAX:
                self.fp16_near_overflow += 1
                self._count("fp16_near_overflow@" + where)
            if all_fin:
                continue
            ok = False
            self._record_detection(where, name, {"dtype": dtype, "shape": shape, "n_nan": n_nan,
                                                 "n_inf": n_inf, "max_abs_finite": max_abs})
        return ok

    def _pose_tensors(self):
        pg = self.slam.pg
        n = max(int(self.slam.n), 0)
        return {"pg.poses_[:n]": pg.poses_[:n], "pg.patches_[:n]": pg.patches_[:n]}

    def _imu_state_ok(self, where):
        """Finiteness of the IMU-side state (velocities, biases, body poses); CPU only."""
        import numpy as np
        st = getattr(self.slam, "state", None)
        if st is None:
            return True
        ok = True
        try:
            for name, seq, get in (("state.vs", st.vs, lambda v: np.asarray(v, dtype=np.float64)),
                                   ("state.bs", st.bs, lambda b: np.asarray(b.vector(), dtype=np.float64)),
                                   ("state.wTbs", st.wTbs, lambda p: np.asarray(p.matrix(), dtype=np.float64))):
                bad = 0
                for item in seq:
                    if item is None:
                        continue
                    if not np.all(np.isfinite(get(item))):
                        bad += 1
                if bad:
                    ok = False
                    self._record_detection(where, name, {"n_bad_entries": bad, "n_entries": len(seq)})
        except Exception as e:  # never break the run
            self._count("tracer_error:imu_state:" + type(e).__name__)
        return ok

    # ---------- wrappers (installed in pipeline order) ----------
    def _wrap_reproject(self):
        orig = self.slam.reproject

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            try:
                self._check("reproject(out)", {"coords": out})
            except Exception as e:
                self._count("tracer_error:reproject:" + type(e).__name__)
            return out

        self.slam.reproject = wrapped

    def _wrap_corr(self):
        orig = self.slam.corr

        def wrapped(coords, *args, **kwargs):
            out = orig(coords, *args, **kwargs)
            try:
                self._check("corr(out)", {"corr": out})
            except Exception as e:
                self._count("tracer_error:corr:" + type(e).__name__)
            return out

        self.slam.corr = wrapped

    def _wrap_network_update(self):
        """Pre-hook checks inputs first, forward hook then checks outputs; both read-only."""
        import torch
        module = self.slam.network.update

        def pre_hook(_module, inputs):
            try:
                names = ("net", "ctx", "corr")
                self._check("network.update(in)", {names[i]: inputs[i]
                                                   for i in range(min(3, len(inputs)))})
            except Exception as e:
                self._count("tracer_error:network.update.in:" + type(e).__name__)
            return None  # keep the original inputs

        def hook(_module, inputs, output):
            try:
                net, (delta, weight, _) = output
                if not self.dtype_logged:
                    _emit("dtype_probe", {
                        "net": str(net.dtype), "delta": str(delta.dtype),
                        "weight": str(weight.dtype),
                        "corr_input": str(inputs[2].dtype) if len(inputs) > 2 else None,
                        "imap_": str(self.slam.imap_.dtype),
                        "fmap1_": str(self.slam.fmap1_.dtype),
                        "update_params": str(next(module.parameters()).dtype),
                        "autocast_enabled_now": _cuda_autocast_enabled(torch)})
                    self.dtype_logged = True  # only once the record was written
                self._check("network.update(out)", {"net": net, "delta": delta, "weight": weight})
            except Exception as e:  # tracer must never break the run
                self._count("tracer_error:network.update.out:" + type(e).__name__)
            return None  # keep the original output

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(hook)

    def _wrap_fastba(self):
        from dpvo import fastba
        if getattr(fastba.BA, "_numtrace_wrapped", False):
            return
        orig = fastba.BA
        tracer = self

        def wrapped(poses, patches, intrinsics, target, weight, *args, **kwargs):
            try:
                before = tracer._check("fastba.BA(in)", {"poses": poses, "patches": patches,
                                                         "target": target, "weight": weight})
            except Exception as e:
                before = None
                tracer._count("tracer_error:fastba.in:" + type(e).__name__)
            out = orig(poses, patches, intrinsics, target, weight, *args, **kwargs)
            try:
                after = tracer._check("fastba.BA(out)", {"poses": poses, "patches": patches})
                if before and not after:
                    tracer._count("fastba.BA:nonfinite_after_finite_inputs")
            except Exception as e:
                tracer._count("tracer_error:fastba.out:" + type(e).__name__)
            return out

        wrapped._numtrace_wrapped = True
        fastba.BA = wrapped

    def _wrap_run_dba(self):
        name = "_DBA__run_DBA"
        if not hasattr(self.slam, name):
            raise AttributeError(name)
        orig = getattr(self.slam, name)

        def wrapped(*args, **kwargs):
            before = self._check("run_DBA(in)", self._pose_tensors()) & self._imu_state_ok("run_DBA(in)")
            out = orig(*args, **kwargs)
            after = self._check("run_DBA(out)", self._pose_tensors()) & self._imu_state_ok("run_DBA(out)")
            if before and not after:
                self._count("run_DBA:nonfinite_after_finite_inputs")
            return out

        setattr(self.slam, name, wrapped)

    def _alignment_inputs(self, t0, t1):
        """What VisualIMUAlignment consumes: visual poses, IMU preintegration and IMU state."""
        import numpy as np
        import torch
        pg = self.slam.pg
        q = pg.poses_[:t1, 3:].detach().double()
        qn = q.norm(dim=-1)
        info = {"t0": int(t0), "t1": int(t1),
                "poses_finite": bool(torch.isfinite(pg.poses_[:t1]).all().item()),
                "quat_norm_min": float(qn.min().item()) if qn.numel() else None,
                "quat_norm_max": float(qn.max().item()) if qn.numel() else None}
        # rotation matrices that the alignment will build from these poses
        try:
            from dpvo.lietorch import SE3
            R = SE3(pg.poses_[:t1]).inv().matrix()[:, :3, :3].detach().double().cpu().numpy()
            orth = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max(axis=(1, 2))
            info["rot_orthonormality_err_max"] = float(np.nanmax(orth)) if orth.size else None
            info["rot_finite"] = bool(np.all(np.isfinite(R)))
        except Exception as e:
            info["rot_error"] = type(e).__name__ + ": " + str(e)[:100]
        dts, p_ok, v_ok = [], True, True
        try:
            for i in range(t0, t1 - 1):
                pim = self.slam.state.preintegrations[i]
                dts.append(float(pim.deltaTij()))
                p_ok &= bool(np.all(np.isfinite(pim.deltaPij())))
                v_ok &= bool(np.all(np.isfinite(pim.deltaVij())))
        except Exception as e:
            info["preint_error"] = type(e).__name__ + ": " + str(e)[:120]
        info.update({"preint_dt_min": min(dts) if dts else None,
                     "preint_dt_max": max(dts) if dts else None,
                     "preint_dP_finite": p_ok, "preint_dV_finite": v_ok})
        try:
            st = self.slam.state
            vs = [np.asarray(st.vs[i], dtype=np.float64) for i in range(t0, t1) if st.vs[i] is not None]
            bs = [np.asarray(st.bs[i].vector(), dtype=np.float64) for i in range(t0, t1)]
            info["state_vs_finite"] = bool(all(np.all(np.isfinite(v)) for v in vs))
            info["state_bs_finite"] = bool(all(np.all(np.isfinite(b)) for b in bs))
        except Exception as e:
            info["state_error"] = type(e).__name__ + ": " + str(e)[:120]
        return info

    def _wrap_vi_alignment(self):
        orig = self.slam.VisualIMUAlignment

        def wrapped(t0, t1, *args, **kwargs):
            self.vi_align_calls += 1
            try:
                inp = self._alignment_inputs(t0, t1)
            except Exception as e:
                inp = {"tracer_error": type(e).__name__ + ": " + str(e)[:120]}
            inp.update({"call": self.vi_align_calls, "frame": self.frame,
                        "ignore_lever": kwargs.get("ignore_lever", args[0] if args else None)})
            _emit("vi_alignment_in", inp)
            for key in ("poses_finite", "rot_finite", "preint_dP_finite", "preint_dV_finite",
                        "state_vs_finite", "state_bs_finite"):
                if inp.get(key) is False:
                    self._record_detection("vi_alignment(in)", key, {})
            try:
                return orig(t0, t1, *args, **kwargs)
            except Exception as e:
                _emit("vi_alignment_exception", {"call": self.vi_align_calls, "frame": self.frame,
                                                 "exc_type": type(e).__name__,
                                                 "exc_msg": str(e)[:200]})
                raise

        self.slam.VisualIMUAlignment = wrapped

    def install(self):
        global _CURRENT
        for fn in (self._wrap_reproject, self._wrap_corr, self._wrap_network_update,
                   self._wrap_fastba, self._wrap_run_dba, self._wrap_vi_alignment):
            try:
                fn()
                self.installed.append(fn.__name__)
            except Exception as e:
                self._count("tracer_install_error:" + fn.__name__ + ":" + type(e).__name__)
        _CURRENT = self
        _emit("trace_installed", {"version": VERSION, "installed": self.installed,
                                  "install_errors": {k: v for k, v in self.counts.items()
                                                     if k.startswith("tracer_install_error")}})
        return self

    # ---------- per-frame and end-of-run ----------
    def observe_input(self, voxel, t):
        """Fold one input frame into the input fingerprint. Bit-exact: the float bit patterns are
        reduced with wrap-around int64 arithmetic, which is associative, so the value does not
        depend on reduction order or device. Read-only."""
        try:
            import struct
            import torch
            v = voxel.detach().contiguous().reshape(-1)
            if v.dtype == torch.float32:
                bits = v.view(torch.int32).to(torch.int64)
            elif v.dtype in (torch.float16, torch.bfloat16):
                bits = v.view(torch.int16).to(torch.int64)
            elif v.dtype == torch.float64:
                bits = v.view(torch.int64)
            else:
                bits = v.to(torch.int64)
            w = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64)
            s1 = int((bits * w).sum().item())
            s2 = int((bits ^ (w * 2654435761)).sum().item())
            self._in_hash.update(struct.pack("<qqq", s1, s2, bits.numel()))
            self._in_hash.update(("%s|%s|%r;" % (tuple(voxel.shape), voxel.dtype, float(t))).encode())
            self.in_frames += 1
            if self.in_frames in (1, 10, 50) or self.in_frames % 100 == 0:
                self.in_checkpoints[str(self.in_frames)] = self._in_hash.hexdigest()[:16]
        except Exception as e:
            self._count("tracer_error:observe_input:" + type(e).__name__)

    def after_frame(self, i):
        """Call after the estimator processed input frame i (0-based)."""
        self.frame = i
        try:
            self._check("after_frame", self._pose_tensors())
            if getattr(self.slam, "imu_enabled", False):
                self._imu_state_ok("after_frame")
        except Exception as e:
            self._count("tracer_error:after_frame:" + type(e).__name__)
        self.frame = i + 1  # events from now on belong to the next input frame

    def finish(self, exc=None):
        if self.finished:
            return
        self.finished = True
        import torch
        stats = {"frames_completed": self.frame, "keyframes_n": int(getattr(self.slam, "n", -1)),
                 "first_detected_nonfinite": self.first, "nonfinite_events": self.events,
                 "fp16_near_overflow": self.fp16_near_overflow, "counts": self.counts,
                 "tracer_errors": sum(v for k, v in self.counts.items() if k.startswith("tracer_")),
                 "installed": self.installed, "vi_align_calls": self.vi_align_calls,
                 "vi_initialized": bool(getattr(self.slam, "imu_enabled", False)),
                 "wall_s": round(time.time() - self.t0, 2),
                 "input_frames": self.in_frames, "input_fingerprint": self._in_hash.hexdigest()[:16],
                 "input_checkpoints": dict(self.in_checkpoints)}
        if torch.cuda.is_available():
            stats["peak_mem_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            stats["peak_mem_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        if exc is not None:
            stats["exception"] = {"type": type(exc).__name__, "msg": str(exc)[:200],
                                  "site": _last_frame(exc)}
        _emit("run_summary", stats)


def _last_frame(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return None
    f = tb[-1]
    return "%s:%d:%s" % (os.path.basename(f.filename), f.lineno, f.name)


def install(slam):
    """Return an installed tracer, or None when FREQVIO_NUMTRACE is not set."""
    if not trace_enabled():
        return None
    return NumTrace(slam).install()


def current_exception():
    """The exception propagating through a finally block, if any."""
    return sys.exc_info()[1]
