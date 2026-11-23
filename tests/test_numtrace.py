"""Tests for utils/numtrace.py with a stand-in estimator (CPU only, no CUDA extensions).

Run from the FreqVIO root:  python tests/test_numtrace.py
"""
import contextlib
import io
import json
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Stand-in for the compiled BA extension: "dpvo.fastba.BA" writes NaN into poses on demand.
_fake_dpvo = types.ModuleType("dpvo")
_fake_fastba = types.ModuleType("dpvo.fastba")
_BA_STATE = {"poison": False, "calls": 0}


def _fake_BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations,
             eff_impl=False):
    _BA_STATE["calls"] += 1
    if _BA_STATE["poison"]:
        poses[0, 1, 0] = float("nan")
    return []


_fake_fastba.BA = _fake_BA
_fake_dpvo.fastba = _fake_fastba
sys.modules.setdefault("dpvo", _fake_dpvo)
sys.modules["dpvo.fastba"] = _fake_fastba

from utils import numtrace  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "script", "diag"))
import precision_pilot  # noqa: E402  (stdlib only; for the control lists the pilot checks)


class _Update(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)

    def forward(self, net, ctx, corr, flow, ii, jj, kk):
        return net, (corr * 1.0, torch.ones_like(corr), None)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.update = _Update()


class _PIM:
    def __init__(self, dt):
        self.dt = dt

    def deltaTij(self):
        return self.dt

    def deltaPij(self):
        return [0.0, 0.0, 0.0]

    def deltaVij(self):
        return [0.0, 0.0, 9.8 * self.dt]


class _Bias:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=float)

    def vector(self):
        return self.v


class _Pose:
    def matrix(self):
        return np.eye(4)


class _Slam:
    def __init__(self):
        self.n = 0
        self.pg = types.SimpleNamespace(poses_=torch.zeros(16, 7), patches_=torch.zeros(16, 4, 3, 3, 3))
        self.pg.poses_[:, 6] = 1.0
        self.imap_ = torch.zeros(4, 4, dtype=torch.float16)
        self.fmap1_ = torch.zeros(1, 2, 2, 2)
        self.network = _Net()
        self.imu_enabled = False
        self.is_initialized = True
        self.state = types.SimpleNamespace(preintegrations=[_PIM(0.05) for _ in range(16)],
                                           vs=[np.zeros(3) for _ in range(16)],
                                           bs=[_Bias(np.zeros(6)) for _ in range(16)],
                                           wTbs=[_Pose() for _ in range(16)])
        self.corr_poison = False

    def reproject(self):
        return torch.zeros(1, 5, 2)

    def corr(self, coords):
        c = torch.randn(1, 5, 2)
        if self.corr_poison:
            c[0, 0, 0] = float("inf")
        return c

    def step(self):
        """One frame in pipeline order: reproject, corr, update operator, visual BA."""
        self.n += 1
        coords = self.reproject()
        corr = self.corr(coords)
        self.network.update(torch.zeros(1, 5, 2), None, corr, None, None, None, None)
        from dpvo import fastba
        fastba.BA(self.pg.poses_.view(1, -1, 7), self.pg.patches_, None, corr, corr, None,
                  None, None, None, 0, self.n, M=4, iterations=2)

    def VisualIMUAlignment(self, t0, t1, ignore_lever, disable_scale=False):
        if not torch.isfinite(self.pg.poses_[:t1]).all():
            raise ValueError("Found zero norm quaternions in `quat`.")
        return "aligned"

    def _DBA__run_DBA(self, **kw):
        return "dba"


def _records(text):
    return [json.loads(l[len("[numtrace] "):]) for l in text.splitlines() if l.startswith("[numtrace] ")]


class NumTraceTest(unittest.TestCase):
    def setUp(self):
        _BA_STATE.update(poison=False, calls=0)
        _fake_fastba.BA = _fake_BA
        for k in (numtrace.ENV_TRACE, numtrace.ENV_FP32, numtrace.ENV_OUT, numtrace.ENV_POLICY):
            os.environ.pop(k, None)
        numtrace.ACTIVE = False
        numtrace._CURRENT = None

    def _install(self, slam):
        os.environ[numtrace.ENV_TRACE] = "1"
        numtrace.ACTIVE = True
        return numtrace.install(slam)

    def test_disabled_by_default(self):
        self.assertIsNone(numtrace.install(_Slam()))
        self.assertTrue(numtrace.update_autocast_enabled())
        numtrace.align_probe("linear_alignment", g=np.full(3, np.nan))  # ACTIVE False: no output
        self.assertIsNone(numtrace._CURRENT)

    def test_all_wrappers_installed(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr = self._install(_Slam())
        rec = next(r for r in _records(buf.getvalue()) if r["kind"] == "trace_installed")
        self.assertEqual(rec["install_errors"], {})
        self.assertEqual(len(tr.installed), 6)
        # the pilot requires exactly these checkpoints; the two lists must not drift apart
        self.assertEqual(tuple(rec["installed"]), precision_pilot.REQUIRED_HOOKS)

    def test_dtype_probe_has_every_field_the_pilot_checks(self):
        slam = _Slam()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr = self._install(slam)
            slam.step()
            slam.step()
            tr.finish(None)
        probes = [r for r in _records(buf.getvalue()) if r["kind"] == "dtype_probe"]
        self.assertEqual(len(probes), 1)                    # logged once
        missing = [k for k in precision_pilot.DTYPE_FIELDS if probes[0].get(k) is None]
        self.assertEqual(missing, [])
        self.assertIs(probes[0]["autocast_enabled_now"], False)   # CPU test, no autocast
        self.assertEqual(probes[0]["imap_"], "torch.float16")

    def test_ba_output_detected(self):
        slam = _Slam()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr = self._install(slam)
            for i in range(5):
                _BA_STATE["poison"] = (i == 3)
                slam.step()
                tr.after_frame(i)
            self.assertEqual(slam.VisualIMUAlignment(0, 1, True), "aligned")   # pose 0 still finite
            with self.assertRaises(ValueError):
                slam.VisualIMUAlignment(0, 5, ignore_lever=True)
            self.assertEqual(slam._DBA__run_DBA(), "dba")                      # wrapper returns original
            try:
                raise ValueError("Found zero norm quaternions in `quat`.")
            except ValueError as e:
                tr.finish(e)
        recs = _records(buf.getvalue())
        kinds = [r["kind"] for r in recs]
        first = next(r for r in recs if r["kind"] == "first_detected_nonfinite")
        self.assertEqual((first["where"], first["tensor"], first["frame"]), ("fastba.BA(out)", "poses", 3))
        self.assertIn("dtype_probe", kinds)
        self.assertIn("vi_alignment_exception", kinds)
        vi_in = [r for r in recs if r["kind"] == "vi_alignment_in"]
        self.assertTrue(vi_in[0]["poses_finite"])
        self.assertFalse(vi_in[1]["poses_finite"])
        self.assertTrue(vi_in[1]["state_vs_finite"])
        self.assertAlmostEqual(vi_in[1]["preint_dt_min"], 0.05)
        summ = next(r for r in recs if r["kind"] == "run_summary")
        self.assertEqual(summ["exception"]["type"], "ValueError")
        self.assertEqual(summ["counts"].get("fastba.BA:nonfinite_after_finite_inputs"), 1)
        self.assertEqual(summ["tracer_errors"], 0)
        self.assertEqual(_BA_STATE["calls"], 5)

    def test_input_nonfinite_is_attributed_to_input_not_output(self):
        """A non-finite correlation must be detected at corr(out) / update(in), never first at update(out)."""
        slam = _Slam()
        slam.corr_poison = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._install(slam)
            slam.step()
        recs = [r for r in _records(buf.getvalue()) if r["kind"] in ("first_detected_nonfinite", "nonfinite")]
        self.assertEqual(recs[0]["where"], "corr(out)")
        self.assertEqual(recs[1]["where"], "network.update(in)")
        self.assertEqual(recs[2]["where"], "network.update(out)")

    def test_alignment_probe_flags_zero_denominator_and_nan(self):
        slam = _Slam()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._install(slam)
            numtrace.align_probe("refine_basis", k=0, g_norm=9.81, bb_norm=0.0)  # finite inputs, zero denominator
            numtrace.align_probe("linear_alignment", A=np.eye(4), x=np.array([1.0, np.nan]), g_norm=float("nan"))
        recs = _records(buf.getvalue())
        probes = [r for r in recs if r["kind"] == "vi_alignment_probe"]
        self.assertEqual(probes[0]["bb_norm"], 0.0)
        self.assertEqual(probes[1]["A"]["cond"], 1.0)
        self.assertFalse(probes[1]["x"]["finite"])
        first = next(r for r in recs if r["kind"] == "first_detected_nonfinite")
        self.assertEqual((first["where"], first["tensor"]), ("vi_alignment:refine_basis", "zero:bb_norm"))
        later = [r for r in recs if r["kind"] == "nonfinite"]
        self.assertEqual(later[0]["tensor"], "g_norm,x")

    def test_imu_state_checked_after_init(self):
        slam = _Slam()
        slam.imu_enabled = True
        slam.state.vs[2] = np.array([np.nan, 0, 0])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tr = self._install(slam)
            tr.after_frame(0)
        first = next(r for r in _records(buf.getvalue()) if r["kind"] == "first_detected_nonfinite")
        self.assertEqual((first["where"], first["tensor"], first["n_bad_entries"]), ("after_frame", "state.vs", 1))

    def test_values_unchanged_by_hooks(self):
        slam = _Slam()
        corr = torch.randn(1, 5, 2)
        ref = slam.network.update(torch.zeros(1, 5, 2), None, corr, None, None, None, None)
        with contextlib.redirect_stdout(io.StringIO()):
            self._install(slam)
            out = slam.network.update(torch.zeros(1, 5, 2), None, corr, None, None, None, None)
        self.assertTrue(torch.equal(ref[1][0], out[1][0]))
        self.assertTrue(torch.equal(ref[1][1], out[1][1]))

    def test_precision_policy(self):
        cfg = types.SimpleNamespace(MIXED_PRECISION=True)
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        rep = numtrace.apply_precision_policy(cfg)
        self.assertTrue(cfg.MIXED_PRECISION)
        self.assertEqual(rep, {"arm": "default", "recorded": False})
        os.environ[numtrace.ENV_TRACE] = "1"   # traced default arm: recorded, still unchanged
        with contextlib.redirect_stdout(io.StringIO()):
            rep = numtrace.apply_precision_policy(cfg)
        self.assertTrue(cfg.MIXED_PRECISION)
        self.assertEqual(rep["arm"], "default")
        self.assertTrue(rep["update_autocast"])
        os.environ[numtrace.ENV_FP32] = "1"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rep = numtrace.apply_precision_policy(cfg)
        self.assertFalse(cfg.MIXED_PRECISION)
        self.assertFalse(numtrace.update_autocast_enabled())
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertEqual(rep["arm"], "fp32_strict")
        self.assertEqual(_records(buf.getvalue())[0]["kind"], "precision_policy")
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("highest")

    def test_input_fingerprint_is_bit_exact(self):
        def run(frames, flip=None, t_shift=0.0, stop=None):
            with contextlib.redirect_stdout(io.StringIO()):
                tr = self._install(_Slam())
            for i, v in enumerate(frames[:stop]):
                v = v.clone()
                if flip is not None and i == flip:
                    b = v.view(torch.int32).reshape(-1)
                    b[7] ^= 1                                  # one bit in one frame
                tr.observe_input(v, 0.05 * i + t_shift)
            return tr
        g = torch.Generator().manual_seed(0)
        frames = [torch.randn(5, 6, 8, generator=g) for _ in range(120)]
        a, b = run(frames), run(frames)
        self.assertEqual(a._in_hash.hexdigest(), b._in_hash.hexdigest())
        self.assertEqual(a.in_frames, 120)
        self.assertNotEqual(a._in_hash.hexdigest(), run(frames, flip=60)._in_hash.hexdigest())
        self.assertNotEqual(a._in_hash.hexdigest(), run(frames, t_shift=1e-6)._in_hash.hexdigest())
        early = run(frames, stop=100)                          # a run that stops early matches on its checkpoints
        self.assertEqual(early.in_checkpoints["100"], a.in_checkpoints["100"])
        self.assertNotEqual(run(frames, flip=60).in_checkpoints["100"], a.in_checkpoints["100"])
        self.assertEqual(run(frames, flip=60).in_checkpoints["50"], a.in_checkpoints["50"])   # before the flip

    def test_split_policies(self):
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, True  # torch defaults
            os.environ[numtrace.ENV_POLICY] = "amp_off"
            cfg = types.SimpleNamespace(MIXED_PRECISION=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rep = numtrace.apply_precision_policy(cfg)
            self.assertEqual(rep["arm"], "amp_off")
            self.assertFalse(cfg.MIXED_PRECISION)
            self.assertFalse(numtrace.update_autocast_enabled())
            self.assertTrue(torch.backends.cudnn.allow_tf32)          # TF32 left alone
            os.environ[numtrace.ENV_POLICY] = "tf32_off"
            cfg = types.SimpleNamespace(MIXED_PRECISION=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rep = numtrace.apply_precision_policy(cfg)
            self.assertEqual(rep["arm"], "tf32_off")
            self.assertTrue(cfg.MIXED_PRECISION)                      # AMP left alone
            self.assertTrue(numtrace.update_autocast_enabled())
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            os.environ[numtrace.ENV_POLICY] = "matmul_tf32_off"
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True
            cfg = types.SimpleNamespace(MIXED_PRECISION=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rep = numtrace.apply_precision_policy(cfg)
            self.assertEqual(rep["arm"], "matmul_tf32_off")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)           # cuDNN left alone
            self.assertTrue(cfg.MIXED_PRECISION)
            os.environ[numtrace.ENV_POLICY] = "cudnn_tf32_off"
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True
            with contextlib.redirect_stdout(io.StringIO()):
                rep = numtrace.apply_precision_policy(types.SimpleNamespace(MIXED_PRECISION=True))
            self.assertEqual(rep["arm"], "cudnn_tf32_off")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)     # matmul left alone
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            # BA-local: global TF32 stays on, inside fastba.BA it is off, restored afterwards
            os.environ[numtrace.ENV_POLICY] = "ba_matmul_fp32"
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True
            seen = {}
            from dpvo import fastba as _fb

            def probe_ba(*a, **k):
                seen["inside"] = torch.backends.cuda.matmul.allow_tf32
                return "ba"
            _fb.BA = probe_ba
            with contextlib.redirect_stdout(io.StringIO()):
                rep = numtrace.apply_precision_policy(types.SimpleNamespace(MIXED_PRECISION=True))
            self.assertTrue(rep["ba_local_matmul_fp32"])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(_fb.BA(), "ba")
            self.assertFalse(seen["inside"])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)      # restored
            _fb.BA = _fake_BA
            os.environ[numtrace.ENV_POLICY] = "fp32_strict"
            self.assertTrue(numtrace.force_fp32())
            os.environ[numtrace.ENV_POLICY] = "amp_off"
            os.environ[numtrace.ENV_FP32] = "1"                       # contradiction -> refuse
            with self.assertRaises(ValueError):
                numtrace.precision_policy()
            os.environ.pop(numtrace.ENV_FP32)
            os.environ[numtrace.ENV_POLICY] = "half"
            with self.assertRaises(ValueError):
                numtrace.precision_policy()
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
            torch.set_float32_matmul_precision("highest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
