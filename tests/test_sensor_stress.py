"""Tests for utils/sensor_stress.py (stdlib unittest + numpy only; Python >= 3.9).

Run from the repository root:  python3 -m unittest tests.test_sensor_stress -v
"""
import json
import math
import os
import re
import subprocess
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import sensor_stress as ss  # noqa: E402

DEG = 180.0 / math.pi
SEQ = "unit_test_seq"
EVIDENCE_FIELDS = ("operator", "module_version", "numpy_version", "sequence_id", "stress_seed",
                   "derived_seeds", "rng", "requested", "realized", "input_format",
                   "input_sha256", "output_sha256", "data_changed")


def synthetic_imu(n=2000, rate_hz=200.0, t0_us=1.0e6, gyro_unit="deg/s"):
    """Deterministic (RNG-free) IMU in the run_DEIO2 format."""
    k = np.arange(n, dtype=np.float64)
    t = t0_us + k * (1.0e6 / rate_hz)
    gyro_rad = np.column_stack([0.1 * np.sin(0.01 * k), 0.2 * np.cos(0.013 * k), 0.05 * np.sin(0.007 * k)])
    acc = np.column_stack([0.3 * np.sin(0.02 * k), 9.81 + 0.1 * np.cos(0.011 * k), 0.2 * np.sin(0.017 * k)])
    gyro = gyro_rad * (DEG if gyro_unit == "deg/s" else 1.0)
    return np.column_stack([t, gyro, acc]).astype(np.float64)


def synthetic_events(n=5000, t0_us=1_000_000, step_us=7):
    """Deterministic event batch with int64 microsecond timestamps (EventSlicer style)."""
    k = np.arange(n, dtype=np.int64)
    return {"t": t0_us + k * step_us, "x": (k * 37) % 640, "y": (k * 11) % 480,
            "p": (k % 2).astype(np.int8), "meta": "passthrough"}


def _fingerprint():
    """Output hashes of all operator families on fixed inputs (used for the cross-process test)."""
    imu = synthetic_imu()
    a, _ = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=3, gyro_bias_rad_s=0.01,
                                   accel_bias_m_s2=[0.05, 0.0, -0.02], gyro_noise_rad_s_sqrthz=2e-3,
                                   accel_noise_m_s2_sqrthz=2e-2)
    b, _ = ss.apply_imu_time_offset(imu, 3.25, (imu[0, 0], imu[-1, 0]))
    sch, _ = ss.make_burst_schedule((imu[0, 0], imu[-1, 0]), 40.0, sequence_id=SEQ, stress_seed=3,
                                    target_fraction=0.1)
    c, _ = ss.apply_imu_burst_dropout(imu, sch)
    d, _ = ss.apply_event_burst_dropout(synthetic_events(), sch)
    e, _ = ss.apply_event_background_noise(synthetic_events(), (1_000_000, 1_035_000), (480, 640),
                                           sequence_id=SEQ, stress_seed=3, ba_rate_hz_per_pixel=0.1,
                                           hot_pixel_fraction=1e-4, hot_pixel_rate_hz=50.0)
    return "|".join([ss.array_sha256(a), ss.array_sha256(b), ss.array_sha256(sch),
                     ss.array_sha256(c), ss.events_sha256(d), ss.events_sha256(e)])


class _Base(unittest.TestCase):
    def assertSameArray(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(ss.array_sha256(a), ss.array_sha256(b))

    def assertEvidence(self, ev, operator, out_hash):
        for f in EVIDENCE_FIELDS:
            self.assertIn(f, ev, "missing evidence field %r" % f)
        self.assertEqual(ev["operator"], operator)
        self.assertEqual(ev["module_version"], ss.SENSOR_STRESS_VERSION)
        self.assertEqual(ev["output_sha256"], out_hash)
        self.assertEqual(ev["data_changed"], ev["input_sha256"] != ev["output_sha256"])
        json.dumps(ev, allow_nan=False)  # must be plain-JSON serializable


# ============================================================================ seeding

class TestSeeding(_Base):
    def test_golden_seed_value(self):
        # Pins the canonical hash payload; changing it silently changes every realization.
        self.assertEqual(ss.derive_seed("x", {"a": 1, "b": [1, 2]}, "seq", 7), 1108344732366439075)

    def test_param_canonicalization(self):
        s1 = ss.derive_seed("op", {"a": 1, "b": (1, 2)}, "seq", 7)
        s2 = ss.derive_seed("op", {"b": np.array([1.0, 2.0]), "a": np.float64(1.0)}, "seq", 7)
        self.assertEqual(s1, s2)

    def test_seed_sensitive_to_every_field(self):
        base = ss.derive_seed("op", {"a": 1.0}, "seq", 7)
        self.assertNotEqual(base, ss.derive_seed("op2", {"a": 1.0}, "seq", 7))
        self.assertNotEqual(base, ss.derive_seed("op", {"a": 1.5}, "seq", 7))
        self.assertNotEqual(base, ss.derive_seed("op", {"a": 1.0}, "seq2", 7))
        self.assertNotEqual(base, ss.derive_seed("op", {"a": 1.0}, "seq", 8))

    def test_seed_input_validation(self):
        for bad in (-1, 1.5, True, "3", None):
            with self.assertRaises(ValueError):
                ss.derive_seed("op", {}, "seq", bad)
        for bad in ("", None, 3):
            with self.assertRaises(ValueError):
                ss.derive_seed("op", {}, bad, 0)
        with self.assertRaises(ValueError):
            ss.derive_seed("op", {"a": float("nan")}, "seq", 0)

    def test_independent_of_global_numpy_rng(self):
        """Eval scripts call np.random.seed(eval_seed); the degraded input must not care."""
        imu = synthetic_imu()
        kw = dict(sequence_id=SEQ, stress_seed=1, gyro_bias_rad_s=0.01, gyro_noise_rad_s_sqrthz=1e-3)
        np.random.seed(0)
        a, _ = ss.apply_imu_bias_noise(imu, **kw)
        np.random.seed(12345)
        np.random.rand(10)
        b, _ = ss.apply_imu_bias_noise(imu, **kw)
        self.assertSameArray(a, b)

    def test_cross_process_bit_identity(self):
        env = dict(os.environ, PYTHONHASHSEED="987")
        code = ("import sys; sys.path.insert(0, %r); "
                "from tests.test_sensor_stress import _fingerprint; print(_fingerprint())" % REPO)
        res = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), _fingerprint())


# ============================================================================ 1. bias + noise

class TestImuBiasNoise(_Base):
    def test_noop_identity(self):
        imu = synthetic_imu()
        for kw in ({}, {"gyro_bias_rad_s": 0.0, "accel_bias_m_s2": [0.0, 0.0, 0.0],
                        "gyro_noise_rad_s_sqrthz": 0.0, "accel_noise_m_s2_sqrthz": 0.0}):
            out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, **kw)
            self.assertSameArray(out, imu)
            self.assertIsNot(out, imu)
            self.assertFalse(ev["data_changed"])
            self.assertEqual(ev["derived_seeds"], {})
            self.assertEqual(ev["realized"]["gyro_bias_mode"], "none")

    def test_no_input_mutation(self):
        imu = synthetic_imu()
        ref = imu.copy()
        ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, gyro_bias_rad_s=0.1,
                                accel_bias_m_s2=0.2, gyro_noise_rad_s_sqrthz=1e-2,
                                accel_noise_m_s2_sqrthz=1e-1)
        self.assertSameArray(imu, ref)

    def test_determinism_two_calls(self):
        imu = synthetic_imu()
        kw = dict(sequence_id=SEQ, stress_seed=5, gyro_bias_rad_s=0.02, accel_bias_m_s2=0.1,
                  gyro_noise_rad_s_sqrthz=1e-3, accel_noise_m_s2_sqrthz=1e-2)
        a, ea = ss.apply_imu_bias_noise(imu, **kw)
        b, eb = ss.apply_imu_bias_noise(imu, **kw)
        self.assertSameArray(a, b)
        self.assertEqual(ea, eb)

    def test_fresh_rng_reproduces_output(self):
        """Documented draw contract: delta = bias + sigma_d * standard_normal((N,3)) per sensor,
        from Generator(PCG64(derive_seed('imu_bias_noise/<sensor>_noise', {...}, seq, seed)))."""
        imu = synthetic_imu()
        sg, sa = 2e-3, 3e-2
        bg = np.array([0.01, -0.02, 0.005])
        out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=9, gyro_bias_rad_s=bg,
                                          gyro_noise_rad_s_sqrthz=sg, accel_noise_m_s2_sqrthz=sa)
        rate = 200.0
        gseed = ss.derive_seed("imu_bias_noise/gyro_noise", {"gyro_noise_density": sg}, SEQ, 9)
        aseed = ss.derive_seed("imu_bias_noise/accel_noise", {"accel_noise_density": sa}, SEQ, 9)
        self.assertEqual(ev["derived_seeds"], {"gyro_noise": gseed, "accel_noise": aseed})
        zg = np.random.Generator(np.random.PCG64(gseed)).standard_normal((imu.shape[0], 3))
        za = np.random.Generator(np.random.PCG64(aseed)).standard_normal((imu.shape[0], 3))
        exp = imu.copy()
        exp[:, 1:4] = imu[:, 1:4] + (bg + sg * math.sqrt(rate) * zg) * DEG
        exp[:, 4:7] = imu[:, 4:7] + (np.zeros(3) + sa * math.sqrt(rate) * za) * 1.0
        self.assertSameArray(out, exp)

    def test_explicit_bias_exact_noise_free(self):
        imu = synthetic_imu()
        bg, ba = np.array([0.01, -0.02, 0.03]), np.array([0.1, 0.0, -0.05])
        out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0,
                                          gyro_bias_rad_s=bg, accel_bias_m_s2=ba)
        np.testing.assert_allclose(out[:, 1:4] - imu[:, 1:4], np.tile(bg * DEG, (len(imu), 1)),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(out[:, 4:7] - imu[:, 4:7], np.tile(ba, (len(imu), 1)), rtol=0, atol=1e-12)
        r = ev["realized"]
        np.testing.assert_allclose(r["gyro_delta_mean_rad_s"], bg, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r["accel_delta_mean_m_s2"], ba, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r["gyro_delta_std_rad_s"], 0.0, atol=1e-12)
        self.assertEqual(r["gyro_bias_mode"], "explicit_vector")
        self.assertEqual(ev["requested"]["gyro_bias"]["direction_mode"], "explicit_vector")
        self.assertSameArray(out[:, 0], imu[:, 0])  # time column untouched
        self.assertEqual(ev["derived_seeds"], {})

    def test_random_direction_bias(self):
        imu = synthetic_imu()
        outs = {}
        for mag in (0.005, 0.02):
            out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=4, gyro_bias_rad_s=mag)
            r = ev["realized"]
            u = np.array(r["gyro_bias_direction_unit"])
            self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(r["gyro_delta_mean_rad_s"])), mag, places=12)
            self.assertEqual(r["gyro_bias_mode"], "random_unit_direction")
            self.assertIn("gyro_bias_direction", ev["derived_seeds"])
            outs[mag] = (u, ev["derived_seeds"]["gyro_bias_direction"])
        # direction is fixed across magnitude levels (depends on sequence + stress seed only)
        np.testing.assert_array_equal(outs[0.005][0], outs[0.02][0])
        self.assertEqual(outs[0.005][1], outs[0.02][1])
        _, ev_other = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=5, gyro_bias_rad_s=0.02)
        self.assertFalse(np.allclose(ev_other["realized"]["gyro_bias_direction_unit"], outs[0.02][0]))

    def test_noise_std_matches_density_sqrt_rate(self):
        n, rate = 400_000, 200.0  # 2000 s at 200 Hz
        imu = synthetic_imu(n=n, rate_hz=rate)
        sg, sa = 1.7e-3, 2.0e-2
        out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=1,
                                          gyro_noise_rad_s_sqrthz=sg, accel_noise_m_s2_sqrthz=sa)
        r = ev["realized"]
        self.assertEqual(r["rate_hz_estimated"], rate)
        exp_g, exp_a = sg * math.sqrt(rate), sa * math.sqrt(rate)
        self.assertAlmostEqual(r["gyro_sigma_discrete_rad_s"], exp_g, places=15)
        dg = (out[:, 1:4] - imu[:, 1:4]) / DEG
        da = out[:, 4:7] - imu[:, 4:7]
        np.testing.assert_allclose(dg.std(axis=0), exp_g, rtol=0.02)
        np.testing.assert_allclose(da.std(axis=0), exp_a, rtol=0.02)
        np.testing.assert_allclose(dg.mean(axis=0), 0.0, atol=5 * exp_g / math.sqrt(n))
        np.testing.assert_allclose(r["gyro_delta_std_rad_s"], exp_g, rtol=0.02)

    def test_rate_estimated_from_timestamps(self):
        imu = synthetic_imu(n=3000, rate_hz=1000.0)
        _, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, gyro_noise_rad_s_sqrthz=1e-3)
        self.assertAlmostEqual(ev["realized"]["rate_hz_estimated"], 1000.0, places=9)
        self.assertAlmostEqual(ev["realized"]["gyro_sigma_discrete_rad_s"], 1e-3 * math.sqrt(1000.0), places=15)

    def test_gyro_unit_conversion(self):
        bg = np.array([0.01, -0.02, 0.03])
        imu_deg = synthetic_imu(gyro_unit="deg/s")
        imu_rad = synthetic_imu(gyro_unit="rad/s")
        od, ed = ss.apply_imu_bias_noise(imu_deg, sequence_id=SEQ, stress_seed=0, gyro_bias_rad_s=bg)
        orad, er = ss.apply_imu_bias_noise(imu_rad, sequence_id=SEQ, stress_seed=0, gyro_bias_rad_s=bg,
                                           gyro_unit="rad/s")
        np.testing.assert_allclose((od - imu_deg)[:, 1:4].mean(axis=0), bg * 180.0 / math.pi, rtol=1e-10)
        np.testing.assert_allclose((orad - imu_rad)[:, 1:4].mean(axis=0), bg, rtol=1e-10)
        self.assertAlmostEqual(float((od - imu_deg)[0, 1]), 0.01 * 57.29577951308232, places=10)
        np.testing.assert_allclose(ed["realized"]["gyro_delta_mean_rad_s"], bg, rtol=1e-10)
        np.testing.assert_allclose(er["realized"]["gyro_delta_mean_rad_s"], bg, rtol=1e-10)
        # same physical noise realization regardless of the array's gyro unit
        kw = dict(sequence_id=SEQ, stress_seed=2, gyro_noise_rad_s_sqrthz=1e-3)
        nd, _ = ss.apply_imu_bias_noise(imu_deg, **kw)
        nr, _ = ss.apply_imu_bias_noise(imu_rad, gyro_unit="rad/s", **kw)
        np.testing.assert_allclose((nd - imu_deg)[:, 1:4] / DEG, (nr - imu_rad)[:, 1:4], rtol=0, atol=1e-12)

    def test_only_targeted_columns_change(self):
        imu = synthetic_imu()
        out, _ = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, gyro_noise_rad_s_sqrthz=1e-3)
        self.assertSameArray(out[:, [0, 4, 5, 6]], imu[:, [0, 4, 5, 6]])
        out, _ = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, accel_bias_m_s2=0.1)
        self.assertSameArray(out[:, :4], imu[:, :4])

    def test_seed_and_sequence_change_output(self):
        imu = synthetic_imu()
        kw = dict(gyro_noise_rad_s_sqrthz=1e-3)
        a, _ = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=0, **kw)
        b, _ = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=1, **kw)
        c, _ = ss.apply_imu_bias_noise(imu, sequence_id="other", stress_seed=0, **kw)
        self.assertNotEqual(ss.array_sha256(a), ss.array_sha256(b))
        self.assertNotEqual(ss.array_sha256(a), ss.array_sha256(c))

    def test_evidence(self):
        imu = synthetic_imu()
        out, ev = ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=3, gyro_bias_rad_s=0.01,
                                          accel_noise_m_s2_sqrthz=1e-2)
        self.assertEvidence(ev, "imu_bias_noise", ss.array_sha256(out))
        self.assertEqual(ev["input_sha256"], ss.array_sha256(imu))
        self.assertTrue(ev["data_changed"])
        self.assertEqual(ev["sequence_id"], SEQ)
        self.assertEqual(ev["stress_seed"], 3)
        self.assertEqual(ev["requested"]["gyro_bias"]["unit"], "rad/s")
        self.assertEqual(ev["requested"]["accel_noise_density"]["unit"], "m/s^2/sqrt(Hz)")
        self.assertEqual(set(ev["derived_seeds"]), {"gyro_bias_direction", "accel_noise"})

    def test_validation(self):
        imu = synthetic_imu()
        kw = dict(sequence_id=SEQ, stress_seed=0)
        with self.assertRaises(TypeError):
            ss.apply_imu_bias_noise(imu.astype(np.float32), **kw)
        with self.assertRaises(TypeError):
            ss.apply_imu_bias_noise(imu.tolist(), **kw)
        with self.assertRaises(ValueError):
            ss.apply_imu_bias_noise(imu[:, :6].copy(), **kw)
        bad = imu.copy(); bad[10, 3] = np.nan
        with self.assertRaises(ValueError):
            ss.apply_imu_bias_noise(bad, **kw)
        bad = imu.copy(); bad[[10, 11], 0] = bad[[11, 10], 0]
        with self.assertRaises(ValueError):
            ss.apply_imu_bias_noise(bad, **kw)
        secs = imu.copy(); secs[:, 0] /= 1e6  # array after run_DEIO2's /1e6
        with self.assertRaises(ValueError):
            ss.apply_imu_bias_noise(secs, **kw)
        for bad_kw in ({"gyro_bias_rad_s": -0.1}, {"gyro_bias_rad_s": [0.1, 0.2]},
                       {"accel_noise_m_s2_sqrthz": -1.0}, {"gyro_unit": "rpm"},
                       {"gyro_noise_rad_s_sqrthz": float("inf")}):
            with self.assertRaises(ValueError):
                ss.apply_imu_bias_noise(imu, **dict(kw, **bad_kw))
        with self.assertRaises(TypeError):
            ss.apply_imu_bias_noise(imu, gyro_bias_rad_s=0.1)  # sequence_id / stress_seed required


# ============================================================================ 2. time offset

class TestImuTimeOffset(_Base):
    def test_exact_shift_and_other_columns_untouched(self):
        imu = synthetic_imu()
        rng = (imu[0, 0], imu[-1, 0])
        for off_ms in (2.5, -7.0, 0.125, 40.0):
            out, ev = ss.apply_imu_time_offset(imu, off_ms, rng)
            self.assertTrue(np.array_equal(out[:, 0] - imu[:, 0], np.full(len(imu), off_ms * 1000.0)))
            self.assertTrue(np.array_equal(out[:, 0], imu[:, 0] + off_ms * 1000.0))
            self.assertSameArray(out[:, 1:], imu[:, 1:])
            self.assertEqual(out.shape, imu.shape)  # nothing dropped, nothing added
            r = ev["realized"]
            self.assertEqual(r["offset_us"], off_ms * 1000.0)
            self.assertEqual(r["max_abs_float_rounding_us"], 0.0)
            self.assertEqual((r["n_dropped"], r["n_extrapolated"]), (0, 0))

    def test_absolute_unix_us_rounding_recorded(self):
        # VECtor keeps absolute Unix microseconds (~1.6e15): float64 spacing there is 0.25 us.
        imu = synthetic_imu(t0_us=1.6e15)
        out, ev = ss.apply_imu_time_offset(imu, 1e-4, (imu[0, 0], imu[-1, 0]))  # 0.1 us request
        self.assertLessEqual(ev["realized"]["max_abs_float_rounding_us"], 0.125)
        out, ev = ss.apply_imu_time_offset(imu, 2.5, (imu[0, 0], imu[-1, 0]))
        self.assertEqual(ev["realized"]["max_abs_float_rounding_us"], 0.0)

    def test_noop(self):
        imu = synthetic_imu()
        for z in (0, 0.0, -0.0):
            out, ev = ss.apply_imu_time_offset(imu, z, (imu[0, 0], imu[-1, 0]))
            self.assertSameArray(out, imu)
            self.assertIsNot(out, imu)
            self.assertFalse(ev["data_changed"])

    def test_no_input_mutation(self):
        imu = synthetic_imu()
        ref = imu.copy()
        ss.apply_imu_time_offset(imu, 12.0, (imu[0, 0], imu[-1, 0]))
        self.assertSameArray(imu, ref)

    def test_edge_accounting(self):
        # IMU 200 Hz on t = 5000*k us, k = 0..2000 (0..10 s); events cover [1 s, 9 s].
        imu = synthetic_imu(n=2001, t0_us=0.0)
        ev_range = (1.0e6, 9.0e6)
        out, ev = ss.apply_imu_time_offset(imu, 1500.0, ev_range)  # +1.5 s
        a = ev["realized"]["after_shift"]
        t2 = imu[:, 0] + 1.5e6
        self.assertEqual(a["n_before_event_range"], 0)
        self.assertEqual(a["n_after_event_range"], 500)   # t2 > 9e6  <=> k > 1500
        self.assertEqual(a["n_inside_event_range"], 1501)  # 1.5e6 <= t2 <= 9e6
        self.assertEqual(a["n_after_event_range"], int(np.sum(t2 > 9.0e6)))
        self.assertEqual(a["uncovered_head_us"], 0.5e6)
        self.assertEqual(a["uncovered_tail_us"], 0.0)
        self.assertFalse(a["covers_event_range"])
        b = ev["realized"]["before_shift"]
        self.assertEqual((b["n_before_event_range"], b["n_after_event_range"]), (200, 200))
        self.assertTrue(b["covers_event_range"])
        _, ev = ss.apply_imu_time_offset(imu, -1500.0, ev_range)
        a = ev["realized"]["after_shift"]
        self.assertEqual(a["n_before_event_range"], 500)
        self.assertEqual(a["uncovered_tail_us"], 0.5e6)
        self.assertEqual(a["imu_last_us"], 10.0e6 - 1.5e6)
        self.assertEqual(sum(a[k] for k in ("n_before_event_range", "n_after_event_range",
                                            "n_inside_event_range")), len(imu))

    def test_evidence(self):
        imu = synthetic_imu()
        out, ev = ss.apply_imu_time_offset(imu, -3.0, (imu[0, 0], imu[-1, 0]), sequence_id=SEQ)
        self.assertEvidence(ev, "imu_time_offset", ss.array_sha256(out))
        self.assertEqual(ev["requested"]["offset"], {"value": -3.0, "unit": "ms",
                                                     "convention": "t_imu' = t_imu + offset"})
        self.assertEqual(ev["derived_seeds"], {})
        self.assertEqual(ev["rng"], "none (deterministic)")

    def test_validation(self):
        imu = synthetic_imu()
        with self.assertRaises(ValueError):
            ss.apply_imu_time_offset(imu, float("nan"), (0.0, 1.0))
        with self.assertRaises(ValueError):
            ss.apply_imu_time_offset(imu, 1.0, (5.0, 5.0))
        with self.assertRaises(TypeError):
            ss.apply_imu_time_offset(imu, 1.0, None)


# ============================================================================ 3. bursts

class TestBurstSchedule(_Base):
    def test_determinism(self):
        kw = dict(sequence_id=SEQ, stress_seed=2, target_fraction=0.1)
        a, ea = ss.make_burst_schedule((0.0, 60e6), 50.0, **kw)
        b, eb = ss.make_burst_schedule((0.0, 60e6), 50.0, **kw)
        self.assertSameArray(a, b)
        self.assertEqual(ea, eb)
        c, _ = ss.make_burst_schedule((0.0, 60e6), 50.0, sequence_id=SEQ, stress_seed=3, target_fraction=0.1)
        self.assertFalse(np.array_equal(a, c))
        # fresh RNG reproduces the documented placement rule
        seed = ea["derived_seeds"]["schedule"]
        k = ea["realized"]["n_bursts"]
        u = np.sort(np.random.Generator(np.random.PCG64(seed)).random(k)) * (60e6 - k * 50e3)
        np.testing.assert_array_equal(a[:, 0], 0.0 + u + np.arange(k) * 50e3)

    def test_target_fraction_long_span(self):
        span = (5.0e6, 5.0e6 + 3600e6)  # 1 h
        for frac, dur_ms in ((0.05, 100.0), (0.123, 37.0), (0.3, 250.0)):
            s, ev = ss.make_burst_schedule(span, dur_ms, sequence_id=SEQ, stress_seed=0, target_fraction=frac)
            r = ev["realized"]
            self.assertLessEqual(abs(r["realized_time_fraction"] - frac), dur_ms * 1e3 / (2 * 3600e6) + 1e-12)
            self.assertAlmostEqual(r["realized_time_fraction"], len(s) * dur_ms * 1e3 / 3600e6, places=9)
            self._check_well_formed(s, span, dur_ms)

    def test_rate_mode(self):
        span = (0.0, 1000e6)
        s, ev = ss.make_burst_schedule(span, 20.0, sequence_id=SEQ, stress_seed=0, rate_hz=0.5)
        self.assertEqual(len(s), 500)
        self.assertAlmostEqual(ev["realized"]["realized_rate_hz"], 0.5, places=12)
        self.assertAlmostEqual(ev["realized"]["realized_time_fraction"], 500 * 20e3 / 1000e6, places=12)
        self._check_well_formed(s, span, 20.0)

    def _check_well_formed(self, s, span, dur_ms):
        self.assertEqual(s.dtype, np.float64)
        self.assertTrue((s[:, 0] >= span[0]).all() and (s[:, 1] <= span[1]).all())
        self.assertTrue((s[1:, 0] >= s[:-1, 1]).all())  # sorted, non-overlapping
        np.testing.assert_allclose(s[:, 1] - s[:, 0], dur_ms * 1e3, rtol=0, atol=1e-3)

    def test_zero_level_empty(self):
        for kw in ({"rate_hz": 0.0}, {"target_fraction": 0.0}):
            s, ev = ss.make_burst_schedule((0.0, 10e6), 50.0, sequence_id=SEQ, stress_seed=0, **kw)
            self.assertEqual(s.shape, (0, 2))
            self.assertEqual(ev["realized"]["realized_time_fraction"], 0.0)

    def test_shared_vs_independent(self):
        kw = dict(sequence_id=SEQ, stress_seed=0, target_fraction=0.1)
        sh = ss.make_burst_schedules((0.0, 60e6), 40.0, mode="shared", **kw)
        self.assertSameArray(sh["imu"][0], sh["events"][0])
        ind = ss.make_burst_schedules((0.0, 60e6), 40.0, mode="independent", **kw)
        self.assertEqual(len(ind["imu"][0]), len(ind["events"][0]))
        self.assertFalse(np.array_equal(ind["imu"][0], ind["events"][0]))
        self.assertFalse(np.array_equal(ind["imu"][0], sh["imu"][0]))
        self.assertEqual(ind["imu"][1]["requested"]["stream"]["value"], "imu")
        again = ss.make_burst_schedules((0.0, 60e6), 40.0, mode="independent", **kw)
        self.assertSameArray(again["events"][0], ind["events"][0])
        with self.assertRaises(ValueError):
            ss.make_burst_schedules((0.0, 60e6), 40.0, mode="both", **kw)

    def test_evidence(self):
        s, ev = ss.make_burst_schedule((0.0, 60e6), 40.0, sequence_id=SEQ, stress_seed=0, rate_hz=1.0)
        self.assertEvidence(ev, "burst_schedule", ss.array_sha256(s))
        self.assertEqual(ev["realized"]["schedule_sha256"], ss.array_sha256(s))
        self.assertEqual(np.array(ev["realized"]["schedule_us"]).tolist(), s.tolist())
        self.assertEqual(ev["requested"]["burst_duration"], {"value": 40.0, "unit": "ms"})

    def test_validation(self):
        kw = dict(sequence_id=SEQ, stress_seed=0)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, rate_hz=1.0, target_fraction=0.1, **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, target_fraction=1.0, **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, rate_hz=20.0, **kw)  # 200 x 50 ms = whole span
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 0.0, rate_hz=1.0, **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, rate_hz=-1.0, **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((0.0, 10e6), 50.0, rate_hz=1.0, stream="gps", **kw)
        with self.assertRaises(ValueError):
            ss.make_burst_schedule((10e6, 0.0), 50.0, rate_hz=1.0, **kw)


class TestBurstMaskAndAppliers(_Base):
    SCHED = np.array([[100.0, 200.0], [300.0, 350.0]])

    def test_mask_half_open_windows(self):
        t = np.arange(0, 500, 10, dtype=np.int64)
        keep = ss.burst_keep_mask(t, self.SCHED)
        removed = t[~keep].tolist()
        self.assertEqual(removed, list(range(100, 200, 10)) + list(range(300, 350, 10)))
        self.assertTrue(keep[t == 200][0] and keep[t == 350][0] and keep[t == 90][0])
        self.assertFalse(keep[t == 100][0] or keep[t == 300][0])
        tf = np.array([99.999, 100.0, 199.999, 200.0, 299.0, 349.5, 350.0])
        self.assertEqual(ss.burst_keep_mask(tf, self.SCHED).tolist(),
                         [True, False, False, True, True, False, True])

    def test_mask_matches_brute_force(self):
        s, _ = ss.make_burst_schedule((0.0, 20e6), 30.0, sequence_id=SEQ, stress_seed=7, target_fraction=0.25)
        t = np.sort(np.random.Generator(np.random.PCG64(1)).uniform(-1e6, 21e6, 50_000))
        inside = np.zeros(t.shape, dtype=bool)
        for a, b in s:
            inside |= (t >= a) & (t < b)
        self.assertTrue(np.array_equal(ss.burst_keep_mask(t, s), ~inside))
        self.assertTrue(np.array_equal(ss.burst_keep_mask(t[::-1], s), ~inside[::-1]))  # order-free

    def test_imu_dropout_exact(self):
        imu = synthetic_imu(n=4001)
        s, sev = ss.make_burst_schedule((imu[0, 0], imu[-1, 0]), 60.0, sequence_id=SEQ, stress_seed=0,
                                        target_fraction=0.2)
        out, ev = ss.apply_imu_burst_dropout(imu, s, sequence_id=SEQ)
        inside = np.zeros(len(imu), dtype=bool)
        for a, b in s:
            inside |= (imu[:, 0] >= a) & (imu[:, 0] < b)
        self.assertSameArray(out, imu[~inside])
        r = ev["realized"]
        self.assertEqual(r["n_removed"], int(inside.sum()))
        self.assertEqual(r["n_in"], len(imu))
        self.assertEqual(r["n_out"], len(out))
        self.assertEqual(r["schedule_sha256"], sev["output_sha256"])
        self.assertEqual(r["n_bursts"], len(s))
        self.assertAlmostEqual(r["time_fraction_removed_in_data_span"], sev["realized"]["realized_time_fraction"],
                               places=12)
        self.assertAlmostEqual(r["fraction_removed"], 0.2, delta=0.01)
        self.assertGreater(r["max_gap_us_out"], r["max_gap_us_in"])
        self.assertEvidence(ev, "imu_burst_dropout", ss.array_sha256(out))

    def test_event_dropout_exact(self):
        evs = synthetic_events()
        ref = {k: v.copy() for k, v in evs.items() if k != "meta"}
        out, ev = ss.apply_event_burst_dropout(evs, self.SCHED + 1_000_000.0)
        t = evs["t"]
        keep = ~(((t >= 1_000_100) & (t < 1_000_200)) | ((t >= 1_000_300) & (t < 1_000_350)))
        for k in ss.EVENT_KEYS:
            self.assertSameArray(out[k], evs[k][keep])
            self.assertSameArray(evs[k], ref[k])  # input untouched
        self.assertEqual(out["meta"], "passthrough")
        self.assertEqual(ev["realized"]["n_removed"], int((~keep).sum()))
        self.assertEqual(ev["realized"]["n_removed"], 14 + 7)  # multiples of 7 in [100,200), [300,350)
        self.assertEqual(ev["input_sha256"], ss.events_sha256(evs))
        self.assertEvidence(ev, "event_burst_dropout", ss.events_sha256(out))
        self.assertTrue(ev["data_changed"])

    def test_noop_and_empty(self):
        imu = synthetic_imu()
        empty = np.empty((0, 2))
        out, ev = ss.apply_imu_burst_dropout(imu, empty)
        self.assertSameArray(out, imu)
        self.assertIsNot(out, imu)
        self.assertFalse(ev["data_changed"])
        out, ev = ss.apply_imu_burst_dropout(imu, [])
        self.assertSameArray(out, imu)
        # windows that fall between samples remove nothing -> identical output, identical hash
        out, ev = ss.apply_imu_burst_dropout(imu, [[imu[3, 0] + 1.0, imu[3, 0] + 2.0]])
        self.assertSameArray(out, imu)
        self.assertFalse(ev["data_changed"])
        evs = synthetic_events()
        o, e = ss.apply_event_burst_dropout(evs, empty)
        for k in ss.EVENT_KEYS:
            self.assertSameArray(o[k], evs[k])
            self.assertIsNot(o[k], evs[k])
        self.assertFalse(e["data_changed"])
        e0 = {k: v[:0] for k, v in synthetic_events().items() if k != "meta"}
        o, e = ss.apply_event_burst_dropout(e0, self.SCHED)
        self.assertEqual(e["realized"]["n_in"], 0)
        self.assertIsNone(e["realized"]["data_span_us"])
        json.dumps(e)

    def test_validation(self):
        evs = synthetic_events()
        with self.assertRaises(ValueError):
            ss.burst_keep_mask(evs["t"], [[0.0, 10.0], [5.0, 20.0]])  # overlapping
        with self.assertRaises(ValueError):
            ss.burst_keep_mask(evs["t"], [[10.0, 10.0]])  # empty window
        with self.assertRaises(ValueError):
            ss.burst_keep_mask(evs["t"], [[20.0, 30.0], [0.0, 10.0]])  # unsorted
        with self.assertRaises(ValueError):
            ss.burst_keep_mask(evs["t"], [1.0, 2.0, 3.0])
        bad = dict(evs); bad["x"] = bad["x"][:-1]
        with self.assertRaises(ValueError):
            ss.apply_event_burst_dropout(bad, self.SCHED)
        bad = dict(evs); bad["t"] = bad["t"][::-1].copy()
        with self.assertRaises(ValueError):
            ss.apply_event_burst_dropout(bad, self.SCHED)
        bad = dict(evs); del bad["p"]
        with self.assertRaises(ValueError):
            ss.apply_event_burst_dropout(bad, self.SCHED)
        bad = dict(evs); bad["t"] = bad["t"].astype(np.float64); bad["t"][3] = np.inf
        with self.assertRaises(ValueError):
            ss.apply_event_burst_dropout(bad, self.SCHED)



# ============================================================================ 4. event background noise

def _rows(ev):
    return np.stack([ev[k].astype(np.float64) for k in ss.EVENT_KEYS], axis=1)


class TestEventBackgroundNoise(_Base):
    HW = (480, 640)
    WIN = (1_000_000, 1_035_000)          # synthetic_events() spans [1_000_000, 1_035_000)
    KW = dict(sequence_id=SEQ, stress_seed=5, ba_rate_hz_per_pixel=0.1,
              hot_pixel_fraction=1e-4, hot_pixel_rate_hz=50.0)

    def test_noop_identity(self):
        evs = synthetic_events()
        for kw in (dict(), dict(ba_rate_hz_per_pixel=0.0, hot_pixel_fraction=0.01, hot_pixel_rate_hz=0.0),
                   dict(hot_pixel_fraction=0.0, hot_pixel_rate_hz=100.0)):
            out, ev = ss.apply_event_background_noise(evs, self.WIN, self.HW, sequence_id=SEQ, stress_seed=5, **kw)
            for k in ss.EVENT_KEYS:
                self.assertSameArray(out[k], evs[k])
                self.assertIsNot(out[k], evs[k])
            self.assertEqual(out["meta"], "passthrough")
            self.assertFalse(ev["data_changed"])
            self.assertNotIn("event_background_noise", ev["derived_seeds"])
            self.assertEvidence(ev, "event_background_noise", ss.events_sha256(out))

    def test_no_input_mutation_and_originals_kept_in_order(self):
        evs = synthetic_events()
        ref = {k: v.copy() for k, v in evs.items() if k != "meta"}
        out, ev = ss.apply_event_background_noise(evs, self.WIN, self.HW, **self.KW)
        for k in ss.EVENT_KEYS:
            self.assertSameArray(evs[k], ref[k])
        self.assertTrue(np.all(np.diff(out["t"]) >= 0))
        n_noise = ev["realized"]["n_background"] + ev["realized"]["n_hot"]
        self.assertGreater(n_noise, 0)
        self.assertEqual(out["t"].shape[0], 5000 + n_noise)
        o_rows, i_rows = _rows(out), _rows(evs)
        j = 0
        for r in o_rows:                    # originals form a subsequence of the output, in order
            if j < len(i_rows) and np.array_equal(r, i_rows[j]):
                j += 1
        self.assertEqual(j, len(i_rows))
        for k in ss.EVENT_KEYS:
            self.assertEqual(out[k].dtype, evs[k].dtype)

    def test_determinism_and_seed_sensitivity(self):
        evs = synthetic_events()
        a, ea = ss.apply_event_background_noise(evs, self.WIN, self.HW, **self.KW)
        b, eb = ss.apply_event_background_noise(evs, self.WIN, self.HW, **self.KW)
        for k in ss.EVENT_KEYS:
            self.assertSameArray(a[k], b[k])
        self.assertEqual(ea, eb)
        np.random.seed(123)                 # global RNG must not matter
        c, _ = ss.apply_event_background_noise(evs, self.WIN, self.HW, **self.KW)
        self.assertEqual(ss.events_sha256(a), ss.events_sha256(c))
        for change in (dict(stress_seed=6), dict(sequence_id="other"), dict(ba_rate_hz_per_pixel=0.2)):
            d, _ = ss.apply_event_background_noise(evs, self.WIN, self.HW, **dict(self.KW, **change))
            self.assertNotEqual(ss.events_sha256(a), ss.events_sha256(d), change)
        e, _ = ss.apply_event_background_noise(evs, (1_000_000, 1_035_001), self.HW, **self.KW)
        self.assertNotEqual(ss.events_sha256(a), ss.events_sha256(e))   # window enters the seed

    def test_rates_match_poisson_expectation(self):
        empty = {k: v[:0] for k, v in synthetic_events().items() if k != "meta"}
        win = (0, 2_000_000)                # 2 s
        out, ev = ss.apply_event_background_noise(empty, win, self.HW, sequence_id=SEQ, stress_seed=1,
                                                  ba_rate_hz_per_pixel=0.05, hot_pixel_fraction=2e-4,
                                                  hot_pixel_rate_hz=100.0, polarity_values=(0, 1))
        r = ev["realized"]
        mu = 0.05 * 480 * 640 * 2.0
        self.assertLess(abs(r["n_background"] - mu), 5 * math.sqrt(mu))
        self.assertEqual(r["n_hot_pixels"], round(2e-4 * 480 * 640))
        mu_hot = 100.0 * 2.0 * r["n_hot_pixels"]
        self.assertLess(abs(r["n_hot"] - mu_hot), 5 * math.sqrt(mu_hot))
        self.assertTrue(np.all((out["t"] >= 0) & (out["t"] < 2_000_000)))
        self.assertTrue(np.all((out["x"] >= 0) & (out["x"] < 640) & (out["y"] >= 0) & (out["y"] < 480)))
        frac_pos = float(np.mean(out["p"] == 1))
        self.assertLess(abs(frac_pos - 0.5), 0.01)

    def test_hot_pixels_fixed_per_sequence(self):
        idx1, i1 = ss.hot_pixel_set(self.HW, 1e-4, sequence_id=SEQ, stress_seed=5)
        idx2, _ = ss.hot_pixel_set(self.HW, 1e-4, sequence_id=SEQ, stress_seed=5)
        self.assertSameArray(idx1, idx2)
        self.assertEqual(len(np.unique(idx1)), len(idx1))
        empty = {k: v[:0] for k, v in synthetic_events().items() if k != "meta"}
        lin = []
        for w in ((0, 500_000), (500_000, 1_000_000)):       # two windows, same hot set
            out, ev = ss.apply_event_background_noise(empty, w, self.HW, sequence_id=SEQ, stress_seed=5,
                                                      hot_pixel_fraction=1e-4, hot_pixel_rate_hz=200.0,
                                                      polarity_values=(0, 1))
            self.assertEqual(ev["realized"]["hot_pixels_sha256"], i1["realized"]["hot_pixels_sha256"])
            lin.append(out["y"].astype(np.int64) * 640 + out["x"])
        for l in lin:
            self.assertTrue(np.isin(l, idx1).all())
        other, _ = ss.hot_pixel_set(self.HW, 1e-4, sequence_id="other", stress_seed=5)
        self.assertFalse(np.array_equal(idx1, other))

    def test_timestamp_dtype_and_window(self):
        evs = synthetic_events()
        out, _ = ss.apply_event_background_noise(evs, (1_000_000.5, 1_035_000.5), self.HW, **self.KW)
        self.assertEqual(out["t"].dtype, np.int64)
        self.assertTrue(np.all((out["t"] >= 1_000_000) & (out["t"] <= 1_035_000)))   # [ceil a, ceil b)
        fl = dict(evs, t=evs["t"].astype(np.float64))
        out, _ = ss.apply_event_background_noise(fl, (1_000_000.25, 1_035_000.25), self.HW, **self.KW)
        noise_t = np.setdiff1d(out["t"], fl["t"])
        self.assertTrue(np.all((noise_t >= 1_000_000.25) & (noise_t < 1_035_000.25)))

    def test_polarity_encoding(self):
        evs = synthetic_events()
        pm = dict(evs, p=np.where(evs["p"] == 1, 1, -1).astype(np.int8))
        out, ev = ss.apply_event_background_noise(pm, self.WIN, self.HW, **self.KW)
        self.assertEqual(set(np.unique(out["p"]).tolist()), {-1, 1})
        self.assertEqual(ev["realized"]["polarity_values"], [-1, 1])
        empty = {k: v[:0] for k, v in evs.items() if k != "meta"}
        with self.assertRaises(ValueError):          # cannot infer the encoding from nothing
            ss.apply_event_background_noise(empty, self.WIN, self.HW, **self.KW)
        out, ev = ss.apply_event_background_noise(empty, self.WIN, self.HW, polarity_values=(-1, 1), **self.KW)
        self.assertEqual(ev["realized"]["polarity_source"], "given")
        self.assertTrue(set(np.unique(out["p"]).tolist()) <= {-1, 1})

    def test_evidence(self):
        evs = synthetic_events()
        out, ev = ss.apply_event_background_noise(evs, self.WIN, self.HW, **self.KW)
        self.assertEvidence(ev, "event_background_noise", ss.events_sha256(out))
        self.assertTrue(ev["data_changed"])
        self.assertIn("event_background_noise", ev["derived_seeds"])
        self.assertIn("event_hot_pixels", ev["derived_seeds"])
        self.assertEqual(ev["requested"]["ba_rate_hz_per_pixel"]["unit"], "Hz/pixel")
        self.assertEqual(ev["realized"]["n_out"], out["t"].shape[0])
        self.assertEqual(ev["realized"]["n_in_outside_window"], 0)
        _, ev2 = ss.apply_event_background_noise(evs, (1_010_000, 1_020_000), self.HW, **self.KW)
        self.assertGreater(ev2["realized"]["n_in_outside_window"], 0)

    def test_validation(self):
        evs = synthetic_events()
        bad = [dict(window_us=(5, 5)), dict(window_us=(0, float("inf"))), dict(sensor_hw=(0, 640)),
               dict(sensor_hw=(480,)), dict(sensor_hw=(480.0, 640)), dict(ba_rate_hz_per_pixel=-1),
               dict(hot_pixel_fraction=1.5), dict(hot_pixel_rate_hz=float("nan")),
               dict(polarity_values=(1, 1)), dict(stress_seed=-1), dict(sequence_id="")]
        for b in bad:
            kw = dict(self.KW, window_us=self.WIN, sensor_hw=self.HW)
            kw.update(b)
            win, hw = kw.pop("window_us"), kw.pop("sensor_hw")
            with self.assertRaises((ValueError, TypeError), msg=str(b)):
                ss.apply_event_background_noise(evs, win, hw, **kw)


# ============================================================================ evidence conventions

class TestEvidenceConventions(_Base):
    def test_hash_changes_iff_data_changes(self):
        imu = synthetic_imu()
        h0 = ss.array_sha256(imu)
        self.assertEqual(h0, ss.array_sha256(imu.copy()))
        self.assertEqual(h0, ss.array_sha256(np.asfortranarray(imu)))  # layout-independent
        x = imu.copy(); x[0, 4] = np.nextafter(x[0, 4], np.inf)
        self.assertNotEqual(h0, ss.array_sha256(x))
        outs = [ss.apply_imu_bias_noise(imu, sequence_id=SEQ, stress_seed=s, gyro_noise_rad_s_sqrthz=1e-3)[1]
                for s in (0, 0, 1)]
        self.assertEqual(outs[0]["output_sha256"], outs[1]["output_sha256"])
        self.assertNotEqual(outs[0]["output_sha256"], outs[2]["output_sha256"])
        for e in outs:
            self.assertEqual(e["input_sha256"], h0)

    def test_evidence_line_parses_like_stress_evidence(self):
        imu = synthetic_imu()
        _, ev = ss.apply_imu_time_offset(imu, 1.0, (imu[0, 0], imu[-1, 0]), sequence_id=SEQ)
        line = ss.format_evidence_line(ev)
        self.assertTrue(line.startswith("[sensor-stress] "))
        kv = dict(re.findall(r"(\w+)=(\S+)", line))  # same regex as script/stress_evidence.py
        self.assertEqual(kv["operator"], "imu_time_offset")
        self.assertEqual(kv["version"], ss.SENSOR_STRESS_VERSION)
        self.assertEqual(kv["out_sha256"], ev["output_sha256"])
        self.assertEqual(kv["changed"], "1")


if __name__ == "__main__":
    unittest.main()
