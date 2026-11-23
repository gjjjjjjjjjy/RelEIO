"""
sensor_stress.py -- physically motivated sensing-disturbance operators for the
RelEIO stress protocol (event noise, IMU bias/noise, event-IMU
time offset, burst missing measurements). Spec: docs/SENSOR_STRESS_OPERATORS.md.

STANDALONE: not wired into any evaluation pipeline. Pure stdlib + numpy, Python >= 3.9.

Array formats (verified against the DEIO loaders, see the spec for file:line):
  IMU    float64 ndarray [N, 7] exactly as handed to utils/eval_utils.run_DEIO2,
         i.e. BEFORE its in-place `_all_imu[:, 0] /= 1e6`:
           col 0     t      microseconds, non-decreasing
           col 1:4   gyro   deg/s  (DEIO convention; devo/dba.py multiplies by pi/180)
           col 4:7   accel  m/s^2
  Events dict of equal-length 1-D arrays "t" (microseconds), "x", "y", "p" -- the form
         utils/event_stress.apply_event_stress acts on. Other keys pass through untouched.

Determinism: every random draw comes from numpy.random.Generator(PCG64(seed)) with
  seed = derive_seed(component, params, sequence_id, stress_seed)      (SHA-256 based).
The stress seed is a protocol constant, independent of the estimator's evaluation seed
(this module never touches numpy's global RNG), so the degraded input is identical
across repeated runs of a condition.

Every applier returns (output, evidence) and never mutates its input.
"""
import hashlib
import json
import math
import numbers

import numpy as np

# Bump whenever operator semantics (draws, order, seeding, edge rules) change.
SENSOR_STRESS_VERSION = "ss-v2"  # v2 adds event background noise; v1 operators unchanged

IMU_NCOLS = 7
IMU_GYRO = slice(1, 4)
IMU_ACC = slice(4, 7)
EVENT_KEYS = ("t", "x", "y", "p")

# Factor converting a gyro quantity given in rad/s into the array's gyro unit.
GYRO_UNIT_FROM_RAD_S = {"deg/s": 180.0 / math.pi, "rad/s": 1.0}


def _imu_format(gyro_unit=None):
    g = gyro_unit if gyro_unit else "not modified; deg/s in the run_DEIO2 format"
    return "float64 [N,7]: t (us), gyro xyz (%s), accel xyz (m/s^2)" % g


EVENT_FORMAT = "dict of equal-length 1-D arrays: t (us), x, y, p"

# Plausible IMU sample interval: catches seconds-vs-microseconds mix-ups
# (e.g. an array that already went through run_DEIO2's /1e6).
_MIN_DT_US, _MAX_DT_US = 10.0, 1.0e6

OP_BIAS_NOISE = "imu_bias_noise"
OP_TIME_OFFSET = "imu_time_offset"
OP_BURST_SCHEDULE = "burst_schedule"
OP_IMU_BURST = "imu_burst_dropout"
OP_EVENT_BURST = "event_burst_dropout"
OP_EVENT_NOISE = "event_background_noise"
OP_HOT_PIXELS = "event_hot_pixels"
BURST_STREAMS = ("shared", "imu", "events")

__all__ = [
    "SENSOR_STRESS_VERSION", "GYRO_UNIT_FROM_RAD_S", "BURST_STREAMS",
    "derive_seed", "make_rng", "array_sha256", "events_sha256", "format_evidence_line",
    "apply_imu_bias_noise", "apply_imu_time_offset",
    "make_burst_schedule", "make_burst_schedules", "burst_keep_mask",
    "apply_imu_burst_dropout", "apply_event_burst_dropout",
    "hot_pixel_set", "apply_event_background_noise",
]


# ----------------------------------------------------------------------------- seeding

def _canon(v):
    """Canonical JSON-able form for seed hashing: all numbers -> float, tuples -> lists."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_canon(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("seed parameters must be finite, got %r" % (v,))
        return f
    if v is None or isinstance(v, str):
        return v
    raise TypeError("unsupported seed parameter type %s" % type(v).__name__)


def _check_seed_inputs(sequence_id, stress_seed):
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be a non-empty str, got %r" % (sequence_id,))
    if isinstance(stress_seed, (bool, np.bool_)) or not isinstance(stress_seed, numbers.Integral) \
            or int(stress_seed) < 0:
        raise ValueError("stress_seed must be a non-negative int, got %r" % (stress_seed,))


def derive_seed(component, params, sequence_id, stress_seed):
    """64-bit seed = first 8 bytes (big-endian) of SHA-256 over canonical JSON of
    {component, params (sorted keys, numbers as float), sequence_id, stress_seed}."""
    _check_seed_inputs(sequence_id, stress_seed)
    payload = json.dumps(
        {"component": str(component), "params": _canon(params),
         "sequence_id": sequence_id, "stress_seed": int(stress_seed)},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def make_rng(seed):
    """The only RNG constructor used by this module."""
    return np.random.Generator(np.random.PCG64(seed))


# ----------------------------------------------------------------------------- hashing / evidence

def array_sha256(a):
    """SHA-256 hex of np.ascontiguousarray(a).tobytes()."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def events_sha256(ev):
    """SHA-256 hex over the raw bytes of ev['t'], ev['x'], ev['y'], ev['p'] (in that order)."""
    h = hashlib.sha256()
    for k in EVENT_KEYS:
        h.update(np.ascontiguousarray(ev[k]).tobytes())
    return h.hexdigest()


def _py(v):
    """numpy -> plain Python so evidence is json.dumps-able."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return _py(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        return float(v)
    return v


def _evidence(operator, requested, realized, input_format, in_sha, out_sha,
              sequence_id=None, stress_seed=None, derived_seeds=None):
    seeds = dict(derived_seeds or {})
    return _py({
        "operator": operator,
        "module_version": SENSOR_STRESS_VERSION,
        "numpy_version": np.__version__,
        "sequence_id": sequence_id,
        "stress_seed": stress_seed,
        "derived_seeds": seeds,
        "rng": "numpy.random.Generator(PCG64(seed))" if seeds else "none (deterministic)",
        "requested": requested,
        "realized": realized,
        "input_format": input_format,
        "input_sha256": in_sha,
        "output_sha256": out_sha,
        "data_changed": in_sha != out_sha,
    })


def format_evidence_line(ev):
    """One greppable log line in the style of event_stress's `[stress-evidence]` line."""
    fields = [("operator", ev["operator"]), ("version", ev["module_version"]),
              ("seq", ev.get("sequence_id")), ("stress_seed", ev.get("stress_seed")),
              ("in_sha256", ev["input_sha256"]), ("out_sha256", ev["output_sha256"]),
              ("changed", int(bool(ev["data_changed"])))]
    return "[sensor-stress] " + " ".join("%s=%s" % kv for kv in fields)


# ----------------------------------------------------------------------------- validation

def _check_imu(imu):
    """Validate the run_DEIO2 IMU format; return interval stats (us)."""
    if not isinstance(imu, np.ndarray):
        raise TypeError("imu must be a numpy.ndarray, got %s" % type(imu).__name__)
    if imu.ndim != 2 or imu.shape[1] != IMU_NCOLS:
        raise ValueError("imu must have shape [N, 7] (t_us, gyro xyz, accel xyz), got %s"
                         % (imu.shape,))
    if imu.dtype != np.float64:
        raise TypeError("imu must be float64 (as produced by the loaders), got %s" % imu.dtype)
    if imu.shape[0] < 2:
        raise ValueError("imu needs at least 2 samples, got %d" % imu.shape[0])
    finite = np.isfinite(imu)
    if not finite.all():
        r, c = np.argwhere(~finite)[0]
        raise ValueError("imu has non-finite values (first at row %d, col %d)" % (r, c))
    t = imu[:, 0]
    bad = t[1:] < t[:-1]
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError("imu timestamps must be non-decreasing: t[%d]=%r > t[%d]=%r"
                         % (i, t[i], i + 1, t[i + 1]))
    dt = np.diff(t)
    med = float(np.median(dt))
    if not (_MIN_DT_US <= med <= _MAX_DT_US):
        raise ValueError("median IMU interval %.6g is implausible for microseconds "
                         "(expected %g..%g us); column 0 must be in us, i.e. pass the array "
                         "BEFORE run_DEIO2's /1e6" % (med, _MIN_DT_US, _MAX_DT_US))
    return {"dt_us_median": med, "dt_us_min": float(dt.min()), "dt_us_max": float(dt.max())}


def _check_events(ev):
    """Validate the event dict format; return the number of events."""
    if not isinstance(ev, dict):
        raise TypeError("events must be a dict with keys %s, got %s" % (EVENT_KEYS, type(ev).__name__))
    missing = [k for k in EVENT_KEYS if k not in ev]
    if missing:
        raise ValueError("events dict is missing keys %s" % missing)
    n = None
    for k in EVENT_KEYS:
        a = ev[k]
        if not isinstance(a, np.ndarray) or a.ndim != 1:
            raise ValueError("events[%r] must be a 1-D numpy array" % k)
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError("events arrays differ in length: len(t)=%d, len(%s)=%d" % (n, k, a.shape[0]))
    t = ev["t"]
    if not (np.issubdtype(t.dtype, np.integer) or np.issubdtype(t.dtype, np.floating)):
        raise TypeError("events['t'] must be integer or float microseconds, got %s" % t.dtype)
    if np.issubdtype(t.dtype, np.floating) and not np.isfinite(t).all():
        raise ValueError("events['t'] has non-finite values")
    bad = t[1:] < t[:-1]
    if bad.any():
        raise ValueError("events['t'] must be non-decreasing (first violation at index %d)"
                         % (int(np.argmax(bad)) + 1))
    return n


def _as_span(span, name):
    try:
        a, b = span
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        raise TypeError("%s must be a (start_us, end_us) pair, got %r" % (name, span))
    if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
        raise ValueError("%s must be finite with end > start, got (%r, %r)" % (name, a, b))
    return a, b


def _as_schedule(schedule):
    """Validate a burst schedule: float64 [K, 2] of sorted, non-overlapping [start, end) us."""
    s = np.asarray(schedule, dtype=np.float64)
    if s.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError("schedule must have shape [K, 2] (t_start_us, t_end_us), got %s" % (s.shape,))
    if not np.isfinite(s).all():
        raise ValueError("schedule has non-finite values")
    if not (s[:, 1] > s[:, 0]).all():
        raise ValueError("schedule windows must have t_end > t_start")
    if not (s[1:, 0] >= s[:-1, 1]).all():
        raise ValueError("schedule windows must be sorted and non-overlapping")
    return s


def _as_nonneg(v, name):
    f = float(v)
    if not math.isfinite(f) or f < 0:
        raise ValueError("%s must be finite and >= 0, got %r" % (name, v))
    return f


def _covered_us(schedule, a, b):
    """Length (us) of the union of schedule windows inside [a, b] (windows are disjoint)."""
    if len(schedule) == 0 or b <= a:
        return 0.0
    return float(np.sum(np.clip(schedule[:, 1], a, b) - np.clip(schedule[:, 0], a, b)))


# ----------------------------------------------------------------------------- 1. IMU bias + noise

def _resolve_bias(name, value, sequence_id, stress_seed, seeds):
    """scalar >= 0 -> magnitude along a seeded random unit direction; 3-vector -> explicit.
    Returns (mode, bias_vector_SI, unit_direction_or_None)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        mag = float(arr)
        if not math.isfinite(mag) or mag < 0:
            raise ValueError("%s bias magnitude must be finite and >= 0 (pass a 3-vector for an "
                             "explicit signed bias), got %r" % (name, value))
        if mag == 0.0:
            return "none", np.zeros(3), None
        # Direction depends only on (sequence, stress seed), not on the magnitude, so a
        # magnitude sweep keeps one fixed direction.
        seed = derive_seed("%s/%s_bias_direction" % (OP_BIAS_NOISE, name), {}, sequence_id, stress_seed)
        seeds["%s_bias_direction" % name] = seed
        z = make_rng(seed).standard_normal(3)
        u = z / np.linalg.norm(z)
        return "random_unit_direction", mag * u, u
    if arr.shape == (3,):
        if not np.isfinite(arr).all():
            raise ValueError("%s bias vector must be finite, got %r" % (name, value))
        nrm = float(np.linalg.norm(arr))
        if nrm == 0.0:
            return "none", np.zeros(3), None
        return "explicit_vector", arr.copy(), arr / nrm
    raise ValueError("%s bias must be a scalar magnitude or a 3-vector, got shape %s" % (name, arr.shape))


def apply_imu_bias_noise(imu, *, sequence_id, stress_seed,
                         gyro_bias_rad_s=0.0, accel_bias_m_s2=0.0,
                         gyro_noise_rad_s_sqrthz=0.0, accel_noise_m_s2_sqrthz=0.0,
                         gyro_unit="deg/s"):
    """Additive constant bias and/or zero-mean white noise on gyro and accel.

    imu: float64 [N,7] (t_us, gyro in `gyro_unit`, accel m/s^2); not modified.
    gyro_bias_rad_s / accel_bias_m_s2: scalar >= 0 = magnitude along a deterministic random
        unit direction (seeded by sequence_id + stress_seed); 3-vector = explicit bias.
    gyro_noise_rad_s_sqrthz / accel_noise_m_s2_sqrthz: continuous-time white-noise density
        (same units as cfg.gyro_noise_sigma / cfg.accel_noise_sigma). Discretized per sample as
        sigma_d = sigma_c * sqrt(rate_hz), rate_hz = 1e6 / median(diff(t_us)).
    gyro_unit: unit of imu[:, 1:4]; "deg/s" is the run_DEIO2 convention. rad/s parameters are
        converted to it. Noise draws are made in SI units, so the physical realization does not
        depend on gyro_unit.
    Returns (imu_out, evidence).
    """
    dt_stats = _check_imu(imu)
    _check_seed_inputs(sequence_id, stress_seed)
    if gyro_unit not in GYRO_UNIT_FROM_RAD_S:
        raise ValueError("gyro_unit must be one of %s, got %r" % (sorted(GYRO_UNIT_FROM_RAD_S), gyro_unit))
    n = imu.shape[0]
    rate_hz = 1.0e6 / dt_stats["dt_us_median"]
    out = imu.copy()
    seeds, requested, realized = {}, {}, {
        "n_samples": n, "rate_hz_estimated": rate_hz,
        "rate_estimator": "1e6 / median(diff(t_us))"}
    realized.update(dt_stats)

    channels = (
        ("gyro", IMU_GYRO, gyro_bias_rad_s, gyro_noise_rad_s_sqrthz,
         GYRO_UNIT_FROM_RAD_S[gyro_unit], "rad/s", "rad_s", "rad/s/sqrt(Hz)"),
        ("accel", IMU_ACC, accel_bias_m_s2, accel_noise_m_s2_sqrthz,
         1.0, "m/s^2", "m_s2", "m/s^2/sqrt(Hz)"),
    )
    for name, cols, bias, density, to_array, unit, tag, density_unit in channels:
        mode, bias_vec, direction = _resolve_bias(name, bias, sequence_id, stress_seed, seeds)
        density = _as_nonneg(density, "%s noise density" % name)
        sigma_d = density * math.sqrt(rate_hz)
        requested["%s_bias" % name] = {"value": _py(np.asarray(bias, dtype=np.float64)),
                                       "unit": unit, "direction_mode": mode}
        requested["%s_noise_density" % name] = {"value": density, "unit": density_unit}

        if mode != "none" or density > 0.0:
            delta = np.zeros((n, 3)) + bias_vec                      # SI units
            if density > 0.0:
                seed = derive_seed("%s/%s_noise" % (OP_BIAS_NOISE, name),
                                   {"%s_noise_density" % name: density}, sequence_id, stress_seed)
                seeds["%s_noise" % name] = seed
                delta = delta + sigma_d * make_rng(seed).standard_normal((n, 3))
            out[:, cols] = imu[:, cols] + delta * to_array
        d_real = (out[:, cols] - imu[:, cols]) / to_array             # back to SI
        realized.update({
            "%s_bias_mode" % name: mode,
            "%s_bias_direction_unit" % name: direction,
            "%s_bias_vector_%s" % (name, tag): bias_vec,
            "%s_sigma_discrete_%s" % (name, tag): sigma_d,
            "%s_delta_mean_%s" % (name, tag): d_real.mean(axis=0),
            "%s_delta_std_%s" % (name, tag): d_real.std(axis=0),
        })
    realized["gyro_unit_in_array"] = gyro_unit
    realized["gyro_rad_s_to_array_factor"] = GYRO_UNIT_FROM_RAD_S[gyro_unit]

    ev = _evidence(OP_BIAS_NOISE, requested, realized, _imu_format(gyro_unit),
                   array_sha256(imu), array_sha256(out), sequence_id, int(stress_seed), seeds)
    return out, ev


# ----------------------------------------------------------------------------- 2. event-IMU time offset

def _edge_counts(t, a, b):
    """Where IMU timestamps t sit relative to the closed event range [a, b] (us)."""
    return {
        "n_before_event_range": int(np.count_nonzero(t < a)),
        "n_after_event_range": int(np.count_nonzero(t > b)),
        "n_inside_event_range": int(np.count_nonzero((t >= a) & (t <= b))),
        "imu_first_us": float(t[0]), "imu_last_us": float(t[-1]),
        "uncovered_head_us": max(0.0, float(t[0]) - a),
        "uncovered_tail_us": max(0.0, b - float(t[-1])),
        "covers_event_range": bool(t[0] <= a and t[-1] >= b),
    }


def apply_imu_time_offset(imu, offset_ms, event_range_us, *, sequence_id=None):
    """Clock/synchronization offset between sensors: t_imu' = t_imu + delta.

    imu: float64 [N,7] (t_us, gyro, accel); not modified. Only column 0 changes.
    offset_ms: signed delta in milliseconds; > 0 means every IMU sample is stamped later than
        it was taken (IMU stream appears delayed relative to the events).
    event_range_us: (start_us, end_us) of the event data the estimator consumes (e.g. first and
        last voxel-window timestamps). Used only for edge accounting: no sample is dropped and
        nothing is extrapolated; counts of samples outside the range are recorded.
    sequence_id: optional, recorded in evidence only (operator is deterministic, no RNG).
    Returns (imu_out, evidence).
    """
    dt_stats = _check_imu(imu)
    off = float(offset_ms)
    if not math.isfinite(off):
        raise ValueError("offset_ms must be finite, got %r" % (offset_ms,))
    a, b = _as_span(event_range_us, "event_range_us")
    delta_us = off * 1000.0
    out = imu.copy()
    if delta_us != 0.0:
        out[:, 0] = imu[:, 0] + delta_us
    shift = out[:, 0] - imu[:, 0]
    realized = {
        "n_samples": imu.shape[0],
        "offset_us": delta_us,
        "realized_shift_us_min": float(shift.min()),
        "realized_shift_us_max": float(shift.max()),
        "max_abs_float_rounding_us": float(np.max(np.abs(shift - delta_us))),
        "event_range_us": [a, b],
        "before_shift": _edge_counts(imu[:, 0], a, b),
        "after_shift": _edge_counts(out[:, 0], a, b),
        "n_dropped": 0, "n_extrapolated": 0,
    }
    realized.update(dt_stats)
    requested = {"offset": {"value": off, "unit": "ms",
                            "convention": "t_imu' = t_imu + offset"},
                 "event_range": {"value": [a, b], "unit": "us"}}
    ev = _evidence(OP_TIME_OFFSET, requested, realized, _imu_format(),
                   array_sha256(imu), array_sha256(out), sequence_id)
    return out, ev


# ----------------------------------------------------------------------------- 3. burst missing measurements

def make_burst_schedule(span_us, burst_ms, *, sequence_id, stress_seed,
                        rate_hz=None, target_fraction=None, stream="shared"):
    """Deterministic outage windows [t_start, t_end) (us) inside span_us.

    K = round(rate_hz * span_s) or K = round(target_fraction * span_us / burst_us) bursts
    (exactly one of rate_hz / target_fraction), each lasting burst_ms, placed uniformly at
    random without overlap: offsets u_(1) <= ... <= u_(K) are sorted U(0, span - K*burst)
    draws and burst i starts at span_start + u_(i) + i*burst. Realized time fraction is
    K*burst/span (exact up to the rounding of K).
    stream: "shared" (one schedule for IMU and events), or "imu"/"events" (independent
        schedules); it enters the seed.
    Returns (schedule float64 [K, 2], evidence).
    """
    a, b = _as_span(span_us, "span_us")
    burst_us = float(burst_ms) * 1000.0
    if not math.isfinite(burst_us) or burst_us <= 0:
        raise ValueError("burst_ms must be finite and > 0, got %r" % (burst_ms,))
    if (rate_hz is None) == (target_fraction is None):
        raise ValueError("give exactly one of rate_hz or target_fraction")
    if stream not in BURST_STREAMS:
        raise ValueError("stream must be one of %s, got %r" % (BURST_STREAMS, stream))
    _check_seed_inputs(sequence_id, stress_seed)
    span = b - a
    if rate_hz is not None:
        rate = _as_nonneg(rate_hz, "rate_hz")
        k = int(round(rate * span / 1.0e6))
        req_key, req = "rate", {"value": rate, "unit": "bursts/s"}
    else:
        frac = _as_nonneg(target_fraction, "target_fraction")
        if frac >= 1.0:
            raise ValueError("target_fraction must be < 1, got %r" % (target_fraction,))
        k = int(round(frac * span / burst_us))
        req_key, req = "target_fraction", {"value": frac, "unit": "fraction of span time"}
    if k * burst_us >= span:
        raise ValueError("%d bursts of %g ms would cover the whole %g s span" % (k, burst_ms, span / 1e6))

    seed_params = {"span_us": [a, b], "burst_ms": float(burst_ms), req_key: req["value"], "stream": stream}
    seed = derive_seed(OP_BURST_SCHEDULE, seed_params, sequence_id, stress_seed)
    if k == 0:
        sched = np.empty((0, 2), dtype=np.float64)
    else:
        u = np.sort(make_rng(seed).random(k)) * (span - k * burst_us)
        base = a + u
        idx = np.arange(k, dtype=np.float64)
        # Floating-point addition is monotonic, so end_i <= start_{i+1} holds exactly.
        sched = np.column_stack([base + idx * burst_us,
                                 np.minimum(base + (idx + 1.0) * burst_us, b)])
    removed = _covered_us(sched, a, b)
    requested = {"span": {"value": [a, b], "unit": "us"},
                 "burst_duration": {"value": float(burst_ms), "unit": "ms"},
                 req_key: req, "stream": {"value": stream, "unit": None}}
    realized = {"n_bursts": k, "span_s": span / 1e6, "removed_us": removed,
                "realized_time_fraction": removed / span,
                "realized_rate_hz": k / (span / 1e6),
                "schedule_sha256": array_sha256(sched), "schedule_us": sched}
    ev = _evidence(OP_BURST_SCHEDULE, requested, realized, "schedule float64 [K,2]: t_start_us, t_end_us",
                   None, array_sha256(sched), sequence_id, int(stress_seed), {"schedule": seed})
    return sched, ev


def make_burst_schedules(span_us, burst_ms, *, mode, sequence_id, stress_seed,
                         rate_hz=None, target_fraction=None):
    """{"imu": (schedule, evidence), "events": (schedule, evidence)}.
    mode="shared": the same schedule for both (common outage, e.g. a bus/host stall);
    mode="independent": separately seeded schedules per stream."""
    kw = dict(sequence_id=sequence_id, stress_seed=stress_seed,
              rate_hz=rate_hz, target_fraction=target_fraction)
    if mode == "shared":
        s = make_burst_schedule(span_us, burst_ms, stream="shared", **kw)
        return {"imu": s, "events": s}
    if mode == "independent":
        return {"imu": make_burst_schedule(span_us, burst_ms, stream="imu", **kw),
                "events": make_burst_schedule(span_us, burst_ms, stream="events", **kw)}
    raise ValueError("mode must be 'shared' or 'independent', got %r" % (mode,))


def burst_keep_mask(t_us, schedule):
    """Boolean mask, True where t_us lies outside every [t_start, t_end) window.
    t_us: 1-D integer/float microseconds (any order). schedule: [K, 2] sorted, disjoint."""
    t = np.asarray(t_us)
    if t.ndim != 1 or not (np.issubdtype(t.dtype, np.integer) or np.issubdtype(t.dtype, np.floating)):
        raise ValueError("t_us must be a 1-D integer or float array, got %s %s" % (t.dtype, t.shape))
    if np.issubdtype(t.dtype, np.floating) and not np.isfinite(t).all():
        raise ValueError("t_us has non-finite values")
    s = _as_schedule(schedule)
    if len(s) == 0 or t.size == 0:
        return np.ones(t.shape, dtype=bool)
    i = np.searchsorted(s[:, 0], t, side="right") - 1      # last window with start <= t
    inside = (i >= 0) & (t < s[np.maximum(i, 0), 1])
    return ~inside


def _burst_realized(t_in, keep, sched):
    n_in = int(t_in.shape[0])
    n_out = int(np.count_nonzero(keep))
    r = {"n_in": n_in, "n_removed": n_in - n_out, "n_out": n_out,
         "fraction_removed": (n_in - n_out) / n_in if n_in else 0.0,
         "n_bursts": int(len(sched)), "schedule_sha256": array_sha256(sched)}
    r.update({"data_span_us": None, "time_fraction_removed_in_data_span": None,
              "n_bursts_touching_data_span": 0})
    if n_in:
        a, b = float(t_in[0]), float(t_in[-1])
        r["data_span_us"] = [a, b]
        if b > a:
            r["time_fraction_removed_in_data_span"] = _covered_us(sched, a, b) / (b - a)
        r["n_bursts_touching_data_span"] = int(np.count_nonzero((sched[:, 1] > a) & (sched[:, 0] <= b)))
    return r


def apply_imu_burst_dropout(imu, schedule, *, sequence_id=None):
    """Remove IMU rows whose timestamp falls in a schedule window (no interpolation/fill).
    imu: float64 [N,7]; not modified. schedule: [K,2] us (from make_burst_schedule).
    Returns (imu_out, evidence)."""
    _check_imu(imu)
    sched = _as_schedule(schedule)
    keep = burst_keep_mask(imu[:, 0], sched)
    out = imu[keep]
    realized = _burst_realized(imu[:, 0], keep, sched)
    realized["max_gap_us_in"] = float(np.diff(imu[:, 0]).max())
    realized["max_gap_us_out"] = float(np.diff(out[:, 0]).max()) if out.shape[0] >= 2 else None
    requested = {"schedule": {"value": "sha256:" + realized["schedule_sha256"], "unit": "us",
                              "n_bursts": realized["n_bursts"]}}
    ev = _evidence(OP_IMU_BURST, requested, realized, _imu_format(),
                   array_sha256(imu), array_sha256(out), sequence_id)
    return out, ev


def apply_event_burst_dropout(ev, schedule, *, sequence_id=None):
    """Remove events whose t falls in a schedule window. ev: dict t (us), x, y, p (any batch,
    e.g. one voxel window; pass the sequence-level schedule). Other keys pass through by
    reference. Returns (ev_out, evidence)."""
    _check_events(ev)
    sched = _as_schedule(schedule)
    keep = burst_keep_mask(ev["t"], sched)
    out = dict(ev)
    for k in EVENT_KEYS:
        out[k] = ev[k][keep]
    realized = _burst_realized(ev["t"], keep, sched)
    requested = {"schedule": {"value": "sha256:" + realized["schedule_sha256"], "unit": "us",
                              "n_bursts": realized["n_bursts"]}}
    evd = _evidence(OP_EVENT_BURST, requested, realized, EVENT_FORMAT,
                    events_sha256(ev), events_sha256(out), sequence_id)
    return out, evd


# ----------------------------------------------------------------------------- event noise

def _as_sensor_hw(sensor_hw):
    try:
        h, w = sensor_hw
    except (TypeError, ValueError):
        raise ValueError("sensor_hw must be (height, width), got %r" % (sensor_hw,))
    for v, name in ((h, "height"), (w, "width")):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) or int(v) <= 0:
            raise ValueError("sensor %s must be a positive int, got %r" % (name, v))
    return int(h), int(w)


def _as_fraction(v, name):
    v = _as_nonneg(v, name)
    if v > 1.0:
        raise ValueError("%s must be in [0, 1], got %r" % (name, v))
    return v


def hot_pixel_set(sensor_hw, hot_pixel_fraction, *, sequence_id, stress_seed):
    """Fixed hot-pixel set for a sequence: sorted int64 linear indices y*W + x.

    The set depends only on (sensor size, fraction, sequence_id, stress_seed), so every
    window and every repeated run of a condition sees the same hot pixels.
    Returns (indices, evidence-like dict)."""
    h, w = _as_sensor_hw(sensor_hw)
    frac = _as_fraction(hot_pixel_fraction, "hot_pixel_fraction")
    n = int(round(frac * h * w))
    params = {"sensor_hw": [h, w], "hot_pixel_fraction": frac}
    if n == 0:
        _check_seed_inputs(sequence_id, stress_seed)
        idx = np.empty(0, dtype=np.int64)
        seeds = {}
    else:
        seed = derive_seed(OP_HOT_PIXELS, params, sequence_id, stress_seed)
        idx = np.sort(make_rng(seed).choice(h * w, size=n, replace=False)).astype(np.int64)
        seeds = {OP_HOT_PIXELS: seed}
    info = _py({"operator": OP_HOT_PIXELS, "module_version": SENSOR_STRESS_VERSION,
                "sequence_id": sequence_id, "stress_seed": stress_seed, "derived_seeds": seeds,
                "requested": {"hot_pixel_fraction": {"value": frac, "unit": "1"},
                              "sensor_hw": {"value": [h, w], "unit": "px"}},
                "realized": {"n_hot_pixels": n, "hot_pixels_sha256": array_sha256(idx)}})
    return idx, info


def _polarity_values(p, polarity_values):
    if polarity_values is not None:
        vals = tuple(int(v) for v in polarity_values)
        if len(vals) != 2 or vals[0] == vals[1]:
            raise ValueError("polarity_values must be two distinct ints (negative, positive), got %r"
                             % (polarity_values,))
        return vals, "given"
    if p.shape[0] == 0:
        raise ValueError("cannot infer the polarity encoding from an empty batch; pass polarity_values")
    if np.any(p < 0):
        return (-1, 1), "inferred"
    return (0, 1), "inferred"


def apply_event_background_noise(ev, window_us, sensor_hw, *, sequence_id, stress_seed,
                                 ba_rate_hz_per_pixel=0.0, hot_pixel_fraction=0.0,
                                 hot_pixel_rate_hz=0.0, polarity_values=None):
    """Add synthetic noise events inside the half-open window [a, b) (microseconds).

    Two components (both off by default):
      background activity  homogeneous Poisson, `ba_rate_hz_per_pixel` per pixel: the count is
                           Poisson(rate * H * W * T); each event has a uniform pixel, a uniform
                           time in [a, b) and a fair-coin polarity.
      hot pixels           the fixed per-sequence set from hot_pixel_set(); each hot pixel emits
                           Poisson(hot_pixel_rate_hz * T) events, uniform in time, fair-coin
                           polarity.
    The window RNG is seeded by (parameters, sensor size, window, sequence_id, stress_seed), so the
    same window always gets the same noise. Call it on DISJOINT windows that tile the stream (for
    example [t_prev, t_cur) of each voxel window); overlapping calls would add noise twice.
    Draw order (part of the semantics): BA count, BA pixels, BA times, BA polarities, hot counts,
    hot times, hot polarities.
    Integer `t` gets integer noise times drawn from [ceil(a), ceil(b)); float `t` gets
    uniform floats in [a, b). x/y/p keep their input dtypes. The merge is a stable sort on t with
    the original events first, so their relative order is unchanged and, at equal timestamps,
    original events precede noise. Other keys pass through by reference.
    Returns (ev_out, evidence).
    """
    n_in = _check_events(ev)
    a, b = _as_span(window_us, "window_us")
    h, w = _as_sensor_hw(sensor_hw)
    ba = _as_nonneg(ba_rate_hz_per_pixel, "ba_rate_hz_per_pixel")
    frac = _as_fraction(hot_pixel_fraction, "hot_pixel_fraction")
    hot_rate = _as_nonneg(hot_pixel_rate_hz, "hot_pixel_rate_hz")
    _check_seed_inputs(sequence_id, stress_seed)
    t_in = ev["t"]
    integer_t = np.issubdtype(t_in.dtype, np.integer)
    if integer_t:
        lo, hi = int(math.ceil(a)), int(math.ceil(b))
        span_s = max(hi - lo, 0) / 1e6
    else:
        span_s = (b - a) / 1e6
    params = {"ba_rate_hz_per_pixel": ba, "hot_pixel_fraction": frac, "hot_pixel_rate_hz": hot_rate,
              "sensor_hw": [h, w], "window_us": [a, b]}
    requested = {"ba_rate_hz_per_pixel": {"value": ba, "unit": "Hz/pixel"},
                 "hot_pixel_fraction": {"value": frac, "unit": "1"},
                 "hot_pixel_rate_hz": {"value": hot_rate, "unit": "Hz/pixel"},
                 "sensor_hw": {"value": [h, w], "unit": "px"},
                 "window_us": {"value": [a, b], "unit": "us"}}
    hot_idx, hot_info = hot_pixel_set((h, w), frac, sequence_id=sequence_id, stress_seed=stress_seed)
    active = span_s > 0 and (ba > 0 or (hot_idx.size > 0 and hot_rate > 0))
    seeds = dict(hot_info["derived_seeds"])
    nt, nx, ny, npol = [], [], [], []
    n_ba = n_hot = 0
    pol_vals, pol_src = (None, "unused")
    if active:
        pol_vals, pol_src = _polarity_values(ev["p"], polarity_values)
        seed = derive_seed(OP_EVENT_NOISE, params, sequence_id, stress_seed)
        seeds[OP_EVENT_NOISE] = seed
        rng = make_rng(seed)

        def times(k):
            if integer_t:
                return rng.integers(lo, hi, size=k, dtype=np.int64)
            return rng.uniform(a, b, size=k)

        n_ba = int(rng.poisson(ba * h * w * span_s)) if ba > 0 else 0
        pix = rng.integers(0, h * w, size=n_ba, dtype=np.int64)
        nt.append(times(n_ba))
        nx.append(pix % w)
        ny.append(pix // w)
        npol.append(rng.integers(0, 2, size=n_ba))
        if hot_idx.size and hot_rate > 0:
            counts = rng.poisson(hot_rate * span_s, size=hot_idx.size)
            n_hot = int(counts.sum())
            pix = np.repeat(hot_idx, counts)
            nt.append(times(n_hot))
            nx.append(pix % w)
            ny.append(pix // w)
            npol.append(rng.integers(0, 2, size=n_hot))
    out = dict(ev)
    if n_ba + n_hot == 0:
        for k in EVENT_KEYS:
            out[k] = ev[k].copy()
    else:
        noise = {"t": np.concatenate(nt).astype(t_in.dtype), "x": np.concatenate(nx).astype(ev["x"].dtype),
                 "y": np.concatenate(ny).astype(ev["y"].dtype),
                 "p": np.where(np.concatenate(npol) == 1, pol_vals[1], pol_vals[0]).astype(ev["p"].dtype)}
        order = np.argsort(np.concatenate([t_in, noise["t"]]), kind="stable")
        for k in EVENT_KEYS:
            out[k] = np.concatenate([ev[k], noise[k]])[order]
    outside = int(np.count_nonzero((t_in < a) | (t_in >= b))) if n_in else 0
    realized = {"n_in": n_in, "n_background": n_ba, "n_hot": n_hot, "n_out": n_in + n_ba + n_hot,
                "window_s": span_s, "expected_background": ba * h * w * span_s,
                "expected_hot": hot_rate * span_s * int(hot_idx.size),
                "n_hot_pixels": int(hot_idx.size), "hot_pixels_sha256": hot_info["realized"]["hot_pixels_sha256"],
                "polarity_values": list(pol_vals) if pol_vals else None, "polarity_source": pol_src,
                "integer_timestamps": bool(integer_t), "n_in_outside_window": outside}
    evd = _evidence(OP_EVENT_NOISE, requested, realized, EVENT_FORMAT,
                    events_sha256(ev), events_sha256(out), sequence_id, stress_seed, seeds)
    return out, evd
