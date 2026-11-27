"""Order-free sensing perturbations for the pipeline comparison.

Every reader of a perturbed recording (DEIO through our data-loading layer, DEIO through its own DAVIS240C pipeline,
SuperEIO through a rosbag) must receive the same perturbed sensor data. The functions here therefore decide from an
event's own content -- its time in whole microseconds on the time base of data/RPGDAVIS_pp (the text release of the
RPG event-camera dataset), its pixel, its polarity -- and a fixed seed, never from the order in which a reader visits
the events or how they are packed into messages or windows. The rosbag writer converts ROS times to that time base
before calling them, so both files get the same decisions. Pure numpy; no state.

  keep_mask(t_us, x, y, p, keep, seq)   thinning: keep an event iff a 64-bit hash of (t_us, x, y, p, seed) maps below
                                        `keep`; the seed depends only on the sequence name and `keep`
  quantize_us(t_us, tq_us)              timestamp quantization on the absolute grid round(t / tq) * tq of that time base
                                        (the grid of utils/event_stress.quantize_timestamps)
  per_second_digest(t_us, x, y, p)      event count and SHA-256 of the sorted (t, x, y, p) tuples in each whole second,
                                        the fingerprint both writers compare
  noise_events(seq, rate_hz, t0, t1)    background activity: a Poisson number of events at rate_hz per pixel over
                                        [t0, t1] (the source recording's first and last event), uniform in time and
                                        pixel, polarity 0/1 with equal probability; drawn once from a fixed seed, so
                                        both writers insert the same events
  IMU clock offset                      imu:<us> shifts every IMU timestamp by <us> microseconds (events, frames and
                                        ground truth keep their times); applied by the writers to imu_data.txt and to
                                        the /dvs/imu header stamps
"""
import hashlib

import numpy as np

VERSION = "r5-ops-2"
H, W = 180, 240     # DAVIS240C

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_INV53 = 1.0 / float(1 << 53)


def seed64(*parts):
    """A 64-bit seed from any printable parts (sequence name, operator, level)."""
    return np.uint64(int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little"))


def _mix64(z):
    """splitmix64 finalizer on a uint64 array (wraps modulo 2**64)."""
    z = np.array(z, dtype=np.uint64, copy=True)
    with np.errstate(over="ignore"):
        z ^= z >> _S30
        z *= _M1
        z ^= z >> _S27
        z *= _M2
        z ^= z >> _S31
    return z


def as_event_arrays(t_us, x, y, p):
    """Integer copies: t in whole microseconds (int64), x and y (int64), polarity 0/1 (int64)."""
    t = np.rint(np.asarray(t_us, dtype=np.float64)).astype(np.int64)
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    p = (np.asarray(p).astype(np.int64) > 0).astype(np.int64)
    if not (len(t) == len(x) == len(y) == len(p)):
        raise ValueError("event arrays differ in length")
    return t, x, y, p


def event_uniform(t_us, x, y, p, seed):
    """A number in [0, 1) per event that depends only on (t_us, x, y, p) and the seed."""
    t, x, y, p = as_event_arrays(t_us, x, y, p)
    pix = (x.astype(np.uint64) << np.uint64(20)) | (y.astype(np.uint64) << np.uint64(1)) | p.astype(np.uint64)
    with np.errstate(over="ignore"):
        k = _mix64(t.astype(np.uint64) * _GOLD ^ np.uint64(seed)) ^ _mix64(pix + np.uint64(seed))
    return (_mix64(k) >> _S11).astype(np.float64) * _INV53


def keep_mask(t_us, x, y, p, keep, seq):
    """Boolean mask of the events kept at density `keep` for sequence `seq` (independent Bernoulli per event)."""
    keep = float(keep)
    if not 0.0 < keep <= 1.0:
        raise ValueError("keep must be in (0, 1]")
    if keep >= 1.0:
        return np.ones(len(np.asarray(t_us)), dtype=bool)
    return event_uniform(t_us, x, y, p, seed64("r5", "keep", seq, keep)) < keep


def quantize_us(t_us, tq_us):
    """Event times rounded to the absolute grid of multiples of tq_us (whole microseconds, int64)."""
    t = np.rint(np.asarray(t_us, dtype=np.float64))
    if tq_us <= 0:
        return t.astype(np.int64)
    return (np.round(t / float(tq_us)) * float(tq_us)).astype(np.int64)


def per_second_digest(t_us, x, y, p):
    """{second: [count, sha256 of the lexicographically sorted (t, x, y, p) tuples in that second]}.

    Seconds are floor(t / 1e6) on the common time base. Two event sets with the same digest hold the same events."""
    t, x, y, p = as_event_arrays(t_us, x, y, p)
    out = {}
    if len(t) == 0:
        return out
    sec = np.floor_divide(t, 1_000_000)
    order = np.argsort(sec, kind="stable")
    sec_sorted = sec[order]
    starts = np.flatnonzero(np.r_[True, np.diff(sec_sorted) != 0])
    ends = np.r_[starts[1:], len(order)]
    for a, b in zip(starts, ends):
        idx = order[a:b]
        tt, xx, yy, pp = t[idx], x[idx], y[idx], p[idx]
        o = np.lexsort((pp, yy, xx, tt))
        h = hashlib.sha256()
        for arr, dt in ((tt[o], "<i8"), (xx[o], "<i4"), (yy[o], "<i4"), (pp[o], "<i1")):
            h.update(np.ascontiguousarray(arr.astype(dt)).tobytes())
        out[int(sec_sorted[a])] = [int(b - a), h.hexdigest()]
    return out


def digest_summary(digest):
    """One SHA-256 over a per-second digest, and the total event count."""
    h = hashlib.sha256()
    total = 0
    for s in sorted(digest):
        n, d = digest[s]
        total += n
        h.update(("%d:%d:%s;" % (s, n, d)).encode())
    return {"events": total, "seconds": len(digest), "sha256": h.hexdigest()}


def noise_events(seq, rate_hz, t0_us, t1_us, h=H, w=W):
    """Background-activity events on [t0_us, t1_us], sorted by (t, x, y, p); identical for the same arguments."""
    rng = np.random.Generator(np.random.PCG64(int(seed64("r5", "noise", seq, float(rate_hz), int(t0_us), int(t1_us)))))
    n = int(rng.poisson(float(rate_hz) * h * w * (int(t1_us) - int(t0_us)) / 1e6))
    t = rng.integers(int(t0_us), int(t1_us) + 1, n, dtype=np.int64)
    x = rng.integers(0, w, n, dtype=np.int64)
    y = rng.integers(0, h, n, dtype=np.int64)
    p = rng.integers(0, 2, n, dtype=np.int64)
    o = np.lexsort((p, y, x, t))
    return t[o], x[o], y[o], p[o]


def imu_offset_us(cond):
    """The IMU time shift of a condition (0 for event-only conditions)."""
    return int(cond[4:]) if cond.startswith("imu:") else 0


def apply_condition(t_us, x, y, p, cond, seq):
    """Apply one condition string to an event stream; returns (t_us, x, y, p) as int64 arrays and a short record.

    cond: "clean" | "keep:<rho>" | "tq:<microseconds>" | "noise:<Hz per pixel>" | "imu:<microseconds>" (events
    unchanged; the writers shift the IMU). The output is sorted by time (stable), so a quantized stream stays
    time-ordered. Noise is drawn over the input's first and last event time."""
    t, x, y, p = as_event_arrays(t_us, x, y, p)
    rec = {"cond": cond, "version": VERSION, "n_in": int(len(t))}
    if cond == "clean":
        pass
    elif cond.startswith("keep:"):
        m = keep_mask(t, x, y, p, float(cond[5:]), seq)
        t, x, y, p = t[m], x[m], y[m], p[m]
    elif cond.startswith("tq:"):
        tq = int(cond[3:])
        tn = quantize_us(t, tq)
        rec["tq_changed"] = int(np.count_nonzero(tn != t))
        t = tn
    elif cond.startswith("noise:"):
        nt, nx, ny, np_ = noise_events(seq, float(cond[6:]), int(t.min()), int(t.max()))
        rec["noise_events"] = int(len(nt))
        t, x, y, p = np.r_[t, nt], np.r_[x, nx], np.r_[y, ny], np.r_[p, np_]
    elif cond.startswith("imu:"):
        rec["imu_offset_us"] = imu_offset_us(cond)
    else:
        raise ValueError("unknown condition %r" % cond)
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    rec["n_out"] = int(len(t))
    return t, x, y, p, rec
