"""
RelEIO stress operators --- event-channel degradation for the reliability protocol.

Two orthogonal, controlled degradation axes applied to the raw event stream
*before* any event representation (voxel grid / time surface / event stack) is
built, so the same operators apply uniformly across the evaluated systems:

  - Event density (keep): randomly retain a `keep` fraction of events. The RNG is
    seeded by the per-window start timestamp `t0_us`, NOT by the evaluation seed.
    Identical arrays, window boundaries and seeds produce the same retained
    events. Each adapter must establish those conditions explicitly.
  - Timestamp quantization (tq): snap timestamps to a `tq_us` grid,
    t = round(t / tq) * tq. Fully deterministic, no RNG.

Order: thinning first, then quantization. The historical window summary is a
diagnostic aggregate; equality does not establish event-byte identity.
"""
import numpy as np

_SEED_MOD = 2 ** 31


def thin_events(t, x, y, p, keep, t0_us):
    """Event-density degradation: randomly keep a `keep` fraction.

    Seeded by the window start timestamp `t0_us` so the kept subset is identical
    across evaluation seeds when the input window is identical. keep=1 is identity.

    Args:
        t, x, y, p: equal-length numpy arrays of one event window.
        keep: retention ratio in [0, 1].
        t0_us: window start timestamp (us), used as the deterministic seed.
    Returns:
        (t, x, y, p) thinned; length-0 arrays if all dropped (caller checks).
    """
    if not np.isfinite(keep) or not 0 <= keep <= 1:
        raise ValueError("keep must be finite and in [0, 1]")
    if any(np.ndim(a) != 1 for a in (t, x, y, p)):
        raise ValueError("Event arrays must be one-dimensional")
    if len({len(a) for a in (t, x, y, p)}) != 1:
        raise ValueError("Event arrays must have equal lengths")
    if keep == 1.0:
        return t, x, y, p
    n = len(t)
    if n == 0:
        return t, x, y, p
    rng = np.random.RandomState(int(t0_us) % _SEED_MOD)
    sel = rng.rand(n) < keep
    return t[sel], x[sel], y[sel], p[sel]


def quantize_timestamps(t, tq_us):
    """Timestamp-quantization degradation: t = round(t / tq) * tq.

    Fully deterministic (no RNG). tq_us=0 returns the input unchanged.
    Does not touch x/y/p.
    """
    if not np.isfinite(tq_us) or tq_us < 0:
        raise ValueError("tq_us must be finite and nonnegative")
    if np.ndim(t) != 1 or not np.all(np.isfinite(t)):
        raise ValueError("Timestamps must be a finite one-dimensional array")
    if tq_us == 0 or len(t) == 0:
        return t
    _t = np.asarray(t, dtype=np.float64)
    return np.round(_t / tq_us) * tq_us


def window_checksum(t, x, y, p, t0_us):
    """Historical aggregate summary of an event window.

    Uses event count plus sums/extrema of t/x/y/p (byte-order independent).
    Different event arrays can share these sums/extrema. Keep this existing
    diagnostic for historical comparison; use direct arrays to audit identity.
    """
    n = len(t)
    if n == 0:
        return f"n0_t0{int(t0_us)}"
    return (f"n{n}_tsum{np.asarray(t, dtype=np.float64).sum():.3f}"
            f"_xsum{int(np.asarray(x).sum())}_ysum{int(np.asarray(y).sum())}"
            f"_psum{int(np.asarray(p).sum())}_tmax{np.asarray(t, dtype=np.float64).max():.1f}")


def apply_event_stress(t, x, y, p, keep, tq_us, t0_us):
    """Apply (thinning -> quantization) to one event window.

    Returns the degraded (t, x, y, p). This is the single entry point a pipeline
    calls before building its event representation. Cross-system equality also
    requires identical units, raw input, window boundaries and t0_us.
    """
    t, x, y, p = thin_events(t, x, y, p, keep, t0_us)
    t = quantize_timestamps(t, tq_us)
    return t, x, y, p
