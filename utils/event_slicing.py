"""Selecting the events of a time window [t0, t1) (numpy only).

The original DAVIS loader (load_utils.mvsec_converted_iterator) used the boolean mask
(ev_t >= t0) & (ev_t < t1) for every frame: a full scan of all events per frame, which is
O(frames x events) and took ~9 of the ~10 minutes of an RPG-DAVIS run. When the timestamps are
non-decreasing, the events of [t0, t1) are exactly the contiguous index range
[searchsorted(t0, 'left'), searchsorted(t1, 'left')): the same events in the same order, found
in O(log N). For any other input (unordered or NaN timestamps) the original mask is kept, so
the selected events are identical in every case. script/diag/check_davis_slicing.py compares
the two on the real sequences.
"""
import numpy as np


def is_time_ordered(ev_t):
    """True if the timestamps are non-decreasing (NaN anywhere makes this False)."""
    ev_t = np.asarray(ev_t)
    return bool(ev_t.size < 2 or np.all(ev_t[1:] >= ev_t[:-1]))


def window_index(ev_t, t0, t1, ordered):
    """Index of the events with t0 <= t < t1: a slice when `ordered`, else the original boolean mask."""
    if ordered:
        return slice(int(np.searchsorted(ev_t, t0, side="left")), int(np.searchsorted(ev_t, t1, side="left")))
    return (ev_t >= t0) & (ev_t < t1)


def take(arrays, idx):
    """Apply window_index's result to several arrays; always returns new arrays (as boolean indexing does)."""
    if isinstance(idx, slice):
        return [a[idx].copy() for a in arrays]
    return [a[idx] for a in arrays]
