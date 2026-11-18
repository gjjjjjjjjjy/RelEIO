#!/usr/bin/env python3
"""Convert declared HDF5 event units/layout to E2VID text without unit guessing."""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress_operators.event_stress import thin_events

SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def convert_events(t, x, y, p, *, time_unit, time_origin, keep, window_us):
    arrays = tuple(np.asarray(a) for a in (t, x, y, p))
    if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("Event arrays must be one-dimensional and equally sized")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Events must contain finite values")
    t, x, y, p = arrays
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Timestamps must be nondecreasing; inspect the source ordering")
    if not np.all(np.isin(p, [-1, 0, 1])):
        raise ValueError("Polarity must use {-1,1} or {0,1}")
    if not np.isfinite(keep) or not 0 <= keep <= 1:
        raise ValueError("keep must be in [0,1]")
    if time_origin not in ("preserve", "zero"):
        raise ValueError("time_origin must be preserve or zero")
    if keep < 1 and (window_us is None or window_us <= 0):
        raise ValueError("Thinning requires a positive --window-us")
    if len(t) == 0:
        return np.array([], dtype=float), x, y, (p > 0).astype(int)
    # Subtract before floating conversion for integer timestamps with a large origin.
    relative = t-t[0] if time_origin == "zero" else t
    seconds = relative.astype(np.float64)*SECONDS_PER_UNIT[time_unit]
    polarity = (p > 0).astype(int)
    if keep == 1:
        return seconds, x, y, polarity
    microseconds = seconds*1e6
    windows = np.floor(microseconds/window_us).astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(windows))+1))
    ends = np.concatenate((starts[1:], [len(t)]))
    parts = []
    for start, end in zip(starts, ends):
        # Pass original seconds through the selector; integer conversion is only
        # for the window seed, never for retained event timestamps.
        parts.append(thin_events(seconds[start:end], x[start:end], y[start:end],
                                 polarity[start:end], keep, int(windows[start])*window_us))
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events-group", required=True, choices=("/", "events"))
    parser.add_argument("--time-unit", required=True, choices=tuple(SECONDS_PER_UNIT))
    parser.add_argument("--time-origin", required=True, choices=("preserve", "zero"),
                        help="zero requires the corresponding shift of frame/IMU/ground-truth timestamps")
    parser.add_argument("--window-us", type=int,
                        help="fixed thinning windows; match this partition explicitly across adapters")
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--keep", type=float, default=1.0)
    args = parser.parse_args()
    if args.W <= 0 or args.H <= 0:
        parser.error("W and H must be positive")
    with h5py.File(args.h5, "r") as stream:
        group = stream[args.events_group]
        t, x, y, p = (np.asarray(group[key]) for key in ("t", "x", "y", "p"))
    if np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
        raise ValueError("Pixel coordinates must be integers")
    if np.any(x < 0) or np.any(x >= args.W) or np.any(y < 0) or np.any(y >= args.H):
        raise ValueError("Pixel coordinates exceed the declared resolution")
    seconds, x, y, polarity = convert_events(
        t, x, y, p, time_unit=args.time_unit, time_origin=args.time_origin,
        keep=args.keep, window_us=args.window_us)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        stream.write(f"{args.W} {args.H}\n")
        values = np.column_stack((seconds, x, y, polarity))
        np.savetxt(stream, values, fmt=["%.17g", "%d", "%d", "%d"])
    print(f"{len(t)} -> {len(seconds)} events; declared {args.time_unit}, origin={args.time_origin}")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
