#!/usr/bin/env python3
"""SuperEIO side of the pipeline comparison: the DAVIS240C rosbag with the same perturbed events DEIO reads.

The DEIO readers take data/R5_ours/<seq>__<tag>/events_left.h5, written by script/diag/r5_make_inputs.py from the text
release of the recording. This script applies the same order-free operators (utils/r5_ops.py) to the /dvs/events
messages of the rosbag of the same recording and proves, second by second, that the bag then holds exactly the events
of that h5 file:
  1. time base: the bag carries ROS times, the h5 file microseconds from the text release. The constant offset is
     taken from the first events and accepted only if the leading events agree exactly as sorted (t, x, y, p)
     tuples, compared over complete timestamp groups.
  2. IMU: /dvs/imu must exist; with that same offset (no separate fit) every message must match a row of the
     copy's imu_data.txt in time (1 us) and in the six values (1e-4).
  3. identity of the clean recording: the per-second digest (count and SHA-256 of the sorted events) of every bag
     event, on the h5 time base, must equal the source digest stored in data/R5_ours/<seq>__clean/r5_evidence.json.
  4. for a perturbed condition: keep or quantize each event (times converted to the h5 base for the decision and
     back); the per-second digest of the written bag must equal the output digest of
     data/R5_ours/<seq>__<tag>/r5_evidence.json, and the written bag, read back, must match the source message by
     message: topic, receive time, type, untouched topics byte for byte, event headers, and the expected events.
Time units: the DAVIS driver stamps an event as its reset time (nanoseconds) plus the device's whole-microsecond clock,
so the events of a recording share one sub-microsecond remainder, which is not zero. The h5 times are the text
release's seconds read as float64, times 1e6 (script/pp_data/pp_rpg_davis.py), rounded to whole microseconds
(utils/r5_ops.py). Two cases, chosen from the h5 time base and recorded as "time_mapping":
  nearest       relative text times (poster_6dof: from 0): exact microseconds; a bag time maps to its nearest
                microsecond. A rewritten timestamp keeps its event's remainder; an added noise event takes the
                remainder of the recording's first event.
  text_seconds  absolute text times (boxes_6dof: about 1.47e9 s): float64 resolves them only to 2^-22 s (0.24 us), so
                the h5 microsecond differs from the nearest one for some events. The bag time is mapped through the
                same chain, reproduced exactly with integers: the correctly rounded double of ns/1e9 (as the text
                parser reads "secs.nsecs"), times 1e6 in float64, rounded. Rewritten and added events get whole
                microseconds (remainder 0), which this chain maps to themselves.
Either way the per-second identity check against the h5 copy proves the mapping event by event; the remainders found
are recorded.
Every message keeps its topic, receive time, header and packet; only the events inside an EventArray change (dropped,
or new timestamp). Other topics (/dvs/imu, images, ground truth) are copied byte for byte. Messages are parsed as raw
bytes (dvs_msgs/Event = uint16 x, uint16 y, uint32 secs, uint32 nsecs, uint8 polarity: 13 bytes), which keeps the
rewrite fast for 10^8 events.

  python3 script/supereio/make_r5_bag.py boxes_6dof --cond clean       # identity check only, writes no bag
  python3 script/supereio/make_r5_bag.py boxes_6dof --cond keep:0.1    # data/rosbags/rpgdavis/<seq>__r5_keep0.1.bag
  python3 script/supereio/make_r5_bag.py boxes_6dof --cond tq:10000    # ...__r5_tq10000us.bag
  python3 script/supereio/make_r5_bag.py boxes_6dof --cond noise:5     # noise events added to the packets they fall in
  python3 script/supereio/make_r5_bag.py boxes_6dof --cond imu:15000   # IMU header stamps +15 ms
Needs python3-rosbag (setup_env.sh) and the R5 folders on the shared disk. Stops (exit 1, no bag kept) on any mismatch;
the evidence <seq>__r5_<tag>.json is written only when every check passed, a failure goes to <...>.failed.json, so a
repeated or failed attempt never replaces a passed record.
"""
import argparse
import hashlib
import itertools
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utils import r5_ops as R   # noqa: E402

EV_TOPIC = "/dvs/events"
IMU_TOPIC = "/dvs/imu"
EVENT = np.dtype([("x", "<u2"), ("y", "<u2"), ("secs", "<u4"), ("nsecs", "<u4"), ("pol", "u1")])
assert EVENT.itemsize == 13


def tag_of(cond):
    if cond == "clean":
        return "clean"
    if cond.startswith("keep:"):
        return "keep" + cond[5:]
    if cond.startswith("tq:"):
        return "tq%sus" % cond[3:]
    if cond.startswith("noise:"):
        return "noise%shz" % cond[6:]
    if cond.startswith("imu:"):
        return "imu%sus" % cond[4:]
    raise ValueError(cond)


# ---- raw dvs_msgs/EventArray ----------------------------------------------------------------------------------
def parse_event_array(buf):
    """(prefix bytes up to the event count, structured array of events). prefix = header, height, width."""
    o = 0
    o += 4 + 8                                     # header.seq, header.stamp
    (n_frame,) = struct.unpack_from("<I", buf, o)
    o += 4 + n_frame                               # header.frame_id
    o += 8                                         # height, width
    prefix = bytes(buf[:o])
    (n,) = struct.unpack_from("<I", buf, o)
    o += 4
    ev = np.frombuffer(buf, dtype=EVENT, count=n, offset=o)
    if o + n * EVENT.itemsize != len(buf):
        raise ValueError("EventArray length mismatch")
    return prefix, ev


def serialize_event_array(prefix, ev):
    return prefix + struct.pack("<I", len(ev)) + np.ascontiguousarray(ev, dtype=EVENT).tobytes()


def _ns(ev):
    return ev["secs"].astype(np.int64) * 1_000_000_000 + ev["nsecs"].astype(np.int64)


def ev_us(ev):
    """Event times in microseconds (int64, to the nearest microsecond) on the bag's ROS time base."""
    return (_ns(ev) + 500) // 1000


TEXT_S_MIN_NS, TEXT_S_MAX_NS = (1 << 30) * 1_000_000_000, (1 << 31) * 1_000_000_000   # seconds in [2^30, 2^31)


def text_seconds_us(ns):
    """Microseconds as pp_rpg_davis.py + r5_ops.py compute them from the text "secs.nsecs" of a time ns (int64): the
    correctly rounded double of ns/1e9 (= m * 2^-22 for seconds in [2^30, 2^31), m = round(ns * 8192 / 1953125),
    never a tie since 1953125 is odd), times 1e6 in float64, rounded to the nearest integer (np.rint)."""
    ns = np.asarray(ns, dtype=np.int64)
    if len(ns) and (ns.min() < TEXT_S_MIN_NS or ns.max() >= TEXT_S_MAX_NS):
        raise ValueError("text_seconds mapping needs times in [2^30, 2^31) s")
    q, r = np.divmod(ns, 1953125)
    b, rem = np.divmod(r * 8192, 1953125)
    m = q * 8192 + b + (2 * rem > 1953125)
    return np.rint(m.astype(np.float64) * 2.0 ** -22 * 1e6).astype(np.int64)


def bag_us(ev, mode="nearest"):
    """Event times in microseconds on the bag's ROS base, as the h5 copy holds them (see the module notes)."""
    return ev_us(ev) if mode == "nearest" else text_seconds_us(_ns(ev))


def time_mapping(first_t_us):
    """nearest for relative h5 times, text_seconds for absolute ones (text seconds at or above 2^30)."""
    return "text_seconds" if first_t_us >= (1 << 30) * 1_000_000 else "nearest"


def rewrite_sub_ns(ev, mode):
    """Remainder (ns) a rewritten event gets: its own under nearest, 0 under text_seconds."""
    return sub_us(ev) if mode == "nearest" else 0


def sub_us(ev):
    """Nanoseconds of each event beyond its microsecond ev_us, in -500..499."""
    ns = _ns(ev)
    return ns - 1000 * ((ns + 500) // 1000)


def set_us(ev, t_us, sub_ns=0):
    """Events with new times: t_us microseconds plus sub_ns nanoseconds (scalar or per event)."""
    out = ev.copy()
    ns = np.asarray(t_us, dtype=np.int64) * 1000 + np.asarray(sub_ns, dtype=np.int64)
    out["secs"] = (ns // 1_000_000_000).astype(np.uint32)
    out["nsecs"] = (ns % 1_000_000_000).astype(np.uint32)
    return out


# ---- checks ---------------------------------------------------------------------------------------------------
def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def leading_events(bag, cutoff_rel_us, mode="nearest"):
    """Events of the first messages, enough to cover the bag's first `cutoff_rel_us` microseconds plus one second
    (packets can arrive slightly out of time order). Rows (t_us on the ROS base, x, y, p), time-sorted."""
    rows, first = [], None
    for _, raw, _ in bag.read_messages(topics=[EV_TOPIC], raw=True):
        _, ev = parse_event_array(raw[1])
        if len(ev) == 0:
            continue
        tb = bag_us(ev, mode)
        if first is not None and tb.min() > first + cutoff_rel_us + 1_000_000:
            break
        first = tb.min() if first is None else min(first, tb.min())
        rows.append(np.stack([tb, ev["x"].astype(np.int64), ev["y"].astype(np.int64), ev["pol"].astype(np.int64)],
                             axis=1))
    a = np.concatenate(rows) if rows else np.zeros((0, 4), np.int64)
    return a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))]


def time_offset(bag, ref, complete_groups=False, mode="nearest"):
    """Offset (us) with t_h5 = t_bag - offset, from the first events.

    `ref` holds the first events of the h5 file. Only complete timestamp groups are compared. If the writer stored
    complete groups (complete_groups=True, r5_make_inputs.py), every event up to the last timestamp of `ref` is
    compared; otherwise the group at that timestamp may be cut and is left out on both sides. Accepted only if
    the two prefixes are identical as sorted (t, x, y, p) tuples. Returns (offset, ok, n_ref, n_bag)."""
    r = np.asarray(ref, dtype=np.int64)
    r = r[np.lexsort((r[:, 3], r[:, 2], r[:, 1], r[:, 0]))]
    cutoff = int(r[-1, 0])
    full = r if complete_groups else r[r[:, 0] < cutoff]
    if len(full) == 0:
        full, complete_groups = r, True
    b = leading_events(bag, cutoff - int(r[0, 0]), mode)
    if len(b) == 0:
        return None, False, len(full), 0
    off = int(b[0, 0] - r[0, 0])
    bb = b.copy()
    bb[:, 0] -= off
    bb = bb[bb[:, 0] <= cutoff] if complete_groups else bb[bb[:, 0] < cutoff]
    return off, bool(np.array_equal(bb, full)), len(full), len(bb)


def imu_check(bag, imu_txt, off, t_tol_us=1.0, v_tol=1e-4):
    """The /dvs/imu messages against imu_data.txt (t_us, wx, wy, wz, ax, ay, az) with the offset found from the
    events (no separate fit). Requires the topic to exist and to match row for row. Returns (ok, report)."""
    t, g, acc = [], [], []
    for _, msg, _ in bag.read_messages(topics=[IMU_TOPIC]):
        t.append(msg.header.stamp.to_nsec())
        g.append((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))
        acc.append((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
    rep = {"messages": len(t)}
    if not t:
        rep["problem"] = "no %s messages" % IMU_TOPIC
        return False, rep
    txt = np.loadtxt(imu_txt)
    txt = txt.reshape(1, -1) if txt.ndim == 1 else txt
    rep["rows_txt"] = int(len(txt))
    if len(txt) != len(t):
        rep["problem"] = "count differs"
        return False, rep
    t_h5 = np.asarray(t, dtype=np.float64) / 1e3 - off
    dt = np.abs(t_h5 - txt[:, 0])
    dg = np.abs(np.asarray(g) - txt[:, 1:4])
    da = np.abs(np.asarray(acc) - txt[:, 4:7])
    rep.update({"max_time_diff_us": float(dt.max()), "max_gyro_diff": float(dg.max()),
                "max_acc_diff": float(da.max())})
    ok = rep["max_time_diff_us"] <= t_tol_us and rep["max_gyro_diff"] <= v_tol and rep["max_acc_diff"] <= v_tol
    if not ok:
        rep["problem"] = "times or values differ beyond %.1f us / %g" % (t_tol_us, v_tol)
    return ok, rep


def shift_stamp(buf, off_us):
    """A raw message whose header stamp (bytes 4..12 after header.seq) is moved by off_us; everything else kept."""
    secs, nsecs = struct.unpack_from("<II", buf, 4)
    ns = secs * 1_000_000_000 + nsecs + int(off_us) * 1000
    b = bytearray(buf)
    struct.pack_into("<II", b, 4, ns // 1_000_000_000, ns % 1_000_000_000)
    return bytes(b)


def noise_plan(bag, cond, seq, off, t0_us, t1_us, mode="nearest"):
    """For noise:<Hz>, the noise events (h5 base) and, per event message in bag order, the slice of them it receives:
    each noise event goes to the first message whose running maximum event time reaches it, so packets keep their
    receive times and every noise event is written exactly once. None for other conditions."""
    if not cond.startswith("noise:"):
        return None
    nz = R.noise_events(seq, float(cond[6:]), t0_us, t1_us)
    run_max, cm, sub0 = [], None, None
    for _, raw, _ in bag.read_messages(topics=[EV_TOPIC], raw=True):
        _, ev = parse_event_array(raw[1])
        if len(ev):
            if sub0 is None:
                sub0 = int(sub_us(ev[:1])[0]) if mode == "nearest" else 0
            m = int(bag_us(ev, mode).max() - off)
            cm = m if cm is None else max(cm, m)
        run_max.append(cm)
    bounds = [0 if c is None else int(np.searchsorted(nz[0], c, side="right")) for c in run_max]
    bounds[-1] = len(nz[0])                       # the last message takes anything left (none if t1 is the last event)
    starts = [0] + bounds[:-1]
    return {"events": nz, "slices": list(zip(starts, bounds)), "sub_ns": sub0 or 0}


def transform(ev, th, cond, seq, off, plan=None, k=None, mode="nearest"):
    """The events of event message number k after `cond` (time decisions on the h5 base th); returns
    (events, new h5 times)."""
    x, y, p = ev["x"].astype(np.int64), ev["y"].astype(np.int64), ev["pol"].astype(np.int64)
    if cond == "clean" or cond.startswith("imu:"):
        return ev, th
    if cond.startswith("keep:"):
        m = R.keep_mask(th, x, y, p, float(cond[5:]), seq)
        return ev[m], th[m]
    if cond.startswith("tq:"):
        t2 = R.quantize_us(th, int(cond[3:]))
        return set_us(ev, t2 + off, rewrite_sub_ns(ev, mode)), t2
    if cond.startswith("noise:"):
        a, b = plan["slices"][k]
        nt, nx, ny, np_ = (arr[a:b] for arr in plan["events"])
        add = np.zeros(b - a, dtype=EVENT)
        add["x"], add["y"], add["pol"] = nx, ny, np_
        add = set_us(add, nt + off, plan["sub_ns"])
        allev = np.concatenate([ev, add])
        allt = np.concatenate([th, nt])
        o = np.argsort(allt, kind="stable")
        return allev[o], allt[o]
    raise ValueError(cond)


def verify_rewrite(src_msgs, out_msgs, cond, seq, off, plan=None, mode="nearest"):
    """Message-by-message comparison of the source and the written bag: same topic, receive time, type and md5;
    untouched topics byte-identical (IMU messages with only the header stamp moved, for imu:<us>); event messages
    with the same header and exactly the transformed events. A message present on one side only is a mismatch."""
    rep = {"messages": 0, "missing_or_extra": 0, "topic_or_time": 0, "type": 0, "other_bytes": 0, "event_header": 0,
           "event_payload": 0}
    end = object()
    k = 0
    imu_off = R.imu_offset_us(cond)
    for s, o in itertools.zip_longest(src_msgs, out_msgs, fillvalue=end):
        rep["messages"] += 1
        if s is end or o is end:
            rep["missing_or_extra"] += 1
            continue
        (ts, rs, tt), (to, ro, t2) = s, o
        if ts != to or tt != t2:
            rep["topic_or_time"] += 1
            if ts == EV_TOPIC:
                k += 1
            continue
        if rs[0] != ro[0] or rs[2] != ro[2]:
            rep["type"] += 1
        if ts != EV_TOPIC:
            want = shift_stamp(rs[1], imu_off) if (ts == IMU_TOPIC and imu_off) else bytes(rs[1])
            if want != bytes(ro[1]):
                rep["other_bytes"] += 1
            continue
        pre_s, ev_s = parse_event_array(rs[1])
        pre_o, ev_o = parse_event_array(ro[1])
        if pre_s != pre_o:
            rep["event_header"] += 1
        exp, _ = transform(ev_s, bag_us(ev_s, mode) - off, cond, seq, off, plan, k, mode)
        k += 1
        if np.ascontiguousarray(exp, dtype=EVENT).tobytes() != np.ascontiguousarray(ev_o, dtype=EVENT).tobytes():
            rep["event_payload"] += 1
    ok = all(v == 0 for k_, v in rep.items() if k_ != "messages")
    return ok, rep


def digest_mismatch(a, b):
    keys = sorted(set(a) | set(b), key=int)
    return [s for s in keys if a.get(s) != b.get(s)]


def run(seq, cond, rosbag, root=ROOT):
    """All checks and, for a perturbed condition, the rewrite. Returns (ok, report)."""
    src = os.path.join(root, "data", "rosbags", "rpgdavis", seq + ".bag")
    d_clean = os.path.join(root, "data", "R5_ours", seq + "__clean")
    d_cond = os.path.join(root, "data", "R5_ours", "%s__%s" % (seq, tag_of(cond)))
    for p in (src, os.path.join(d_clean, "r5_evidence.json"), os.path.join(d_cond, "r5_evidence.json"),
              os.path.join(d_clean, "imu_data.txt")):
        if not os.path.isfile(p):
            return False, {"problem": "%s missing" % p}
    clean = json.load(open(os.path.join(d_clean, "r5_evidence.json")))
    want = json.load(open(os.path.join(d_cond, "r5_evidence.json")))
    rep = {"seq": seq, "cond": cond, "ops_version": R.VERSION, "bag": os.path.relpath(src, root)}
    # the manifests describe the h5 files the DEIO readers take; prove those files are still the ones described
    rep["h5_files"] = {}
    for d, ev in ((d_clean, clean), (d_cond, want)):
        h5 = os.path.join(d, "events_left.h5")
        got = sha256(h5) if os.path.isfile(h5) else None
        rep["h5_files"][os.path.relpath(h5, root)] = {"sha256": got, "manifest": ev["output"].get("events_sha256")}
        if got is None or got != ev["output"].get("events_sha256"):
            return False, dict(rep, problem="%s missing or differs from its manifest" % os.path.relpath(h5, root))
    out_path = None
    mode = time_mapping(int(clean["source"]["first_t_us"]))
    rep["time_mapping"] = mode
    with rosbag.Bag(src, "r") as bag:
        off, ok_off, n_ref, n_bag = time_offset(bag, clean["source"]["first_events"],
                                                clean["source"].get("first_events_complete_groups", False), mode)
        rep["time_offset"] = {"offset_us": off, "prefix_equal": ok_off, "events_ref": n_ref, "events_bag": n_bag}
        if not ok_off:
            return False, rep
        ok_imu, rep["imu"] = imu_check(bag, os.path.join(d_clean, "imu_data.txt"), off)
        if not ok_imu:
            return False, rep
        plan = noise_plan(bag, cond, seq, off, clean["source"]["first_t_us"], clean["source"]["last_t_us"], mode)
        if plan is not None:
            rep["noise_events"] = int(len(plan["events"][0]))
        imu_off = R.imu_offset_us(cond)
        out_bag = None
        if cond != "clean":
            out_path = os.path.join(root, "data", "rosbags", "rpgdavis", "%s__r5_%s.bag" % (seq, tag_of(cond)))
            if os.path.exists(out_path):
                return False, dict(rep, problem="%s exists; move it away first" % out_path)
            out_bag = rosbag.Bag(out_path + ".tmp", "w")
        S, O, n_msgs, k = ([], [], [], []), ([], [], [], []), {}, 0
        sub_hist = np.zeros(1000, dtype=np.int64)    # remainders -500..499 ns of the source events
        try:
            for topic, raw, t in bag.read_messages(raw=True):
                n_msgs[topic] = n_msgs.get(topic, 0) + 1
                if topic != EV_TOPIC:
                    if out_bag is not None:
                        if topic == IMU_TOPIC and imu_off:
                            raw = (raw[0], shift_stamp(raw[1], imu_off), raw[2], raw[3], raw[4])
                        out_bag.write(topic, raw, t, raw=True)
                    continue
                prefix, ev = parse_event_array(raw[1])
                th = bag_us(ev, mode) - off
                sub_hist += np.bincount(sub_us(ev) + 500, minlength=1000)
                for L, v in zip(S, (th, ev["x"], ev["y"], ev["pol"])):
                    L.append(np.asarray(v, dtype=np.int64))
                if out_bag is None:
                    continue
                ev2, t2 = transform(ev, th, cond, seq, off, plan, k, mode)
                k += 1
                for L, v in zip(O, (t2, ev2["x"], ev2["y"], ev2["pol"])):
                    L.append(np.asarray(v, dtype=np.int64))
                out_bag.write(topic, (raw[0], serialize_event_array(prefix, ev2), raw[2], raw[3], raw[4]), t, raw=True)
        finally:
            if out_bag is not None:
                out_bag.close()
    rep["messages"] = n_msgs
    nzb = np.nonzero(sub_hist)[0]
    top = nzb[np.argsort(-sub_hist[nzb], kind="stable")][:5]
    rep["sub_microsecond_remainder_ns"] = {"distinct": int(len(nzb)),
                                           "most_common": [[int(v) - 500, int(sub_hist[v])] for v in top]}
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)   # noqa: E731
    d_src = {str(s): v for s, v in R.per_second_digest(*[cat(L) for L in S]).items()}
    bad = digest_mismatch(d_src, clean["source"]["per_second"])
    rep["identity_clean"] = {"seconds": len(d_src), "n_mismatched": len(bad), "first_mismatched": bad[:20],
                             "events_bag": int(sum(v[0] for v in d_src.values())),
                             "events_h5": clean["source"]["digest"]["events"]}
    ok = not bad
    if out_path is not None:
        d_out = {str(s): v for s, v in R.per_second_digest(*[cat(L) for L in O]).items()}
        bad2 = digest_mismatch(d_out, want["output"]["per_second"])
        rep["identity_output"] = {"seconds": len(d_out), "n_mismatched": len(bad2), "first_mismatched": bad2[:20],
                                  "events_bag": int(sum(v[0] for v in d_out.values())),
                                  "events_h5": want["output"]["digest"]["events"]}
        with rosbag.Bag(src, "r") as sb, rosbag.Bag(out_path + ".tmp", "r") as ob:
            ok_rw, rep["rewrite"] = verify_rewrite(sb.read_messages(raw=True), ob.read_messages(raw=True), cond, seq,
                                                   off, plan, mode)
        # the IMU SuperEIO will read against the IMU file the DEIO readers take for this condition
        with rosbag.Bag(out_path + ".tmp", "r") as ob:
            ok_imu2, rep["imu_output"] = imu_check(ob, os.path.join(d_cond, "imu_data.txt"), off)
        ok = ok and not bad2 and ok_rw and ok_imu2
        if ok:
            os.replace(out_path + ".tmp", out_path)
            rep["output_bag"] = os.path.relpath(out_path, root)
            rep["output_sha256"] = sha256(out_path)
        else:
            os.remove(out_path + ".tmp")
    rep["source_sha256"] = sha256(src)
    return ok, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq")
    ap.add_argument("--cond", required=True,
                    help="clean | keep:<rho> | tq:<microseconds> | noise:<Hz per pixel> | imu:<microseconds>")
    a = ap.parse_args()
    import rosbag   # noqa: E402  (SuperEIO environment only)
    ok, rep = run(a.seq, a.cond, rosbag)
    rep["passed"] = bool(ok)
    # the evidence file is written only for a passed check; a failed or repeated attempt never replaces it
    base = os.path.join(ROOT, "data", "rosbags", "rpgdavis", "%s__r5_%s" % (a.seq, tag_of(a.cond)))
    js = base + ".json" if ok else base + ".failed.json"
    with open(js, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in rep if k not in ("messages",)}, indent=1)[:3000])
    print("[r5-bag] %s %s: %s; evidence %s" % (a.seq, a.cond, "PASSED" if ok else "STOPPED", os.path.relpath(js, ROOT)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
