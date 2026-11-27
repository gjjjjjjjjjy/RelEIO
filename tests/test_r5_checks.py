#!/usr/bin/env python3
"""Regression tests for the input checks of the pipeline comparison (numpy only; h5py and rosbag are replaced by
small in-memory stand-ins). Each counterexample that once passed or failed wrongly is kept here:
  1. DEIO's rebuilt inputs: wrong event coordinates, a missing IMU file, a missing ground-truth file must fail
  2. a bag without /dvs/imu must fail; IMU times or values off must fail; a rewritten bag with a changed untouched
     message, receive time or missing message must fail
  3. identical streams whose 2000th and 2001st events share a timestamp must pass the time-offset check
  4. an event that leaves all windows after quantization is counted as "left", not dropped from the count
Run: python3 tests/test_r5_checks.py   (or pytest)
"""
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---- stand-ins ---------------------------------------------------------------------------------------------------
class _DS:
    def __init__(self, a):
        self.a = np.asarray(a)
        self.dtype, self.shape = self.a.dtype, self.a.shape

    def __getitem__(self, k):
        return self.a[k]


class _H5File:
    def __init__(self, path, mode="r"):
        self.path, self.mode, self.d = path, mode, {}
        if mode == "r":
            with np.load(path, allow_pickle=False) as z:
                self.d = {k.replace("|", "/"): _DS(z[k]) for k in z.files}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.mode == "w":
            with open(self.path, "wb") as fh:
                np.savez(fh, **{k.replace("/", "|"): v.a for k, v in self.d.items()})

    def __getitem__(self, k):
        return self.d[k]

    def create_dataset(self, k, data):
        self.d[k] = _DS(data)


fake_h5py = types.ModuleType("h5py")
fake_h5py.File = _H5File
sys.modules["h5py"] = fake_h5py

_BAGS = {}


class _Stamp:
    def __init__(self, ns):
        self.ns = int(ns)

    def to_nsec(self):
        return self.ns


def _imu_raw(stamp_ns, k, v):
    """sensor_msgs/Imu bytes: header (seq, stamp, frame_id), orientation, covariances, angular velocity, acceleration."""
    b = struct.pack("<III", k, int(stamp_ns) // 1_000_000_000, int(stamp_ns) % 1_000_000_000)
    b += struct.pack("<I", 3) + b"imu" + struct.pack("<4d", 0, 0, 0, 1) + struct.pack("<9d", *[0.0] * 9)
    b += struct.pack("<3d", *v[0:3]) + struct.pack("<9d", *[0.0] * 9)
    b += struct.pack("<3d", *v[3:6]) + struct.pack("<9d", *[0.0] * 9)
    return b


def _imu_obj(b):
    secs, nsecs = struct.unpack_from("<II", b, 4)
    (n,) = struct.unpack_from("<I", b, 12)
    o = 16 + n + 32 + 72
    g = struct.unpack_from("<3d", b, o)
    a = struct.unpack_from("<3d", b, o + 24 + 72)
    ns = types.SimpleNamespace
    return ns(header=ns(stamp=_Stamp(secs * 1_000_000_000 + nsecs)), angular_velocity=ns(x=g[0], y=g[1], z=g[2]),
              linear_acceleration=ns(x=a[0], y=a[1], z=a[2]))


class _Bag:
    def __init__(self, path, mode="r"):
        self.path, self.mode = os.path.realpath(path), mode
        if mode == "w":
            _BAGS[self.path] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        if self.mode == "w":
            open(self.path, "ab").close()

    def write(self, topic, msg, t=None, raw=False):
        _BAGS[self.path].append((topic, msg, t, None))

    def read_messages(self, topics=None, raw=False):
        for topic, r, t, obj in _BAGS[os.path.realpath(self.path)]:
            if topics and topic not in topics:
                continue
            if not raw and topic == "/dvs/imu":
                obj = _imu_obj(bytes(r[1]))
            yield topic, (r if raw else obj), t


fake_rosbag = types.ModuleType("rosbag")
fake_rosbag.Bag = _Bag


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


from utils import r5_ops as R   # noqa: E402
CU = _load("r5_check_upstream", "script/diag/r5_check_upstream.py")
MB = _load("make_r5_bag", "script/supereio/make_r5_bag.py")
CW = _load("r5_check_windows", "script/diag/r5_check_windows.py")
RA = _load("r5_accept", "script/supereio/r5_accept.py")


# ---- synthetic recording -----------------------------------------------------------------------------------------
def _stream(n=60000, seed=0, span=4_000_000, ties_at=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(1000, span, n))
    if ties_at is not None:
        t[ties_at] = t[ties_at - 1]
        t = np.sort(t)
    return t, rng.integers(0, 240, n), rng.integers(0, 180, n), rng.integers(0, 2, n)


def _imu(span=4_000_000):
    t = np.arange(500, span, 1000).astype(float)
    v = np.random.default_rng(5).normal(size=(len(t), 6))
    return np.column_stack([t, v])


def _gt(span=4_000_000):
    t = np.arange(-2000, span, 5000).astype(float) + 0.25
    return np.column_stack([t, np.random.default_rng(6).normal(size=(len(t), 7))])


def _make_folders(tmp, seq="s", cond="keep:0.1", abs_sub=None):
    """abs_sub: h5 times absolute, as pp_rpg_davis.py makes them from a text release in absolute seconds (boxes_6dof):
    the float chain of the bag time (OFF + t) us + abs_sub ns; IMU times absolute too."""
    t, x, y, p = _stream()
    imu = _imu()
    if abs_sub is not None:
        t = MB.text_seconds_us((t + OFF) * 1000 + abs_sub)
        imu[:, 0] += OFF
    src = os.path.join(tmp, "data", "RPGDAVIS_pp", seq)
    os.makedirs(src)
    np.savetxt(os.path.join(src, "imu_data.txt"), imu)
    np.savetxt(os.path.join(src, "mocap_data.txt"), _gt())
    img = np.arange(800, 4_000_000, 40000).astype(float)
    np.savetxt(os.path.join(src, "image_timestamps.txt"), np.column_stack([np.arange(len(img)), img, img]))
    folders = {}
    for c in ("clean", cond):
        tt, xx, yy, pp, rec = R.apply_condition(t, x, y, p, c, seq)
        d = os.path.join(tmp, "data", "R5_ours", "%s__%s" % (seq, MB.tag_of(c)))
        os.makedirs(d)
        for name in ("imu_data.txt", "mocap_data.txt", "image_timestamps.txt"):
            shutil.copy(os.path.join(src, name), d)
        if R.imu_offset_us(c):
            imu = np.loadtxt(os.path.join(src, "imu_data.txt"))
            imu[:, 0] += R.imu_offset_us(c)
            np.savetxt(os.path.join(d, "imu_data.txt"), imu)
        with _H5File(os.path.join(d, "events_left.h5"), "w") as f:
            f.create_dataset("t", tt.astype(float)); f.create_dataset("x", xx.astype(np.float32))
            f.create_dataset("y", yy.astype(np.float32)); f.create_dataset("p", pp)
        ds = R.per_second_digest(t, x, y, p)
        do = R.per_second_digest(tt, xx, yy, pp)
        n_first = int(np.searchsorted(t, t[1999], side="right"))       # as r5_make_inputs.py: whole groups
        ev = {"cond": c, "source": {"first_t_us": int(t[0]), "last_t_us": int(t[-1]), "first_events": [[int(t[i]), int(x[i]), int(y[i]), int(p[i])] for i in range(n_first)],
                         "first_events_complete_groups": True,
                         "per_second": {str(k): v for k, v in ds.items()}, "digest": R.digest_summary(ds)},
              "output": {"per_second": {str(k): v for k, v in do.items()}, "digest": R.digest_summary(do),
                         "events_sha256": MB.sha256(os.path.join(d, "events_left.h5"))}}
        json.dump(ev, open(os.path.join(d, "r5_evidence.json"), "w"))
        folders[c] = d
    return src, folders, (t, x, y, p)


def _ref_folder(tmp):
    """The trusted calibration of DEIO's own inputs (stands in for data/DAVIS240C_upstream/<seq>)."""
    ref = os.path.join(tmp, "ref")
    if not os.path.isdir(ref):
        os.makedirs(ref)
        open(os.path.join(ref, "calib_undist_left.txt"), "w").write("171.3 171.6 120.9 89.2")
        yy, xx = np.mgrid[0:180, 0:240]
        m = np.stack([xx + 0.37 * np.sin(yy / 30.0), yy + 0.21 * np.cos(xx / 40.0)], axis=-1).astype(np.float32)
        with _H5File(os.path.join(ref, "rectify_map_left.h5"), "w") as f:
            f.create_dataset("rectify_map", m)
    return ref


def _make_upstream(ours, src, up):
    """What pp_davis240c_from_converted.py writes, built here without cv2."""
    os.makedirs(up)
    ref = _ref_folder(os.path.dirname(up))
    with _H5File(os.path.join(ours, "events_left.h5"), "r") as f:
        t, x, y, p = f["t"][:], f["x"][:], f["y"][:], f["p"][:]
    gt = np.loadtxt(os.path.join(ours, "mocap_data.txt"))
    gt = gt[gt[:, 0] > 0]
    img = np.round(np.loadtxt(os.path.join(ours, "image_timestamps.txt"))[:, 1])
    t0 = min(gt[0, 0], img[0], t[0])
    for name in ("calib_undist_left.txt", "rectify_map_left.h5"):
        shutil.copy(os.path.join(ref, name), up)
    open(os.path.join(up, "raw_tss_imgs_ns_left.txt"), "w").writelines("%d\n" % round(v * 1e3) for v in img)
    open(os.path.join(up, "tss_imgs_us_left.txt"), "w").writelines(f"{v - t0:.012f}\n" for v in img)
    for name, times in (("raw_gt_stamped_ns_left.txt", gt[:, 0]), ("gt_stamped_left.txt", gt[:, 0] - t0)):
        with open(os.path.join(up, name), "w") as fh:
            for ts, pose in zip(times, gt[:, 1:8]):
                fh.write(f"{ts} " + " ".join(str(v) for v in pose) + "\n")
    imu = np.loadtxt(os.path.join(ours, "imu_data.txt"))      # the rebuild reads its --src folder, i.e. the copy
    with open(os.path.join(up, "imu_data.csv"), "w") as fh:
        fh.write("#timestamp [ns],w,w,w,a,a,a\n")
        for r in imu:
            fh.write(",".join(map(str, [int(round(r[0] * 1e3))] + list(r[1:7]))) + "\n")
    with _H5File(os.path.join(up, "evs_left.h5"), "w") as f:
        f.create_dataset("events/t", (np.asarray(t, float) - t0).astype(np.uint32))
        f.create_dataset("events/x", np.asarray(x).astype(np.uint16))
        f.create_dataset("events/y", np.asarray(y).astype(np.uint16))
        f.create_dataset("events/p", np.asarray(p).astype(np.uint8))
        f.create_dataset("ms_to_idx", CU.ms_to_idx(np.asarray(t, float) - t0).astype(np.uint64))   # pp: float times


OFF = 1_468_939_993_000_000


def _make_bag(path, stream, with_imu=True, imu_shift_us=0.0, sub_ns=0):
    """A fake bag; sub_ns is the recording's sub-microsecond remainder (the DAVIS driver: reset time in ns plus whole
    device microseconds), on events and IMU stamps alike."""
    t, x, y, p = stream
    msgs = []
    cuts = np.linspace(0, len(t), 41).astype(int)
    imu = _imu()
    k = 0
    prefix = struct.pack("<I", 1) + struct.pack("<II", 0, 0) + struct.pack("<I", 4) + b"dvs0" + \
        struct.pack("<II", 180, 240)

    def imu_msg(k):
        r = imu[k]
        raw = ("sensor_msgs/Imu", _imu_raw((r[0] + OFF + imu_shift_us) * 1000 + sub_ns, k, r[1:7]), "md5i", None,
               None)
        return ("/dvs/imu", raw, _Stamp((r[0] + OFF) * 1000 + 7), None)

    for a, b in zip(cuts[:-1], cuts[1:]):
        ev = np.zeros(b - a, dtype=MB.EVENT)
        ev["x"], ev["y"], ev["pol"] = x[a:b], y[a:b], p[a:b]
        ev = MB.set_us(ev, t[a:b] + OFF, sub_ns)
        rt = _Stamp((t[b - 1] + OFF) * 1000)
        msgs.append(("/dvs/events", ("dvs_msgs/EventArray", MB.serialize_event_array(prefix, ev), "md5e", None, None),
                     rt, None))
        while with_imu and k < len(imu) and imu[k, 0] <= t[b - 1]:
            msgs.append(imu_msg(k))
            k += 1
    while with_imu and k < len(imu):
        msgs.append(imu_msg(k))
        k += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    _BAGS[os.path.realpath(path)] = msgs


# ---- 1. DEIO's rebuilt inputs ------------------------------------------------------------------------------------
def test_upstream_check():
    tmp = tempfile.mkdtemp()
    try:
        src, F, _ = _make_folders(tmp)
        up = os.path.join(tmp, "up")
        _make_upstream(F["keep:0.1"], src, up)
        ok, rep = CU.check_upstream(F["keep:0.1"], up, src, ref=os.path.join(tmp, "ref"))
        assert ok, rep
        # wrong coordinates
        with _H5File(os.path.join(up, "evs_left.h5"), "r") as f:
            d = {k: f[k][:] for k in ("events/t", "events/x", "events/y", "events/p")}
        d["events/x"] = d["events/x"].copy()
        d["events/x"][100] = (d["events/x"][100] + 1) % 240
        with _H5File(os.path.join(up, "evs_left.h5"), "w") as f:
            for k, v in d.items():
                f.create_dataset(k, v)
        ok, rep = CU.check_upstream(F["keep:0.1"], up, src, ref=os.path.join(tmp, "ref"))
        assert not ok and "events" in rep["failed"], rep
        # missing IMU file, missing ground truth
        for name in ("imu_data.csv", "gt_stamped_left.txt"):
            up2 = os.path.join(tmp, "up_" + name)
            _make_upstream(F["keep:0.1"], src, up2)
            os.remove(os.path.join(up2, name))
            ok, rep = CU.check_upstream(F["keep:0.1"], up2, src, ref=os.path.join(tmp, "ref"))
            assert not ok and name in rep["missing_files"], rep
    finally:
        shutil.rmtree(tmp)


def _rewrite_h5(path, **change):
    with _H5File(path, "r") as f:
        d = {k: f[k][:] for k in f.d}
    for k, v in change.items():
        if v is None:
            d.pop(k.replace("__", "/"))
        else:
            d[k.replace("__", "/")] = v
    with _H5File(path, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, v)


def test_upstream_calibration():
    tmp = tempfile.mkdtemp()
    try:
        src, F, _ = _make_folders(tmp)
        ours = F["keep:0.1"]
        up = os.path.join(tmp, "up_zero_focal")
        _make_upstream(ours, src, up)
        open(os.path.join(up, "calib_undist_left.txt"), "w").write("0 0 120.9 89.2")
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert not ok and "calibration" in rep["failed"], rep
        up = os.path.join(tmp, "up_placeholder_map")
        _make_upstream(ours, src, up)
        _rewrite_h5(os.path.join(up, "rectify_map_left.h5"), rectify_map=np.zeros((180, 240, 2), np.float32))
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert not ok and not rep["calib"]["rectify_map_equal"], rep
        up = os.path.join(tmp, "up_ok")
        _make_upstream(ours, src, up)
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "no_such_reference"))
        assert not ok and rep["calib"]["problem"] == "reference calibration missing", rep
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert ok, rep
    finally:
        shutil.rmtree(tmp)


def test_upstream_frames_and_ms_index():
    tmp = tempfile.mkdtemp()
    try:
        src, F, _ = _make_folders(tmp)
        ours = F["keep:0.1"]
        # frame times cut to one line
        up = os.path.join(tmp, "up_frames")
        _make_upstream(ours, src, up)
        lines = open(os.path.join(up, "raw_tss_imgs_ns_left.txt")).readlines()
        open(os.path.join(up, "raw_tss_imgs_ns_left.txt"), "w").writelines(lines[:1])
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert not ok and "frames" in rep["failed"], rep
        # ms_to_idx missing, and altered
        for case, val in (("missing", None), ("altered", "shift")):
            up = os.path.join(tmp, "up_ms_" + case)
            _make_upstream(ours, src, up)
            if val is None:
                _rewrite_h5(os.path.join(up, "evs_left.h5"), ms_to_idx=None)
            else:
                with _H5File(os.path.join(up, "evs_left.h5"), "r") as f:
                    m = f["ms_to_idx"][:].copy()
                m[5] += 1
                _rewrite_h5(os.path.join(up, "evs_left.h5"), ms_to_idx=m)
            ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
            assert not ok and rep["events"]["ms_to_idx"] == case.replace("altered", "differs"), rep
    finally:
        shutil.rmtree(tmp)


# ---- 2 and 3. bag ------------------------------------------------------------------------------------------------
def _bag_case(with_imu=True, imu_shift_us=0.0, cond="keep:0.1", sub_ns=0, absolute=False):
    """absolute=True: the boxes_6dof case, h5 times from absolute text seconds (the bag built from the same events)."""
    tmp = tempfile.mkdtemp()
    src, F, stream = _make_folders(tmp, cond=cond, abs_sub=sub_ns if absolute else None)
    if absolute:
        stream = _stream()
    _make_bag(os.path.join(tmp, "data", "rosbags", "rpgdavis", "s.bag"), stream, with_imu, imu_shift_us, sub_ns)
    return tmp


def test_text_seconds_chain_matches_decimal_parse():
    rng = np.random.default_rng(3)
    ns = (OFF * 1000 + rng.integers(0, 60_000_000_000, 20000)).astype(np.int64)
    want = np.rint(np.array([float("%d.%09d" % (v // 10**9, v % 10**9)) for v in ns]) * 1e6).astype(np.int64)
    assert np.array_equal(MB.text_seconds_us(ns), want)
    assert (MB.text_seconds_us(ns) != (ns + 500) // 1000).any()        # it is not the nearest microsecond
    T = (ns + 500) // 1000
    assert np.array_equal(MB.text_seconds_us(T * 1000), T)              # whole microseconds map to themselves
    assert MB.time_mapping(0) == "nearest" and MB.time_mapping(int(OFF)) == "text_seconds"


def test_bag_absolute_h5_times():
    """boxes_6dof: h5 times from absolute text seconds; every check passes through the text_seconds mapping, the
    nearest microsecond does not reproduce them, and rewritten events are written on whole microseconds."""
    for cond in ("keep:0.1", "tq:10000", "noise:5", "imu:15000"):
        tmp = _bag_case(cond=cond, sub_ns=635, absolute=True)
        try:
            path = os.path.join(tmp, "data", "rosbags", "rpgdavis", "s.bag")
            ref = json.load(open(os.path.join(tmp, "data", "R5_ours", "s__clean", "r5_evidence.json")))["source"]
            assert not MB.time_offset(_Bag(path), ref["first_events"], True, mode="nearest")[1]
            ok, rep = MB.run("s", "clean", fake_rosbag, root=tmp)
            assert ok and rep["time_mapping"] == "text_seconds" and rep["time_offset"]["offset_us"] == 0, rep
            ok, rep = MB.run("s", cond, fake_rosbag, root=tmp)
            assert ok and rep["identity_output"]["n_mismatched"] == 0 and rep["rewrite"]["event_payload"] == 0, (cond, rep)
            subs = set()
            for topic, raw, _ in _Bag(os.path.join(tmp, "data", "rosbags", "rpgdavis",
                                                   "s__r5_%s.bag.tmp" % MB.tag_of(cond))).read_messages(raw=True):
                if topic == "/dvs/events":
                    subs |= set(MB.sub_us(MB.parse_event_array(raw[1])[1]).tolist())
            want = {0} if cond.startswith("tq:") else {0, -365} if cond.startswith("noise:") else {-365}
            assert subs == want, (cond, sorted(subs))
        finally:
            shutil.rmtree(tmp)


def test_bag_times_with_sub_microsecond_remainder():
    """Real DAVIS bags: every event at reset time + whole microseconds, so a constant remainder (here 104 ns, rounded
    down, and 700 ns, rounded up). All checks pass, and the written bag keeps the remainder on every event."""
    for sub in (104, 700):
        for cond in ("keep:0.1", "tq:10000", "noise:5", "imu:15000"):
            tmp = _bag_case(cond=cond, sub_ns=sub)
            try:
                ok, rep = MB.run("s", "clean", fake_rosbag, root=tmp)
                assert ok and rep["sub_microsecond_remainder_ns"]["distinct"] == 1, (sub, rep)
                want = sub - 1000 if sub >= 500 else sub
                assert rep["sub_microsecond_remainder_ns"]["most_common"][0][0] == want, rep
                ok, rep = MB.run("s", cond, fake_rosbag, root=tmp)
                assert ok and rep["identity_output"]["n_mismatched"] == 0, (sub, cond, rep)
                out = os.path.join(tmp, "data", "rosbags", "rpgdavis", "s__r5_%s.bag" % MB.tag_of(cond))
                subs = set()
                assert os.path.isfile(out)
                for topic, raw, _ in _Bag(out + ".tmp").read_messages(raw=True):   # stand-in keeps the .tmp key
                    if topic == "/dvs/events":
                        subs |= set(MB.sub_us(MB.parse_event_array(raw[1])[1]).tolist())
                assert subs == {want}, (sub, cond, sorted(subs)[:5])
            finally:
                shutil.rmtree(tmp)


def test_bag_requires_imu():
    tmp = _bag_case(with_imu=False)
    try:
        ok, rep = MB.run("s", "clean", fake_rosbag, root=tmp)
        assert not ok and "problem" in rep["imu"], rep
    finally:
        shutil.rmtree(tmp)


def test_bag_imu_time_off():
    tmp = _bag_case(imu_shift_us=500.0)
    try:
        ok, rep = MB.run("s", "clean", fake_rosbag, root=tmp)
        assert not ok and rep["imu"]["max_time_diff_us"] >= 499, rep
    finally:
        shutil.rmtree(tmp)


def test_bag_clean_and_rewrite_pass():
    for cond in ("keep:0.1", "tq:10000", "noise:5", "imu:15000"):
        tmp = _bag_case(cond=cond)
        try:
            ok, rep = MB.run("s", "clean", fake_rosbag, root=tmp)
            assert ok, rep
            ok, rep = MB.run("s", cond, fake_rosbag, root=tmp)
            assert ok and rep["identity_output"]["n_mismatched"] == 0, rep
        finally:
            shutil.rmtree(tmp)


def test_verify_rewrite_detects_changes():
    tmp = _bag_case()
    try:
        ok, rep = MB.run("s", "keep:0.1", fake_rosbag, root=tmp)
        assert ok, rep
        src = list(_Bag(os.path.join(tmp, "data", "rosbags", "rpgdavis", "s.bag")).read_messages(raw=True))
        outp = os.path.join(tmp, "data", "rosbags", "rpgdavis", "s__r5_keep0.1.bag")
        assert os.path.isfile(outp)          # renamed from .tmp only after every check passed
        out = [(m[0], m[1], m[2]) for m in _BAGS[os.path.realpath(outp + ".tmp")]]   # stand-in keeps the .tmp key
        assert MB.verify_rewrite(iter(src), iter(out), "keep:0.1", "s", OFF)[0]
        i = next(k for k, m in enumerate(out) if m[0] == "/dvs/imu")
        bad_bytes = list(out)
        bad_bytes[i] = (out[i][0], (out[i][1][0], b"changed", out[i][1][2], None, None), out[i][2])
        assert not MB.verify_rewrite(iter(src), iter(bad_bytes), "keep:0.1", "s", OFF)[0]
        bad_time = list(out)
        bad_time[i] = (out[i][0], out[i][1], _Stamp(123))
        assert not MB.verify_rewrite(iter(src), iter(bad_time), "keep:0.1", "s", OFF)[0]
        assert not MB.verify_rewrite(iter(src), iter(out[:-1]), "keep:0.1", "s", OFF)[0]
    finally:
        shutil.rmtree(tmp)


def test_bag_checks_h5_against_manifest():
    for change in ("missing", "modified"):
        tmp = _bag_case()
        try:
            h5 = os.path.join(tmp, "data", "R5_ours", "s__keep0.1", "events_left.h5")
            if change == "missing":
                os.remove(h5)
            else:
                with _H5File(h5, "r") as f:
                    x = f["x"][:].copy()
                x[3] = (x[3] + 1) % 240
                _rewrite_h5(h5, x=x)
            ok, rep = MB.run("s", "keep:0.1", fake_rosbag, root=tmp)
            assert not ok and "differs from its manifest" in rep.get("problem", ""), rep
        finally:
            shutil.rmtree(tmp)


def test_rewrite_imu_shift_checked():
    tmp = _bag_case(cond="imu:15000")
    try:
        ok, rep = MB.run("s", "imu:15000", fake_rosbag, root=tmp)
        assert ok and rep["imu_output"]["max_time_diff_us"] <= 1.0, rep
        src = [(m[0], m[1], m[2]) for m in _BAGS[os.path.realpath(os.path.join(tmp, "data", "rosbags", "rpgdavis",
                                                                                "s.bag"))]]
        # an output whose IMU was not shifted is caught message by message
        assert not MB.verify_rewrite(iter(src), iter(src), "imu:15000", "s", OFF)[0]
        # and an IMU shifted by the wrong amount fails the IMU check against the copy
        moved = [(tp, (r[0], MB.shift_stamp(r[1], 5000), r[2], r[3], r[4]) if tp == "/dvs/imu" else r, ts, None)
                 for tp, r, ts in src]
        path = os.path.join(tmp, "wrong.bag")
        open(path, "wb").close()
        _BAGS[os.path.realpath(path)] = moved
        ok2, rep2 = MB.imu_check(_Bag(path), os.path.join(tmp, "data", "R5_ours", "s__imu15000us", "imu_data.txt"), OFF)
        assert not ok2 and rep2["max_time_diff_us"] >= 9999, rep2
    finally:
        shutil.rmtree(tmp)


def test_noise_rewrite_inserts_each_event_once():
    tmp = _bag_case(cond="noise:5")
    try:
        ok, rep = MB.run("s", "noise:5", fake_rosbag, root=tmp)
        n_clean = rep["identity_clean"]["events_bag"]
        assert ok and rep["identity_output"]["events_bag"] == n_clean + rep["noise_events"], rep
    finally:
        shutil.rmtree(tmp)


def test_upstream_imu_condition():
    tmp = tempfile.mkdtemp()
    try:
        src, F, _ = _make_folders(tmp, cond="imu:15000")
        ours = F["imu:15000"]
        up = os.path.join(tmp, "up")
        _make_upstream(ours, src, up)
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert ok and rep["imu"]["offset_us"] == 15000, rep
        shutil.copy(os.path.join(src, "imu_data.txt"), ours)          # copy left unshifted
        ok, rep = CU.check_upstream(ours, up, src, ref=os.path.join(tmp, "ref"))
        assert not ok and "imu" in rep["failed"], rep
    finally:
        shutil.rmtree(tmp)


def test_offset_prefix_all_same_timestamp():
    n = 60000
    rng = np.random.default_rng(9)
    t = np.sort(np.r_[np.full(2100, 5000), rng.integers(6000, 4_000_000, n - 2100)])
    x, y, p = rng.integers(0, 240, n), rng.integers(0, 180, n), rng.integers(0, 2, n)
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "b.bag")
        _make_bag(path, (t, x, y, p), with_imu=False)
        n_first = int(np.searchsorted(t, t[1999], side="right"))      # 2100: the whole group
        ref = [[int(t[i]), int(x[i]), int(y[i]), int(p[i])] for i in range(n_first)]
        off, ok, n_ref, n_bag = MB.time_offset(_Bag(path), ref, complete_groups=True)
        assert ok and n_ref == n_bag == 2100, (off, ok, n_ref, n_bag)
    finally:
        shutil.rmtree(tmp)


def test_offset_prefix_with_tie_at_2000():
    t, x, y, p = _stream(ties_at=2000)
    assert t[1999] == t[2000]
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "b.bag")
        _make_bag(path, (t, x, y, p), with_imu=False)
        ref = [[int(t[i]), int(x[i]), int(y[i]), int(p[i])] for i in range(2000)]
        off, ok, n_ref, n_bag = MB.time_offset(_Bag(path), ref)
        assert ok and off == OFF and n_ref == n_bag, (off, ok, n_ref, n_bag)
    finally:
        shutil.rmtree(tmp)


# ---- 4. window moves ---------------------------------------------------------------------------------------------
def test_window_moves_counts_events_leaving():
    tss = np.array([0.0, 100.0, 300.0])          # windows [0,100), [100,200), [300,400); gap [200,300)
    t = np.array([50.0, 150.0, 199.0, 350.0])
    t_new = np.array([50.0, 150.0, 250.0, 350.0])   # the third event falls into the gap
    r = CW.window_moves(t, t_new, tss, 100.0)
    assert r["left"] == 1 and r["crossed"] == 0 and r["entered"] == 0, r
    t_new2 = np.array([50.0, 150.0, 305.0, 350.0])  # now it lands in another window
    r = CW.window_moves(t, t_new2, tss, 100.0)
    assert r["crossed"] == 1 and r["left"] == 0, r
    assert abs(r["share_changed"] - 0.25) < 1e-12, r
    # one event in a window before, three more enter: the share is a proportion (at most 1)
    tss2 = np.array([0.0])
    r = CW.window_moves(np.array([50.0, 150.0, 160.0, 170.0]), np.array([50.0, 60.0, 70.0, 80.0]), tss2, 100.0)
    assert r["entered"] == 3 and 0.0 <= r["share_changed"] <= 1.0, r


# ---- 5. SuperEIO CPU blocks and acceptance ---------------------------------------------------------------------
def _topo_2x48x2():
    """192 logical CPUs: 2 sockets x 48 cores x 2 threads; thread 2 of core c on socket s is CPU 96 + 48 s + c."""
    return [(48 * s + c + 96 * h, c, s) for s in range(2) for c in range(48) for h in range(2)]


def test_cpu_blocks():
    topo = _topo_2x48x2()
    b = RA.make_blocks(topo, {c for c, _, _ in topo})
    assert len(b) == 12 and all(len(x) == 16 for x in b), [len(x) for x in b]
    flat = [c for x in b for c in x]
    assert len(flat) == len(set(flat))
    sock = {c: s for c, _, s in topo}
    assert all(len({sock[c] for c in x}) == 1 for x in b)
    # a container that may use only CPUs 0-31 and their siblings 96-127: 32 whole cores, 4 blocks
    b = RA.make_blocks(topo, set(range(0, 32)) | set(range(96, 128)))
    assert len(b) == 4 and all(len(x) == 16 for x in b)
    # siblings outside the container: no whole core, no block
    assert RA.make_blocks(topo, set(range(0, 64))) == []


ENG = ["aaaa  ./super_eventpoint.engine", "bbbb  ./super_eventmatch.engine", "cccc  ./super_eventpoint.onnx"]
CFG = ("extrinsicRotation: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
       "extrinsicTranslation: !!opencv-matrix\n   rows: 3\n   cols: 1\n   dt: d\n   data: [0, 0, 0]\n")


def _attempt_dir(root, name, cpus, time_limited=36, solves=700, se3=1.2, summary=True, bag_rc=0, restarts=0,
                 reboots=0, deaths=0, poses=50, engines=ENG, seq="boxes_6dof", cond="clean", idx=1, variant="orig",
                 bag=None, run=None):
    """A run folder as run_seq.sh and eval_traj.py leave it (bag paths under `root`)."""
    d = os.path.join(root, name)
    os.makedirs(d)
    if summary:
        lines = ["sequence %s  tag r5%s%s  run %s" % (seq, cond, idx, run or d),
                 "variant %s" % variant,
                 "bag %s" % (bag or RA.bag_of(seq, cond, root)),
                 "machine h nproc 192 cpus %s load_at_start 0.1 gpu 0 ros_port 11311 parallel_slots 8" % cpus,
                 "rosbag play exit code: %d" % bag_rc if bag_rc is not None else "",
                 "odometry poses (evio_result_no_loop.csv): %d" % poses,
                 "event_detector restarts: %d" % restarts,
                 "failure detections (system reboot!): %d" % reboots,
                 "process deaths: %d" % deaths]
        if solves is not None:
            lines.append("solver: solves %d stopped by the time limit %d" % (solves, time_limited))
        open(os.path.join(d, "summary.txt"), "w").write("\n".join(lines) + "\n")
        with open(os.path.join(d, "evio_result_no_loop.csv"), "w") as fh:
            for k in range(poses):
                fh.write("%d,%f,0,0,1,0,0,0,0\n" % (1_000_000_000 + 50_000_000 * k, 0.01 * k))
        open(os.path.join(d, "config.yaml"), "w").write(CFG)
    if engines is not None:
        open(os.path.join(d, "engines.sha256"), "w").write("\n".join(engines) + "\n")
    if se3 is not None:
        json.dump({"se3_all": {"mpe_percent": se3}}, open(os.path.join(d, "eval.json"), "w"))
    return d


def test_accept_smoke():
    tmp = tempfile.mkdtemp()
    try:
        ref = sorted(ENG)
        blocks = [list(range(0, 8)) + list(range(96, 104)), list(range(8, 16)) + list(range(104, 112))]
        c0, c1 = "0-7,96-103", "8-15,104-111"
        plan = [("boxes_6dof", "clean", "1"), ("boxes_6dof", "keep0.1", "1")]
        chk = lambda rows: RA.check("smoke", plan, rows, blocks, ref, root=tmp)  # noqa: E731
        C = lambda n, **kw: ("boxes_6dof", "clean", "1", _attempt_dir(tmp, n, c0, **kw), 0, 0)  # noqa: E731
        Kp = lambda n, rc=0, **kw: ("boxes_6dof", "keep0.1", "1",  # noqa: E731
                                     _attempt_dir(tmp, n, c1, cond="keep0.1", **kw), rc, 1)
        good = [C("a"), Kp("b")]
        ok, rec = chk(good)
        assert ok and rec["outcomes"] == {"completed": 2}, rec["problems"]
        assert all(x["as_planned"] is True for x in rec["attempts"]), rec["attempts"]   # stored in the record
        # rosbag play failed but a scorable trajectory was left: undetermined, the smoke stops
        ok, rec = chk([good[0], Kp("c", bag_rc=1)])
        assert not ok and rec["attempts"][1]["outcome"].startswith("undetermined: rosbag play"), rec
        # play exit code not recorded, a ROS process died, run_seq.sh exited 1: all undetermined
        for n, kw, rc in (("u1", {"bag_rc": None}, 0), ("u2", {"deaths": 1}, 0), ("u3", {}, 1)):
            ok, rec = chk([good[0], Kp(n, rc=rc, **kw)])
            assert not ok and rec["attempts"][1]["outcome"].startswith("undetermined"), (n, rec["attempts"][1])
        # the estimator restarted on clean input with a scorable trajectory: failed, and a clean smoke must complete
        ok, rec = chk([C("d", reboots=1), good[1]])
        assert not ok and rec["attempts"][0]["outcome"] == "failed: restart or reboot", rec
        # the same under a perturbation is a result: listed, not a reason to stop
        ok, rec = chk([good[0], Kp("e", restarts=1, se3=None)])
        assert ok and rec["algorithm_failures"] == ["boxes_6dof keep0.1 1: failed: restart or reboot"], rec
        # completed but eval_traj.py gave no score; a missing attempt; 24 CPUs instead of 16; other engines; none
        assert not chk([good[0], Kp("f", se3=None)])[0]
        assert not chk(good[:1])[0]
        ok, rec = chk([("boxes_6dof", "clean", "1", _attempt_dir(tmp, "g", "0-11,96-107"), 0, 0), good[1]])
        assert not ok and any("CPU set" in p for p in rec["problems"]), rec["problems"]
        ok, rec = chk([good[0], Kp("h", engines=ENG[:2] + ["dddd  ./x.onnx"])])
        assert not ok and any("engines" in p for p in rec["problems"]), rec["problems"]
        assert not chk([good[0], Kp("i", engines=None)])[0]
        # summary.txt says the attempt played the clean boxes bag, the list says keep0.1: rejected
        ok, rec = chk([good[0], ("boxes_6dof", "keep0.1", "1", _attempt_dir(tmp, "j", c1), 0, 1)])
        assert not ok and any("summary.txt says" in p and "bag" in p for p in rec["problems"]), rec["problems"]
        assert rec["attempts"][1]["as_planned"] is False
        # another sequence's run, another variant, a summary naming another folder
        ok, rec = chk([good[0], Kp("k", seq="poster_6dof", bag=RA.bag_of("boxes_6dof", "keep0.1", tmp))])
        assert not ok and any("seq poster_6dof" in p for p in rec["problems"]), rec["problems"]
        assert not chk([good[0], Kp("l", variant="ctrl")])[0]
        assert not chk([good[0], Kp("m", run=os.path.join(tmp, "a"))])[0]
        # one folder listed for both attempts
        ok, rec = chk([good[0], ("boxes_6dof", "keep0.1", "1", good[0][3], 0, 1)])
        assert not ok and any("listed for 2 attempts" in p for p in rec["problems"]), rec["problems"]
        # the clean boxes smoke attempt: no checkpoint at 700 solves, or outside the single-run range 35-44
        ok, rec = chk([C("n", solves=None), good[1]])
        assert not ok and any("scheduling" in p for p in rec["problems"]), rec["problems"]
        assert not chk([C("o", time_limited=50), good[1]])[0]
        assert RA.parse_cpu_list("0-2,7") == {0, 1, 2, 7}
    finally:
        shutil.rmtree(tmp)


def test_accept_main_scheduling_check():
    tmp = tempfile.mkdtemp()
    try:
        ref = sorted(ENG)
        blocks = [list(range(0, 8)) + list(range(96, 104))]
        c0 = "0-7,96-103"
        tags = ("clean", "keep0.1", "tq10000us")
        plan = [(s, t, str(i)) for i in range(1, 11) for s in RA.SEQS for t in tags]
        chk = lambda pl, rows: RA.check("main", pl, rows, blocks, ref, root=tmp)  # noqa: E731
        tl = [36, 38, 40, 41, 39, 37, 42, 40, 38, 39]
        rows = []
        for s, t, i in plan:
            kw = {"time_limited": tl[int(i) - 1]} if (s, t) == ("boxes_6dof", "clean") else {}
            if (s, t, i) == ("boxes_6dof", "keep0.1", "1"):
                kw.update(reboots=1, se3=None)
            rows.append((s, t, i, _attempt_dir(tmp, "%s_%s_%s" % (s, t, i), c0, seq=s, cond=t, idx=i, **kw), 0, 0))
        ok, rec = chk(plan, rows)
        sc = rec["scheduling_check"]
        assert ok and sc["passed"] and sc["with_checkpoint_700"] == 10 and sc["planned"] == 10, rec["problems"]
        assert rec["algorithm_failures"] == ["boxes_6dof keep0.1 1: failed: restart or reboot"]
        assert rec["distinct_run_folders"] == 60
        cb = [k for k, r in enumerate(rows) if r[:2] == ("boxes_6dof", "clean")]
        # only one of the ten clean boxes attempts recorded the checkpoint: not valid
        one = list(rows)
        for k in cb[1:]:
            s, t, i = rows[k][:3]
            one[k] = (s, t, i, _attempt_dir(tmp, "n" + i, c0, idx=i, solves=None), 0, 0)
        ok, rec = chk(plan, one)
        assert not ok and rec["scheduling_check"]["with_checkpoint_700"] == 1, rec["scheduling_check"]
        # a checkpoint at another solve count does not count; a median outside 35-44 fails
        other = list(rows)
        other[cb[-1]] = ("boxes_6dof", "clean", "10", _attempt_dir(tmp, "o", c0, idx=10, solves=650), 0, 0)
        assert not chk(plan, other)[0]
        hot = list(rows)
        for k in cb:
            i = rows[k][2]
            hot[k] = ("boxes_6dof", "clean", i, _attempt_dir(tmp, "h" + i, c0, idx=i, time_limited=60), 0, 0)
        ok, rec = chk(plan, hot)
        assert not ok and not rec["scheduling_check"]["passed"]
        # an undetermined attempt is kept and counted apart; it does not count as completed
        und = list(rows)
        k = rows.index(next(r for r in rows if r[:3] == ("poster_6dof", "tq10000us", "3")))
        und[k] = ("poster_6dof", "tq10000us", "3",
                  _attempt_dir(tmp, "x", c0, seq="poster_6dof", cond="tq10000us", idx=3, bag_rc=2), 0, 0)
        ok, rec = chk(plan, und)
        assert ok and rec["outcomes"] == {"completed": 58, "failed": 1, "undetermined": 1}, rec["outcomes"]
        # the same clean folder under every attempt name
        same = [(s, t, i, rows[0][3], 0, 0) for s, t, i in plan]
        ok, rec = chk(plan, same)
        assert not ok and rec["distinct_run_folders"] == 1, rec["problems"][:2]
        # a reduced plan (K=1 left in the shell) is not the main batch, even if all of it completed
        small = [x for x in plan if x[2] == "1"]
        ok, rec = chk(small, [r for r in rows if r[2] == "1"])
        assert not ok and any("full main plan" in p for p in rec["problems"]), rec["problems"]
        assert not chk([x for x in plan if x[0] == "boxes_6dof"], [r for r in rows if r[0] == "boxes_6dof"])[0]
    finally:
        shutil.rmtree(tmp)


def test_accept_outcome_matches_bias_experiment_rule():
    """r5_accept.classify gives the outcome cross_system_eval.py gives to the same attempt."""
    sys.path.insert(0, os.path.join(ROOT, "script", "diag"))
    CS = _load("cross_system_eval", "script/diag/cross_system_eval.py")
    tmp = tempfile.mkdtemp()
    try:
        cases = [({}, 0), ({"bag_rc": 1}, 0), ({"bag_rc": None}, 0), ({"deaths": 1}, 0), ({"poses": 5}, 0),
                 ({"restarts": 1}, 0), ({"reboots": 2}, 0), ({"deaths": 1, "reboots": 1}, 0), ({}, 1),
                 ({"summary": False}, 1), ({"bag_rc": 1, "poses": 3}, 0), ({"poses": 0, "reboots": 1}, 0)]
        lines = []
        for k, (kw, rc) in enumerate(cases):
            d = _attempt_dir(tmp, "r%d" % k, "0-7", **kw)
            lines.append("boxes_6dof clean %d %s rc=%d slot=0" % (k + 1, d, rc))
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("\n".join(lines) + "\n")
        theirs = [x["status"] for x in CS.load_supereio(lst)]
        ours = [RA.read_attempt(os.path.join(tmp, "r%d" % k), rc)["outcome"] for k, (kw, rc) in enumerate(cases)]
        norm = lambda o: o.replace(" (check)", "").split(" (exit")[0]  # noqa: E731
        assert [norm(o) for o in theirs] == [norm(o) for o in ours], list(zip(theirs, ours))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print("%d tests passed" % len(names))
