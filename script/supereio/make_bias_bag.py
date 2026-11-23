#!/usr/bin/env python3
"""SuperEIO side of the cross-system IMU-bias check: a copy of a DAVIS240C rosbag with a constant gyro bias added.

The bias is the vector DEIO gets for cell gbias:<d> (script/diag/precision_pilot.py): (+d, -d, +d) deg/s on the
gyro axes, written in rad/s (sensor_msgs/Imu convention) into every /dvs/imu message. Every other message is copied
byte for byte with its original receive time; the IMU header stamps are unchanged.
Before writing, the script checks that DEIO's IMU file for the sequence (data/RPGDAVIS_pp/<seq>/imu_data.txt, gyro in
rad/s per script/eval_deio/mvsec.py) has the same gyro axes as the bag, so that the same vector means the same
physical bias in both systems.

  python3 script/supereio/make_bias_bag.py boxes_6dof --gbias-deg-s 5
Output: data/rosbags/rpgdavis/<seq>__gbias<d>.bag and <same>.json (vector, messages changed, realised change,
axis check, sha256 of input and output). Needs python3-rosbag (setup_env.sh).
Stops (exit 1, no bag) unless: DEIO's IMU file exists and has the same axes and units; at least one IMU message was
changed; reading the written bag back, every IMU message has the source stamps and gyro + exactly the bias vector, and
every other message is unchanged in number. An existing bag is reused only if its evidence file asks for the same
vector, passed these checks, and both sha256 still match; otherwise it stops (use --force to rebuild).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

SIGNS = (1.0, -1.0, 1.0)   # the same as GBIAS_SIGNS in script/diag/precision_pilot.py


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def axis_check(bag_gyro_t_us, bag_gyro, txt_path):
    """Median |difference| per axis between the bag gyro and DEIO's imu_data.txt gyro at the nearest time, after
    removing a constant time offset (the text file may use another time origin)."""
    if not os.path.exists(txt_path):
        return {"checked": False, "why": "missing %s" % txt_path}
    txt = np.loadtxt(txt_path)
    txt = txt[np.argsort(txt[:, 0])]
    off = bag_gyro_t_us[0] - txt[0, 0]
    t = txt[:, 0] + off
    j = np.clip(np.searchsorted(t, bag_gyro_t_us), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - bag_gyro_t_us) < np.abs(t[j] - bag_gyro_t_us), j - 1, j)
    d = np.abs(txt[j, 1:4] - bag_gyro)
    med = np.median(d, axis=0)
    scale = np.median(np.abs(bag_gyro), axis=0)
    return {"checked": True, "rows_bag": int(len(bag_gyro)), "rows_txt": int(len(txt)), "time_offset_us": float(off),
            "median_abs_diff_rad_s": med.tolist(), "median_abs_gyro_rad_s": scale.tolist(),
            "same_axes_and_units": bool(np.all(med <= 0.01 * np.maximum(scale, 1e-3)))}


def imu_rows(bag_path, topic):
    import rosbag
    rows = []
    with rosbag.Bag(bag_path) as b:
        for _, m, _ in b.read_messages(topics=[topic]):
            rows.append([m.header.stamp.to_nsec(), m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                         m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        n_all = b.get_message_count()
    return np.array(rows, dtype=np.float64), n_all


def verify_written(src, out, topic, vec):
    """Read both bags back: same IMU stamps, accel unchanged, gyro changed by exactly vec, same message count."""
    a, n_a = imu_rows(src, topic)
    b, n_b = imu_rows(out, topic)
    if len(a) == 0 or len(a) != len(b) or n_a != n_b:
        return False, "message counts differ (IMU %d vs %d, all %d vs %d)" % (len(a), len(b), n_a, n_b)
    if not np.array_equal(a[:, 0], b[:, 0]) or not np.array_equal(a[:, 4:7], b[:, 4:7]):
        return False, "IMU stamps or accelerations changed"
    dev = np.abs((b[:, 1:4] - a[:, 1:4]) - np.asarray(vec))
    if dev.max() > 1e-12:
        return False, "gyro change differs from the bias vector by up to %.3g rad/s" % dev.max()
    return True, {"imu_messages": int(len(a)), "all_messages": int(n_a), "max_abs_deviation_rad_s": float(dev.max()),
                  "per_axis_mean_change_rad_s": (b[:, 1:4] - a[:, 1:4]).mean(axis=0).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence")
    ap.add_argument("--gbias-deg-s", type=float, required=True, help="d in (+d, -d, +d) deg/s")
    ap.add_argument("--imu-topic", default="/dvs/imu")
    ap.add_argument("--force", action="store_true", help="rewrite even if the output exists")
    a = ap.parse_args()
    import rosbag   # python3-rosbag

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = os.path.join(root, "data", "rosbags", "rpgdavis", a.sequence + ".bag")
    tag = ("%g" % a.gbias_deg_s).replace(".", "p")
    out = os.path.join(root, "data", "rosbags", "rpgdavis", "%s__gbias%s.bag" % (a.sequence, tag))
    if not a.gbias_deg_s > 0:
        sys.exit("--gbias-deg-s must be > 0")
    vec = [s * a.gbias_deg_s * math.pi / 180.0 for s in SIGNS]
    if os.path.exists(out) and not a.force:
        try:
            ev = json.load(open(out + ".json"))
        except (OSError, ValueError):
            sys.exit("%s exists without a readable evidence file: rebuild with --force" % out)
        ok = (np.allclose(ev.get("gyro_bias_rad_s", []), vec, rtol=0, atol=1e-15)
              and (ev.get("axis_check") or {}).get("same_axes_and_units") is True
              and (ev.get("verification") or {}).get("imu_messages", 0) > 0
              and ev.get("source_sha256") == sha256(src) and ev.get("output_sha256") == sha256(out))
        if not ok:
            sys.exit("%s does not match its evidence (vector, checks or sha256): rebuild with --force" % out)
        print("reused (vector, checks and sha256 verified): %s" % out)
        return

    t_us, gyro = [], []
    with rosbag.Bag(src) as ib:
        for _, m, _ in ib.read_messages(topics=[a.imu_topic]):
            t_us.append(m.header.stamp.to_nsec() / 1e3)
            gyro.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
    chk = axis_check(np.array(t_us), np.array(gyro),
                     os.path.join(root, "data", "RPGDAVIS_pp", a.sequence, "imu_data.txt"))
    print("axis check against DEIO's imu_data.txt:", json.dumps(chk))
    if not chk.get("checked"):
        sys.exit("cannot check the gyro axes against DEIO's IMU file (%s): stopped" % chk.get("why"))
    if not chk["same_axes_and_units"]:
        sys.exit("the bag gyro and DEIO's imu_data.txt differ in axes or units: the same vector would not be the "
                 "same physical bias; stopped")

    tmp = out + ".partial"
    n_imu, n_other, deltas = 0, 0, []
    with rosbag.Bag(src) as ib, rosbag.Bag(tmp, "w") as ob:
        for topic, msg, t in ib.read_messages(raw=True):
            if topic == a.imu_topic:
                m = msg[4]()
                m.deserialize(msg[1])
                before = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
                m.angular_velocity.x += vec[0]
                m.angular_velocity.y += vec[1]
                m.angular_velocity.z += vec[2]
                deltas.append([m.angular_velocity.x - before[0], m.angular_velocity.y - before[1],
                               m.angular_velocity.z - before[2]])
                ob.write(topic, m, t)
                n_imu += 1
            else:
                ob.write(topic, msg, t, raw=True)
                n_other += 1
    if n_imu == 0:
        os.remove(tmp)
        sys.exit("no %s message in %s: stopped" % (a.imu_topic, src))
    good, ver = verify_written(src, tmp, a.imu_topic, vec)
    if not good:
        os.remove(tmp)
        sys.exit("written bag failed the read-back check: %s" % ver)
    os.replace(tmp, out)
    ev = {"sequence": a.sequence, "source": src, "output": out, "imu_topic": a.imu_topic,
          "gbias_deg_s_per_axis": a.gbias_deg_s, "signs": SIGNS, "gyro_bias_rad_s": vec,
          "imu_messages_changed": n_imu, "other_messages_copied_raw": n_other,
          "realised_mean_delta_rad_s": np.mean(deltas, axis=0).tolist() if deltas else None,
          "axis_check": chk, "verification": ver, "source_sha256": sha256(src), "output_sha256": sha256(out)}
    with open(out + ".json", "w") as fh:
        json.dump(ev, fh, indent=2)
    print("wrote %s: %d IMU messages changed by %s rad/s, %d other messages copied"
          % (out, n_imu, ["%.6f" % v for v in ev["realised_mean_delta_rad_s"]], n_other))


if __name__ == "__main__":
    main()
