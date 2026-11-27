#!/usr/bin/env python3
"""Check that DEIO's own DAVIS240C inputs, rebuilt from a perturbed copy, hold exactly the copy's sensor data.

Compares the files DEIO's entry script davis240c.py reads (data/R5_up/<seq>__<tag>) with the copy they were built
from (data/R5_ours/<seq>__<tag>) and with the unperturbed source (data/RPGDAVIS_pp/<seq>):
  files     every file of pp_davis240c_from_converted.py exists and is not empty
  calib     calib_undist_left.txt and rectify_map_left.h5 (the undistorted intrinsics and the per-pixel map DEIO's
            reader applies) equal those of the trusted inputs of the earlier pipeline comparison,
            data/DAVIS240C_upstream/<seq>, value for value; a missing reference fails
  t0        recovered as raw_tss_imgs_ns_left.txt[0] / 1e3 - tss_imgs_us_left.txt[0], the way davis240c.py does
  frames    the whole column of frame times, against the source the rebuild uses (data/RPGDAVIS_unzip/<seq>/images.txt
            if present, else the copy's image_timestamps.txt): raw_tss_imgs_ns = round(t * 1e3), tss_imgs_us = t - t0
  events    evs_left.h5 (events/t = t - t0 truncated to uint32, x, y, p) against the copy: the per-second digest of the
            copy's events, with t mapped by the same truncation, must equal the digest of the file; events/t must be
            non-decreasing, and ms_to_idx (the reader's millisecond index) must equal the index recomputed from the
            stored times (the algorithm of utils/event_utils.compute_ms_to_idx)
  IMU       the copy's imu_data.txt equals the source's, with every time shifted by <us> for the condition imu:<us>;
            imu_data.csv has one row per sample, t_ns = round(t_us * 1e3) and the same six values
  ground    the copy's mocap_data.txt equals the source's; raw_gt_stamped_ns_left.txt equals the source poses with
  truth     t > 0; gt_stamped_left.txt equals it shifted by t0 (within 1e-3 us)
Exit 0 only if everything agrees.

  python3 script/diag/r5_check_upstream.py data/R5_ours/boxes_6dof__keep0.1 data/R5_up/boxes_6dof__keep0.1 \
      data/RPGDAVIS_pp/boxes_6dof
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utils import r5_ops as R   # noqa: E402

UP_FILES = ("calib_undist_left.txt", "rectify_map_left.h5", "evs_left.h5", "tss_imgs_us_left.txt",
            "raw_tss_imgs_ns_left.txt", "gt_stamped_left.txt", "raw_gt_stamped_ns_left.txt", "imu_data.csv")


def _load_txt(path, **kw):
    a = np.loadtxt(path, **kw)
    return a.reshape(1, -1) if a.ndim == 1 else a


def ms_to_idx(t_us):
    """utils/event_utils.compute_ms_to_idx(t_us * 1e3, 0) in numpy: the first event index of every millisecond."""
    tss_ns = np.asarray(t_us, dtype=np.float64) * 1e3
    ms_end = int(np.floor(tss_ns.max()) / 1e6)
    win = np.arange(0, ms_end + 1, 1).astype(np.uint64)
    return np.searchsorted(tss_ns, win * 1e6, side="left", sorter=np.argsort(tss_ns, kind="stable"))


def frame_source(ours, seq):
    """The frame times the rebuild uses (pp_davis240c_from_converted.py): raw images.txt if present, else the copy's
    image_timestamps.txt (second column). Rounded to whole microseconds, as there."""
    raw = os.path.join(ROOT, "data", "RPGDAVIS_unzip", seq, "images.txt") if seq else ""
    if raw and os.path.isfile(raw):
        img = np.array([float(l.split()[0]) * 1e6 for l in open(raw) if l.strip()])
        name = raw
    else:
        img = _load_txt(os.path.join(ours, "image_timestamps.txt"))[:, 1]
        name = os.path.join(ours, "image_timestamps.txt")
    return np.round(img), name


def check_calibration(up, ref):
    """The undistorted intrinsics and the rectify map against the trusted reference folder."""
    import h5py
    rep = {"reference": ref}
    if not ref or not all(os.path.isfile(os.path.join(ref, f)) for f in ("calib_undist_left.txt",
                                                                         "rectify_map_left.h5")):
        rep["problem"] = "reference calibration missing"
        return False, rep
    k_up = np.loadtxt(os.path.join(up, "calib_undist_left.txt")).ravel()
    k_ref = np.loadtxt(os.path.join(ref, "calib_undist_left.txt")).ravel()
    rep["intrinsics"] = k_up.tolist()
    rep["intrinsics_equal"] = bool(k_up.shape == k_ref.shape and np.array_equal(k_up, k_ref))
    with h5py.File(os.path.join(up, "rectify_map_left.h5"), "r") as f:
        m_up = np.asarray(f["rectify_map"][:])
    with h5py.File(os.path.join(ref, "rectify_map_left.h5"), "r") as f:
        m_ref = np.asarray(f["rectify_map"][:])
    rep["rectify_map_shape"] = list(m_up.shape)
    rep["rectify_map_equal"] = bool(m_up.shape == m_ref.shape and np.array_equal(m_up, m_ref))
    return rep["intrinsics_equal"] and rep["rectify_map_equal"], rep


def check_upstream(ours, up, src, seq=None, ref=None):
    """(ok, report) for one rebuilt folder. seq selects the raw image times (None: the copy's image_timestamps.txt);
    ref is the trusted folder of DEIO's own inputs for the calibration check (default data/DAVIS240C_upstream/<seq>)."""
    import h5py
    rep, bad = {"ours": ours, "up": up, "src": src}, []
    missing = [f for f in UP_FILES if not (os.path.isfile(os.path.join(up, f)) and os.path.getsize(os.path.join(up, f)))]
    rep["missing_files"] = missing
    if missing:
        return False, rep
    if ref is None and seq:
        ref = os.path.join(ROOT, "data", "DAVIS240C_upstream", seq)
    ok_cal, rep["calib"] = check_calibration(up, ref)
    if not ok_cal:
        bad.append("calibration")
    raw_img_ns = np.loadtxt(os.path.join(up, "raw_tss_imgs_ns_left.txt")).ravel()
    tss = np.loadtxt(os.path.join(up, "tss_imgs_us_left.txt")).ravel()
    t0 = float(raw_img_ns[0] / 1e3 - tss[0])
    rep["t0_us"] = t0

    # frame times ---------------------------------------------------------------------------------------------
    img, img_src = frame_source(ours, seq)
    rep["frames"] = {"source": img_src, "frames_source": int(len(img)), "frames_raw": int(len(raw_img_ns)),
                     "frames_rel": int(len(tss))}
    ok_fr = len(img) == len(raw_img_ns) == len(tss)
    if ok_fr:
        rep["frames"]["raw_ns_exact"] = bool(np.array_equal(raw_img_ns, np.round(img * 1e3)))
        rep["frames"]["rel_max_err_us"] = float(np.max(np.abs(tss - (img - t0))))
        ok_fr = rep["frames"]["raw_ns_exact"] and rep["frames"]["rel_max_err_us"] <= 1e-6
    if not ok_fr:
        bad.append("frames")

    # events -------------------------------------------------------------------------------------------------
    with h5py.File(os.path.join(up, "evs_left.h5"), "r") as f:
        tu = np.asarray(f["events/t"][:]).astype(np.int64)
        xu, yu, pu = f["events/x"][:], f["events/y"][:], f["events/p"][:]
        try:
            msi = np.asarray(f["ms_to_idx"][:]).astype(np.int64)
        except KeyError:
            msi = None
    with h5py.File(os.path.join(ours, "events_left.h5"), "r") as f:
        to, xo, yo, po = f["t"][:], f["x"][:], f["y"][:], f["p"][:]
    to, xo, yo, po = R.as_event_arrays(to, xo, yo, po)
    t_expect = np.floor(to.astype(np.float64) - t0).astype(np.int64)     # float -> uint32 truncation in the rebuild
    d_up = R.per_second_digest(tu, xu, yu, pu)
    d_ex = R.per_second_digest(t_expect, xo, yo, po)
    mism = sorted(s for s in set(d_up) | set(d_ex) if d_up.get(s) != d_ex.get(s))
    rep["events"] = {"up": int(len(tu)), "ours": int(len(to)), "mismatched_seconds": len(mism), "first": mism[:10]}
    rep["events"]["time_ordered"] = bool(np.all(np.diff(tu) >= 0))
    rep["events"]["ms_to_idx"] = "missing" if msi is None else (
        "equal" if np.array_equal(msi, ms_to_idx(tu)) else "differs")
    if mism or len(tu) != len(to) or not rep["events"]["time_ordered"] or rep["events"]["ms_to_idx"] != "equal":
        bad.append("events")

    # IMU ----------------------------------------------------------------------------------------------------
    imu_src = _load_txt(os.path.join(src, "imu_data.txt"))
    cond = "clean"
    if os.path.isfile(os.path.join(ours, "r5_evidence.json")):
        cond = json.load(open(os.path.join(ours, "r5_evidence.json"))).get("cond", "clean")
    off = R.imu_offset_us(cond)
    imu_src = imu_src.copy()
    imu_src[:, 0] += off                         # the IMU the copy must hold: the source, shifted for imu:<us>
    imu_ours = _load_txt(os.path.join(ours, "imu_data.txt"))
    imu_up = _load_txt(os.path.join(up, "imu_data.csv"), delimiter=",", comments="#")
    rep["imu"] = {"cond": cond, "offset_us": off, "rows_src": int(len(imu_src)), "rows_up": int(len(imu_up)),
                  "copy_equals_expected": bool(np.array_equal(imu_ours, imu_src))}
    ok_imu = rep["imu"]["copy_equals_expected"] and len(imu_up) == len(imu_src) and imu_up.shape[1] == 7
    if ok_imu:
        rep["imu"]["t_ns_exact"] = bool(np.array_equal(imu_up[:, 0], np.round(imu_src[:, 0] * 1e3)))
        rep["imu"]["max_abs_value_diff"] = float(np.max(np.abs(imu_up[:, 1:7] - imu_src[:, 1:7])))
        ok_imu = rep["imu"]["t_ns_exact"] and rep["imu"]["max_abs_value_diff"] == 0.0
    if not ok_imu:
        bad.append("imu")

    # ground truth -------------------------------------------------------------------------------------------
    gt_src_all = _load_txt(os.path.join(src, "mocap_data.txt"))
    gt_ours = _load_txt(os.path.join(ours, "mocap_data.txt"))
    gt_src = gt_src_all[gt_src_all[:, 0] > 0]
    raw_gt = _load_txt(os.path.join(up, "raw_gt_stamped_ns_left.txt"))
    rel_gt = _load_txt(os.path.join(up, "gt_stamped_left.txt"))
    rep["gt"] = {"copy_equals_source": bool(np.array_equal(gt_ours, gt_src_all)), "rows_src": int(len(gt_src)),
                 "rows_up": int(len(raw_gt))}
    ok_gt = rep["gt"]["copy_equals_source"] and raw_gt.shape == gt_src.shape and rel_gt.shape == gt_src.shape
    if ok_gt:
        rep["gt"]["raw_equal"] = bool(np.array_equal(raw_gt, gt_src))
        rep["gt"]["rel_max_time_err_us"] = float(np.max(np.abs(rel_gt[:, 0] + t0 - gt_src[:, 0])))
        rep["gt"]["rel_poses_equal"] = bool(np.array_equal(rel_gt[:, 1:], gt_src[:, 1:]))
        ok_gt = rep["gt"]["raw_equal"] and rep["gt"]["rel_max_time_err_us"] <= 1e-3 and rep["gt"]["rel_poses_equal"]
    if not ok_gt:
        bad.append("ground truth")
    rep["failed"] = bad
    return not bad, rep


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: r5_check_upstream.py <R5_ours folder> <R5_up folder> <source folder data/RPGDAVIS_pp/<seq>>")
    ok, rep = check_upstream(*sys.argv[1:4], seq=os.path.basename(os.path.normpath(sys.argv[3])))
    print(json.dumps(rep, indent=1))
    print("[r5-upstream] %s %s" % ("PASSED" if ok else "FAILED:", "" if ok else ", ".join(rep.get("failed") or
                                                                                     ["missing files"])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
