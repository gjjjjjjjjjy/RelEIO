#!/usr/bin/env python3
"""DEIO's DAVIS240C inputs (the files pp_davis240c.py writes) from the same recording without ROS.

pp_davis240c.py reads the rosbag with ROS 1 and holds every event in a Python list (more than 10 GB for boxes_6dof);
the run environment has no ROS. The recording is the same as the text release of the RPG event-camera dataset that
the main matrix uses (data/RPGDAVIS_pp/<seq>, written by pp_rpg_davis.py from events.txt, imu.txt, groundtruth.txt,
images.txt). This script writes, from those files, exactly the outputs DEIO's entry script davis240c.py and loader
davis240c_evs_iterator read, following every step of pp_davis240c.py that changes the data:
  calib_undist_left.txt     cv2.getOptimalNewCameraMatrix(alpha=0) of the intrinsics and distortion hard-coded in
                            pp_davis240c.py (left camera), as there
  rectify_map_left.h5       utils.load_utils.compute_rmap_vector (the function pp_davis240c.py calls)
  evs_left.h5               utils.event_utils.write_evs_arr_to_h5 of (x, y, t - t0, p), polarity 0/1 as there
  tss_imgs_us_left.txt      image times - t0, "%.012f"
  raw_tss_imgs_ns_left.txt  image times in ns (davis240c.py derives the IMU time offset from the first one)
  gt_stamped_left.txt       ground-truth times - t0 and the pose x y z qx qy qz qw (T_marker_cam0 = I, as there)
  raw_gt_stamped_ns_left.txt  ground truth with the original times (as there, in microseconds despite the name)
  imu_data.csv              "t_ns,wx,wy,wz,ax,ay,az" with the header line of pp_davis240c.py
with t0 = min(first ground-truth time, first image time, first event time), as in pp_davis240c.py. Event times are
rounded to whole microseconds (the camera's resolution). Ground-truth rows with t <= 0 are left out, as
deio_se3_eval.load_gt does for the main matrix, so both pipelines are scored against the same poses. Image times come
from the raw images.txt when data/RPGDAVIS_unzip/<seq>/images.txt exists, otherwise from image_timestamps.txt
(which drops one frame at each end that falls outside the IMU and ground-truth range); the script says which.
Only the visualizations of pp_davis240c.py (undistorted images, time surfaces) are not written: DEIO's event
pipeline does not read them.

  python3 script/pp_data/pp_davis240c_from_converted.py --seq boxes_6dof --outdir <scratch>/boxes_6dof
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import cv2      # noqa: E402
import h5py     # noqa: E402
from utils.event_utils import write_evs_arr_to_h5   # noqa: E402
from utils.load_utils import compute_rmap_vector    # noqa: E402

H, W = 180, 240
# left camera of pp_davis240c.py: fx, fy, cx, cy, k1, k2, p1, p2
INTRINSICS = [199.092366542, 198.82882047, 132.192071378, 110.712660011,
              -0.368436311798, 0.150947243557, -0.000296130534385, -0.000759431726241]
IMU_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    src = os.path.join(ROOT, "data", "RPGDAVIS_pp", a.seq)
    raw_images = os.path.join(ROOT, "data", "RPGDAVIS_unzip", a.seq, "images.txt")
    os.makedirs(a.outdir, exist_ok=True)

    with h5py.File(os.path.join(src, "events_left.h5"), "r") as f:
        t_ev = np.round(f["t"][:]).astype(np.float64)
        x, y, p = f["x"][:], f["y"][:], f["p"][:]
    assert np.all(np.diff(t_ev) >= 0), "events not sorted by time"
    imu = np.loadtxt(os.path.join(src, "imu_data.txt"))           # t_us wx wy wz ax ay az
    gt = np.loadtxt(os.path.join(src, "mocap_data.txt"))          # t_us x y z qx qy qz qw
    gt = gt[gt[:, 0] > 0]
    if os.path.isfile(raw_images):
        img_us = np.array([float(l.split()[0]) * 1e6 for l in open(raw_images) if l.strip()])
        img_src = raw_images
    else:
        img_us = np.loadtxt(os.path.join(src, "image_timestamps.txt"))[:, 1]
        img_src = os.path.join(src, "image_timestamps.txt") + " (one frame dropped at each end)"
    img_us = np.round(img_us)
    assert np.all(np.diff(img_us) > 0) and np.all(np.diff(gt[:, 0]) >= 0)

    fx, fy, cx, cy, k1, k2, p1, p2 = INTRINSICS
    Kdist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.asarray([k1, k2, p1, p2])
    K_new, _ = cv2.getOptimalNewCameraMatrix(Kdist, dist, (W, H), alpha=0, newImgSize=(W, H))
    with open(os.path.join(a.outdir, "calib_undist_left.txt"), "w") as f:
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
    _, K_new_evs = compute_rmap_vector(Kdist, dist, a.outdir, "left", H=H, W=W)
    assert np.all(abs(K_new_evs - K_new) < 1e-5)

    t0 = min(gt[0, 0], img_us[0], t_ev[0])
    with open(os.path.join(a.outdir, "raw_tss_imgs_ns_left.txt"), "w") as f:
        f.writelines("%d\n" % round(t * 1e3) for t in img_us)
    with open(os.path.join(a.outdir, "tss_imgs_us_left.txt"), "w") as f:
        f.writelines(f"{t - t0:.012f}\n" for t in img_us)
    for name, times in (("raw_gt_stamped_ns_left.txt", gt[:, 0]), ("gt_stamped_left.txt", gt[:, 0] - t0)):
        with open(os.path.join(a.outdir, name), "w") as f:
            for ts, pose in zip(times, gt[:, 1:8]):
                f.write(f"{ts} " + " ".join(str(v) for v in pose) + "\n")
    with open(os.path.join(a.outdir, "imu_data.csv"), "w") as f:
        f.write(IMU_HEADER)
        for r in imu:
            f.write(",".join(map(str, [int(round(r[0] * 1e3))] + list(r[1:7]))) + "\n")
    evs = np.stack([x.astype(np.float64), y.astype(np.float64), t_ev - t0, p.astype(np.float64)], 1)
    del x, y, p
    write_evs_arr_to_h5(evs, os.path.join(a.outdir, "evs_left.h5"))

    with h5py.File(os.path.join(a.outdir, "evs_left.h5"), "r") as f:      # read back
        n, tt = f["events/t"].shape[0], f["events/t"][-1]
        assert n == len(evs) and abs(tt - evs[-1, 2]) <= 1, "events written incorrectly"
    print("%s: %d events (t0 %.0f us, last %.3f s after t0), %d images from %s, %d IMU samples, %d ground-truth poses"
          % (a.seq, len(evs), t0, evs[-1, 2] / 1e6, len(img_us), img_src, len(imu), len(gt)))
    print("  undistorted intrinsics fx %.3f fy %.3f cx %.3f cy %.3f -> %s" % (K_new[0, 0], K_new[1, 1], K_new[0, 2],
                                                                           K_new[1, 2], a.outdir))


if __name__ == "__main__":
    main()
