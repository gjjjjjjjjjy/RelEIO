# Per-run records

Every file is extracted from the run ledgers by the scripts in `script/diag/` and is not edited by hand.
One row is one run unless stated otherwise. Errors are the mean position error (MPE) in percent of the
ground-truth path length. `SHA256SUMS` lists the checksum of each file.

Common values:

- `arm`: `default` is the container default (the variable `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` is set);
  `no_tf32_override` is the same environment with that variable unset.
- `cell`: input condition. `clean`; `keep:0.1` event density 0.1; `tq:10000` timestamp quantization 10 ms;
  `patch:16` patch budget 16.
- `scene`: dataset and sequence. `vector/desk_normal1`, `vector/corner_slow1`, `hku/vicon_dark1`,
  `tumvie/mocap-3d-trans`, `fpv/indoor_forward_3`, `rpgdavis/boxes_6dof`, `rpgdavis/poster_6dof`,
  `rpgdavis/shapes_6dof` (`rpgdavis` is DAVIS240C).

## All rounds (1412 runs)

### `all_rounds_attribution_20261123.csv`: the DEIO runs of all rounds
Written by `script/diag/attribution_table.py` over all ledgers. Columns as in
`round4_all_attribution_*.csv` below; the rows of the main matrix are identical in both files.

| `ledger` begins with | Round | Runs |
|---|---|---|
| `precision_pilot_` | round 1 (pilot): `desk_normal1`, clean, arms `default` and `fp32_strict` (full FP32) | 48 |
| `round2A_` | round 2A: `desk_normal1`, clean, arms `default`, `amp_off` (mixed precision off), `tf32_off`, `fp32_strict` | 64 |
| `round2B_` | round 2B: the other seven sequences, clean, arms `default` and `fp32_strict` | 140 |
| `round2C_` | round 2C, the precision switches of Table III: `default`, `cudnn_tf32_off`, `no_tf32_override`, `matmul_tf32_off`, `ba_matmul_fp32` (TF32 off in the bundle-adjustment call only) | 80 |
| `round3a_` | round 3A: 14 conditions on five sequences, both arms, on the RTX 4090 machine | 280 |
| `hardware_l40s_`, `hardware_h200_`, `hardware_h100_` | replication on three further GPU models | 120 |
| `round4_` | round 4, the main matrix | 640 |
| `imu_bias_deio_` | DEIO's runs of the bias experiment (`gbias:5` is the biased condition) | 40 |

In this file the clean condition is labelled `base` in rounds 1-2, `keep:1.0` in round 3A and in the hardware
replication, and `clean` in the main matrix and the bias experiment. Further conditions of round 3A: `keep:0.3`,
`patch:32`, `patch:64`. The eight untraced reference runs of round 1 and the voxel-cache check are not part of
the extracted records.

## Main matrix (8 sequences x 4 input conditions x 2 arms x 10 seeds = 640 runs)

### `round4_all_attribution_20261121_023641.csv`: every run, 640 rows
Written by `script/diag/attribution_table.py`.

| Column | Meaning |
|---|---|
| `run_id` | identifier of the run in the ledger |
| `arm`, `cell`, `scene` | as above |
| `gpu` | index of the GPU on the machine (eight RTX 4090) |
| `outcome`, `category` | `completed`, or `numerical_failure:<site>` (a crash); `category` is the outcome without the site |
| `mpe` | error written by the estimator's own evaluation (rule A), empty for a crash |
| `wall_s` | wall time of the run in seconds, tracer included |
| `ba_tf32` | whether TF32 matrix multiplication was on inside the visual bundle adjustment, as read in the process |
| `first_where`, `first_tensor`, `first_frame` | checkpoint, tensor, and frame at which the tracer first detected a non-finite value |
| `first_imu_enabled`, `first_vi_initialized` | whether IMU factors were in use, and whether the visual-only initialization had completed, at that detection (the second name is historical: it does not refer to the visual-inertial alignment) |
| `next_detections` | the following detections |
| `ba_nonfinite_from_finite_inputs` | calls of the visual bundle adjustment that returned a non-finite output from finite input |
| `frames_completed`, `keyframes`, `vi_align_calls`, `vi_initialized` | progress of the run: frames, keyframes, calls of the visual-inertial alignment, and whether that alignment completed (IMU factors in use at the end) |
| `chol_flags` | number of Cholesky factorizations that reported a failure |
| `chol_first_where`, `chol_first_call`, `chol_first_info` | site, call index, and status code of the first such report |
| `chol_first_S_finite`, `chol_first_y_finite`, `chol_first_dX_finite` | whether the Schur complement, the right-hand side, and the solution were finite at that report |
| `chol_vs_detection` | whether the first Cholesky report came before (`chol_first`) or after (`detection_first`) the first detection in the log |
| `chol_info_values` | the status codes reported |
| `ledger` | name of the ledger the run belongs to |
| `peak_alloc_mb`, `peak_reserved_mb` | peak memory of the PyTorch allocator, in MiB (bytes / 2^20; the column names say mb for historical reasons) |
| `device_mem_mib` | largest device-memory sample of the process (every 30 s; a lower bound of the peak) |

`round4_main_all_details_20261121_023641.csv` holds the same runs without the last eight columns
(written by `script/diag/pilot_details.py`).

### `deio_se3_round4_steps_20261121.csv`: every completed run under four scoring rules, 592 rows
Written by `script/diag/deio_se3_eval.py` from the saved trajectories. The 48 crashed runs have no row.

| Column | Meaning |
|---|---|
| `run`, `arm`, `cell`, `scene`, `seed`, `ledger` | identification; `seed` is the run seed (2000-2009) |
| `deio_rule` | rule A, the rule of DEIO's evaluation code: Sim(3) fitted on the first 1000 poses (all poses on UZH-FPV), matching within 1 s |
| `sim3_1000_20ms` | rule B: rule A with matching within 20 ms |
| `sim3_all` | rule C: rule B fitted on all matched poses |
| `se3_all` | rule D, the main rule: rule C without scale (SE(3)) |
| `sim3_scale` | scale of the Sim(3) fit on all matched poses |
| `dist_ratio` | median ratio of the distance traveled between poses 1 s apart, estimate over ground truth (no alignment) |
| `gt_spread` | mean distance of the matched ground truth from its centroid, in percent of the path length (the score of an estimate that does not move) |
| `est_extent_m`, `gt_extent_m` | largest distance from the first pose, estimate and ground truth, in metres |
| `ledger_mpe` | the error the estimator wrote at run time (equals `mpe` above) |
| `score` | `ok` if the trajectory could be scored |

### Derived tables
- `round4_main_table_se3.csv`: one row per sequence, input condition and arm (64 rows): runs, crashes, completed
  runs, runs with a rule D error at most 100%, medians under rules D and A, runs with a fitted scale below 0.5.
  Written by `script/diag/round4_reliability.py`.
- `round4_main_table_20261121.csv`: one row per sequence and input condition (32 rows) with the failures of both
  arms and the Newcombe interval of their difference, as fixed before the run. Written by
  `script/diag/round4_analysis.py`.

### `round4_input_identity_20261121_from_cluster.txt`
Output of `script/diag/compare_inputs.py` on the ledger of the main matrix: the input fingerprints of the 24
distinct inputs, compared across arms and seeds.

### `deio_round4_scores_paper_rule_20261125.csv`: the same 592 runs with a further rigid rule
Written by `script/diag/deio_se3_eval.py`; `script/diag/compare_rescore.py` checks that every run and its rules A-D equal
`deio_se3_round4_steps_20261121.csv`. Added columns:

| Column | Meaning |
|---|---|
| `se3_first5s` | SE(3) fitted on the matched poses within 5 s of the first estimated pose, error over all matched poses |
| `matched_20ms`, `matched_first5s` | matched poses (20 ms), and those within the first 5 s |
| `estimates` | poses in the saved trajectory |

### `scoring_faults_20261125.csv`: known faults injected into the ground truth, 72 rows
Written by `script/diag/scoring_checks.py faults`. One row per sequence (`sequence`) and fault (`fault`: `none`,
`rigid` (yaw 30 deg, shift 1 m), `scale0.5`, `scale2`, `drift2`, `drift20` (linear from mid-time to 2% or 20% of the
path length), `diverge200` (exponential from mid-time to 200%), `stationary` (the first pose throughout), `freeze`
(the mid-time pose from then on)). `A`, `C`, `D`, `first5s`: MPE (% of the path length) under rules A, C, D and the
rigid first-5-s rule, empty if not computable; `CR_1`, `CR_10`: correct rate at 1% and 10%.

### `crt_round4_20261125.csv`: correct rate of every run of the main matrix, 640 rows
Written by `script/diag/scoring_checks.py crt`. CR follows OpenLORIS (Shi et al., ICRA 2020, Eq. 2) with one rigid
alignment of the whole trajectory and the position criterion only.

| Column | Meaning |
|---|---|
| `run`, `arm`, `cell`, `scene`, `seed`, `ledger`, `outcome` | identification and outcome |
| `crt_status` | `ok`, or `not computable (...)` for a crashed run without trajectory |
| `CR_1`, `CR_10` | correct rate at 1% and 10% of the path length: correct time (each pose at most 1 s) over the ground-truth time span |
| `CRT_1`, `CRT_10` | the same over the time span from the first estimate (Eq. 3) |
| `rule_A`, `rule_D` | MPE under rules A and D, as in `deio_se3_round4_steps_20261121.csv` |
| `aoe_median_deg_xyzw`, `aoe_median_deg_wxyz` | median orientation error after the rigid alignment, estimate quaternion read as x y z w (the file format) or w x y z |
| `offset_spread_deg_xyzw`, `offset_spread_deg_wxyz`, `offset_spread_deg` | median angle of the per-pose rotation between estimate and ground truth from its mean (`offset_spread_deg` = `_xyzw`) |
| `unmatched` | estimated poses without a ground-truth pose within 20 ms (counted as incorrect) |

## Comparison of DAVIS240C data pipelines (40 runs)

`davis_pipelines_scores_20261125_093426.csv`: columns as `deio_round4_scores_paper_rule_20261125.csv`. `scene` is
`rpgdavis_up/<sequence>` for DEIO's own preprocessing, entry script and configuration, and `rpgdavis/<sequence>` for
our data-loading layer (as in the main matrix); `boxes_6dof` and `poster_6dof`, clean input, override removed, seeds
2000-2009, one RTX 4090. `davis_pipelines_details_20261125_093426.csv`: the same runs with the columns of
`round4_main_all_details_20261121_023641.csv` (written by `script/diag/pilot_details.py`).

## Replication on three further GPU models (40 runs each)

`hardware_l40s_*`, `hardware_h200_*`, `hardware_h100_*_details_*.csv`: VECtor `corner_slow1` under event density
0.1 and on clean input, ten runs per input condition and arm. Columns as in the main matrix without the last eight;
these files carry no `cell` column. The same runs with their input condition are in
`all_rounds_attribution_20261123.csv`.

## Common gyroscope bias (80 runs)

`imu_bias_cross_system_pinned_20261121.csv`, written by `script/diag/cross_system_eval.py`.

| Column | Meaning |
|---|---|
| `system` | `DEIO` (override removed) or `SuperEIO` (upstream configuration, CPU-pinned batch) |
| `seq` | `boxes_6dof` or `poster_6dof` (DAVIS240C) |
| `cond` | `clean`, or `gbias`: a constant gyroscope bias of (+5, -5, +5) deg/s |
| `rep` | run seed (DEIO) or repetition index (SuperEIO) |
| `run`, `status` | identifier and outcome of the run |
| `se3_W` | rule D error on the time window common to both systems |
| `se3_W_first5s` | the same with the alignment fitted on the first 5 s of the window |
| `sim3_W`, `sim3_W_scale` | error and scale of a Sim(3) fit on the window |
| `dist_ratio_1s` | distance ratio over 1 s, as above |
| `matched` | number of matched poses |

## Three readers of the same perturbed recordings (round 5; 260 runs and attempts)

DAVIS240C `boxes_6dof` and `poster_6dof`. Each perturbation was applied once to the whole event stream of the text
release (`utils/r5_ops.py`), and the same events were given to three readers: DEIO's own DAVIS240C pipeline (input
files rebuilt from each perturbed copy), DEIO through our data-loading layer, and SuperEIO (rewritten rosbags).
DEIO: override removed, seeds 4000-4009, eight RTX 4090 GPUs. SuperEIO: its authors' configuration, eight parallel
instances with 16 logical CPUs each, eight RTX 4090 GPUs.

`r5_eval_20261126.csv`, written by `script/diag/r5_eval.py`: one row per DEIO run or SuperEIO attempt.

| Column | Meaning |
|---|---|
| `reader` | `deio_own` (DEIO's own pipeline), `deio_ours` (our data-loading layer), `supereio` |
| `seq` | `boxes_6dof` or `poster_6dof` |
| `cond` | `clean`; `keep0.1` event density 0.1; `tq10000us` timestamp quantization 10 ms; `noise5hz` background noise of 5 Hz per pixel; `imu15000us` IMU timestamps shifted by +15 ms (the last two for `deio_own` and `supereio` only) |
| `rep` | run seed (DEIO) or attempt index 1-10 (SuperEIO) |
| `run` | run identifier (DEIO) or run folder (SuperEIO) |
| `status` | `completed`; `failed` (DEIO: numerical failure; SuperEIO: fewer than 10 poses, a restart, or a "system reboot"); `undetermined` (SuperEIO: `run_seq.sh` or `rosbag play` exit code not 0, or a ROS process that died) |
| `D`, `C` | error on the time window common to the three readers, rigid (SE(3)) and similarity (Sim(3)) alignment fitted on the same matched poses (20 ms matching) |
| `A` | DEIO only: DEIO's evaluation code on the whole trajectory |
| `scale` | scale of the similarity fit |
| `matched_W`, `path_W_m` | matched poses in the window, ground-truth path length in the window (m) |
| `D_full` | rigid alignment on the whole trajectory |

`D` and `C` of an undetermined SuperEIO attempt that left a trajectory are given for information; such attempts are
not counted as completed. `r5_eval_20261126_readouts.json`: the common window of each sequence (bag time in
microseconds, and the runs that set its two ends), the offset that moves each DEIO reader's ground-truth time to the
bag's, and the readouts per reader, sequence, and condition.

`r5_supereio_main_20261126_152538.json`, written by `script/supereio/r5_accept.py`: the acceptance record of the
SuperEIO batch. Per attempt: outcome, the identity read from its summary (sequence, tag, run folder, variant, rosbag;
paths relative to the repository), CPU set and its check, hash of the TensorRT engines and its check, solves stopped
by the time limit at the 700th solve, and the rule D error of the whole run. For the batch: validity, the scheduling
check, and whether the engines are those of the bias experiment.

`r5_deio_scores_20261126_152156.csv` and `r5_deio_details_20261126_152156.csv`: the 160 DEIO runs of the RTX 4090
batch, with the columns of `davis_pipelines_scores_*` and `davis_pipelines_details_*`; `scene` is
`r5up/<seq>__<cond>` (DEIO's own pipeline) or `r5ours/<seq>__<cond>` (our layer). `..._20261126_132849.csv`: the same
plan run on eight L40S GPUs (160 runs), kept as a record and not used in the paper's numbers.

`r5_evidence/`: identity of the input across the three readers.

| File | Content |
|---|---|
| `copy_<seq>__<cond>.json` (10) | the perturbed copy (`script/diag/r5_make_inputs.py`): hash of the source, per-second digest (number of events and SHA-256 of the sorted events) of the source and of the output. The first events of the recording, raw data of the dataset, are omitted here (`first_events_omitted`) |
| `<seq>__r5_<cond>.json` (10) | the rosbag check (`script/supereio/make_r5_bag.py`): time offset and time mapping, IMU check, per-second identity with the copy, message-by-message comparison of the rewritten rosbag with the original |
| `check_upstream_<seq>__<cond>_*.log` (10) | DEIO's own input files read back and compared with the copy (`script/diag/r5_check_upstream.py`) |
| `check_windows_<seq>_*.log` (2) | the per-window perturbation of the main matrix and the whole-stream perturbation compared over the windows of our layer (`script/diag/r5_check_windows.py`): events whose window membership changes, as a share of the events that lie in a window before or after quantization |
