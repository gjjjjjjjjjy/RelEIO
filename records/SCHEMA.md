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
