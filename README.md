# RelEIO: reliability measurement of learned event-inertial odometry

Per-run records, analysis scripts, and protocol code for

> **Reliability Measurement of Learned Event-Inertial Odometry: Numerical, Runtime, and Scoring Effects Beyond Sensing Degradation**
> Jiaying Guo, Zhengjie Wang, Wenxing Long, and Jun Ying (corresponding author: junying@shnu.edu.cn)

This is the artifact of the revised manuscript (revision of TIM-26-07641, submitted to IEEE Transactions on
Instrumentation and Measurement). The artifact of the first submission is kept unchanged in
[`first_submission/`](first_submission/) and at the tag `first-submission`; the revised paper does not use its records.

## Check the main results

Python 3.9 or later, standard library only:

```bash
python3 script/headline_numbers.py
```

The script recomputes 48 quantities from the records and prints each next to the value stated in the paper; it exits
with status 1 if any differs. Among them:

| Result | Value | Records |
|---|---|---|
| Runs of DEIO that crashed with the container's TF32 override / with the override removed | 48/320 / 0/320 | `round4_all_attribution_*.csv` |
| Override-removed runs within 10% of the path length under the rule of DEIO's evaluation code (rule A) / a rigid alignment (rule D) | 313 / 227 of 320 | `deio_se3_round4_steps_*.csv` |
| Runs that pass rule A and fail rule D at 10% / the reverse | 86 / 0 | same |
| Runs above the path length under rules A, C, D | 3, 0, 69 | same |
| Crashes with / without the override on three further GPU models (L40S, H200, H100) | 10/20, 10/20, 11/20 / 0/20 each | `hardware_*_details_*.csv` |

## Regenerate the figures and tables

NumPy and Matplotlib are needed. Outputs are written to `output/`.

```bash
python3 script/make_fig1_protocol.py     # Fig. 1 (schematic)
python3 script/diag/round4_reliability.py \
    --outcomes records/round4_all_attribution_20261121_023641.csv \
    --scores records/deio_se3_round4_steps_20261121.csv --outdir output     # Fig. 2, per-condition table
python3 script/make_main_figures.py      # Fig. 3, Fig. 4, Fig. S1
python3 script/make_supp_figures.py      # Fig. S2, Fig. S3
python3 script/diag/make_supp_tables.py  # Tables S4-S8 and S10, and the numbers the text quotes from them
```

| Paper | File written | Source |
|---|---|---|
| Fig. 1 | `output/figures/Fig1_protocol.pdf` | drawn by the script; not a data figure |
| Fig. 2 | `output/figs/round4_reliability.pdf` | main matrix, all 640 runs |
| Fig. 3 | `output/figures/Fig4_scoring.pdf` | 320 override-removed runs, rules A and D |
| Fig. 4 | `output/figures/Fig5_bias.pdf` | gyroscope-bias runs of DEIO and SuperEIO |
| Fig. S1 | `output/figures/Fig2_numerics.pdf` | main matrix; panel (b) see *Coverage* |
| Fig. S2, S3 | `output/figures/FigS1_*.pdf`, `FigS2_*.pdf` | main matrix |
| Table IV | printed by `script/headline_numbers.py` | main matrix |
| Tables S4-S8, S10 | `output/tables/supp_*.tex` | main matrix, bias runs |

The file names carry the figure numbers of an earlier draft; the table above gives the numbers of the paper.

## Contents

| Path | Content |
|---|---|
| `records/` | per-run records of the revised paper; columns in [`records/SCHEMA.md`](records/SCHEMA.md), checksums in `records/SHA256SUMS` |
| `script/headline_numbers.py`, `script/make_*.py`, `script/diag/round4_reliability.py`, `script/diag/make_supp_tables.py` | analysis: numbers, figures, supplementary tables |
| `script/diag/` | protocol code: run ledger (`run_ledger.py`), launchers of the precision arms and of the main matrix (`precision_pilot.py`, `precision_matrix.py`, `run_round*.sh`, `run_hardware.sh`, `run_imu_bias.sh`), control verification and input comparison (`compare_inputs.py`, `check_env.py`, `check_imu_units.py`), extraction of the records from the ledgers (`attribution_table.py`, `pilot_details.py`), and scoring under rules A-D (`deio_se3_eval.py`, `cross_system_eval.py`) |
| `script/run_*_sweep.py`, `script/stress_evidence.py` | runners that the launchers call, and the check of the realized degradation of each run |
| `utils/sensor_stress.py`, [`docs/SENSOR_STRESS_OPERATORS.md`](docs/SENSOR_STRESS_OPERATORS.md) | degradation operators (event density, timestamp quantization, gyroscope bias) and their definition |
| `utils/numtrace.py` | numerical tracer: first non-finite tensor, status of each Cholesky factorization |
| `script/supereio/` | build, launch, and evaluation scripts for the SuperEIO port, and the bias-bag generator |
| `revision/plans/` | run plans of the main matrix, frozen before execution |
| `tests/` | unit tests of the operators, the ledger, the launch plans, and the tracer |
| `first_submission/` | artifact of the first submission (records of DEIO, DEVO, RAMP-VO, E2VID+TartanVO; tools; operators) |

## Coverage

What can be recomputed from this repository alone, and what cannot:

- **From the records here:** every number of the main matrix (crashes, reliability under rules A-D, Table IV, the
  paired comparison of the arms, where the crashes start), the supplementary tables S4-S8 and S10, Figs. 2-4 and
  S1(a)-S3, the totals per GPU and arm of the replication on three further GPU models, and the gyroscope-bias
  results.
- **Not included in this release:** the raw run ledgers (one directory per run with the request, the recorded
  environment, the input fingerprints, the tracer output, the log, and the trajectory), from which the records
  were extracted with the scripts above; and the per-run records of the diagnostic rounds, of the precision-switch
  matrix (Table III), of round 3A on the RTX 4090, and of the single SuperEIO runs (Tables S1-S3 and S9). Those
  tables report counts taken from the ledgers. The split of each GPU's failures into the two input conditions in
  Fig. S1(b) and Table S3 comes from the same ledgers; the per-GPU records here carry no condition label, so the
  figure script holds that split as constants and checks only the totals against the records.
- **The estimator is not included.** The launchers run inside a checkout of DEIO with our data-loading layer and
  with the probe hooks that call `utils/numtrace.py`; the supplementary material (Section S2) describes its
  differences from the upstream code. Network weights and datasets are not redistributed.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests of the operators, the ledger, and the launch plans need NumPy. `tests/test_numtrace.py` needs PyTorch,
and one test of the voxel-cache namespace is skipped outside an estimator checkout.

## License

MIT; see [`LICENSE`](LICENSE).
