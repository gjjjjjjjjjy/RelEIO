# RelEIO: reliability measurement of learned event-inertial odometry

RelEIO is a repeated-run protocol for measuring how reliably a learned event-inertial odometry (EIO) system
delivers a usable trajectory, together with the per-run records of a study of two systems (DEIO and SuperEIO) and
the code that produced and analyzes them. It accompanies the paper

> **Reliability Measurement of Learned Event-Inertial Odometry: Numerical, Runtime, and Scoring Effects Beyond Sensing Degradation**
> Jiaying Guo, Zhengjie Wang, Wenxing Long, and Jun Ying (corresponding author: junying@shnu.edu.cn)

The protocol treats a reported trajectory error as a measurement result whose conditions must be stated:

- every run is recorded with its computing environment, precision flags, and input fingerprints;
- outcomes (completed, numerical failure, undetermined) are classified from how a run ends (exit status, exception,
  whether the estimator wrote a finite result), not from the size of its error;
- completed runs are scored under a stated rule, and reliability is the share of *all* runs that complete within an
  error tolerance, so crashes and inaccurate completions share one denominator.

## Check the main results

Python 3.9 or later, standard library only:

```bash
python3 script/headline_numbers.py
```

The script recomputes 99 quantities from the records and prints each next to the value stated in the paper; it exits
with status 1 if any differs. Among them:

| Result | Value | Records |
|---|---|---|
| Runs of DEIO that crashed with the container's TF32 override / with the override removed | 48/320 / 0/320 | `round4_all_attribution_*.csv` |
| Override-removed runs within 10% of the path length under the rule of DEIO's evaluation code (rule A) / a rigid alignment (rule D) | 313 / 227 of 320 | `deio_se3_round4_steps_*.csv` |
| Runs that pass rule A and fail rule D at 10% / the reverse | 86 / 0 | same |
| Runs above the path length under rules A, C, D | 3, 0, 69 | same |
| Precision switches (Table III): crashes with TF32 matrix multiplication on / off inside the bundle adjustment | 8/16, 8/16 / 0/16, 0/16, 0/16 | `all_rounds_attribution_*.csv` |
| Four GPU models under reduced event density / clean input, with the override (Table S3) | 33/40 / 7/40; 0/40 without it | same |

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

File names do not follow the figure numbers of the paper; this table gives the correspondence.

| Paper | File written | Source |
|---|---|---|
| Fig. 1 | `output/figures/Fig1_protocol.pdf` | drawn by the script; not a data figure |
| Fig. 2 | `output/figs/round4_reliability.pdf` | main matrix, all 640 runs |
| Fig. 3 | `output/figures/Fig4_scoring.pdf` | 320 override-removed runs, rules A and D |
| Fig. 4 | `output/figures/Fig5_bias.pdf` | gyroscope-bias runs of DEIO and SuperEIO |
| Fig. S1 | `output/figures/Fig2_numerics.pdf` | main matrix; panel (b): round 3A and the three GPU-replication ledgers |
| Fig. S2, S3 | `output/figures/FigS1_*.pdf`, `FigS2_*.pdf` | main matrix |
| Tables III, IV, S1-S3 | printed and checked by `script/headline_numbers.py` | all rounds |
| Tables S4-S8, S10 | `output/tables/supp_*.tex` | main matrix, GPU replication, bias runs |

## Contents

| Path | Content |
|---|---|
| `records/` | per-run records; columns in [`records/SCHEMA.md`](records/SCHEMA.md), checksums in `records/SHA256SUMS` |
| `script/headline_numbers.py`, `script/make_*.py`, `script/diag/round4_reliability.py`, `script/diag/make_supp_tables.py` | analysis: numbers, figures, supplementary tables |
| `script/diag/` | protocol code: run ledger (`run_ledger.py`), launchers of the precision arms and of the main matrix (`precision_pilot.py`, `precision_matrix.py`, `run_round*.sh`, `run_hardware.sh`, `run_imu_bias.sh`), control verification and input comparison (`compare_inputs.py`, `check_env.py`, `check_imu_units.py`), extraction of the records from the ledgers (`attribution_table.py`, `pilot_details.py`), scoring under rules A-D (`deio_se3_eval.py`, `cross_system_eval.py`), and the run plans of the main matrix, fixed before execution (`plans/`) |
| `script/run_*_sweep.py`, `script/stress_evidence.py` | runners that the launchers call, and the check of the realized degradation of each run |
| `utils/sensor_stress.py`, [`docs/SENSOR_STRESS_OPERATORS.md`](docs/SENSOR_STRESS_OPERATORS.md) | degradation operators and their definition |
| `utils/numtrace.py` | numerical tracer: first non-finite tensor, status of each Cholesky factorization |
| `script/supereio/` | build, launch, and evaluation scripts for the SuperEIO port, and the bias-bag generator |
| `tests/` | unit tests of the operators, the ledger, the launch plans, and the tracer |

## Coverage

What can be recomputed from this repository alone, and what cannot:

- **From the records here:** 1412 DEIO runs in ten ledgers (the diagnostic rounds, the
  precision-switch matrix, round 3A, the main matrix, the replication on three further GPU models, and DEIO's
  side of the bias experiment) and the SuperEIO runs of the bias experiment. This covers the crash counts of
  every round and arm (Tables III and S1-S3, apart from the two checks named below), every number of the main matrix (reliability under rules A-D,
  Table IV, the paired comparison of the arms, where the crashes start), the supplementary tables S4-S8 and S10,
  and all figures.
- **Not included:** the raw run ledgers (one directory per run with the request, the recorded environment, the
  input fingerprints, the tracer output, the log, and the trajectory), from which the records were extracted with
  the scripts above; two small checks of Table S1 that are outside the extracted records (the eight untraced runs
  of round 1 and the voxel-cache check); and the logs of the single SuperEIO runs on clean input, which Table S9
  summarizes.
- **The estimator is not included.** The launchers run inside a checkout of DEIO with our data-loading layer and
  with the probe hooks that call `utils/numtrace.py`; the supplementary material of the paper describes its
  differences from the upstream code. In the scripts, `FreqVIO` is the name of that checkout and `FREQVIO_*` the
  prefix of its environment switches. Network weights and datasets are not redistributed.

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests of the operators, the ledger, and the launch plans need NumPy. `tests/test_numtrace.py` needs PyTorch,
and one test of the voxel-cache namespace is skipped outside an estimator checkout.

## License

MIT; see [`LICENSE`](LICENSE).
