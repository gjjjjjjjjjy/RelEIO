# RelEIO: repeated-run evaluation under sensing and computational stress

Code and analysis tools for “Reliability Benchmarking of Learned
Event-Inertial Odometry Under Sensing and Computational Stress”.

Jiaying Guo, Zhengjie Wang, Wenxing Long, and Jun Ying
(corresponding author: junying@shnu.edu.cn).

Jiaying Guo, Wenxing Long and Jun Ying: College of Information, Mechanical and
Electrical Engineering, Shanghai Normal University. Zhengjie Wang: School of
Engineering, Westlake University, and Shanghai Innovation Institute, DELTA Lab.

The released primary records contain 480 DEIO runs in 96 conditions, with five
runs per condition. They describe execution outcomes of the evaluated
implementation. Statistical tools are standard; architectural causes and
application-level reliability require additional evidence.

## Run locally

From the repository root, using Python 3.9 or later:

~~~bash
python tools/final_recompute.py
python tools/make_paper_tables.py
python tools/make_crosspipeline_table.py
python tools/make_final_figures.py
~~~

The first three commands use the Python standard library. Figures require NumPy
and Matplotlib. Input and output defaults are repository-relative, independent of
the shell working directory. Each command also accepts explicit --records and
--out paths. A missing or malformed input raises an error.

Results go to reports/: statistics.json, condition/threshold/success-curve CSVs,
matched-system CSVs, LaTeX tables, and revised PDF/SVG figures. Figure generators
cover the revised Fig. 1–4, S5–S6 and an empirical success curve. Other original
supplementary figures have no generator here.

[运行手册](docs/INTEGRATION.md) includes verified CPU commands,
the explicit GPU adapter interface, and the external dependencies needed for model inference. [Record definitions](records/SCHEMA.md) states each statistic's population
and limitations.

## GPU execution and tests

The original external model checkouts, custom adapters, raw datasets, weights and
full run logs are absent from this checkout. The matrix runner is tested with
controlled processes; actual GPU models have not been rerun during this revision.

~~~bash
python -m unittest discover -s tests -v
~~~

Operator tests require NumPy, PyTorch, h5py and Numba. The matrix runner records
every process's command, exit code, stdout, stderr and wall time. A process failure
is recorded and other independent jobs continue; the overall command exits
nonzero if any process failed. Exit codes require diagnosis before being assigned
a scientific outcome.

## Systems and source coverage

| Records | Interpretation |
|---|---|
| deio_keep/time/patch.csv | 480 primary DEIO runs |
| loopclosure.csv | 60 DEIO configuration-variant runs |
| devo.csv | 45 IMU-off runs, sharing the DEIO front-end |
| rampvo.csv | 15 event+frame runs |
| e2vid_tartanvo.csv | 27 reconstruction+VO records, three repeats per cell |

Pretrained weights and raw datasets are not redistributed. License: MIT.
