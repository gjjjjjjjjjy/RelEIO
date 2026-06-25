# Reliability Benchmark for Learned Event-Inertial Odometry Under Stress

Artifact accompanying the paper *"Reliability Benchmarking of Learned
Event-Inertial Odometry Under Sensing and Computational Stress."* This repository
provides the per-run records, stress operators, and reproduction scripts for the
repeated-run reliability measurement protocol described in the paper.

> **Anonymized for review.** Author/affiliation information and raw datasets are
> intentionally omitted. Pretrained weights and datasets are not redistributed;
> download instructions and preprocessing scripts are referenced in the paper.

## Contents

```
records/            # per-run measurement records (one row per sequence x level x seed)
  deio_keep.csv     #   DEIO, event-density axis (160 runs)
  deio_time.csv     #   DEIO, timestamp-precision axis (160 runs)
  deio_patch.csv    #   DEIO, visual-budget axis (160 runs)
  devo.csv          #   DEVO (IMU-off ablation), event-density axis (45 runs)
  loopclosure.csv   #   DEIO + loop-closure variant, event-density axis (60 runs)
  rampvo.csv        #   RAMP-VO, event-density axis (15 runs)        [see note]
  e2vid_tartanvo.csv#   E2VID+TartanVO, event-density axis (27 runs) [see note]
stress_operators/   # the degradation operators (event density / timestamp / patch)
  event_stress.py   #   thinning + timestamp quantization, seeded by frame timestamp
  event_utils.py    #   voxel-grid construction
scripts/            # recompute tables/figures from records
  final_recompute.py
  make_final_figures.py
  make_paper_tables.py
  make_crosspipeline_table.py
```

## Record schema

Each row in `records/*.csv` is one independent run:

| column | meaning |
|---|---|
| `dataset`, `scene` | sequence identifier |
| `keep` / `tq_us` / `patch` | stress level on the relevant axis |
| `seed` | evaluation seed |
| `status` | `ok` / `crash` (numerical) ; catastrophic = `ok` with MPE > 100% |
| `mpe` | mean position error (%), blank on numerical crash |
| `stress_version`, `actual_keep`, `tq_changed`, `full_stream_hash` | per-run evidence: the realized degradation and a full-stream input hash, so each run's degradation is independently verifiable |

DEIO/DEVO/loop-closure report MPE (%); RAMP-VO and E2VID+TartanVO report
absolute trajectory error (ATE). Cross-system results compare **failure patterns
and repeated-run reliability, not absolute accuracy** across heterogeneous metrics.

## Three-class outcome

Every run is classified as **valid** (completes, MPE <= 100%),
**numerical-crash** (zero-norm quaternion / NaN), or **catastrophic-completion**
(completes but MPE > 100%, i.e. diverged). Failure = crash + catastrophic.

## Reproduce

```bash
python scripts/final_recompute.py     # three-axis merge -> headline numbers
python scripts/make_paper_tables.py   # regenerate tables
python scripts/make_final_figures.py  # regenerate figures
```

## Stress operators

Degradation is seeded by the **frame timestamp**, not the evaluation seed, so for
a fixed level the degraded input is **bit-identical across repeated runs**
(verified by `full_stream_hash`). This isolates degradation from estimator
variability. See `stress_operators/event_stress.py`.
