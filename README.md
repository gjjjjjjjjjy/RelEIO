# RelEIO: A Reliability Benchmark for Learned Event-Inertial Odometry Under Stress

Artifact for the paper *"Reliability Benchmarking of Learned Event-Inertial
Odometry Under Sensing and Computational Stress."*

Learned event(-inertial) odometry is usually evaluated by a single trajectory
error on a clean benchmark sequence. **RelEIO** instead measures *reliability*:
it runs each system repeatedly under controlled sensing/computational
degradation and reports a three-class outcome distribution with confidence
intervals, exposing failures that a single run hides.

> **Anonymized for review.** Author/affiliation information is omitted. Pretrained
> weights and raw datasets are not redistributed; see `docs/INTEGRATION.md` for
> how to obtain each system and dataset.

## What's here

```
records/             per-run measurement records (the benchmark data)
  *.csv              one row per (sequence, stress level, seed); see SCHEMA.md
  SCHEMA.md          column definitions + three-class outcome + metric note
stress_operators/    the controlled degradation operators (the protocol core)
  event_stress.py    event thinning + timestamp quantization (seeded by frame ts)
  event_utils.py     event voxel-grid construction
scripts/             per-system run scripts (.sh)
  run_deio.sh  run_devo.sh  run_rampvo.sh  run_e2vid_tartanvo.sh
tools/               analysis (.py): recompute the paper's tables/figures from records
docs/
  INTEGRATION.md     how to obtain and wire each of the four systems (reproducible)
```

## The protocol in one paragraph

Three orthogonal **stress axes** perturb the input or compute budget: event
density (keep a fraction of events), timestamp precision (quantize timestamps),
and visual budget (number of tracked patches). Each run is classified
**valid / numerical-crash / catastrophic-completion**. Every condition cell is
run with `K>=5` seeds as independent processes; we report the three outcome counts
with a Wilson confidence interval, plus median/worst-case/IQR over valid runs.
Crucially, the degradation is **seeded by the frame timestamp, not the evaluation
seed**, so the degraded input is bit-identical across repeated runs (verified by
`full_stream_hash`) --- isolating degradation from estimator variability.

## Reproduce the paper numbers from the released records

No GPU needed --- the records are provided.

```bash
python tools/final_recompute.py        # headline numbers (three axes)
python tools/make_paper_tables.py       # tables
python tools/make_final_figures.py      # figures
python tools/make_crosspipeline_table.py
```

## Reproduce the runs (GPU)

Each system must apply the RelEIO stress operators to the raw events before
building its own representation. See **`docs/INTEGRATION.md`** for environment
setup, known pitfalls, and the stress hook per system, then:

```bash
export DEIO_ROOT=... DATA_ROOT=...      # see INTEGRATION.md for each system's vars
bash scripts/run_deio.sh  <sequences>
bash scripts/run_devo.sh  <sequences>
bash scripts/run_rampvo.sh <sequences>
bash scripts/run_e2vid_tartanvo.sh <sequences>
```

## Systems evaluated

| System | Lineage | Modality | Metric |
|---|---|---|---|
| DEIO (primary) | patch-tracking | event + IMU | MPE % |
| DEVO (IMU-off ablation) | patch-tracking | event-only | MPE % |
| RAMP-VO | patch-tracking | event + frame | ATE m |
| E2VID + TartanVO | reconstruction + VO (non-DPVO) | event -> frames | ATE m |

Cross-system results compare **failure patterns and repeated-run reliability**,
not absolute accuracy across these heterogeneous metrics.

## License

MIT (see `LICENSE`).
