# Released record schema and interpretation

Each row is a recorded run. Original CSVs are preserved.

| File | System | Axis | Error |
|---|---|---|---|
| deio_keep/time/patch.csv | DEIO | density / time quantization / patch count | MPE % |
| devo.csv | DEIO IMU-off ablation | density | MPE % |
| loopclosure.csv | DEIO loop-closure variant | density | MPE % |
| rampvo.csv | RAMP-VO, event + frame | density | ATE m |
| e2vid_tartanvo.csv | reconstruction + VO | density | ATE m |

Identifiers are dataset, scene, axis level and seed (rep for E2VID+TartanVO).
status is ok or crash. A completed row requires a finite nonnegative error; a
crash row has an empty error. Unknown states and duplicate identifiers raise.
A missing completed-run metric is a malformed record, not an inferred crash.

For MPE records, the operational outcomes are:
- numerical crash;
- completed with MPE > 100%;
- completed with MPE <= 100%.

The historical words catastrophic and valid refer to the last two threshold
categories. They do not establish mission failure or navigation acceptability.
Error medians, worst values and IQR use the within-threshold completed subset;
its size is reported. No qualifying observations yields an undefined statistic,
represented by null in JSON or an empty CSV field. Failure intervals use the
Wilson method with all runs as denominator.

Cross-system comparisons use numerical crashes only, on each pair's common
sequence/keep cells. MPE and ATE retain their units. No common ATE catastrophic
threshold or complete-trajectory coverage criterion is available in these CSVs.

The analysis writes Q1/Q2/Q3 definitions to reports/statistics.json.
See docs/INTEGRATION.md for their interpretation. Q3 ties are exact ties at the
precision stored in the CSV; original higher-precision outputs are unavailable.
The empirical success curve counts all runs in the denominator; status=ok is
the released completion indicator, not independently verified output coverage.

## Input evidence fields

stress_version, req_keep, req_tq_us describe the requested operator.
n_in/n_out/actual_keep and tq_changed/tq_total/tq_changed_ratio describe recorded
event counts. Blank fields remain missing evidence.

first_window_checksum and full_stream_hash retain historical values. The
available historical implementation aggregates window counts/sums/extrema.
Equal summaries do not establish byte-identical event streams. Blank fields indicate missing evidence.
These fields cannot locate a numerical divergence within the pipeline.

## Available and missing records

There are 480 primary runs, 60 loop-closure runs, 45 IMU-off runs, 15 RAMP-VO
runs and 27 E2VID+TartanVO rows. Runtime, GPU peak memory,
complete output trajectories and failure timestamps are absent from these CSVs.

The schema documents evidence in this release. An execution.csv produced by the
new matrix runner is a process log and must not be silently relabeled as this
scientific per-run dataset.
