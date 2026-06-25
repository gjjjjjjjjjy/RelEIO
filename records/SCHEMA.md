# Record schema

Each row in `records/*.csv` is one independent run.

| file | system | axis | metric |
|---|---|---|---|
| `deio_keep.csv` / `deio_time.csv` / `deio_patch.csv` | DEIO (event+IMU, primary) | event density / timestamp quant. / visual budget | MPE % |
| `devo.csv` | DEVO (event-only, IMU-off ablation) | event density | MPE % |
| `loopclosure.csv` | DEIO + loop-closure backend (config variant) | event density | MPE % |
| `rampvo.csv` | RAMP-VO (event+frame, independently trained) | event density | ATE m |
| `e2vid_tartanvo.csv` | E2VID + TartanVO (reconstruction+VO, non-DPVO) | event density | ATE m |

| column | meaning |
|---|---|
| `dataset`, `scene` | sequence identifier |
| `pipeline` | (DEVO/loop-closure only) configuration tag |
| `keep` / `tq_us` / `patch` | stress level on the relevant axis |
| `seed` | evaluation seed (DEIO/DEVO/RAMP-VO) |
| `rep` | repeat index (E2VID+TartanVO, which has no evaluation seed) |
| `status` | `ok` / `crash` (numerical: zero-norm quaternion / NaN) |
| `mpe` | mean position error (%); blank on numerical crash; `ok` with mpe>100% = catastrophic completion |
| `ate` | (RAMP-VO / E2VID+TartanVO) absolute trajectory error (m), Sim(3)-aligned; **blank on crash** (no placeholder value) |
| `matched` | (E2VID+TartanVO) number of timestamp-associated pose pairs used for ATE |
| `stress_version` | degradation operator version (evidence) |
| `req_keep`, `req_tq_us` | requested stress level |
| `n_in`, `n_out`, `actual_keep` | events before/after thinning, realized keep ratio (evidence) |
| `tq_changed` | timestamps changed by quantization (evidence) |
| `full_stream_hash` | rolling hash of the degraded event stream: identical across seeds at a fixed level, different across levels |

**Three-class outcome.** valid = completes, mpe<=100%; numerical-crash =
zero-norm/NaN; catastrophic-completion = completes but mpe>100% (diverged).
Failure = crash + catastrophic.

**Metrics differ by system.** DEIO/DEVO/loop-closure report MPE (%); RAMP-VO and
E2VID+TartanVO report ATE (m). Cross-system comparison is on failure patterns and
repeated-run reliability, **not** absolute accuracy across heterogeneous metrics.
