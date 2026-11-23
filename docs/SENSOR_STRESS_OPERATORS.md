# Sensor-stress operators (`utils/sensor_stress.py`, version `ss-v2`)

> `ss-v2` adds §4 (event background noise). The `ss-v1` operators are unchanged: same draws, same outputs (their tests and golden values did not change).

There are three families of physically meaningful sensing disturbances: IMU bias/noise, event–IMU
time offset, and burst missing measurements. **Status: standalone module.** It is not wired into
any `script/eval_deio/*` runner yet. Tests: `python3 -m unittest tests.test_sensor_stress -v`.
All `file:line` references point to commit `4f70c7b`. The working-tree copies of `devo/dba.py` and
`utils/eval_utils.py` were being edited by a separate diagnosis, so their lines may drift.

## Input formats

**IMU**: a `float64` ndarray `[N, 7]`, in the form the dataset scripts hand to
`utils/eval_utils.run_DEIO2`. That is the array *before* `run_DEIO2` runs `_all_imu[:, 0] /= 1e6`
in place (`eval_utils.py:62`).

| col | quantity | unit |
|-----|----------|------|
| 0   | timestamp | µs, non-decreasing |
| 1:4 | gyro x,y,z | **deg/s** (`devo/dba.py:848,856,860,866,1219,1225,1231` apply `[1:4]/180*math.pi`) |
| 4:7 | accel x,y,z | m/s² (passed unchanged to `append_imu`, `devo/multi_sensor.py:86`) |

How each loader produces deg/s (checked in code only; no data files are present locally):

| dataset | loader | gyro in file | conversion | time base |
|---|---|---|---|---|
| VECtor | `script/eval_deio/vector.py:24-31` | deg/s (written by `script/pp_data/pp_vector.py:74`, rad/s→deg/s) | none | absolute Unix µs (`pp_vector.py:73`, s×1e6, no t0 subtraction) |
| TUM-VIE | `tumvie.py:75-90` | rad/s (comment `:82`) | ×180/π at `:83` | inferred from magnitude by `_to_us` (`:42-67`), then `-time_offset_us` and `t>0` filter (`:77-78`) |
| HKU | `hku.py:24-31` | deg/s (written by `pp_hku_rosbags.py:176`) | none | relative to bag t0 (`pp_hku_rosbags.py:137-139`) |
| DAVIS240C | `davis240c.py:169-180` | rad/s (csv header, `pp_davis240c.py:43`) | ×180/π at `:177` | ns→µs `:171`, `-t_offset_us` `:173`, `t>0` filter `:175` |
| UZH-FPV | `uzh-fpv.py:180-191` | rad/s (raw `imu.txt`, columns 1..7) | ×180/π at `:188` | s→µs `:182`, `-t0_us` `:184`, `t>0` filter `:186` |
| MVSEC | `mvsec.py:46-58` | rad/s | ×180/π at `:54` | µs |

Every loader therefore produces deg/s by the time `run_DEIO2` runs. The bias/noise operator takes
`gyro_unit="deg/s"` (the default) or `"rad/s"` for other uses. Validation rejects any array that is
not float64 `[N,7]`, is non-finite, is unsorted, or whose median sample interval falls outside
10 µs..1 s. The last check catches an array already divided by 1e6.

**Events**: a dict of equal-length 1-D arrays `t` (µs, int or float, non-decreasing), `x`, `y`, `p`.
This is the form that `utils/event_stress.apply_event_stress` acts on (`load_utils.py:190`). Other
keys pass through by reference.

## Determinism and seeding

- Each random component uses `numpy.random.Generator(PCG64(seed))`. The seed is
  `seed = derive_seed(component, params, sequence_id, stress_seed)`, taken as the first 8 bytes
  (big-endian) of the SHA-256 of canonical JSON (sorted keys, all numbers as float).
- `stress_seed` is an explicit protocol constant (an int ≥ 0). It is *independent of the
  estimator's evaluation seed*. The module never touches `np.random.*` global state (tested), so the
  degraded input is identical across repeated runs of a condition.
- Bit identity holds across calls and processes (tested). It was also checked across numpy 1.26.4
  and 2.0.2. `numpy_version` is recorded so cross-environment differences can be detected.
- Zero/no-op parameters return a copy that is `np.array_equal` to the input with the same dtype and
  hash. No operator mutates its input.

## Evidence (every applier returns `(output, evidence)`; evidence is plain JSON)

Common fields: `operator`, `module_version`, `numpy_version`, `sequence_id`, `stress_seed`,
`derived_seeds` (component → int; `{}` when deterministic), `rng`, `requested`
(each entry is `{value, unit, ...}`), `realized`, `input_format`, `input_sha256`, `output_sha256`
(the SHA-256 of `np.ascontiguousarray(a).tobytes()`; for events, of the t,x,y,p bytes in that
order), and `data_changed`. `format_evidence_line(ev)` gives a `[sensor-stress] k=v ...` line. That
line parses with the same regex as `script/stress_evidence.py`.

---

## 1. IMU bias and white noise: `apply_imu_bias_noise`

- **Physical meaning.** Two error terms in the inertial sensor output. The first is a constant
  turn-on bias, such as an uncalibrated or temperature offset. The second is broadband white
  measurement noise (angle/velocity random walk).
- **Acts on.** IMU columns 1:4 and/or 4:7. Timestamps are never touched. A column changes only if
  its sensor has a nonzero perturbation.
- **Parameters.**
  - `gyro_bias_rad_s` (rad/s) and `accel_bias_m_s2` (m/s²). A **scalar ≥ 0** is a magnitude along a
    deterministic random unit direction (`direction_mode="random_unit_direction"`). That direction
    is seeded only by `(sequence_id, stress_seed)`, so a magnitude sweep keeps one direction. A
    **3-vector** is an explicit signed bias (`"explicit_vector"`).
  - `gyro_noise_rad_s_sqrthz` (rad/s/√Hz) and `accel_noise_m_s2_sqrthz` (m/s²/√Hz) are
    continuous-time densities. They share units with `cfg.gyro_noise_sigma` and
    `cfg.accel_noise_sigma`, which GTSAM uses as continuous covariances (`multi_sensor.py:46-57`), so
    levels can be stated as multiples of the estimator's assumed density. The discretization is
    `σ_d = σ_c·√rate` with `rate = 1e6 / median(diff(t_us))`. The noise is drawn in SI units per
    sensor as `σ_d · standard_normal((N,3))`, then converted to the array unit (rad/s→deg/s ×180/π).
  - `sequence_id`, `stress_seed` (required) and `gyro_unit` (`"deg/s"` default, or `"rad/s"`).
- **Realized evidence.** `rate_hz_estimated`, dt median/min/max, the bias mode, unit direction and
  vector (SI), `σ_d` (SI), and the per-axis mean and std of `out − in` (SI).
- **What it is NOT.** It is not a bias random walk: the bias is constant over the sequence, while
  the estimator models a random walk with `cfg.*_bias_sigma`. It is not colored, quantization, or
  scale-factor/misalignment error. It does not change the estimator's assumed noise parameters
  (`cfg` stays untouched).

## 2. Event–IMU time offset: `apply_imu_time_offset`

- **Physical meaning.** A constant clock or synchronization offset between the event camera and the
  IMU, as from an unmodelled transport latency or a calibration error in `t_d`. Convention:
  **`t_imu' = t_imu + δ`**. δ > 0 means each IMU sample is stamped δ later than it was taken, so the
  IMU appears delayed relative to the events.
- **Acts on.** IMU column 0 only. The event, frame, and GT timestamps are unchanged, so the
  trajectory-vs-GT association used for ATE stays intact. Only the fusion is misaligned.
- **Parameters.** `offset_ms` (signed, ms). `event_range_us = (start, end)` is the event span the
  estimator consumes, such as the first and last voxel-window timestamps, and is used only for
  accounting. `sequence_id` is optional and only recorded.
- **Edge handling.** No sample is dropped and nothing is extrapolated (`n_dropped = n_extrapolated
  = 0`). The evidence records, before and after the shift, counts of samples before, after, and
  inside `[start, end]`. It also records `uncovered_head_us`, `uncovered_tail_us`, and
  `covers_event_range`. `max_abs_float_rounding_us` records float64 rounding of `t + δ`. It is 0
  for the relative time bases. For VECtor's absolute ~1.6e15 µs stamps the float spacing is
  0.25 µs.
- **What it is NOT.** It is **not timestamp quantization** (`FREQVIO_TIME_QUANT_US`). Quantization
  rounds *event* stamps to a grid, which changes resolution: the error is bounded by q/2 with zero
  mean, and the relative alignment is unchanged. The offset keeps resolution and moves *alignment*
  by a constant. It is also not the `--time_offset_us` option in `tumvie.py:227,77,95`. That option
  is a calibration knob that shifts IMU and GT *together*, with the opposite sign
  (`t' = t − offset`).

## 3. Burst missing measurements: `make_burst_schedule(s)`, `burst_keep_mask`, `apply_imu_burst_dropout`, `apply_event_burst_dropout`

- **Physical meaning.** Temporally contiguous total outages: dropped USB/bus packets, a host or
  driver stall, or buffer overflow. All samples inside a window are lost. Nothing is interpolated or
  filled.
- **Schedule.** `make_burst_schedule(span_us, burst_ms, *, sequence_id, stress_seed, rate_hz=None |
  target_fraction=None, stream)` returns a float64 `[K,2]` array of `[t_start, t_end)` µs inside
  `span_us`. The burst count is `K = round(rate_hz·span_s)` or
  `K = round(target_fraction·span/burst)` (ties to even), so the count is fixed, not Poisson. The
  bursts are placed uniformly without overlap. With offsets `u_(1..K)` as sorted
  `U(0, span − K·burst)` draws, `start_i = span_start + u_(i) + i·burst`. The realized time fraction
  is `K·burst/span`, which is within `burst/(2·span)` of the target. If K·burst ≥ span, the function
  raises an error.
- **Shared vs independent.** `make_burst_schedules(..., mode="shared")` gives one schedule
  (`stream="shared"`) for IMU and events, modelling a common outage. `mode="independent"` gives two
  schedules seeded with `stream="imu"` and `stream="events"`. The stream enters the seed. To degrade
  only one stream, apply only that applier.
- **Mask.** `burst_keep_mask(t_us, schedule)` returns True outside every half-open window. It works
  on any 1-D int or float timestamp array, in any order.
- **Appliers.** IMU rows or events with `t` inside a window are removed. For events, call per batch
  (for example per voxel window) with the *sequence-level* schedule.
- **Evidence.** The schedule evidence holds `n_bursts`, `realized_time_fraction`,
  `realized_rate_hz`, `removed_us`, `schedule_sha256` and the full `schedule_us`. The applier
  evidence holds `n_in`, `n_removed`, `n_out`, `fraction_removed`, `n_bursts`, `schedule_sha256`,
  `data_span_us`, `time_fraction_removed_in_data_span`, and `n_bursts_touching_data_span`. IMU
  evidence adds `max_gap_us_in` and `max_gap_us_out`.
- **What it is NOT.** It is not event thinning (`FREQVIO_EVENT_KEEP`), which removes events i.i.d.
  at constant density. It is not voxel dropout (`utils/event_degrade.py`), which is spatial. It is
  not IMU down-sampling. It does not shorten bursts at the span edges: windows always lie inside
  `span_us`. To protect initialization, pass a span that starts after a guard interval.

---

## 4. Event background noise: `hot_pixel_set`, `apply_event_background_noise`

- **Physical meaning.** Events that do not come from scene brightness changes. Two components:
  *background activity* (spatially uniform, temporally random events; in real sensors mainly
  junction-leakage and photon shot noise, which increase in low light and at higher temperature),
  and *hot pixels* (a small fixed set of defective pixels that fire at a high rate). The rate unit is
  Hz per pixel, the same unit as the shot-noise rate parameter of the v2e event simulator. Levels
  must be chosen from sensor measurements or the literature; every source must be checked before it
  is cited.
- **Model.** For the half-open window `[a, b)` of length `T`:
  background count `~ Poisson(r_BA · H · W · T)`, each event with a uniform pixel, a uniform time
  and a fair-coin polarity; hot pixels are `round(f · H · W)` pixels drawn once per sequence
  (`hot_pixel_set`, seeded by sensor size, fraction, `sequence_id`, `stress_seed`), each emitting
  `Poisson(r_hot · T)` events. Draw order is fixed (BA count, BA pixels, BA times, BA polarities,
  hot counts, hot times, hot polarities) and is part of the semantics.
- **Seeding and tiling.** The window RNG seed includes the parameters, the sensor size and the
  window `[a, b)`. The same window therefore always receives the same noise, independent of the
  evaluation seed. Call the operator on **disjoint windows that tile the stream** (for example
  `[t_prev, t_cur)` of each voxel window). Overlapping calls would add noise twice. Changing the
  window boundaries changes the noise realisation, so boundaries must be the same across the
  conditions being compared.
- **Formats.** Integer `t` gets integer noise times in `[ceil(a), ceil(b))`. Float `t` gets uniform
  floats in `[a, b)`. `x`, `y`, `p` keep their dtypes. The polarity encoding is inferred from the
  batch (`{-1, 1}` if any value is negative, else `{0, 1}`). For an empty batch it must be passed as
  `polarity_values`. The merge is a stable sort on `t` with the original events first, so their
  relative order is unchanged and, at equal timestamps, originals precede noise.
- **Evidence.** `requested` holds the three rates/fraction, the sensor size and the window.
  `realized` holds `n_in`, `n_background`, `n_hot`, `n_out`, `window_s`, `expected_background`,
  `expected_hot`, `n_hot_pixels`, `hot_pixels_sha256`, `polarity_values`, `polarity_source`,
  `integer_timestamps`, and `n_in_outside_window` (input events outside `[a, b)`, which signals a
  mismatch between the call window and the batch).
- **What it is NOT.** It does not model refractory periods, illumination-dependent leak events,
  spatially correlated noise or noise that depends on scene activity. It does not remove or move
  real events. Proposed order when wired in (not decided yet): noise is part of the sensor output,
  so it goes before event thinning and timestamp quantization.

---

## Integration notes (facts about the current pipeline; nothing here is changed)

- `run_DEIO2` mutates the caller's IMU array in place (`eval_utils.py:62`). Apply the operators and
  record their evidence before the call.
- `VIO_update` indexes `all_imu[cur_imu_ii]` with no bounds check (`dba.py:1215-1231`). Each frame
  needs an IMU sample with `t ≥` that frame's time. A negative offset, or a burst at the end, that
  leaves `uncovered_tail_us > 0` relative to the last frame can raise `IndexError`. Decide in the
  protocol whether that counts as a failure or the range should be trimmed. `init_IMU` skips IMU
  samples before the first frame (`dba.py:831-836`).
- `append_imu` treats a gap > 0.025 s as an "IMU gap" and switches to `params_loose` (100× noise
  covariance) for that preintegration (`multi_sensor.py:88-94`, `:62-66`). IMU bursts longer than
  about 25 ms therefore exercise a different estimator code path. Use `max_gap_us_out`.
- If an event burst empties a voxel window, `read_batch_as_voxel` returns `None` and the frame is
  skipped (`load_utils.py:181-193`, `:224-226`). In that case event bursts also remove frames, not
  only events.
- The voxel disk cache key (`event_stress.stress_tag`) holds only keep, tq, and the stress version.
  Event-side sensor stress must enter that key when it is wired in, or cached clean voxels will be
  reused.
