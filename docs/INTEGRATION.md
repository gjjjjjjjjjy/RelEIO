# Integrating the evaluated systems

RelEIO applies one repeated-run reliability protocol to four odometry systems
spanning two architectural lineages. This document describes how each system is
obtained, set up, and wired to the shared stress operators
(`stress_operators/event_stress.py`), so the benchmark is reproducible.

The single integration contract is: **a system must apply the RelEIO stress
operators to the raw event stream before building its own event representation.**
Degradation is seeded by the frame timestamp (not the evaluation seed), so the
degraded input is bit-identical across repeated runs.

Reported metric differs by system (MPE % for the DPVO-family event/event+IMU
systems, ATE m for RAMP-VO and E2VID+TartanVO); cross-system comparison is on
**failure behavior and repeated-run reliability**, not absolute accuracy.

---

## 1. DEIO (event+IMU, primary)  &  2. DEVO (event-only, IMU-off ablation)

- **Source.** DEIO (Deep Event Inertial Odometry); DEVO is the same learned
  front-end with the IMU factor disabled.
- **Environment.** Python + CUDA; custom CUDA ops (correlation / fast bundle
  adjustment / lietorch backends). On recent toolchains (PyTorch 2.x, CUDA 12.x)
  the kernels need a one-line API fix: `AT_DISPATCH_*(x.type(), ...)` ->
  `x.scalar_type()`; build a single SM arch (e.g. `TORCH_CUDA_ARCH_LIST=8.9`) to
  cut compile time.
- **Weights.** Zero-shot pretrained weights (`DEVO.pth`); DEIO reuses the DEVO
  event front-end.
- **Stress hook.** The dataset iterators call `apply_event_stress(...)` before
  voxelization. `RELEIO_EVENT_KEEP` / `RELEIO_TIME_QUANT_US` select the level;
  the visual-budget axis varies the patches-per-frame config.
- **DEVO** = run with the IMU factor off (`ENABLE_IMU=False`), same iterators.
- **Run.** `scripts/run_deio.sh`, `scripts/run_devo.sh`.

## 3. RAMP-VO (event+frame, independently trained)

- **Source.** RAMP-VO (deep VO with events and frames). Vendored as a subdir;
  build in an **isolated venv** because its CUDA ops share names with DEIO's and
  would otherwise clash.
- **Environment.** Same `.type()->.scalar_type()` fix; additionally
  `torch.load(..., weights_only=False)` (PyTorch 2.6+ default change); place
  `eigen-3.4.0` under `thirdparty/`. The hardcoded `CUDA_VISIBLE_DEVICES="8"` in
  the eval entry must be removed.
- **Data.** Needs intensity frames + events (DAVIS sequences). Convert each
  sequence to the RAMP-VO layout (`image_left/*.png`, `events.h5` with an
  `events` group, `timestamps.txt`, `pose_left.txt`, `K.yaml`); the training-set
  index pickle is not needed for inference (skip it in `just_validation` mode).
- **Stress hook.** Inject the RelEIO thinning into `events_from_indices(...)`
  (after slicing the raw events, before the event representation). Frames are
  left untouched.
- **Run.** `scripts/run_rampvo.sh`. Records: `records/rampvo.csv`.

## 4. E2VID + TartanVO (non-DPVO: reconstruction + VO)

A two-stage, offline pipeline in **two separate environments** (E2VID is older
than TartanVO and they conflict on torch/numpy):

1. **E2VID** (event-to-video reconstruction).
   - Env: pin `numpy<2` (the released weights need the old `torch.from_numpy`
     path); use headless OpenCV (`opencv-python-headless`) on servers; replace
     deprecated `np.int/np.float/np.bool`.
   - Input is a text file: first line `W H`, then `t x y polarity` per line with
     `t` in **seconds** and polarity in `{0,1}`. Convert your `events.h5` with
     `tools/h5_to_e2vid_txt.py` (which also injects the RelEIO event thinning, so
     degradation is applied to the reconstruction input).
   - Output: PNG frames + `timestamps.txt` at a fixed frame rate
     (`--fixed_duration --window_duration 33`).
2. **TartanVO** (learned VO on the reconstructed frames).
   - Env: its PWC-Net flow uses cupy JIT kernels; on recent cupy replace
     `cupy.util.memoize`->`cupy.memoize` and
     `cupy.cuda.compile_with_cache(code).get_function(fn)`->
     `cupy.RawModule(code=code).get_function(fn)`. The released code is py2-era:
     fix `dict.has_key`, `dict.keys()[i]`, and scipy `Rotation.as_dcm/from_dcm`
     -> `as_matrix/from_matrix`.
   - Feed the E2VID frames + event-camera intrinsics; evaluate with Sim(3)
     alignment (monocular has no metric scale). Repeat each cell several times to
     capture residual GPU-nondeterminism spread.

- **Run.** `scripts/run_e2vid_tartanvo.sh`. Records: `records/e2vid_tartanvo.csv`.

---

## Two further systems (not included as quantitative baselines)

- A recent **stereo-event learned method** requires right-camera event data not
  available in our processed form.
- A classical **stereo-event-inertial pipeline** requires a ROS toolchain
  incompatible with our environment.

Both are reported as attempted baselines; integrating them is left to future work.
