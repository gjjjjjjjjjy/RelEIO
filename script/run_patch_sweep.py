"""低-patch退化矩阵 (第二条退化轴, 与event_keep/timestamp正交): PATCHES_PER_FRAME (stress-v2).

退化模型容量(视觉预算/后端可用因子数): PATCHES_PER_FRAME=96/64/32/16 (96=default/clean).
keep固定1.0(不thin事件), tq=0(不量化), 改 PATCHES_PER_FRAME via --opts. DEIO+IMU统一.

stress-v2 口径 (与 keep/time runner 同源): cell级调度(每cell=(ds,scene,patch)内5seed串行,
crash不互扰), per-run超时, cell/seed异常隔离, 证据门控(汇总拒非v2/非crash日志, CSV带evidence列).
注: patch轴退化不经过event-stress算子(改的是patch数不是事件流), 故keep=1.0时各run仍有
[stress-evidence] version=v2 (actual_keep=1.0), 用于统一证据口径 + 区分真crash/脏日志.

用法 (4卡):
  python script/run_patch_sweep.py --gpus 0 1 2 3
  python script/run_patch_sweep.py --summary-only
"""
import argparse, os, subprocess, tempfile, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.run_degradation_sweep import DATASETS, parse_mpe, preflight_torch
from script import stress_evidence as _ev

PATCHES = ["96", "64", "32", "16"]   # 96=default(clean), 逐步减视觉预算
SEEDS = [1000, 1001, 1002, 1003, 1004]
LOGDIR = "logs/patch_sweep_v2"  # stress v2: 新目录, 旧 logs/patch_sweep 归档为 logs/patch_sweep_legacy_v1
FORCE = False


def run_one(ds, evalpy, config, inputdir, scene, extra, patch, seed, gpu):
    log = os.path.join(LOGDIR, f"{ds}_{scene}_patch{patch}_seed{seed}.log")
    if not FORCE and os.path.isfile(log) and os.path.getsize(log) > 0:
        txt = open(log, errors="ignore").read()
        done = "MPE is" in txt or "zero norm" in txt or "[nan nan nan]" in txt
        if done and _ev.is_v2_valid(log):
            return "skip"
    splitf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    splitf.write(scene + "\n"); splitf.close()
    # keep=1.0(不thin), tq=0(不量化); 退化轴=PATCHES_PER_FRAME via --opts
    env = dict(os.environ, FREQVIO_USE_FNO="0", FREQVIO_FREQ_WEIGHT="0",
               FREQVIO_EVENT_KEEP="1.0", FREQVIO_TIME_QUANT_US="0", FREQVIO_NO_VOXCACHE="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               PYTHONPATH=os.getcwd(), CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = ["python", evalpy, "--inputdir", inputdir, "--config", config,
           "--val_split", splitf.name, "--enable_event", "--network", "DEVO.pth",
           "--stride", "1", "--trials", "1", "--seed", str(seed),
           "--opts", "PATCHES_PER_FRAME", patch] + extra
    _timeout = float(os.environ.get("FREQVIO_RUN_TIMEOUT", "1800"))
    try:
        with open(log, "w") as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=_timeout)
        ret = r.returncode
    except subprocess.TimeoutExpired:
        with open(log, "a") as lf:
            lf.write(f"\n[runner-timeout] 超过 {_timeout}s 被杀, 标记 crash, 继续下一个\n")
        ret = -9
    finally:
        os.unlink(splitf.name)
    return "ok" if ret == 0 else "crash"


def run_cell_task(args):
    """cell级调度: cell=(ds,scene,patch)固定一GPU, 内5seed串行独立子进程(crash不互扰)."""
    ds, scene, patch, seeds, gpu = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    for seed in seeds:
        try:
            status = run_one(ds, evalpy, config, inputdir, scene, extra, patch, seed, gpu)
            log = os.path.join(LOGDIR, f"{ds}_{scene}_patch{patch}_seed{seed}.log")
            mpe = parse_mpe(log)
            tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
            print(f"[GPU{gpu}][{status:5}] {ds}/{scene} patch={patch} seed={seed} → {tag}", flush=True)
        except Exception as _e:
            print(f"[GPU{gpu}][ERROR] {ds}/{scene} patch={patch} seed={seed}: {_e} (已隔离)", flush=True)
    return f"{ds}/{scene}/patch{patch}"


def summarize(datasets, patches, seeds):
    print(f"\n{'='*95}\n低-patch退化矩阵汇总 (seed-wise, stress v2)\n{'='*95}")
    print(f"{'dataset/seq':28} {'patch':6} {'success':9} {'crash%':8} {'median':9} {'worst':9} {'IQR':8}")
    csv_path = os.path.join(LOGDIR, "per_seed_v2.csv")
    rejected = []
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["dataset", "scene", "patch", "seed", "status", "mpe"] + _ev.evidence_csv_fields())
        for ds in datasets:
            for scene in DATASETS[ds][3]:
                for patch in patches:
                    logs_exist = []
                    for s in seeds:
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_patch{patch}_seed{s}.log")
                        if not os.path.isfile(lg):
                            continue
                        if not _ev.is_countable(lg):  # 有v2证据 或 真crash 才计入
                            rejected.append(lg)
                            continue
                        logs_exist.append(s)
                    if not logs_exist:
                        continue
                    vals = []
                    for s in logs_exist:
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_patch{patch}_seed{s}.log")
                        mpe = parse_mpe(lg)
                        vals.append(mpe)
                        status = "crash" if mpe is None else "ok"
                        w.writerow([ds, scene, patch, s, status,
                                    "" if mpe is None else f"{mpe:.4f}"] + _ev.evidence_row(lg))
                    done = [v for v in vals if v is not None]
                    n = len(logs_exist); n_crash = sum(1 for v in vals if v is None)
                    med = np.median(done) if done else float("nan")
                    wst = max(done) if done else float("nan")
                    iqr = (np.percentile(done, 75) - np.percentile(done, 25)) if len(done) >= 2 else float("nan")
                    print(f"{ds+'/'+scene:28} {patch:6} {len(done)}/{n:<7} {n_crash/n*100:>6.0f}%  {med:>8.3f} {wst:>8.3f} {iqr:>7.3f}")
    print(f"\n[csv] 写出 {csv_path} (仅含 v2 有效日志)")
    if rejected:
        print(f"[拒绝] {len(rejected)} 个日志既无v2证据又非真crash(脏日志/未生效/跑断), 已排除出 CSV (前5个):")
        for r in rejected[:5]:
            print(f"   - {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--patches", nargs="+", default=PATCHES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--logdir", default=None, help="覆盖默认 logs/patch_sweep_v2")
    ap.add_argument("--force", action="store_true", help="忽略已有log重跑(默认断点续)")
    a = ap.parse_args()
    global LOGDIR, FORCE
    if a.logdir:
        LOGDIR = a.logdir
    FORCE = a.force
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[runner] LOGDIR={LOGDIR} FORCE={FORCE} stress_version=v2 (patch轴)")

    if not a.summary_only:
        preflight_torch()
        cells = []
        for ds in a.datasets:
            for scene in DATASETS[ds][3]:
                if a.scenes and scene not in a.scenes:
                    continue
                for patch in a.patches:
                    cells.append((ds, scene, patch))
        print(f"共 {len(cells)} 个cell({len(cells)*len(a.seeds)} runs), cell级调度到 {len(a.gpus)} 张GPU并行")
        job_args = [(ds, scene, patch, list(a.seeds), a.gpus[i % len(a.gpus)])
                    for i, (ds, scene, patch) in enumerate(cells)]
        with ProcessPoolExecutor(max_workers=len(a.gpus)) as ex:
            futs = [ex.submit(run_cell_task, ja) for ja in job_args]
            done_n = 0
            for f in as_completed(futs):
                try:
                    cell = f.result(); done_n += 1
                    print(f"[cell-done {done_n}/{len(futs)}] {cell}", flush=True)
                except Exception as _e:
                    print(f"[cell-error] 某cell异常(已隔离, 继续其余): {_e}", flush=True)
    summarize(a.datasets, a.patches, a.seeds)


if __name__ == "__main__":
    main()
