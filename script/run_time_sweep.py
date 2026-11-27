"""时间退化矩阵 (第三条退化轴, 与event_keep/low-patch正交): timestamp量化.

退化事件时间戳精度: FREQVIO_TIME_QUANT_US=0/1000/5000/10000 (=0/1/5/10ms).
模拟低时间分辨率传感器/时间戳噪声. 与event_keep(密度)、low-patch(视觉预算)第三个正交维度.

设置同前两轴: keep=1.0(不thin), patch=default, DEIO+IMU统一, 细粒度seed-wise.
q=0 = clean = 与event_keep clean / patch96 三方交叉验证一致.

用法:
  python script/run_time_sweep.py --gpus 0 1 2 3
  python script/run_time_sweep.py --summary-only
  python script/run_time_sweep.py --datasets tumvie --quants 0 --seeds 1000   # 小验证
"""
import argparse, os, subprocess, tempfile, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script.run_degradation_sweep import DATASETS, MAIN_DATASETS, parse_mpe, preflight_torch
from script import stress_evidence as _ev

QUANTS = ["0", "1000", "5000", "10000"]   # us: 0(clean)/1ms/5ms/10ms
SEEDS = [1000, 1001, 1002, 1003, 1004]
LOGDIR = "logs/time_sweep_v2"  # stress v2: 新目录, 旧 logs/time_sweep 归档为 logs/time_sweep_legacy_invalid
FORCE = False


def run_one(ds, evalpy, config, inputdir, scene, extra, quant, seed, gpu):
    log = os.path.join(LOGDIR, f"{ds}_{scene}_tq{quant}_seed{seed}.log")
    # 断点续: 已完成 且含 v2 证据 才跳过. --force 或缺 v2 证据则重跑.
    if not FORCE and os.path.isfile(log) and os.path.getsize(log) > 0:
        txt = open(log, errors="ignore").read()
        done = "MPE is" in txt or "zero norm" in txt or "[nan nan nan]" in txt
        if done and _ev.is_v2_valid(log):
            return "skip"
    splitf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    splitf.write(scene + "\n"); splitf.close()
    env = dict(os.environ, FREQVIO_USE_FNO="0", FREQVIO_FREQ_WEIGHT="0",
               FREQVIO_EVENT_KEEP="1.0", FREQVIO_TIME_QUANT_US=quant, FREQVIO_NO_VOXCACHE="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               PYTHONPATH=os.getcwd(), CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = ["python", evalpy, "--inputdir", inputdir, "--config", config,
           "--val_split", splitf.name, "--enable_event", "--network", "DEVO.pth",
           "--stride", "1", "--trials", "1", "--seed", str(seed)] + extra
    # 防卡死: 单run超时(默认1800s, env FREQVIO_RUN_TIMEOUT 可调). 超时杀掉标crash, 不堵塞cell队列.
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
    """cell级调度(stress-v2缓存最优): cell=(ds,scene,quant)固定一张GPU, 内部5seed串行起独立子进程.
    第1seed BUILD voxel缓存, 后4seed HIT → 省5x切片. seed间独立进程(crash不互扰)."""
    ds, scene, quant, seeds, gpu = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    for seed in seeds:  # 串行: 第1个BUILD缓存, 后续HIT
        try:  # 单seed任何意外不中断同cell其余seed
            status = run_one(ds, evalpy, config, inputdir, scene, extra, quant, seed, gpu)
            log = os.path.join(LOGDIR, f"{ds}_{scene}_tq{quant}_seed{seed}.log")
            mpe = parse_mpe(log)
            tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
            print(f"[GPU{gpu}][{status:5}] {ds}/{scene} tq={quant}us seed={seed} → {tag}", flush=True)
        except Exception as _e:
            print(f"[GPU{gpu}][ERROR] {ds}/{scene} tq={quant} seed={seed}: {_e} (已隔离)", flush=True)
    return f"{ds}/{scene}/tq{quant}"


def run_eval_task(args):
    """细粒度(旧, 保留)."""
    ds, scene, quant, seed, gpu = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    status = run_one(ds, evalpy, config, inputdir, scene, extra, quant, seed, gpu)
    log = os.path.join(LOGDIR, f"{ds}_{scene}_tq{quant}_seed{seed}.log")
    mpe = parse_mpe(log)
    tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
    print(f"[GPU{gpu}][{status:5}] {ds}/{scene} tq={quant}us seed={seed} → {tag}", flush=True)
    return status


def summarize(datasets, quants, seeds):
    print(f"\n{'='*95}\n时间退化矩阵汇总 (timestamp量化, seed-wise, stress v2)\n{'='*95}")
    print(f"{'dataset/seq':28} {'tq_us':7} {'success':9} {'crash%':8} {'median':9} {'worst':9} {'IQR':8}")
    csv_path = os.path.join(LOGDIR, "per_seed_v2.csv")
    rejected = []
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["dataset", "scene", "tq_us", "seed", "status", "mpe"] + _ev.evidence_csv_fields())
        for ds in datasets:
            for scene in DATASETS[ds][3]:
                for quant in quants:
                    logs_exist = []
                    for s in seeds:
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_tq{quant}_seed{s}.log")
                        if not os.path.isfile(lg):
                            continue
                        # 计入条件: 有v2证据(正常) 或 真crash(有效失败状态). 只拒"既无证据又非crash"的脏日志.
                        if not _ev.is_countable(lg):
                            rejected.append(lg)
                            continue
                        logs_exist.append(s)
                    if not logs_exist:
                        continue
                    vals = []
                    for s in logs_exist:
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_tq{quant}_seed{s}.log")
                        mpe = parse_mpe(lg)
                        vals.append(mpe)
                        status = "crash" if mpe is None else "ok"
                        w.writerow([ds, scene, quant, s, status,
                                    "" if mpe is None else f"{mpe:.4f}"] + _ev.evidence_row(lg))
                    done = [v for v in vals if v is not None]
                    n = len(logs_exist); n_crash = sum(1 for v in vals if v is None)
                    med = np.median(done) if done else float("nan")
                    wst = max(done) if done else float("nan")
                    iqr = (np.percentile(done, 75) - np.percentile(done, 25)) if len(done) >= 2 else float("nan")
                    print(f"{ds+'/'+scene:28} {quant:7} {len(done)}/{n:<7} {n_crash/n*100:>6.0f}%  {med:>8.3f} {wst:>8.3f} {iqr:>7.3f}")
    print(f"\n[csv] 写出 {csv_path} (仅含 v2 有效日志)")
    if rejected:
        print(f"[拒绝] {len(rejected)} 个日志既无v2证据又非真crash(脏日志/未生效/跑断), 已排除出 CSV (前5个):")
        for r in rejected[:5]:
            print(f"   - {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS))
    ap.add_argument("--quants", nargs="+", default=QUANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--logdir", default=None, help="覆盖默认 logs/time_sweep_v2")
    ap.add_argument("--force", action="store_true", help="忽略已有log重跑(默认断点续)")
    a = ap.parse_args()
    global LOGDIR, FORCE
    if a.logdir:
        LOGDIR = a.logdir
    FORCE = a.force
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[runner] LOGDIR={LOGDIR} FORCE={FORCE} stress_version=v2")

    if not a.summary_only:
        preflight_torch()
        # cell级调度(stress-v2缓存最优): cell=(ds,scene,quant)内5seed串行(第1BUILD后4HIT); cell间并行.
        cells = []
        for ds in a.datasets:
            for scene in DATASETS[ds][3]:
                if a.scenes and scene not in a.scenes:
                    continue
                for quant in a.quants:
                    cells.append((ds, scene, quant))
        print(f"共 {len(cells)} 个cell({len(cells)*len(a.seeds)} runs), cell级调度到 {len(a.gpus)} 张GPU并行")
        job_args = [(ds, scene, quant, list(a.seeds), a.gpus[i % len(a.gpus)])
                    for i, (ds, scene, quant) in enumerate(cells)]
        with ProcessPoolExecutor(max_workers=len(a.gpus)) as ex:
            futs = [ex.submit(run_cell_task, ja) for ja in job_args]
            done_n = 0
            for f in as_completed(futs):
                # 防呆: 单cell任何意外异常只记录, 绝不中断其余cell(无人值守一晚必备)
                try:
                    cell = f.result()
                    done_n += 1
                    print(f"[cell-done {done_n}/{len(futs)}] {cell}", flush=True)
                except Exception as _e:
                    print(f"[cell-error] 某cell异常(已隔离, 继续其余): {_e}", flush=True)
    summarize(a.datasets, a.quants, a.seeds)


if __name__ == "__main__":
    main()
