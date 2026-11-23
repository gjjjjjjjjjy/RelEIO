"""退化矩阵 batch runner (多GPU并行版): 5数据集 × event_keep × 多seed.

每个(数据集,序列)任务分到一张GPU, 8卡并行. 任务内串行(缓存让重复seed秒级).
crash不停, 已有log跳过(断点续), 自动汇总 success/crash_rate/median/worst/IQR.

用法:
  python script/run_degradation_sweep.py                         # 全跑, 自动分8卡
  python script/run_degradation_sweep.py --gpus 0 1 2 3          # 用指定4卡
  python script/run_degradation_sweep.py --datasets vector       # 只跑vector
  python script/run_degradation_sweep.py --keeps 1.0 0.5 --seeds 1000 1001  # debug子集
  python script/run_degradation_sweep.py --summary-only          # 只看表
"""
import argparse, os, subprocess, re, tempfile, sys, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script import stress_evidence as _ev


def preflight_torch():
    """跑实验前自检: torch能否import. 新实例libucc未配会让所有eval崩(误记100%crash污染数据).
    防呆: 子进程测import torch, 失败则提示修libucc并退出, 不让坏环境污染整个矩阵."""
    r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("=" * 70)
        print("✗ import torch 失败! 跑实验前必须先修环境, 否则所有eval会崩成假crash污染数据.")
        print("  错误:", (r.stderr.strip().splitlines() or ["?"])[-1])
        if "ucs_config_doc_nop" in r.stderr or "libucc" in r.stderr:
            print("  → libucc未配(新实例/换节点必现). 修复:")
            print('    echo "/opt/hpcx/ucx/lib" > /etc/ld.so.conf.d/zz-hpcx-ucx.conf')
            print('    echo "/opt/hpcx/ucc/lib" >> /etc/ld.so.conf.d/zz-hpcx-ucx.conf')
            print("    ldconfig")
        print("=" * 70)
        sys.exit(1)
    print(f"[preflight] torch OK, {r.stdout.strip()} GPUs")

# 数据集: (eval脚本, config, inputdir, [序列], 额外参数)
DATASETS = {
    "vector":  ("script/eval_deio/vector.py",   "config/vector.yaml",  "data/VECtor",  ["desk_normal1", "corner_slow1"], []),
    "hku":     ("script/eval_deio/hku.py",      "config/hku.yaml",     "data/HKU",     ["vicon_dark1"], []),
    "tumvie":  ("script/eval_deio/tumvie.py",   "config/tumvie.yaml",  "data/TUM-VIE", ["mocap-3d-trans"], []),
    "rpgdavis":("script/eval_deio/mvsec.py",    "config/mvsec.yaml",   "data/RPGDAVIS_pp", ["shapes_6dof", "boxes_6dof", "poster_6dof"],
                ['--scene_suffix', '', '--t_imu_cam_identity', '--H', '180', '--W', '240']),
    "fpv":     ("script/eval_deio/uzh-fpv.py",  "config/uzhfpv.yaml",  "data/UZHFPV",  ["indoor_forward_3"], []),
}
KEEPS = ["1.0", "0.5", "0.3", "0.1"]
SEEDS = [1000, 1001, 1002, 1003, 1004]
LOGDIR = "logs/sweep_v2"  # stress v2: 新目录, 旧 logs/sweep 归档为 logs/sweep_legacy_v1
FORCE = False  # --force: 忽略已有 log 重跑 (默认断点续跑)


def run_one(ds, evalpy, config, inputdir, scene, extra, keep, seed, gpu):
    log = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{seed}.log")
    # 断点续: 已有 log 且完成 且含 v2 证据 才跳过. --force 或缺 v2 证据则重跑(防旧无效日志被当完成).
    if not FORCE and os.path.isfile(log) and os.path.getsize(log) > 0:
        txt = open(log, errors="ignore").read()
        done = "MPE is" in txt or "zero norm" in txt or "[nan nan nan]" in txt
        if done and _ev.is_v2_valid(log):
            return "skip"
    splitf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    splitf.write(scene + "\n"); splitf.close()
    # NO_VOXCACHE: 切voxel直接跑(~1min/eval), 比读写16GB大cache快, 不占磁盘, 无竞争
    env = dict(os.environ, FREQVIO_USE_FNO="0", FREQVIO_FREQ_WEIGHT="0",
               FREQVIO_EVENT_KEEP=keep, FREQVIO_NO_VOXCACHE="1",
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


def run_seq_task(args):
    """一个(数据集,序列)的完整keep×seed矩阵, 跑在一张GPU上(串行, 缓存加速)."""
    ds, scene, gpu, keeps, seeds = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    results = []
    for keep in keeps:
        for seed in seeds:
            status = run_one(ds, evalpy, config, inputdir, scene, extra, keep, seed, gpu)
            log = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{seed}.log")
            mpe = parse_mpe(log)
            tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
            print(f"[GPU{gpu}][{status:5}] {ds}/{scene} keep={keep} seed={seed} → {tag}", flush=True)
            results.append((ds, scene, keep, seed, mpe))
    return results


def run_cell_task(args):
    """cell级调度(stress-v2缓存最优): 一个cell=(ds,scene,keep)固定一张GPU, 内部5seed串行起独立子进程.
    第1个seed BUILD磁盘voxel缓存, 后4个seed HIT秒级读 → voxel切片省5x. seed间仍独立进程(crash不互扰).
    顶层函数(可pickle, 供ProcessPoolExecutor)."""
    ds, scene, keep, seeds, gpu = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    for seed in seeds:  # 串行: 保证第1个先BUILD缓存, 后续HIT
        try:  # 单seed任何意外不中断同cell其余seed
            status = run_one(ds, evalpy, config, inputdir, scene, extra, keep, seed, gpu)
            log = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{seed}.log")
            mpe = parse_mpe(log)
            tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
            print(f"[GPU{gpu}][{status:5}] {ds}/{scene} keep={keep} seed={seed} → {tag}", flush=True)
        except Exception as _e:
            print(f"[GPU{gpu}][ERROR] {ds}/{scene} keep={keep} seed={seed}: {_e} (已隔离)", flush=True)
    return f"{ds}/{scene}/keep{keep}"


def run_eval_task(args):
    """细粒度(旧, 保留): 一个(ds,scene,keep,seed,gpu)独立eval."""
    ds, scene, keep, seed, gpu = args
    evalpy, config, inputdir, _, extra = DATASETS[ds]
    status = run_one(ds, evalpy, config, inputdir, scene, extra, keep, seed, gpu)
    log = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{seed}.log")
    mpe = parse_mpe(log)
    tag = "CRASH" if mpe is None else f"MPE={mpe:.3f}"
    print(f"[GPU{gpu}][{status:5}] {ds}/{scene} keep={keep} seed={seed} → {tag}", flush=True)
    return status


def parse_mpe(log):
    if not os.path.isfile(log):
        return None
    txt = open(log, errors="ignore").read()
    if "zero norm" in txt or "[nan nan nan]" in txt or "nan, nan" in txt:
        return None
    m = re.findall(r"MPE is ([\d.]+)", txt)
    return float(m[-1]) if m else None


def summarize(datasets, keeps, seeds):
    print(f"\n{'='*95}\n退化矩阵汇总 (seed-wise, stress v2)\n{'='*95}")
    print(f"{'dataset/seq':28} {'keep':6} {'success':9} {'crash%':8} {'median':9} {'worst':9} {'IQR':8}")
    csv_path = os.path.join(LOGDIR, "per_seed_v2.csv")
    rejected = []
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["dataset", "scene", "keep", "seed", "status", "mpe"] + _ev.evidence_csv_fields())
        for ds in datasets:
            for scene in DATASETS[ds][3]:
                for keep in keeps:
                    logs_exist = []
                    for s in seeds:
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{s}.log")
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
                        lg = os.path.join(LOGDIR, f"{ds}_{scene}_keep{keep}_seed{s}.log")
                        mpe = parse_mpe(lg)
                        vals.append(mpe)
                        status = "crash" if mpe is None else "ok"
                        w.writerow([ds, scene, keep, s, status,
                                    "" if mpe is None else f"{mpe:.4f}"] + _ev.evidence_row(lg))
                    done = [v for v in vals if v is not None]
                    n = len(logs_exist); n_crash = sum(1 for v in vals if v is None)
                    med = np.median(done) if done else float("nan")
                    wst = max(done) if done else float("nan")
                    iqr = (np.percentile(done, 75) - np.percentile(done, 25)) if len(done) >= 2 else float("nan")
                    print(f"{ds+'/'+scene:28} {keep:6} {len(done)}/{n:<7} {n_crash/n*100:>6.0f}%  {med:>8.3f} {wst:>8.3f} {iqr:>7.3f}")
    print(f"\n[csv] 写出 {csv_path} (仅含 v2 有效日志)")
    if rejected:
        print(f"[拒绝] {len(rejected)} 个日志既无v2证据又非真crash(脏日志/未生效/跑断), 已排除出 CSV (前5个):")
        for r in rejected[:5]:
            print(f"   - {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--keeps", nargs="+", default=KEEPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--scenes", nargs="+", default=None, help="只跑指定序列")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--logdir", default=None, help="覆盖默认 logs/sweep_v2")
    ap.add_argument("--force", action="store_true", help="忽略已有log重跑(默认断点续)")
    a = ap.parse_args()
    global LOGDIR, FORCE
    if a.logdir:
        LOGDIR = a.logdir
    FORCE = a.force
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[runner] LOGDIR={LOGDIR} FORCE={FORCE} stress_version=v2")

    if not a.summary_only:
        preflight_torch()  # 防呆: torch不能import就别跑(否则假crash污染矩阵)
        # cell级调度(stress-v2缓存最优): 每个cell=(ds,scene,keep)是一个任务, 内部5seed串行
        # (第1 BUILD后4 HIT省5x voxel切片); cell之间并行分到各GPU. seed仍独立子进程(crash不互扰).
        cells = []
        for ds in a.datasets:
            for scene in DATASETS[ds][3]:
                if a.scenes and scene not in a.scenes:
                    continue
                for keep in a.keeps:
                    cells.append((ds, scene, keep))
        print(f"共 {len(cells)} 个cell({len(cells)*len(a.seeds)} runs), cell级调度到 {len(a.gpus)} 张GPU并行")
        job_args = [(ds, scene, keep, list(a.seeds), a.gpus[i % len(a.gpus)])
                    for i, (ds, scene, keep) in enumerate(cells)]
        with ProcessPoolExecutor(max_workers=len(a.gpus)) as ex:
            futs = [ex.submit(run_cell_task, ja) for ja in job_args]
            done_n = 0
            for f in as_completed(futs):
                # 防呆: 单个cell任何意外异常只记录, 绝不中断其余cell(无人值守一晚的必备保险)
                try:
                    cell = f.result()
                    done_n += 1
                    print(f"[cell-done {done_n}/{len(futs)}] {cell}", flush=True)
                except Exception as _e:
                    print(f"[cell-error] 某cell异常(已隔离, 继续其余): {_e}", flush=True)
    summarize(a.datasets, a.keeps, a.seeds)


if __name__ == "__main__":
    main()
