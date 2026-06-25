"""
event_stress.py — 跨算法统一的事件通道退化算子 (AlgorithmAdapter 第一块基石).

把原本内联在 utils/load_utils.py:read_batch_as_voxel 里的两条事件级退化轴抽成
**纯函数**, 让 DEIO / DEVO / RAMP-VO / ESVIO 任意管线在生成自己的事件表示
(voxel / time-surface / event-stack / 左右双流) **之前** 注入逐比特一致的退化.

设计契约 (对齐论文协议的因果基石, 不可改):
  - event density (rho, FREQVIO_EVENT_KEEP): 事件级随机保留 keep 比例.
    种子 = RandomState(int(t0_us) % 2**31), **按帧时间戳 seed, 跨 evaluation seed 字节一致**.
    → 退化输入对所有 eval seed 相同, per-seed 方差纯来自算法内部采样.
  - timestamp quantization (tq, FREQVIO_TIME_QUANT_US): t = round(t/q)*q, 量化到 q 微秒网格.
    **完全确定性**, 无随机种子.
  - 两轴正交; 先 thinning 再 quantization, 顺序与原 read_batch_as_voxel 一致 (load_utils.py:197-218).

逐比特等价性: 本模块复刻 load_utils.py:200-218 的精确逻辑 (RandomState 种子式、
np.round 网格), 因此对单目 voxel 管线 (DEIO/DEVO) 的退化结果与已有 480-run 完全一致.
mvsec/rpgdavis 分支原先缺 tq (load_utils.py:134 只有 thinning) —— 改用本模块即一并修复该不一致.
"""
import os
import numpy as np

# 环境变量名 (与现有 load_utils.py 完全一致, 不要改名, 否则破坏与 480-run 的可比性)
ENV_KEEP = "FREQVIO_EVENT_KEEP"      # rho: 1.0/0.5/0.3/0.1, 不设=1.0(不退化)
ENV_TIME_QUANT = "FREQVIO_TIME_QUANT_US"  # tq: 1000/5000/10000 us, 不设或0=不量化
ENV_NO_VOXCACHE = "FREQVIO_NO_VOXCACHE"   # =1 完全禁用磁盘/内存 voxel 缓存

# stress 版本号: 缓存键/文件名/日志都带上它. 任何改动退化语义(算子/顺序/种子)时 +1,
# 旧缓存与旧日志因版本号不同而自动失效, 杜绝"命中旧 clean 缓存把退化静默跳过"的污染.
# v2 = 修复 tq 缓存键缺失 bug 后的统一算子 (event_stress.apply_event_stress).
STRESS_VERSION = "v2"

_SEED_MOD = 2 ** 31


def no_voxcache():
    """是否禁用 voxel 缓存. 所有 iterator 都必须查它 (修复: 旧 tumvie/mvsec/fpv 不查)."""
    return os.environ.get(ENV_NO_VOXCACHE, "0") == "1"


def stress_tag():
    """统一的 stress 标签, 用于缓存键 / 缓存文件名 / 日志. 含 keep + tq + version.

    例: 'keep1.0_tq5000_stressv2'. 任何 iterator 构造缓存键/文件名都拼它,
    保证 keep 和 tq 都进键 —— 单点修复, 杜绝某个 iterator 漏放 tq.
    """
    keep, tq = get_stress_params()
    keep = round(keep, 4)
    tq = int(tq)
    return f"keep{keep}_tq{tq}_stress{STRESS_VERSION}"


def stress_cache_components():
    """返回进缓存 key tuple 的 stress 分量: (keep, tq_us, STRESS_VERSION)."""
    keep, tq = get_stress_params()
    return (round(keep, 4), int(tq), STRESS_VERSION)


def log_stress_config(where=""):
    """每次运行打印当前 stress 配置 (第5步要求的留痕)."""
    keep, tq = get_stress_params()
    nc = no_voxcache()
    print(f"[event-stress] version={STRESS_VERSION} keep={keep} tq_us={int(tq)} "
          f"no_voxcache={int(nc)} {where}", flush=True)


def input_checksum(ev, t0_us):
    """对一批(退化后)事件算稳定 checksum, 用于第6步验证:
    - 相同 stress、不同 eval seed: checksum 相同 (因退化只由 t0 种子定);
    - tq=0 vs tq=10000: checksum 不同;
    - clean: 与重构前逐比特一致.
    用事件数 + t/x/y/p 的和与极值, 避免依赖 hashlib 的字节序. 纯诊断, 不进热路径.
    """
    import numpy as _np
    t, x, y, p = ev["t"], ev["x"], ev["y"], ev["p"]
    n = len(t)
    if n == 0:
        return f"n0_t0{int(t0_us)}"
    return (f"n{n}_tsum{_np.asarray(t, dtype=_np.float64).sum():.3f}"
            f"_xsum{int(_np.asarray(x).sum())}_ysum{int(_np.asarray(y).sum())}"
            f"_psum{int(_np.asarray(p).sum())}_tmax{_np.asarray(t, dtype=_np.float64).max():.1f}")


def get_stress_params():
    """从环境变量读当前 stress 配置. 返回 (keep, tq_us)."""
    keep = float(os.environ.get(ENV_KEEP, "1.0"))
    tq_us = float(os.environ.get(ENV_TIME_QUANT, "0"))
    return keep, tq_us


# ====================== stress-v2 安全磁盘缓存 (B调度+A缓存混合) ======================
# 同一 cell 的 5 个 seed 退化输入完全一致(退化只由帧时间戳定), voxel 只需切 1 次:
# 第 1 个 seed BUILD 缓存, 后 4 个 HIT 读盘. 但每个 seed 是独立 Python 子进程, 模块级
# 内存缓存不跨进程, 故必须磁盘缓存. 三道安全机制: 文件锁 / 原子写 / 缓存证据 metadata.
# cache HIT 时必须 register_cached_evidence() 把退化证据重新注册, 否则后 4 个 seed
# 没跑 apply_event_stress, 防呆系统看不到证据会判它无效.

def _evidence_metadata(extra=None):
    """打包当前进程 stress 证据 + cell 标识, 存进缓存文件 (cache HIT 时恢复)."""
    ev = get_evidence()
    keep, tq = get_stress_params()
    meta = {
        "stress_version": STRESS_VERSION,
        "stress_tag": stress_tag(),
        "keep": keep, "tq_us": int(tq),
        "n_in": ev["n_in_total"], "n_out": ev["n_out_total"],
        "actual_keep": ev["actual_keep_ratio"],
        "tq_changed": ev["tq_changed_total"], "tq_total": ev["tq_total"],
        "first_window_checksum": ev["first_window_checksum"],
        "full_stream_hash": ev.get("full_stream_hash"),
        "frames": ev["frames"],
    }
    if extra:
        meta.update(extra)
    return meta


def register_cached_evidence(meta):
    """cache HIT 时调: 把缓存里存的 stress 证据重新注入进程证据累加器.
    这样 finalize_run_evidence 仍能打印 [stress-evidence] 且断言/CSV 正常 ——
    后 4 个 seed 虽未执行 apply_event_stress, 防呆系统也能看到退化证据."""
    if not meta:
        return
    # 安全校验: 缓存的 stress_tag 必须与当前请求一致, 否则是读错缓存(防呆)
    cur = stress_tag()
    if meta.get("stress_tag") != cur:
        raise AssertionError(
            f"[stress-cache] HIT 的缓存 stress_tag={meta.get('stress_tag')} != 当前请求 {cur}. "
            f"缓存键错误, 拒绝使用!")
    _EVIDENCE["frames"] = meta.get("frames", 0)
    _EVIDENCE["n_in_total"] = meta.get("n_in", 0)
    _EVIDENCE["n_out_total"] = meta.get("n_out", 0)
    _EVIDENCE["tq_total"] = meta.get("tq_total", 0)
    _EVIDENCE["tq_changed_total"] = meta.get("tq_changed", 0)
    _EVIDENCE["first_window_checksum"] = meta.get("first_window_checksum")
    _EVIDENCE["full_stream_hash"] = meta.get("full_stream_hash")
    _EVIDENCE["keep"] = meta.get("keep")
    _EVIDENCE["tq_us"] = meta.get("tq_us")
    print(f"[stress-cache] cache=HIT 已恢复证据 stress_tag={cur} "
          f"actual_keep={meta.get('actual_keep')} tq_changed={meta.get('tq_changed')}", flush=True)


def load_or_build_cache(cache_path, build_fn, extra_meta=None):
    """安全的 stress-v2 缓存读/建. 文件锁 + 原子写 + 证据 metadata.

    流程:
      1. 无锁快路径: 缓存已存在 -> 直接读, register evidence, 打印 cache=HIT.
      2. 否则取文件锁 (fcntl.flock); 拿到锁后**再查一次**(可能别的进程刚建好) -> HIT.
      3. 仍无 -> build_fn() 切 voxel (此过程 apply_event_stress 累加真实证据);
         打包 {metadata + data_list} 原子写 (tmp.<pid> -> os.replace); 打印 cache=BUILD.
    Args:
        cache_path: 正式缓存文件路径.
        build_fn: 无参函数, 返回 data_list (切 voxel; 内部会调 apply_event_stress 累加证据).
        extra_meta: 附加进 metadata 的 dict (sequence/side/stride/shape 等).
    Returns:
        data_list.
    """
    import pickle, fcntl

    def _read(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "data_list" in obj:
            register_cached_evidence(obj)
            print(f"[stress-cache] cache=HIT {os.path.basename(path)}", flush=True)
            return obj["data_list"]
        # 兼容意外的裸 data_list (无 metadata): 拒绝, 因无证据无法过防呆
        raise AssertionError(f"[stress-cache] {path} 无 stress metadata, 拒绝 (无证据无法验证退化生效)")

    # 1) 无锁快路径
    if os.path.isfile(cache_path):
        return _read(cache_path)

    # 2) 取锁 (锁文件 = cache_path + .lock, 避免锁正式文件本身)
    lock_path = cache_path + ".lock"
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)  # 阻塞直到拿到独占锁
        # 拿到锁后再查一次: 可能别的进程在我等锁期间刚建好
        if os.path.isfile(cache_path):
            return _read(cache_path)

        # 3) 我是 builder: 切 voxel (build_fn 内部 apply_event_stress 累加真实证据)
        reset_evidence()
        data_list = build_fn()
        meta = _evidence_metadata(extra_meta)
        meta["data_list"] = data_list
        # 原子写: tmp.<pid> -> os.replace (中途被杀不留可误读的半文件)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)  # 原子替换
        print(f"[stress-cache] cache=BUILD {os.path.basename(cache_path)} "
              f"(切 {len(data_list)} voxel, 已原子落盘)", flush=True)
        return data_list
    # 锁在 with 退出时自动释放


def thin_events(t, x, y, p, keep, t0_us):
    """事件密度退化: 随机保留 keep 比例. 种子按帧时间戳 t0_us (跨 eval seed 一致).

    Args:
        t, x, y, p: 同长度的原始事件数组 (numpy).
        keep: 保留比例 in (0, 1]. >=1.0 时原样返回.
        t0_us: 该帧/批的起始时间戳 (us), 用作 thinning 的确定性种子.
    Returns:
        (t, x, y, p) thinned. 若全部丢弃返回长度0数组 (调用方需判空).
    """
    if keep >= 1.0:
        return t, x, y, p
    n = len(t)
    if n == 0:
        return t, x, y, p
    rng = np.random.RandomState(int(t0_us) % _SEED_MOD)
    sel = rng.rand(n) < keep
    return t[sel], x[sel], y[sel], p[sel]


def quantize_timestamps(t, tq_us):
    """时间戳量化退化: t = round(t/q)*q. 完全确定性, 无种子.

    Args:
        t: 原始事件时间戳数组 (us).
        tq_us: 量化网格 (us). <=0 时原样返回.
    Returns:
        量化后的 t (float64). 不改 x/y/p.
    """
    if tq_us <= 0 or len(t) == 0:
        return t
    _t = np.asarray(t, dtype=np.float64)
    return np.round(_t / tq_us) * tq_us


# 防呆: 运行时断言开关 (默认开). 设 FREQVIO_STRESS_ASSERT=0 关闭 (仅极端性能场景).
ENV_ASSERT = "FREQVIO_STRESS_ASSERT"
# 进程级证据累加器: 跨所有帧汇总 stress 前后事件数, 供 runner 在 run 结束时落盘/断言.
_EVIDENCE = {
    "frames": 0, "n_in_total": 0, "n_out_total": 0,
    "tq_changed_total": 0, "tq_total": 0,
    "first_window_checksum": None, "keep": None, "tq_us": None,
    "full_stream_hash": None,  # 全流滚动hash: 区分"首窗同但全程不同输入"(确定性审计用)
}


def reset_evidence():
    """每个 run 开始时调一次 (或进程级自动累加亦可)."""
    for k in _EVIDENCE:
        _EVIDENCE[k] = 0
    _EVIDENCE["first_window_checksum"] = None
    _EVIDENCE["keep"] = None
    _EVIDENCE["tq_us"] = None
    _EVIDENCE["full_stream_hash"] = None


def _roll_hash(prev, ev, t0_us):
    """滚动 hash: 把每窗的退化后特征(事件数+t/x/y/p的和与极值)累进一个稳定签名.
    用 hashlib.sha256(上一hash + 本窗特征串). 不对全部事件逐个hash(1.7亿事件太慢),
    用窗级聚合特征已足以区分两次run的完整输入流是否逐窗一致."""
    import hashlib
    feat = input_checksum(ev, t0_us)  # 复用窗级特征串(n+tsum+xsum+...)
    base = (prev or "") + "|" + feat
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def get_evidence():
    """返回当前累加的 stress 证据 (供 runner 落盘 + 终检断言)."""
    ev = dict(_EVIDENCE)
    n_in, n_out = ev["n_in_total"], ev["n_out_total"]
    ev["actual_keep_ratio"] = (n_out / n_in) if n_in > 0 else None
    ev["tq_changed_ratio"] = (ev["tq_changed_total"] / ev["tq_total"]) if ev["tq_total"] > 0 else None
    ev["stress_version"] = STRESS_VERSION
    return ev


def _assert_on():
    return os.environ.get(ENV_ASSERT, "1") == "1"


def apply_event_stress(ev, t0_us, keep=None, tq_us=None, record=True):
    """对一批原始事件施加 (thinning -> quantization) 两条退化轴.

    这是 AlgorithmAdapter 调用的统一入口: 任何管线在生成事件表示前调它一次.
    带防呆: 累加 stress 前后事件数/tq 改变数/首窗 checksum, 并做逐窗运行时断言.

    Args:
        ev: dict, 必须含键 "t","x","y","p" (numpy 数组, 同长度). 其余键透传.
        t0_us: 该批起始时间戳 (us), thinning 种子.
        keep, tq_us: 退化参数; None 时从环境变量读 (保持与现有 sweep runner 一致).
        record: 是否累加进程级证据 (热路径里默认 True; 诊断用例可关).
    Returns:
        新的事件 dict (thinning 后变长, quantization 后 t 变值). 透传其余键.
        若 thinning 后为空, 返回的 dict 各数组长度为 0 —— 调用方按"该帧无事件"处理.
    """
    if keep is None or tq_us is None:
        _k, _q = get_stress_params()
        keep = _k if keep is None else keep
        tq_us = _q if tq_us is None else tq_us

    t, x, y, p = ev["t"], ev["x"], ev["y"], ev["p"]
    n_in = len(t)
    t, x, y, p = thin_events(t, x, y, p, keep, t0_us)
    n_after_thin = len(t)

    tq_changed = 0
    if n_after_thin > 0 and tq_us > 0:
        _t_before = np.asarray(t, dtype=np.float64)
        t = quantize_timestamps(t, tq_us)
        tq_changed = int(np.count_nonzero(np.asarray(t, dtype=np.float64) != _t_before))

    out = dict(ev)
    out.update({"t": t, "x": x, "y": y, "p": p})

    if record:
        _EVIDENCE["frames"] += 1
        _EVIDENCE["n_in_total"] += n_in
        _EVIDENCE["n_out_total"] += n_after_thin
        _EVIDENCE["keep"] = keep
        _EVIDENCE["tq_us"] = int(tq_us)
        if tq_us > 0 and n_after_thin > 0:
            _EVIDENCE["tq_total"] += n_after_thin
            _EVIDENCE["tq_changed_total"] += tq_changed
        if _EVIDENCE["first_window_checksum"] is None and n_after_thin > 0:
            _EVIDENCE["first_window_checksum"] = input_checksum(out, t0_us)
        # 全流滚动hash: 每窗累进, 用于确定性审计(区分首窗同但全程不同的输入流)
        _EVIDENCE["full_stream_hash"] = _roll_hash(_EVIDENCE["full_stream_hash"], out, t0_us)

    # 逐窗运行时断言 (防呆: 让"环境变量设了但没生效"立即暴露, 而非静默)
    if _assert_on() and n_in > 0:
        # tq>0: 所有(被量化的)时间戳必须落在 q 网格上
        if tq_us > 0 and n_after_thin > 0:
            _tt = np.asarray(t, dtype=np.float64)
            _resid = np.abs(_tt / tq_us - np.round(_tt / tq_us))
            assert np.all(_resid < 1e-6), (
                f"[stress-assert] tq={tq_us}us 但有时间戳未落网格 (max resid {_resid.max():.3g}). "
                f"退化未正确施加!")
    return out


def finalize_run_evidence(where="", strict=True):
    """run 结束(整序列跑完)时调一次: 用累加证据做全局防呆断言 + 打印一行可审计证据.

    全局断言 (覆盖逐窗断言够不到的"整体没生效"情形):
      - keep<1: stress 后总事件数必须明显下降 (实际保留比例 ≈ keep, 容差宽松).
      - tq>0: 至少一部分时间戳确实发生了改变 (否则量化是 no-op = 未生效).
    打印的证据行可被 runner 抓取存进 per-run 元数据 (stress_version/keep/tq/比例/checksum/缓存名).

    strict: True=断言生效(正常路径防呆). False=只落证据不抛断言(在finally里crash路径用,
            避免证据不完整(如崩在首帧前frames=0)的断言掩盖原始crash异常).
    返回 evidence dict.
    """
    ev = get_evidence()
    keep, tq_us = ev["keep"], ev["tq_us"]
    line = (f"[stress-evidence] version={ev['stress_version']} where={where} "
            f"keep={keep} tq_us={tq_us} frames={ev['frames']} "
            f"n_in={ev['n_in_total']} n_out={ev['n_out_total']} "
            f"actual_keep={ev['actual_keep_ratio']} "
            f"tq_changed={ev['tq_changed_total']}/{ev['tq_total']} "
            f"tq_changed_ratio={ev['tq_changed_ratio']} "
            f"first_window_checksum={ev['first_window_checksum']} "
            f"full_stream_hash={ev.get('full_stream_hash')}")
    print(line, flush=True)

    if strict and _assert_on() and ev["n_in_total"] > 0:
        if keep is not None and keep < 1.0:
            r = ev["actual_keep_ratio"]
            # 实际保留比例应接近 keep; 给宽容差(各帧事件数差异大). 关键是"明显下降"而非精确.
            assert r is not None and r < 0.9 * 1.0 and r < keep + 0.15, (
                f"[stress-assert] keep={keep} 但实际保留比例={r:.3f} 未明显下降. 退化未生效!")
        if tq_us is not None and tq_us > 0:
            assert ev["tq_changed_total"] > 0, (
                f"[stress-assert] tq={tq_us}us 但没有任何时间戳改变. 量化未生效 (可能命中clean缓存)!")
    return ev


def apply_event_stress_stereo(ev_left, ev_right, t0_us, keep=None, tq_us=None):
    """双目变体 (ESVIO 用): 对左右事件流施加 **相同强度** 的退化.

    协议规范要求: ESVIO 左右事件流按同一强度、确定性规则各自退化.
    - thinning: 左右各自用按帧时间戳的种子. 注意左右用 **同一 t0_us 种子** 会让两路
      丢弃模式相关; 论文规范要的是"同一强度"(同 keep)而非"同一掩码", 因此左右用
      **不同但确定性** 的种子 (右路 t0_us 偏移一个固定常量), 保证 (a)同强度 (b)确定性
      (c)左右非完全相关. 若审稿要求严格对称, 可改为同种子 —— 这是一个明确的设计选择点.
    - quantization: 确定性, 左右用同一网格.

    Returns:
        (ev_left_stressed, ev_right_stressed).
    """
    if keep is None or tq_us is None:
        _k, _q = get_stress_params()
        keep = _k if keep is None else keep
        tq_us = _q if tq_us is None else tq_us
    left = apply_event_stress(ev_left, t0_us, keep=keep, tq_us=tq_us)
    # 右路种子偏移固定常量, 保证确定性且与左路不完全相关 (同强度不同掩码)
    right = apply_event_stress(ev_right, int(t0_us) + 1, keep=keep, tq_us=tq_us)
    return left, right
