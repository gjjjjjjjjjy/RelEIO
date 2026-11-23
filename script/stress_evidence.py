"""runner 侧 stress 证据解析 / 校验 (stress v2 配套).

eval 进程在每个 run 结束时打印一行 (utils/event_stress.finalize_run_evidence):
  [stress-evidence] version=v2 where=... keep=0.3 tq_us=0 frames=1161 n_in=... n_out=...
  actual_keep=0.301 tq_changed=.../... tq_changed_ratio=... first_window_checksum=n123_...

本模块从 log 抓这行, 解析成 dict, 并提供:
  - parse_evidence(log): 返回证据 dict 或 None;
  - is_v2_valid(log): 日志是否含 version=v2 的证据 (汇总时拒绝非 v2 日志, 防止旧无效日志混入 CSV);
  - evidence_csv_fields(): CSV 证据列名;
  - evidence_row(log): 给定 log 返回 CSV 证据字段值 (缺失填空).
"""
import os
import re

_RE = re.compile(r"\[stress-evidence\]\s+(.*)")
_KV = re.compile(r"(\w+)=(\S+)")

CSV_FIELDS = ["stress_version", "req_keep", "req_tq_us",
              "n_in", "n_out", "actual_keep", "tq_changed", "tq_total",
              "tq_changed_ratio", "first_window_checksum", "full_stream_hash"]


def parse_evidence(log):
    """解析 log 里最后一行 [stress-evidence]. 返回 dict 或 None."""
    if not log or not os.path.isfile(log):
        return None
    txt = open(log, errors="ignore").read()
    hits = _RE.findall(txt)
    if not hits:
        return None
    kv = dict(_KV.findall(hits[-1]))  # 取最后一条(整序列终检那条)
    return kv


def is_v2_valid(log):
    """日志是否含 version=v2 的 stress 证据. 汇总时据此拒绝旧/无证据日志."""
    ev = parse_evidence(log)
    return bool(ev) and ev.get("version") == "v2"


def is_real_crash(log):
    """日志是否是真实的算法 numerical crash (zero norm / nan).
    crash 的 run 崩在轨迹算出来前, 不会有 [stress-evidence] 行, 但它是有效的失败状态,
    必须计入失败率, 不能当"无证据脏日志"拒掉. 这与旧/未生效日志(既无 crash 也无证据)区分开."""
    if not log or not os.path.isfile(log):
        return False
    txt = open(log, errors="ignore").read()
    return ("zero norm" in txt or "[nan nan nan]" in txt or "nan, nan" in txt
            or "Found zero norm quaternions" in txt)


def is_countable(log):
    """该日志是否应计入统计: 要么有 v2 证据(valid/正常完成), 要么是真 crash(有效失败状态).
    只有"既无 v2 证据、又非真 crash"的(旧日志/未生效/跑断)才被拒绝出统计."""
    return is_v2_valid(log) or is_real_crash(log)


def evidence_csv_fields():
    return list(CSV_FIELDS)


def evidence_row(log):
    """返回与 CSV_FIELDS 对齐的证据值列表 (缺失填空字符串)."""
    ev = parse_evidence(log) or {}
    return [
        ev.get("version", ""),
        ev.get("keep", ""),
        ev.get("tq_us", ""),
        ev.get("n_in", ""),
        ev.get("n_out", ""),
        ev.get("actual_keep", ""),
        ev.get("tq_changed", "").split("/")[0] if ev.get("tq_changed") else "",
        ev.get("tq_changed", "").split("/")[1] if "/" in ev.get("tq_changed", "") else "",
        ev.get("tq_changed_ratio", ""),
        ev.get("first_window_checksum", ""),
        ev.get("full_stream_hash", ""),
    ]
