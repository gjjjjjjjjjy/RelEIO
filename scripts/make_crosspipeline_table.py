"""跨管线(pipeline)对照表: DEIO(主) vs DEVO event-only vs loop-closure.

读三个 pipeline 的 per_seed CSV, 按 (dataset/scene, keep) 聚合 success/crash%/median/worst MPE,
只列三者(或所选 pipeline)共同覆盖的 (序列, keep) cell, 做公平 within-cell 对照.

口径(写论文遵守):
- DEVO = IMU-off ablation(同前端同权重), 不是独立算法.
- 不做跨 pipeline 绝对精度排名(模态/配置不同); 只看同一 cell 下各 pipeline 的 failure 行为是否不同.
- 退化与失败**非单调**(中间档可能比 clean 更稳), 叙述勿用单调"退化→更糟".

用法:
  python script/make_crosspipeline_table.py            # 默认读 paper/data + 跨算法 CSV
  python script/make_crosspipeline_table.py --md out.md # 额外写 markdown 表
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "paper", "data")

# pipeline 显示名 -> CSV 路径(相对 paper/data). DEIO 主线 keep 轴无 pipeline 列, 视作 deio.
SOURCES = [
    ("DEIO",          "keep_per_seed_v2_final.csv",      "deio"),
    ("DEVO(evt-only)", "devo_eventonly_per_seed_v2.csv", "eventonly"),
    ("loop-closure",  "loopclosure_per_seed_v2_final.csv", "loopclosure"),
]


def load_csv(path, pipeline_label):
    """读一个 per_seed CSV -> dict[(ds,scene,keep)] -> list of (status, mpe)."""
    rows = defaultdict(list)
    if not os.path.isfile(path):
        return rows, False
    with open(path) as f:
        for r in csv.DictReader(f):
            ds, scene, keep = r["dataset"], r["scene"], r["keep"]
            status = r.get("status", "")
            mpe = r.get("mpe", "")
            mpe = float(mpe) if mpe not in ("", None) else None
            rows[(ds, scene, keep)].append((status, mpe))
    return rows, True


def agg(seed_rows):
    """一个 cell 的 (success/n, crash%, median, worst). crash = mpe is None 或 status==crash."""
    n = len(seed_rows)
    mpes = [m for (s, m) in seed_rows if m is not None and s != "crash"]
    nc = n - len(mpes)
    med = np.median(mpes) if mpes else float("nan")
    wst = max(mpes) if mpes else float("nan")
    return len(mpes), n, nc / n * 100 if n else float("nan"), med, wst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None, help="额外写出 markdown 表到此路径")
    a = ap.parse_args()

    loaded = []
    for label, fname, pl in SOURCES:
        path = os.path.join(DATA, fname)
        rows, ok = load_csv(path, pl)
        loaded.append((label, rows, ok, fname))
        print(f"[{'OK ' if ok else 'MISS'}] {label:16} <- {fname}"
              + ("" if ok else "  (缺文件, 该pipeline列留空)"))

    # 只对至少有2个pipeline覆盖的 cell 做对照(优先三者都有的). key=(ds,scene,keep).
    all_keys = set()
    for _, rows, ok, _ in loaded:
        if ok:
            all_keys |= set(rows.keys())
    # 排序: 按 dataset, scene, keep(降序, 1.0->0.1)
    def keep_sort(k):
        try:
            return -float(k[2])
        except ValueError:
            return 0
    keys = sorted(all_keys, key=lambda k: (k[0], k[1], keep_sort(k)))

    headers = ["dataset/scene", "keep"]
    for label, _, _, _ in loaded:
        headers += [f"{label}\nsucc", "crash%", "med", "worst"]

    # 文本表
    print("\n" + "=" * 110)
    print("跨管线对照 (success/n | crash% | median MPE | worst MPE)  —— 同 cell within比较, 非绝对排名")
    print("=" * 110)
    line = f"{'dataset/scene':22} {'keep':5}"
    for label, _, _, _ in loaded:
        line += f" | {label:16}"
    print(line)
    print("-" * 110)

    md_rows = []
    for k in keys:
        ds, scene, keep = k
        cells = []
        line = f"{ds+'/'+scene:22} {keep:5}"
        for label, rows, ok, _ in loaded:
            if ok and k in rows:
                ns, n, cr, med, wst = agg(rows[k])
                cell = f"{ns}/{n} {cr:.0f}% {med:.2f}/{wst:.2f}"
            else:
                cell = "—"
            cells.append(cell)
            line += f" | {cell:16}"
        print(line)
        md_rows.append((f"{ds}/{scene}", keep, cells))

    print("-" * 110)
    print("注: cell = success/n  crash%  median/worst MPE(%). '—'=该pipeline未覆盖此cell.")
    print("DEVO=IMU-off消融(非独立算法). 退化与失败非单调(中间档可能比clean稳).")

    if a.md:
        labels = [l for l, _, _, _ in loaded]
        with open(a.md, "w") as f:
            f.write("# 跨管线对照表 (DEIO / DEVO event-only / loop-closure)\n\n")
            f.write("> within-cell 比较(同序列同 keep 档), **非**绝对精度排名(模态/配置不同)。\n")
            f.write("> DEVO = IMU-off 消融(同前端同权重), 非独立算法。\n")
            f.write("> cell = success/n · crash% · median/worst MPE(%)。退化与失败**非单调**。\n\n")
            hdr = "| dataset/scene | keep | " + " | ".join(labels) + " |\n"
            sep = "|" + "---|" * (2 + len(labels)) + "\n"
            f.write(hdr); f.write(sep)
            for ds_scene, keep, cells in md_rows:
                f.write(f"| {ds_scene} | {keep} | " + " | ".join(cells) + " |\n")
        print(f"\n[md] 写出 {a.md}")


if __name__ == "__main__":
    main()
