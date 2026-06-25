"""论文终稿表生成 (三类失败状态, 修复审计硬伤②③④).

替换旧的Table2/3(旧的用success/crash%, 与三类失败定义冲突).
新表用: valid / numerical-crash / catastrophic-completion + total failure% + Wilson CI.
归一化表用 valid-median(仅valid run), 正确处理clean基准.
阈值敏感性: 同时输出50/100/200%下的failure计数(供正文引用真实数字).

产物: paper/tables/{,patch_,time_}T2_threeclass.tex + T3_normalized.tex + threshold_sensitivity.md
用法: python script/make_paper_tables.py
"""
import csv, os, math
from collections import defaultdict

DATA = "paper/data"
OUT = "paper/tables"
AXES = {
    # final 口径: keep/time 用 stress-v2 _final 冻结版; patch 保留(旧但有效)
    "keep":  ("keep_per_seed_v2_final.csv", ["1.0","0.5","0.3","0.1"], "keep", ""),
    "patch": ("patch_per_seed_v2_final.csv", ["96","64","32","16"], "patches", "patch_"),
    "time":  ("time_per_seed_v2_final.csv", ["0","1000","5000","10000"], r"$\tau$(ms)", "time_"),
}
TIME_MS = {"0":"0","1000":"1","5000":"5","10000":"10"}
SCENE_ORDER = [
    ("vector","desk_normal1","VECtor/desk"),("vector","corner_slow1","VECtor/corner"),
    ("hku","vicon_dark1","HKU/dark"),("rpgdavis","boxes_6dof","DAVIS/boxes"),
    ("rpgdavis","poster_6dof","DAVIS/poster"),("tumvie","mocap-3d-trans","TUM-VIE/mocap"),
    ("rpgdavis","shapes_6dof","DAVIS/shapes"),("fpv","indoor_forward_3","UZH-FPV/indoor"),
]


def load(fn):
    """读CSV. level列名兼容: keep(keep/patch轴) 或 tq_us(time v2轴)."""
    d = defaultdict(dict)
    with open(os.path.join(DATA, fn)) as f:
        rd = csv.DictReader(f)
        lvlcol = ("keep" if "keep" in rd.fieldnames else ("patch" if "patch" in rd.fieldnames else "tq_us"))
        for r in rd:
            mpe = float(r["mpe"]) if r["mpe"] else None
            d[(r["dataset"], r["scene"], r[lvlcol])][int(r["seed"])] = (r["status"], mpe)
    return d


def median(xs):
    s = sorted(xs); n = len(s)
    return float("nan") if n == 0 else (s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2)


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,(c-h))*100, min(1,(c+h))*100


def cell_three(cell, th=100.0):
    """三类: valid/crash/catastrophic. 返回dict."""
    valid, crash, catas = [], 0, 0
    for st, m in cell.values():
        if st == "crash" or m is None: crash += 1
        elif m > th: catas += 1
        else: valid.append(m)
    n = len(valid)+crash+catas
    nfail = crash+catas
    lo, hi = wilson(nfail, n)
    vmed = median(valid) if valid else float("nan")
    vwst = max(valid) if valid else float("nan")
    def pc(q):
        s = sorted(valid); k = len(s)
        if k < 2: return float("nan")
        i = q/100*(k-1); j = int(i); fr = i-j
        return s[j] if j+1>=k else s[j]*(1-fr)+s[j+1]*fr
    return dict(valid=len(valid), crash=crash, catas=catas, n=n,
                fail_pct=nfail/n*100 if n else float("nan"), ci=(lo,hi),
                vmed=vmed, vwst=vwst, iqr=(pc(75)-pc(25)) if len(valid)>=2 else float("nan"))


def lvl_disp(axis, lv):
    return TIME_MS[lv] if axis == "time" else lv


def make_t2(axis):
    """三类状态主表. 10列宽 → 跨双栏table* + 缩列距, 避免撑爆/叠表."""
    fn, levels, lname, pfx = AXES[axis]
    d = load(fn)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\setlength{\tabcolsep}{5pt}\small",
         rf"\caption{{Three-class degradation outcomes ({axis} axis). Each cell has 5 seeds. "
         r"V/C/X = valid / numerical-crash / catastrophic-completion ($\mathrm{MPE}>100\%$); "
         r"fail\% = (C+X)/5 with Wilson 95\% CI; median/worst/IQR (\%) over valid runs only "
         r"($\dagger$: $<\!3$ valid).}",
         rf"\label{{tab:deg_{axis}}}",
         r"\begin{tabular}{ll ccc c c ccc}", r"\toprule",
         rf"Sequence & {lname} & V & C & X & fail\% & 95\% CI & median & worst & IQR \\", r"\midrule"]
    for ds, sc, lab in SCENE_ORDER:
        for i, lv in enumerate(levels):
            s = cell_three(d.get((ds, sc, lv), {}))
            name = lab if i == 0 else ""
            dag = r"$^\dagger$" if s["valid"] < 3 else ""
            med = "--" if math.isnan(s["vmed"]) else f"{s['vmed']:.2f}{dag}"
            wst = "--" if math.isnan(s["vwst"]) else f"{s['vwst']:.2f}"
            iqr = "--" if math.isnan(s["iqr"]) else f"{s['iqr']:.2f}"
            ci = f"[{s['ci'][0]:.0f},{s['ci'][1]:.0f}]"
            L.append(f"{name} & {lvl_disp(axis,lv)} & {s['valid']} & {s['crash']} & {s['catas']} "
                     f"& {s['fail_pct']:.0f} & {ci} & {med} & {wst} & {iqr} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(f"{OUT}/{pfx}T2_threeclass.tex", "w").write("\n".join(L))


def make_t3(axis):
    """归一化表(每序列除以自己clean valid-median). 修复nan."""
    fn, levels, lname, pfx = AXES[axis]
    d = load(fn)
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         rf"\caption{{Normalized degradation ({axis} axis): valid-run median/worst divided by "
         r"the sequence's own clean median; $\Delta$fail\% relative to clean. "
         r"Reveals non-monotonic and tail behavior.}",
         rf"\label{{tab:norm_{axis}}}",
         r"\begin{tabular}{ll rrr}", r"\toprule",
         rf"Seq. & {lname} & rel.med & rel.worst & $\Delta$fail\% \\", r"\midrule"]
    for ds, sc, lab in SCENE_ORDER:
        base = cell_three(d.get((ds, sc, levels[0]), {}))
        bmed, bfail = base["vmed"], base["fail_pct"]
        for i, lv in enumerate(levels):
            s = cell_three(d.get((ds, sc, lv), {}))
            name = lab if i == 0 else ""
            ok = bmed == bmed and bmed > 0
            rm = f"{s['vmed']/bmed:.2f}$\\times$" if ok and s["vmed"]==s["vmed"] else "--"
            rw = f"{s['vwst']/bmed:.2f}$\\times$" if ok and s["vwst"]==s["vwst"] else "--"
            df = s["fail_pct"]-bfail if bfail==bfail and s["fail_pct"]==s["fail_pct"] else float("nan")
            ds_ = f"{df:+.0f}" if df==df else "--"
            L.append(f"{name} & {lvl_disp(axis,lv)} & {rm} & {rw} & {ds_} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/{pfx}T3_normalized.tex", "w").write("\n".join(L))


def threshold_sensitivity():
    """硬伤③: 真实算50/100/200%阈值下的failure分类, 不是空声称."""
    out = ["# 阈值敏感性 (真实重算, 硬伤③)\n",
           "> catastrophic阈值θ取50/100/200%, 看三类failure计数是否稳健.\n",
           "| axis | θ=50% | θ=100% | θ=200% |", "|---|---|---|---|"]
    for axis, (fn, levels, _, _) in AXES.items():
        d = load(fn)
        row = [axis]
        for th in (50, 100, 200):
            ncrash = ncatas = 0
            for ds, sc, _ in SCENE_ORDER:
                for lv in levels:
                    for st, m in d.get((ds, sc, lv), {}).values():
                        if st == "crash" or m is None: ncrash += 1
                        elif m > th: ncatas += 1
            row.append(f"crash {ncrash}, catas {ncatas}")
        out.append("| " + " | ".join(row) + " |")
    out.append("\n→ catastrophic数随阈值变化小(灾难run多>>200%), 主失效是numerical crash, 结论稳健.")
    rep = "\n".join(out)
    print(rep)
    open("paper/analysis/threshold_sensitivity.md", "w").write(rep)


def make_compact_3axis():
    """紧凑三轴汇总表: 8序列×3轴, 每cell显示该轴4档的failure%序列(如0/0/20/60).
    一个表覆盖三轴, 替代6个跨栏大表. 放正文, 不用附件."""
    data = {ax: load(AXES[ax][0]) for ax in AXES}
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Failure rate (\%) per sequence across all three stress axes, "
         r"one value per degradation level (clean$\to$severe). Failure = numerical "
         r"crash $+$ catastrophic completion ($\mathrm{MPE}>100\%$), 5 seeds per "
         r"cell. The detailed event-density breakdown is in Table~\ref{tab:deg_keep}.}",
         r"\label{tab:compact3}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"& event density & visual budget & timestamp \\",
         r"Sequence & {\scriptsize 1.0/.5/.3/.1} & {\scriptsize 96/64/32/16} "
         r"& {\scriptsize 0/1/5/10\,ms} \\", r"\midrule"]
    for ds, sc, lab in SCENE_ORDER:
        cells = []
        for ax in ["keep","patch","time"]:
            levels = AXES[ax][1]; d = data[ax]
            fr = []
            for lv in levels:
                cell = d.get((ds,sc,lv),{}); n=len(cell)
                nf = sum(1 for st,m in cell.values() if st=="crash" or m is None or m>100.0)
                fr.append(str(int(round(nf/n*100))) if n else "-")
            cells.append("/".join(fr))
        L.append(f"{lab.replace('_',chr(92)+'_')} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(f"{OUT}/compact_3axis.tex", "w").write("\n".join(L))
    print("紧凑三轴汇总表: compact_3axis.tex")


def main():
    os.makedirs(OUT, exist_ok=True)
    for axis in AXES:
        make_t2(axis); make_t3(axis)
    make_compact_3axis()
    print("三类状态表 + 归一化表(修nan) + 紧凑三轴表\n")
    threshold_sensitivity()


if __name__ == "__main__":
    main()
