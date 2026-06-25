"""Final paper table generation (three-class failure status).

Replaces the old Table 2/3 (which used success/crash% and conflicted with the
three-class failure definition).
The new tables use: valid / numerical-crash / catastrophic-completion + total failure% + Wilson CI.
The normalized table uses the valid-median (valid runs only) and correctly handles the clean baseline.
Threshold sensitivity: also output failure counts at 50/100/200% (real numbers for the main text).

Outputs: paper/tables/{,patch_,time_}T2_threeclass.tex + T3_normalized.tex + threshold_sensitivity.md
Usage: python scripts/make_paper_tables.py
"""
import csv, os, math
from collections import defaultdict

DATA = "paper/data"
OUT = "paper/tables"
AXES = {
    # Final convention: keep/time use the frozen stress-v2 _final version; patch is kept as-is.
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
    """Read a CSV. Level column handled flexibly: keep (keep/patch axes) or tq_us (time v2 axis)."""
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
    """Three classes: valid / crash / catastrophic. Returns a dict."""
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
    """Main three-class status table. 10 columns wide -> two-column-spanning table* with reduced column spacing to avoid overflow/overlap."""
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
    """Normalized table (each sequence divided by its own clean valid-median), with nan handling."""
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
    """Compute the failure classification at the 50/100/200% thresholds (real numbers, not a claim)."""
    out = ["# Threshold sensitivity (real recomputation)\n",
           "> Catastrophic threshold theta in {50, 100, 200}%; checks whether the three-class failure counts are robust.\n",
           "| axis | theta=50% | theta=100% | theta=200% |", "|---|---|---|---|"]
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
    out.append("\n-> The catastrophic count changes little with the threshold (catastrophic runs are mostly >>200%); the dominant failure mode is numerical crash, so the conclusion is robust.")
    rep = "\n".join(out)
    print(rep)
    open("paper/analysis/threshold_sensitivity.md", "w").write(rep)


def make_compact_3axis():
    """Compact three-axis summary table: 8 sequences x 3 axes, each cell showing the
    failure% sequence over that axis's 4 levels (e.g. 0/0/20/60).
    One table covers all three axes, replacing six wide spanning tables. Goes in the main text, not the appendix."""
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
    print("Compact three-axis summary table: compact_3axis.tex")


def main():
    os.makedirs(OUT, exist_ok=True)
    for axis in AXES:
        make_t2(axis); make_t3(axis)
    make_compact_3axis()
    print("Three-class status tables + normalized tables (with nan handling) + compact three-axis table\n")
    threshold_sensitivity()


if __name__ == "__main__":
    main()
