"""Final paper figures (journal style, redrawn from CSVs), replacing the draft figures.

Journal aesthetics: sans-serif fonts, a professional (colour-blind-friendly)
palette, compact sizes, vector PDF, consistent styling.
Three-class failure definition (consistent with the protocol): failure = crash OR MPE>100%.

Outputs: paper/figures/final_*.pdf
Usage: python scripts/make_final_figures.py
"""
import csv, os, math, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nature_style import apply_nature_style, WONG, C_VALID, C_CATAS, C_CRASH
apply_nature_style()

DATA = "paper/data"
OUT = "paper/figures"
TH = 100.0  # catastrophic threshold

AXES = {
    # Final convention: keep/time use the frozen stress-v2 _final version; patch is kept as-is.
    "keep":  ("keep_per_seed_v2_final.csv", ["1.0","0.5","0.3","0.1"], "event keep ratio"),
    "patch": ("patch_per_seed_v2_final.csv", ["96","64","32","16"], "patches / frame"),
    "time":  ("time_per_seed_v2_final.csv", ["0","1000","5000","10000"], "timestamp quant. (ms)"),
}
TIME_LABEL = {"0":"0","1000":"1","5000":"5","10000":"10"}  # display us as ms
SCENE_ORDER = [
    ("vector","desk_normal1"),("vector","corner_slow1"),("hku","vicon_dark1"),
    ("rpgdavis","boxes_6dof"),("rpgdavis","poster_6dof"),("tumvie","mocap-3d-trans"),
    ("rpgdavis","shapes_6dof"),("fpv","indoor_forward_3"),
]
SLABEL = {
    ("vector","desk_normal1"):"VECtor/desk",("vector","corner_slow1"):"VECtor/corner",
    ("hku","vicon_dark1"):"HKU/dark",("rpgdavis","boxes_6dof"):"DAVIS/boxes",
    ("rpgdavis","poster_6dof"):"DAVIS/poster",("tumvie","mocap-3d-trans"):"TUM-VIE/mocap",
    ("rpgdavis","shapes_6dof"):"DAVIS/shapes",("fpv","indoor_forward_3"):"UZH-FPV/indoor",
}


def load(axis):
    fn, levels, _ = AXES[axis]
    d = defaultdict(dict)
    with open(os.path.join(DATA, fn)) as f:
        rd = csv.DictReader(f)
        lvlcol = ("keep" if "keep" in rd.fieldnames else ("patch" if "patch" in rd.fieldnames else "tq_us"))  # time v2 column is tq_us
        for r in rd:
            mpe = float(r["mpe"]) if r["mpe"] else None
            d[(r["dataset"], r["scene"], r[lvlcol])][int(r["seed"])] = (r["status"], mpe)
    return d, levels


def is_fail(st, m):
    return st == "crash" or m is None or m > TH


def level_labels(axis, levels):
    if axis == "time":
        return [TIME_LABEL[l] for l in levels]
    return levels


def fig_heatmap(axis):
    """Failure-rate heatmap (three failure classes), journal style."""
    d, levels = load(axis)
    M = np.full((len(SCENE_ORDER), len(levels)), np.nan)
    for i, sc in enumerate(SCENE_ORDER):
        for j, lv in enumerate(levels):
            cell = d.get((sc[0], sc[1], lv), {})
            n = len(cell)
            if n:
                M[i, j] = sum(is_fail(st, m) for st, m in cell.values())/n*100
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    im = ax.imshow(M, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(level_labels(axis, levels))
    ax.set_yticks(range(len(SCENE_ORDER)))
    ax.set_yticklabels([SLABEL[s] for s in SCENE_ORDER])
    ax.set_xlabel(AXES[axis][2])
    for i in range(len(SCENE_ORDER)):
        for j in range(len(levels)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 55 else "0.2")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("failure rate (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(f"{OUT}/final_{axis}_heatmap.pdf"); plt.close(fig)


def fig_perseed(axis):
    """Per-seed scatter for representative sequences (valid blue / catastrophic orange / crash red x), log scale."""
    d, levels = load(axis)
    rep = [("vector","desk_normal1"),("hku","vicon_dark1"),
           ("rpgdavis","shapes_6dof"),("fpv","indoor_forward_3")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=False)
    hv = hc = hx = None  # legend handles
    panel_tags = ["(a)","(b)","(c)","(d)"]
    for pi, (ax, sc) in enumerate(zip(axes, rep)):
        allv = [m for lv in levels for st, m in d.get((sc[0],sc[1],lv),{}).values()
                if st == "ok" and m is not None]
        if not allv: continue
        ymin = max(min(allv)*0.5, 1e-2); ymax = max(allv)*3
        ax.set_yscale("log"); ax.set_ylim(ymin*0.5, ymax)
        cy = ymin*0.7
        for j, lv in enumerate(levels):
            cell = d.get((sc[0],sc[1],lv),{})
            valid = [m for st,m in cell.values() if st=="ok" and m is not None and m<=TH]
            catas = [m for st,m in cell.values() if st=="ok" and m is not None and m>TH]
            ncrash = sum(1 for st,m in cell.values() if st=="crash" or m is None)
            xs = lambda n: [j+(k-n/2)*0.05 for k in range(n)]
            if valid:
                s = ax.scatter(xs(len(valid)), valid, s=18, c=C_VALID, alpha=0.85,
                               edgecolors="white", linewidths=0.4, zorder=3)
                hv = hv or s
            if catas:
                s = ax.scatter(xs(len(catas)), catas, s=22, c=C_CATAS, marker="^", zorder=4)
                hc = hc or s
            for c in range(ncrash):
                s = ax.scatter(j+(c-ncrash/2)*0.06, cy, marker="x", s=22, c=C_CRASH, linewidths=1.2, zorder=4)
                hx = hx or s
        ax.axhline(ymin*0.9, color="0.7", ls=":", lw=0.6)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(level_labels(axis,levels), fontsize=7)
        ax.set_title(f"{panel_tags[pi]} {SLABEL[sc]}", fontsize=8); ax.set_xlim(-0.5, len(levels)-0.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("MPE (%, log scale)", fontsize=8)
    fig.text(0.5, -0.06, AXES[axis][2], ha="center", fontsize=8)
    # Single shared legend (three classes) at the top of the figure
    handles = [h for h in [hv, hc, hx] if h is not None]
    labs = [l for h,l in zip([hv,hc,hx],["valid","catastrophic (>100%)","numerical crash"]) if h is not None]
    fig.legend(handles, labs, loc="upper center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/final_{axis}_perseed.pdf"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for axis in AXES:
        fig_heatmap(axis)
        fig_perseed(axis)
    print(f"Final figures written to {OUT}/final_*.pdf:")
    for f in sorted(os.listdir(OUT)):
        if f.startswith("final_"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
