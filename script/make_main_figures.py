#!/usr/bin/env python3
"""Data figures of the revised manuscript (Figs. 2, 4, 5), drawn from the committed per-run CSV files.

  Fig2_numerics  (used in the supplementary) (a) numerical failures per condition, both arms, pooled over 8 sequences (Wilson 95%), with the
                     per-sequence shares as small dots; (b) the replication on four GPU models
  Fig4_scoring   every override-removed run: rule A (DEIO's evaluation code) against rule D (SE(3), no scale)
  Fig5_bias      every run of the common-gyroscope-bias experiment, both systems, two sequences

Inputs (records directory, see script/_layout.py): round4_all_attribution_*.csv, deio_se3_round4_steps_*.csv,
imu_bias_cross_system_pinned_*.csv, all_rounds_attribution_*.csv (every DEIO run of every round, with ledger and
input condition; Fig. 2(b) uses its round 3A runs on the RTX 4090 and the three hardware-replication ledgers).

  python3 script/make_main_figures.py          # writes Fig{2,4,5}_*.pdf/.png into the figures directory
"""
import collections
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diag"))
import _layout  # noqa: E402
import nature_style as ns  # noqa: E402
import precision_pilot as P  # noqa: E402

ns.apply_nature_style()
AN, FIG = _layout.RECORDS, _layout.FIGURES
os.makedirs(FIG, exist_ok=True)
COL = 3.5                                        # IEEE single-column width (in)
C_DEF, C_OFF = ns.NAT_ORANGE, ns.NAT_BLUE        # container default / override removed, as in Fig. 3
GREY = "#6b7178"
# datasets: hue + marker (validated palette; the red/green pair is separated by marker shape)
DS = collections.OrderedDict([
    ("tumvie", ("TUM-VIE", "#2F5E9E", "o")),
    ("hku", ("HKU", "#E08A1E", "s")),
    ("fpv", ("UZH-FPV", "#16865A", "^")),
    ("vector", ("VECtor", "#C0392B", "D")),
    ("rpgdavis", ("DAVIS240C", "#7E57C2", "v")),
])
CELLS = [("clean", "clean"), ("keep:0.1", r"$\rho$"), ("tq:10000", r"$\tau$"), ("patch:16", r"$P$")]


def one(pattern):
    f = sorted(glob.glob(os.path.join(AN, pattern)))
    assert len(f) == 1, (pattern, f)
    return list(csv.DictReader(open(f[0])))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    print("wrote %s.pdf" % _layout.rel(os.path.join(FIG, name)))


def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


# ---------------------------------------------------------------- Fig. 2: numerical environment
att = one("round4_all_attribution_*.csv")
assert len(att) == 640
fail = collections.Counter((r["scene"], r["cell"], r["arm"]) for r in att if r["category"] == "numerical_failure")
scenes = sorted({r["scene"] for r in att})
fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))
for i, (cell, lab) in enumerate(CELLS):
    for arm, c, dx in (("default", C_DEF, -0.17), ("no_tf32_override", C_OFF, 0.17)):
        k = sum(fail[(s, cell, arm)] for s in scenes)
        lo, hi = P.wilson(k, 80)
        per = [fail[(s, cell, arm)] / 10 for s in scenes]
        jit = np.linspace(-0.07, 0.07, len(per))
        a.scatter(i + dx + jit, 100 * np.array(per), s=5, color=c, alpha=0.35, lw=0, zorder=2)
        a.errorbar(i + dx, 100 * k / 80, yerr=[[100 * (k / 80 - lo)], [100 * (hi - k / 80)]], fmt="o", ms=3.6,
                   color=c, mec="white", mew=0.5, elinewidth=1.0, capsize=0, zorder=4)
        if arm == "default":
            a.text(i + dx - 0.12, 100 * k / 80, "%d" % k, fontsize=6.3, ha="right", va="center", color=ns.INK)
a.set_xticks(range(4))
a.set_xticklabels([l for _, l in CELLS], fontsize=7)
a.set_xlim(-0.55, 3.55)
a.set_ylim(-3, 100)
a.set_ylabel("runs with a crash (%)")
a.yaxis.grid(True, color=ns.GREY_FAINT, lw=0.4)
a.legend(handles=[Line2D([], [], color=C_DEF, marker="o", ls="", ms=3.6, label="container default"),
                  Line2D([], [], color=C_OFF, marker="o", ls="", ms=3.6, label="override removed")],
         loc="upper right", fontsize=6.3, handletextpad=0.3, borderaxespad=0.2)
panel(a, "(a)")
# (b) four GPU models, corner_slow1: event density 0.1 and clean (F17, F18, F24)
allr = [r for r in one("all_rounds_attribution_*.csv") if r["scene"] == "vector/corner_slow1"]
gpus = []
for name, ledger in [("RTX 4090", "round3a_"), ("L40S", "hardware_l40s_"), ("H200", "hardware_h200_"), ("H100", "hardware_h100_")]:
    k = {}
    for cell in ("keep:0.1", "keep:1.0"):                       # reduced event density, clean
        for arm in ("default", "no_tf32_override"):
            g = [r for r in allr if r["ledger"].startswith(ledger) and r["cell"] == cell and r["arm"] == arm]
            assert len(g) == 10, (name, cell, arm, len(g))
            k[cell, arm] = sum(r["outcome"].startswith("numerical_failure") for r in g)
    assert k["keep:0.1", "no_tf32_override"] == 0 and k["keep:1.0", "no_tf32_override"] == 0, (name, k)
    gpus.append((name, k["keep:0.1", "default"], k["keep:1.0", "default"]))
assert gpus == [("RTX 4090", 8, 1), ("L40S", 9, 1), ("H200", 7, 3), ("H100", 9, 2)], gpus   # supplementary Table S3
y = np.arange(len(gpus))[::-1]
for yy, (g, kd, kc) in zip(y, gpus):
    b.plot([0, kd], [yy + 0.13] * 2, color=C_DEF, lw=2.2, solid_capstyle="butt")
    b.plot([0, kc], [yy - 0.13] * 2, color=C_DEF, lw=2.2, alpha=0.45, solid_capstyle="butt")
    first = yy == y[0]
    b.text(kd + 0.25, yy + 0.13, "%d" % kd + (r"  $\rho\!=\!0.1$" if first else ""), fontsize=6.3, va="center")
    b.text(kc + 0.25, yy - 0.13, "%d" % kc + ("  clean" if first else ""), fontsize=6.3, va="center")
b.set_yticks(y)
b.set_yticklabels([g for g, _, _ in gpus], fontsize=6.5)
b.set_xlim(0, 13)
b.set_xticks([0, 5, 10])
b.set_ylim(-0.95, len(gpus) - 0.4)
b.set_xlabel("failures in 10 runs")
b.text(12.6, -0.72, "override removed: 0 in every cell", fontsize=6.0, color=C_OFF, ha="right", va="center")
panel(b, "(b)")
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.92)
save(fig, "Fig2_numerics")

# ---------------------------------------------------------------- Fig. 4: scoring rule
steps = one("deio_se3_round4_steps_*.csv")
assert len(steps) == 592
fig, a = plt.subplots(figsize=(COL, 2.2))
off = [r for r in steps if r["arm"] == "no_tf32_override"]
for key, (lab, c, mk) in DS.items():
    g = [r for r in off if r["scene"].startswith(key)]
    a.scatter([float(r["se3_all"]) for r in g], [float(r["deio_rule"]) for r in g], s=9, marker=mk, color=c,
              lw=0.3, edgecolors="white", alpha=0.9, label=lab, zorder=3)
lim = (0.03, 1e5)
mx = max(float(r["se3_all"]) for r in off)
assert mx < lim[1], mx   # every run inside the axes
a.plot(lim, lim, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
a.axvline(10, color=ns.INK, lw=0.5, ls=(0, (1.5, 1.5)))
a.axhline(10, color=ns.INK, lw=0.5, ls=(0, (1.5, 1.5)))
n86 = sum(float(r["deio_rule"]) <= 10 < float(r["se3_all"]) for r in off)
nrev = sum(float(r["se3_all"]) <= 10 < float(r["deio_rule"]) for r in off)
assert nrev == 0
a.fill_between([10, lim[1]], lim[0], 10, color=ns.NAT_RED, alpha=0.06, lw=0, zorder=0)
a.text(5e3, 0.06, "%d runs: within 10%% under\nrule A, beyond under rule D" % n86, fontsize=6.3, ha="right",
       va="bottom", color=ns.NAT_RED)
a.text(8e3, 3e3, "$y = x$", fontsize=6.3, color=GREY, rotation=40, ha="center", va="center")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(*lim)
a.set_ylim(*lim)
a.set_xlabel("MPE, rule D: SE(3), no scale (%)")
a.set_ylabel("MPE, rule A: DEIO's evaluation (%)")
a.legend(loc="upper left", fontsize=6.0, handletextpad=0.1, borderaxespad=0.3, labelspacing=0.25, markerscale=1.1)
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.97)
save(fig, "Fig4_scoring")

# ---------------------------------------------------------------- Fig. 5: common gyroscope bias
bias = one("imu_bias_cross_system_pinned_*.csv")
assert len(bias) == 80 and all(r["status"] == "completed" for r in bias)
g = collections.defaultdict(list)
for r in bias:
    g[(r["system"], r["seq"], r["cond"])].append(float(r["se3_W"]))
groups = [("DEIO", "boxes_6dof"), ("DEIO", "poster_6dof"), ("SuperEIO", "boxes_6dof"), ("SuperEIO", "poster_6dof")]
fig, a = plt.subplots(figsize=(COL, 1.7))
C_CLEAN, C_BIAS = ns.INK, ns.NAT_RED
for i, key in enumerate(groups):
    for cond, c, dx, mk in (("clean", C_CLEAN, -0.17, "o"), ("gbias", C_BIAS, 0.17, "^")):
        v = np.array(sorted(g[key + (cond,)]))
        assert len(v) == 10
        jit = np.linspace(-0.07, 0.07, len(v))
        a.scatter(i + dx + jit, v, s=11, marker=mk, facecolors="none", edgecolors=c, lw=0.8, zorder=3)
        med = np.median(v)
        a.plot([i + dx - 0.12, i + dx + 0.12], [med, med], color=c, lw=1.4, zorder=4)
a.axhline(100, color=GREY, lw=0.5, ls=(0, (2, 2)))
a.text(3.62, 100, "path length", fontsize=6.0, color=GREY, ha="right", va="bottom")
a.set_yscale("log")
a.set_ylim(0.15, 6e3)
a.set_xticks(range(4))
a.set_xticklabels(["DEIO\nboxes", "DEIO\nposter", "SuperEIO\nboxes", "SuperEIO\nposter"], fontsize=6.5)
a.set_xlim(-0.55, 3.65)
a.set_ylabel("MPE, rule D (%)")
for xs in (0.5, 1.5, 2.5):
    a.axvline(xs, color=ns.GREY_FAINT, lw=0.6)
a.legend(handles=[Line2D([], [], color=C_CLEAN, marker="o", mfc="none", ls="", ms=3.5, label="clean"),
                  Line2D([], [], color=C_BIAS, marker="^", mfc="none", ls="", ms=3.5,
                         label="gyroscope bias"),
                  Line2D([], [], color=GREY, lw=1.4, label="median")],
         loc="upper right", fontsize=6.0, ncol=3, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.21, top=0.97)
save(fig, "Fig5_bias")
