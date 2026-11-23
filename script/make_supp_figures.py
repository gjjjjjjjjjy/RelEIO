#!/usr/bin/env python3
"""Figures of the supplementary material, drawn from the committed per-run CSV files of the round-4 main matrix.

  FigS1_reliability_per_sequence  R(eps) per sequence (40 runs per arm: 4 conditions x 10 seeds), both arms,
                                  scale-free SE(3) rule (solid) and rule A of DEIO's evaluation code (dashed)
  FigS2_runs                      every run: SE(3) MPE of completed runs and numerical failures (crosses on
                                  top), per sequence, condition and arm

  python3 script/make_supp_figures.py          # writes paper/figures/FigS{1,2}_*.pdf/.png
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
import _layout  # noqa: E402
import nature_style as ns  # noqa: E402

ns.apply_nature_style()
AN, FIG = _layout.RECORDS, _layout.FIGURES
os.makedirs(FIG, exist_ok=True)
C_DEF, C_OFF = ns.NAT_ORANGE, ns.NAT_BLUE
GREY = "#6b7178"
SEQ = [("tumvie/mocap-3d-trans", "TUM-VIE mocap-3d-trans"), ("hku/vicon_dark1", "HKU vicon_dark1"),
       ("fpv/indoor_forward_3", "UZH-FPV indoor_forward_3"), ("vector/corner_slow1", "VECtor corner_slow1"),
       ("vector/desk_normal1", "VECtor desk_normal1"), ("rpgdavis/shapes_6dof", "DAVIS240C shapes_6dof"),
       ("rpgdavis/poster_6dof", "DAVIS240C poster_6dof"), ("rpgdavis/boxes_6dof", "DAVIS240C boxes_6dof")]
CELLS = [("clean", "clean"), ("keep:0.1", r"$\rho$"), ("tq:10000", r"$\tau$"), ("patch:16", r"$P$")]
ARMS = [("default", C_DEF, "container default"), ("no_tf32_override", C_OFF, "override removed")]


def one(pattern):
    f = sorted(glob.glob(os.path.join(AN, pattern)))
    assert len(f) == 1, (pattern, f)
    return list(csv.DictReader(open(f[0])))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    print("wrote %s.pdf" % _layout.rel(os.path.join(FIG, name)))


att = one("round4_all_attribution_*.csv")
steps = {r["run"]: r for r in one("deio_se3_round4_steps_*.csv")}
runs = []
for r in att:
    s = steps.get(r["run_id"])
    runs.append(dict(scene=r["scene"], cell=r["cell"], arm=r["arm"], cat=r["category"],
                     D=float(s["se3_all"]) if s else np.inf, A=float(s["deio_rule"]) if s else np.inf))
assert len(runs) == 640 and sum(r["cat"] == "completed" for r in runs) == 592

# ---- S1: reliability per sequence
eps = np.logspace(-1.3, 4.3, 600)
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.5), sharex=True, sharey=True)
for ax, (key, title) in zip(axes.flat, SEQ):
    for arm, c, _ in ARMS:
        g = [r for r in runs if r["scene"] == key and r["arm"] == arm]
        assert len(g) == 40
        for rule, ls in (("D", "-"), ("A", (0, (3, 2)))):
            v = np.array([r[rule] if r["cat"] == "completed" else np.inf for r in g])
            ax.plot(eps, [(v <= e).mean() for e in eps], color=c, lw=1.0, ls=ls)
    ax.axvline(100, color=GREY, lw=0.4, ls=(0, (1.5, 1.5)))
    ax.set_xscale("log")
    ax.set_title(title.replace("_", r"\_") if False else title, fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.grid(True, color=ns.GREY_FAINT, lw=0.4)
for ax in axes[1]:
    ax.set_xlabel("tolerance on MPE (%)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$R(\varepsilon)$, 40 runs")
fig.legend(handles=[Line2D([], [], color=C_OFF, lw=1.0, label="override removed"),
                    Line2D([], [], color=C_DEF, lw=1.0, label="container default"),
                    Line2D([], [], color=ns.INK, lw=1.0, label="rule D (SE(3), no scale)"),
                    Line2D([], [], color=ns.INK, lw=1.0, ls=(0, (3, 2)), label="rule A (DEIO's evaluation)")],
           loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.93, wspace=0.12, hspace=0.35)
save(fig, "FigS1_reliability_per_sequence")

# ---- S2: every run
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharey=True)
top = 3e4
for ax, (key, title) in zip(axes.flat, SEQ):
    for i, (cell, _) in enumerate(CELLS):
        for arm, c, _ in ARMS:
            dx = -0.18 if arm == "default" else 0.18
            g = [r for r in runs if r["scene"] == key and r["cell"] == cell and r["arm"] == arm]
            assert len(g) == 10
            comp = sorted(r["D"] for r in g if r["cat"] == "completed")
            jit = np.linspace(-0.08, 0.08, max(len(comp), 1))[:len(comp)]
            inside = [(x, v) for x, v in zip(i + dx + jit, comp) if v <= top]
            above = [x for x, v in zip(i + dx + jit, comp) if v > top]   # drawn at the top edge
            if inside:
                ax.scatter(*zip(*inside), s=6, facecolors="none", edgecolors=c, lw=0.6, zorder=3)
            if above:
                ax.scatter(above, [top] * len(above), s=10, marker="^", facecolors=c, edgecolors=c, lw=0.4,
                           zorder=4, clip_on=False)
            nf = sum(r["cat"] == "numerical_failure" for r in g)
            if nf:
                ax.scatter(i + dx + np.linspace(-0.08, 0.08, nf), [top * 2.2] * nf, marker="x", s=7, c=ns.NAT_RED,
                           lw=0.7, zorder=3, clip_on=False)
    ax.axhline(100, color=GREY, lw=0.4, ls=(0, (1.5, 1.5)))
    ax.set_yscale("log")
    ax.set_ylim(0.03, top)
    ax.set_xticks(range(4))
    ax.set_xticklabels([l for _, l in CELLS], fontsize=7)
    ax.set_xlim(-0.6, 3.6)
    ax.set_title(title, fontsize=7, pad=9)
for ax in axes[:, 0]:
    ax.set_ylabel("MPE, rule D (%)")
fig.legend(handles=[Line2D([], [], color=C_OFF, marker="o", mfc="none", ls="", ms=3.5, label="override removed"),
                    Line2D([], [], color=C_DEF, marker="o", mfc="none", ls="", ms=3.5, label="container default"),
                    Line2D([], [], color=ns.NAT_RED, marker="x", ls="", ms=3.5, label="numerical failure (above the panel)")],
           loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.92, wspace=0.12, hspace=0.45)
save(fig, "FigS2_runs")
