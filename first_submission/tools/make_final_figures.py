#!/usr/bin/env python3
"""Generate revised vector figures from the released records."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from analysis import AXES, OUTPUT, RECORDS, SCENES, SYSTEMS, load_main, report, success_curve, summarize

BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, out, name):
    fig.savefig(out/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out/f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(9.3, 3.6))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    labels = [
        (0.1, 2.5, "Declared condition\nsequence, sensing stress,\ncompute setting"),
        (3.5, 2.5, "Explicit input preparation\nevents / IMU, units,\nwindowing and representation"),
        (6.9, 2.5, "Repeated execution\nrecord actual seed, environment,\nexit status and trajectory"),
        (6.9, 0.6, "Recorded outcome\ncrash / completed above threshold /\ncompleted within threshold"),
        (3.5, 0.6, "Condition-level report\ncounts and Wilson interval;\nconditional error summaries"),
        (0.1, 0.6, "Interpretation\nobserved benchmark conditions;\nmechanism requires diagnosis"),
    ]
    for x, y, label in labels:
        ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.05, boxstyle="round,pad=0.05",
                                  facecolor="#EFF6FA", edgecolor=BLUE))
        ax.text(x+1.45, y+.525, label, ha="center", va="center", fontsize=8.2)
    for start, end in [((3.05, 3), (3.4, 3)), ((6.45, 3), (6.8, 3)),
                       ((8.35, 2.4), (8.35, 1.75)), ((6.8, 1.1), (6.45, 1.1)),
                       ((3.4, 1.1), (3.05, 1.1))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=BLUE))
    save(fig, out, "Fig1")


def heatmaps(axes, out):
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    for ax, (axis, data) in zip(axs, axes.items()):
        levels = AXES[axis][2]
        matrix = np.array([[100*summarize(data[(*scene, level)])["failure_rate"] for level in levels] for scene in SCENES])
        im = ax.imshow(matrix, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(levels)), [str(int(x)//1000) for x in levels] if axis == "time" else levels)
        ax.set_yticks(range(len(SCENES)), list(SCENES.values()))
        ax.set_xlabel("Quantization (ms)" if axis == "time" else AXES[axis][3])
        for i in range(len(SCENES)):
            for j in range(len(levels)):
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                        color="white" if matrix[i,j] >= 60 else "black", fontsize=8)
    fig.colorbar(im, ax=axs, label="Observed failure (%)", shrink=.8)
    save(fig, out, "Fig2")


def single_run(result, out):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    labels = ["Event\ndensity", "Patch\nbudget", "Timestamp\nquantization"]
    q1 = [result["axes"][axis]["q1"] for axis in AXES]
    axs[0].bar(range(3), [100*r["mean_miss_fraction"] for r in q1], color=BLUE)
    for i, row in enumerate(q1):
        axs[0].text(i, 100*row["mean_miss_fraction"]+2, f"{row['partial_cells']} partial cells", ha="center", fontsize=8)
    axs[0].set(title="Q1: conditional empirical miss fraction", ylabel="Observed fraction (%)", ylim=(0, 100))
    q3 = [result["axes"][axis]["q3"][0] for axis in AXES]
    axs[1].bar(range(3), [100*r["opposite_fraction"] for r in q3], color=ORANGE)
    for i, row in enumerate(q3):
        axs[1].text(i, 100*row["opposite_fraction"]+2, f"{row['opposite']}/{row['directed_pairs']}\n{row['ties']} ties separate", ha="center", fontsize=8)
    axs[1].set(title="Q3: strict opposite MPE direction", ylabel="Directed comparisons (%)", ylim=(0, 100))
    for ax in axs:
        ax.set_xticks(range(3), labels)
    save(fig, out, "Fig3")


def matched(result, out):
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.2), layout="constrained")
    for ax, comparison in zip(axs.flat, SYSTEMS):
        data = [r for r in result["matched_systems"] if r["comparison"] == comparison]
        levels = sorted({r["keep"] for r in data}, key=float, reverse=True)
        for index, (system, color) in enumerate([("DEIO", BLUE), (comparison, ORANGE)]):
            for i, level in enumerate(levels):
                rows = [r for r in data if r["system"] == system and r["keep"] == level]
                n, crash = sum(r["n"] for r in rows), sum(r["crash"] for r in rows)
                rate = crash/n
                ax.plot(i+(index-.5)*.18, rate*100, "o", color=color,
                        label=system if i == 0 else None)
                ax.annotate(f"{crash}/{n}", (i+(index-.5)*.18, rate*100), xytext=(0, -14 if index == 0 else 8),
                            textcoords="offset points", ha="center", fontsize=7)
        scenes = len({(r["dataset"], r["scene"]) for r in data})
        ax.set(title=f"Matched with {comparison} ({scenes} scenes)", ylim=(-10, 112),
               xlabel="Event keep ratio", ylabel="Numerical crash (%)")
        ax.set_xticks(range(len(levels)), levels)
        ax.legend(fontsize=7, loc="upper right")
    save(fig, out, "Fig4")


def loopclosure(result, out):
    data = [r for r in result["matched_systems"] if r["comparison"] == "DEIO + loop closure"]
    scenes = sorted({(r["dataset"], r["scene"]) for r in data})
    fig, axs = plt.subplots(1, len(scenes), figsize=(9, 3.3), layout="constrained")
    levels = ["1.0", "0.5", "0.3", "0.1"]
    for ax, scene in zip(axs, scenes):
        for system, color in [("DEIO", BLUE), ("DEIO + loop closure", ORANGE)]:
            rows = [next(r for r in data if (r["dataset"], r["scene"]) == scene and r["keep"] == lv and r["system"] == system) for lv in levels]
            ax.plot(range(4), [100*r["crash_rate"] for r in rows], "o-", color=color, label=system)
        ax.set(title=SCENES[scene], ylim=(-5, 105), xlabel="Event keep ratio", ylabel="Numerical crash (%)")
        ax.set_xticks(range(4), levels)
    axs[-1].legend(fontsize=7)
    save(fig, out, "FigS5")


def sensitivity(result, out):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    thresholds = [50, 100, 200]
    colors = [BLUE, ORANGE, GRAY]
    for index, axis in enumerate(AXES):
        rows = [next(r for r in result["threshold_sensitivity"] if r["axis"] == axis and r["threshold_mpe"] == t) for t in thresholds]
        for ax, metric in zip(axs, ["crash", "extreme"]):
            ax.bar(np.arange(3)+(index-1)*.23, [r[metric] for r in rows], width=.23, color=colors[index], label=AXES[axis][3])
    for ax in axs:
        ax.set_xticks(range(3), thresholds)
        ax.set_xlabel("MPE threshold (%)")
        ax.set_ylabel("Run count")
    axs[0].set_title("Numerical crashes")
    axs[1].set_title("Completed above threshold")
    axs[0].legend(fontsize=7)
    save(fig, out, "FigS6")


def curves(axes, out):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, (axis, data) in zip(axs, axes.items()):
        for level in AXES[axis][2]:
            runs = [run for key, values in data.items() if key[2] == level for run in values]
            rows = success_curve(runs)
            positive = [r for r in rows if r["threshold_mpe"] > 0]
            ax.step([r["threshold_mpe"] for r in positive], [100*r["successes"]/r["n"] for r in positive],
                    where="post", label=level)
        ax.set(xscale="log", ylim=(0, 103), title=AXES[axis][3],
               xlabel="MPE tolerance (%)", ylabel="Completed within tolerance (%)")
        ax.legend(title="Level (us for time)", fontsize=7, title_fontsize=7)
    save(fig, out, "success_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=OUTPUT/"figures")
    args = parser.parse_args()
    axes = load_main(args.records)
    result = report(args.records)
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline(args.out)
    heatmaps(axes, args.out)
    single_run(result, args.out)
    matched(result, args.out)
    loopclosure(result, args.out)
    sensitivity(result, args.out)
    curves(axes, args.out)
    print(f"Seven revised PDF/SVG figures: {args.out.resolve()}")


if __name__ == "__main__":
    main()
