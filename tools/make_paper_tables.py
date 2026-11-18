#!/usr/bin/env python3
"""Generate revised LaTeX tables from the shared descriptive analysis."""
import argparse
from pathlib import Path
from analysis import AXES, OUTPUT, RECORDS, SCENES, report


def display(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def table(caption, label, columns, header, rows, wide=False):
    environment = "table*" if wide else "table"
    return "\n".join([
        rf"\begin{{{environment}}}[t]", r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header+r" \\",
        r"\midrule", *[row+r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}", "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=OUTPUT/"tables")
    args = parser.parse_args()
    result = report(args.records)
    args.out.mkdir(parents=True, exist_ok=True)
    cells = {(r["axis"], r["dataset"], r["scene"], r["level"]): r for r in result["cells"]}
    for axis, (_, _, levels, name) in AXES.items():
        rows = []
        for scene, label in SCENES.items():
            for i, level in enumerate(levels):
                s = cells[(axis, *scene, level)]
                interval = f"[{100*s['failure_ci_low']:.1f}, {100*s['failure_ci_high']:.1f}]"
                level_text = str(int(level)//1000) if axis == "time" else level
                median = display(s["median_mpe"])
                if s["within_threshold"] < 3:
                    median += r"$^\dagger$"
                rows.append(" & ".join([
                    label if i == 0 else "", level_text, str(s["n"]), str(s["within_threshold"]),
                    str(s["crash"]), str(s["extreme"]), f"{100*s['failure_rate']:.0f}", interval,
                    median, display(s["worst_mpe"]), display(s["iqr_mpe"]),
                ]))
        caption = (name + r": observed outcomes. W/C/X denote completed within the 100\% MPE threshold, "
                   r"numerical crash, and completed above the threshold. Failure is C+X. "
                   r"Intervals are Wilson 95\%; median/worst/IQR use W runs only. "
                   r"$\dagger$ marks fewer than three W runs. Timestamp levels are in ms.")
        text = table(caption, f"tab:revised_{axis}", "llrrrrrrrrr",
                     r"Sequence & Level & $n$ & W & C & X & Fail (\%) & 95\% CI & Median & Worst & IQR", rows, True)
        (args.out/f"{axis}_outcomes.tex").write_text(text, encoding="utf-8")

    rows = []
    for scene, label in SCENES.items():
        values = []
        for axis in AXES:
            values.append("/".join(f"{100*cells[(axis,*scene,lv)]['failure_rate']:.0f}" for lv in AXES[axis][2]))
        rows.append(" & ".join([label, *values]))
    caption = (r"Observed failure percentages at the four levels of each axis. "
               r"Density: 1/.5/.3/.1; patches: 96/64/32/16; quantization: 0/1/5/10 ms. "
               r"Five runs per cell in these released records. These curves are exploratory.")
    (args.out/"Table1.tex").write_text(table(caption, "tab:revised_compact", "lccc",
        "Sequence & Density & Patches & Time", rows), encoding="utf-8")

    rows = []
    for axis in AXES:
        stats = result["axes"][axis]
        q1 = stats["q1"]
        for i, q3 in enumerate(stats["q3"]):
            q1text = f"{100*q1['mean_miss_fraction']:.1f} ({q1['partial_cells']} cells)" if i == 0 else ""
            rows.append(" & ".join([AXES[axis][3] if i == 0 else "", q1text,
                f"{q3['clean_level']} to {q3['degraded_level']}", str(q3["opposite"]),
                str(q3["directed_pairs"]), str(q3["ties"]), display(100*q3["opposite_fraction"], 1) if q3["opposite_fraction"] is not None else "--"]))
    caption = (r"Descriptive Q1 and Q3. Q1 is conditional on observed partial-failure cells. "
               r"Q3 compares completed-run MPE directions with same-pool medians; ties are separate. "
               r"Quantization levels here are in microseconds. Pair combinations reuse runs; no pair-level confidence interval is inferred.")
    (args.out/"single_run_descriptive.tex").write_text(table(caption, "tab:revised_q", "llrrrrr",
        r"Axis & Q1 (\%) & Comparison & Opposite & Directed & Ties & Q3 (\%)", rows, True), encoding="utf-8")

    rows = [f"{AXES[r['axis']][3]} & {r['threshold_mpe']} & {r['n']} & {r['crash']} & {r['extreme']}" for r in result["threshold_sensitivity"]]
    extreme_totals = [sum(r["extreme"] for r in result["threshold_sensitivity"] if r["threshold_mpe"] == t)
                      for t in (50, 100, 200)]
    (args.out/"threshold_sensitivity.tex").write_text(table(
        rf"Threshold sensitivity. Extreme completions total {extreme_totals[0]}, {extreme_totals[1]} and {extreme_totals[2]} at 50, 100 and 200\% MPE; "
        f"the numerical crash count is {result['totals']['crash']} throughout. Thresholds are operational, not application safety limits.",
        "tab:revised_threshold", "lrrrr", r"Axis & Threshold (\%) & $n$ & Crashes & Extreme", rows), encoding="utf-8")
    print(f"Tables: {args.out.resolve()}")


if __name__ == "__main__":
    main()
