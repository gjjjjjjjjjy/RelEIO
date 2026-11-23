#!/usr/bin/env python3
"""Fig. 1 of the paper: the RelEIO measurement pipeline (schematic, vector).

Five stages on light cards with a numbered header, one drawing each, and a one-line caption; the figure caption
carries the explanation. Drawn with matplotlib primitives at the final printed size (IEEE two-column width,
7.16 in). No data is read: the trajectories, records and the R(eps) curve are sketches, not results.

  python3 script/make_fig1_protocol.py            # writes paper/figures/Fig1_protocol.pdf and .png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _layout  # noqa: E402
from nature_style import INK, NAT_BLUE, NAT_GREEN, NAT_ORANGE, NAT_RED, apply_nature_style  # noqa: E402

apply_nature_style()
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]   # Arial has a separate bold face
OUT = os.path.join(_layout.FIGURES, "Fig1_protocol")

W, H = 7.16, 1.95
CARD = "#F2F4F7"
GREY = "#6b7178"
LINE = "#B9C0C8"
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

CW, GAP = 1.24, (W - 0.08 - 5 * 1.24) / 4
X0 = [0.04 + i * (CW + GAP) for i in range(5)]
Y0, Y1 = 0.05, 1.90
HEAD_Y = 1.72
MID = 0.98                                   # vertical centre of the drawing area (0.40 .. 1.56)


def text(x, y, s, size=6.5, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=size, zorder=8, **kw)


STAGES = [("Condition", r"input, estimator, $E$, $S$"),
          ("Repeated runs", r"$K$ runs on identical input"),
          ("Record, classify", "settings checked per run"),
          ("Score", "rigid fit, no scale"),
          ("Reliability", r"$R_c(\varepsilon)$ over all $K$ runs")]
for i, (x0, (title, cap)) in enumerate(zip(X0, STAGES)):
    ax.add_patch(FancyBboxPatch((x0, Y0), CW, Y1 - Y0, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=CARD, ec="none", zorder=1))
    ax.add_patch(Circle((x0 + 0.14, HEAD_Y), 0.075, fc=NAT_BLUE, ec="none", zorder=3))
    text(x0 + 0.14, HEAD_Y - 0.003, str(i + 1), size=6.5, color="white", fontweight="bold")
    text(x0 + 0.26, HEAD_Y, title, size=7.2, ha="left", fontweight="bold")
    text(x0 + CW / 2, 0.20, cap, size=6.5, color=GREY)
for x0 in X0[:-1]:                           # chevrons between cards
    cx = x0 + CW + GAP / 2
    ax.add_patch(Polygon([[cx - 0.045, MID + 0.07], [cx + 0.05, MID], [cx - 0.045, MID - 0.07]], closed=True,
                         fc="#8C96A1", ec="none", zorder=2))

rng = np.random.RandomState(11)

# ---- 1. condition: event stream as time slices, IMU trace, environment and scoring-rule tags ---------------
x0 = X0[0]
for k in range(5):                           # five slices of the event stream, receding in time
    ox, oy, s = x0 + 0.14 + 0.085 * k, 1.04 + 0.055 * k, 0.40
    ax.add_patch(FancyBboxPatch((ox, oy), s, s * 0.62, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=LINE, lw=0.6, zorder=3 + k))
    u = np.linspace(0.12, 0.88, 9)           # an edge moving across the sensor
    xe = ox + s * (0.25 + 0.10 * k + 0.04 * np.sin(5 * u))
    ye = oy + s * 0.62 * u
    pol = rng.rand(u.size) > 0.45
    ax.scatter(xe, ye, s=2.2, c=[NAT_BLUE if p else NAT_RED for p in pol], lw=0, zorder=3 + k)
text(x0 + 1.02, 1.49, "$t$", size=6.5, color=GREY)
ax.annotate("", xy=(x0 + 0.98, 1.43), xytext=(x0 + 0.80, 1.33),
            arrowprops=dict(arrowstyle="-|>", lw=0.6, color=GREY, mutation_scale=5))
tt = np.linspace(0, 1, 120)                  # IMU trace
ax.plot(x0 + 0.14 + 0.96 * tt, 0.88 + 0.045 * np.sin(2 * np.pi * 2.6 * tt) * np.exp(-0.4 * tt), color=NAT_GREEN,
        lw=0.9, zorder=4)
text(x0 + 1.10, 0.99, "IMU", size=5.6, color=NAT_GREEN, ha="right")
text(x0 + 0.14, 1.585, "events", size=5.6, color=GREY, ha="left")
for k, (lab, sub_) in enumerate([("$E$", "environment"), ("$S$", "scoring rule")]):
    py = 0.60 - 0.19 * k
    ax.add_patch(FancyBboxPatch((x0 + 0.12, py), 1.00, 0.15, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=LINE, lw=0.6, zorder=4))
    text(x0 + 0.22, py + 0.075, lab, size=7.0)
    text(x0 + 0.33, py + 0.075, sub_, size=5.6, color=INK, ha="left")

# ---- 2. repeated runs: smooth fan from a common start -------------------------------------------------------
x0 = X0[1]
t = np.linspace(0, 1, 200)
bx = x0 + 0.16 + 0.92 * t
by = 0.62 + 0.62 * t + 0.16 * np.sin(np.pi * t)
for k in range(7):
    dev = (0.06 * np.sin(2.2 * np.pi * t + k) + 0.10 * (k - 3) / 3) * t ** 1.6
    ax.plot(bx, by + dev, color=NAT_BLUE, lw=0.9, alpha=0.55 + 0.06 * k, zorder=4, solid_capstyle="round")
div = by + 0.55 * np.clip(t - 0.35, 0, None) ** 1.8 * 2.2                        # a divergent run
m = div < 1.56
ax.plot(bx[m], div[m], color=NAT_ORANGE, lw=1.0, zorder=5)
fail = by - 0.30 * t ** 1.4                                                       # a run that stops
mf = t < 0.58
ax.plot(bx[mf], fail[mf], color=NAT_RED, lw=1.0, zorder=5)
ax.scatter([bx[mf][-1]], [fail[mf][-1]], marker="x", s=24, c=NAT_RED, lw=1.3, zorder=6)
ax.scatter([bx[0]], [by[0]], s=12, c=INK, lw=0, zorder=6)

# ---- 3. record and classify: one row per run, status at the right; one run excluded -------------------------
x0 = X0[2]
rows = [("o", NAT_BLUE), ("o", NAT_BLUE), ("x", NAT_RED), ("o", NAT_BLUE), ("excl", GREY), ("?", GREY),
        ("o", NAT_BLUE)]
rh, top = 0.135, 1.50
for k, (mk, c) in enumerate(rows):
    y = top - k * (rh + 0.035)
    excluded = mk == "excl"
    ax.add_patch(FancyBboxPatch((x0 + 0.12, y - rh), 0.82, rh, boxstyle="round,pad=0,rounding_size=0.025",
                                fc="white", ec=LINE, lw=0.5, ls=(0, (2, 1.5)) if excluded else "-", zorder=3))
    for j, w in enumerate((0.18, 0.22, 0.16)):            # record fields
        fx = x0 + 0.17 + sum((0.18, 0.22, 0.16)[:j]) + 0.03 * j
        ax.plot([fx, fx + w], [y - rh / 2] * 2, color="#D5DAE0" if not excluded else "#E6E9ED", lw=2.2,
                solid_capstyle="round", zorder=4)
    sx = x0 + 1.06
    if mk == "o":
        ax.scatter([sx], [y - rh / 2], s=16, c=c, lw=0, zorder=5)
    elif mk == "x":
        ax.scatter([sx], [y - rh / 2], marker="x", s=18, c=c, lw=1.2, zorder=5)
    elif mk == "?":
        text(sx, y - rh / 2, "?", size=7, color=c, fontweight="bold")
    else:
        text(sx, y - rh / 2, "excl.", size=4.8, color=c)

# ---- 4. score: estimate aligned to ground truth by a rigid transform; position errors ----------------------
x0 = X0[3]
cx, cy = x0 + CW / 2, MID + 0.02
th = np.linspace(0, 2 * np.pi, 240)
gx = cx + 0.42 * np.sin(th)
gy = cy + 0.30 * np.sin(th) * np.cos(th) * 1.6 + 0.05 * np.cos(th)
ex = gx + 0.035 * np.sin(2 * th + 0.5) + 0.02
ey = gy + 0.035 * np.cos(3 * th) - 0.015
for k in range(0, th.size, 16):
    ax.plot([gx[k], ex[k]], [gy[k], ey[k]], color=NAT_RED, lw=0.7, zorder=4)
ax.plot(gx, gy, color=INK, lw=1.0, zorder=5)
ax.plot(ex, ey, color=NAT_BLUE, lw=1.0, ls=(0, (2.6, 1.3)), zorder=5)
text(x0 + 0.17, 1.47, "ground truth", size=5.5, ha="left")
ax.plot([x0 + 0.66, x0 + 0.78], [1.47, 1.47], color=INK, lw=1.0)
text(x0 + 0.17, 1.37, "estimate", size=5.5, ha="left", color=NAT_BLUE)
ax.plot([x0 + 0.66, x0 + 0.78], [1.37, 1.37], color=NAT_BLUE, lw=1.0, ls=(0, (2.6, 1.3)))

# ---- 5. reliability curve ------------------------------------------------------------------------------------
x0 = X0[4]
px0, py0, pw, ph = x0 + 0.22, 0.46, 0.90, 1.02
for gy_ in (0.25, 0.5, 0.75, 1.0):
    ax.plot([px0, px0 + pw], [py0 + gy_ * ph * 0.9] * 2, color="#E2E6EA", lw=0.5, zorder=2)
ax.plot([px0, px0 + pw], [py0, py0], color=INK, lw=0.6, zorder=4)
ax.plot([px0, px0], [py0, py0 + ph], color=INK, lw=0.6, zorder=4)
K, done = 20, 16
xs = np.sort(np.clip(rng.lognormal(-1.2, 0.55, done), 0.04, 0.9))
sxp = lambda v: px0 + v * pw                                     # noqa: E731
syp = lambda r: py0 + r * ph * 0.9                               # noqa: E731
stx = [sxp(0)] + [sxp(v) for v in xs for _ in (0, 1)] + [sxp(1.0)]
sty = [syp(0)] + [syp(r) for k in range(done) for r in (k / K, (k + 1) / K)] + [syp(done / K)]
ax.fill_between(stx, py0, sty, color=NAT_BLUE, alpha=0.12, lw=0, zorder=3)
ax.plot(stx, sty, color=NAT_BLUE, lw=1.1, zorder=5, solid_joinstyle="miter")
ax.plot([px0, px0 + pw], [syp(1), syp(1)], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=4)
ax.annotate("", xy=(px0 + pw - 0.05, syp(done / K)), xytext=(px0 + pw - 0.05, syp(1)),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color=NAT_RED, shrinkA=0, shrinkB=0), zorder=6)
k_star = 10
es = 0.5 * (xs[k_star - 1] + xs[k_star])
ax.plot([sxp(es)] * 2, [py0, syp(k_star / K)], color=INK, lw=0.6, ls=(0, (1.5, 1.5)), zorder=4)
ax.plot([px0, sxp(es)], [syp(k_star / K)] * 2, color=INK, lw=0.6, ls=(0, (1.5, 1.5)), zorder=4)
ax.scatter([sxp(es)], [syp(k_star / K)], s=10, c=INK, lw=0, zorder=6)
text(sxp(es), py0 - 0.07, r"$\varepsilon^{*}$", size=6.5)
text(px0 - 0.04, syp(1), "1", size=6.0, ha="right", color=GREY)
text(px0 - 0.04, py0, "0", size=6.0, ha="right", color=GREY)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".pdf", bbox_inches=None, pad_inches=0)
fig.savefig(OUT + ".png", dpi=300, bbox_inches=None, pad_inches=0)
print("wrote", _layout.rel(OUT + ".pdf"))
