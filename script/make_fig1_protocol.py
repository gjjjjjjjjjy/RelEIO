#!/usr/bin/env python3
"""Fig. 1 of the paper: the RelEIO measurement pipeline (schematic, vector).

Five framed panels, left to right: the condition (sensor input, estimator configuration, environment E, scoring
rule S), the K repeated runs, the per-run record with its check and outcome, the scoring of a completed run, and
the reliability curve. Drawn with matplotlib primitives at the final printed size (IEEE two-column width,
7.16 in). No data is read: the event cloud, the IMU traces, the trajectories, the record and the R(eps) curve
are simulated sketches, not results.

  python3 script/make_fig1_protocol.py            # writes paper/figures/Fig1_protocol.pdf and .png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _layout  # noqa: E402
from nature_style import INK, NAT_BLUE, NAT_GREEN, NAT_ORANGE, NAT_PURPLE, NAT_RED, apply_nature_style  # noqa: E402

apply_nature_style()
plt.rcParams.update({"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],     # Arial has a separate bold face
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})
MONO = ["Menlo", "DejaVu Sans Mono"]
OUT = os.path.join(_layout.FIGURES, "Fig1_protocol")

W, H = 7.16, 1.95
FRAME = "#8F98A1"                            # panel outline
HEAD = "#EDF0F3"                             # header band
GREY = "#687079"                             # secondary text
RULE = "#CDD2D8"                             # hairlines
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

WIDTHS = [1.64, 1.18, 1.40, 1.14, 1.24]
GAP = (W - 0.04 - sum(WIDTHS)) / 4
X0 = [0.02 + sum(WIDTHS[:i]) + i * GAP for i in range(5)]
Y0, Y1 = 0.02, 1.93
TOP = Y1 - 0.24                              # lower edge of the header band
MID = (Y0 + TOP) / 2


def text(x, y, s, size=5.8, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, zorder=8, **kw)


def hline(xa, xb, y, color=RULE, lw=0.5):
    ax.plot([xa, xb], [y, y], color=color, lw=lw, zorder=3, solid_capstyle="butt")


def check(x, y, a=0.028, color=GREY):
    ax.plot([x - a, x - 0.3 * a, x + a], [y, y - 0.75 * a, y + 0.85 * a], color=color, lw=0.9, zorder=6,
            solid_capstyle="round", solid_joinstyle="round")


TITLES = ["Condition", "Repeated runs", "Record and classify", "Score", "Reliability"]
CLIP = []
for x0, w, title in zip(X0, WIDTHS, TITLES):
    box = FancyBboxPatch((x0, Y0), w, Y1 - Y0, boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec="none",
                         zorder=1)
    ax.add_patch(box)
    band = Rectangle((x0, TOP), w, Y1 - TOP, fc=HEAD, ec="none", zorder=1.2)
    ax.add_patch(band)
    band.set_clip_path(box)
    ax.add_patch(FancyBboxPatch((x0, Y0), w, Y1 - Y0, boxstyle="round,pad=0,rounding_size=0.04", fc="none",
                                ec=FRAME, lw=0.6, zorder=9))
    hline(x0, x0 + w, TOP, color=FRAME, lw=0.4)
    text(x0 + w / 2, (TOP + Y1) / 2 - 0.004, title, size=7.2, fontweight="bold")
    inner = Rectangle((x0 + 0.02, Y0 + 0.02), w - 0.04, TOP - Y0 - 0.04, fc="none", ec="none")
    ax.add_patch(inner)
    CLIP.append(inner)
for x0, w in zip(X0[:-1], WIDTHS[:-1]):      # arrows between panels
    ax.annotate("", xy=(x0 + w + GAP - 0.012, MID), xytext=(x0 + w + 0.012, MID), zorder=9,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#4B525A", mutation_scale=6.5, shrinkA=0,
                                shrinkB=0))

rng = np.random.RandomState(7)

# ---- 1. condition: sensor input (event cloud in x, y, t and IMU traces), estimator, environment, scoring rule --
x0, w = X0[0], WIDTHS[0]
EX, EY, LT, LV = x0 + 0.34, 1.03, 0.92, 0.38
DU = (0.25, 0.145)                           # image x axis, drawn receding


def proj(t, u, v):
    """Oblique view of (time, image x, image y)."""
    return EX + LT * t + DU[0] * u, EY + LV * v + DU[1] * u


def polygon_edges(vertices, per_len):
    """Points on the edges of a counter-clockwise polygon and their outward normals."""
    pts, nrm = [], []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        d = b - a
        n = max(2, int(np.hypot(*d) * per_len))
        s = (np.arange(n) + 0.5) / n
        pts.append(a + s[:, None] * d)
        nrm.append(np.tile(np.array([d[1], -d[0]]) / np.hypot(*d), (n, 1)))
    return np.vstack(pts), np.vstack(nrm)


ang = np.linspace(0, 2 * np.pi, 22, endpoint=False)
SHAPES = [np.array([[-0.34, -0.30], [-0.06, -0.34], [-0.22, -0.02]]),                       # triangle
          np.array([[0.04, -0.20], [0.32, -0.20], [0.32, 0.08], [0.04, 0.08]]),             # square
          np.c_[-0.12 + 0.13 * np.cos(ang), 0.24 + 0.13 * np.sin(ang)]]                     # disc
P, N = map(np.vstack, zip(*[polygon_edges(s, 150) for s in SHAPES]))


def image_points(t):
    """The shapes under a smooth camera motion: positions in the unit image and rotated normals."""
    th = 0.55 * np.sin(2 * np.pi * 0.8 * t + 0.4)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    off = np.array([0.5 + 0.13 * np.sin(2 * np.pi * 1.25 * t + 0.6), 0.5 + 0.10 * np.sin(2 * np.pi * 1.9 * t)])
    return P @ rot.T + off, N @ rot.T


ev_t, ev_u, ev_v, ev_p = [], [], [], []
for tk in np.linspace(0.0, 1.0, 260):
    q, n = image_points(tk)
    vel = (image_points(tk + 1e-3)[0] - image_points(tk - 1e-3)[0]) / 2e-3
    vn = np.sum(vel * n, axis=1)                                   # speed of the edge along its normal
    fire = rng.rand(vn.size) < 0.085 * np.abs(vn)
    ev_t.append(np.full(fire.sum(), tk) + rng.uniform(-0.002, 0.002, fire.sum()))
    ev_u.append(q[fire, 0] + rng.normal(0, 0.006, fire.sum()))
    ev_v.append(q[fire, 1] + rng.normal(0, 0.006, fire.sum()))
    ev_p.append(vn[fire] < 0)                                      # dark shape leaving a pixel: ON event
ev_t, ev_u, ev_v, ev_p = map(np.concatenate, (ev_t, ev_u, ev_v, ev_p))
nz = 110                                                           # background noise events
ev_t = np.r_[ev_t, rng.rand(nz)]
ev_u = np.r_[ev_u, rng.rand(nz)]
ev_v = np.r_[ev_v, rng.rand(nz)]
ev_p = np.r_[ev_p, rng.rand(nz) > 0.5]
keep = (ev_t > 0) & (ev_t < 1) & (ev_u > 0.01) & (ev_u < 0.99) & (ev_v > 0.01) & (ev_v < 0.99)
ev_t, ev_u, ev_v, ev_p = ev_t[keep], ev_u[keep], ev_v[keep], ev_p[keep]

BOXC = "#A9B1BA"
for (a, b) in [((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 1, 0), (0, 0, 0))]:   # hidden edges
    ax.plot(*zip(proj(*a), proj(*b)), color=BOXC, lw=0.35, ls=(0, (2, 2)), zorder=3)
order = np.argsort(-ev_u)                                          # far side first
ex, ey = proj(ev_t[order], ev_u[order], ev_v[order])
ax.scatter(ex, ey, s=0.55, c=np.where(ev_p[order], NAT_RED, NAT_BLUE), lw=0, alpha=0.9, zorder=4)
for (a, b) in [((0, 0, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
               ((1, 0, 0), (1, 1, 0)), ((1, 1, 0), (1, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((1, 0, 0), (1, 0, 1)),
               ((0, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 1))]:
    ax.plot(*zip(proj(*a), proj(*b)), color=BOXC, lw=0.45, zorder=5, solid_capstyle="round")
text(EX + LT / 2, EY - 0.052, "$t$", size=6.2, color=GREY)
text(EX - 0.045, EY + LV / 2, "$y$", size=6.2, color=GREY)
text(proj(1, 0.5, 0)[0] + 0.045, proj(1, 0.5, 0)[1] - 0.03, "$x$", size=6.2, color=GREY)
text(x0 + 0.09, 1.60, "events", ha="left")
ax.scatter([x0 + 0.45, x0 + 0.66], [1.60, 1.60], s=3.2, c=[NAT_RED, NAT_BLUE], lw=0, zorder=6)
text(x0 + 0.48, 1.60, "on", size=5.2, color=GREY, ha="left")
text(x0 + 0.69, 1.60, "off", size=5.2, color=GREY, ha="left")

tt = np.linspace(0, 1, 260)                                        # three gyroscope axes
for k, c in enumerate((NAT_GREEN, NAT_PURPLE, "#8A929B")):
    f, ph = rng.uniform(1.0, 9.0, 7), rng.uniform(0, 2 * np.pi, 7)
    y = sum(np.sin(2 * np.pi * fi * tt + pi) / (1 + 0.35 * i) for i, (fi, pi) in enumerate(zip(np.sort(f), ph)))
    y = y / np.abs(y).max() + rng.normal(0, 0.03, tt.size)
    ax.plot(EX + LT * tt, 0.795 + 0.066 * y, color=c, lw=0.55, zorder=4, solid_joinstyle="round")
text(x0 + 0.09, 0.795, "IMU", ha="left")
ax.plot([EX, EX + LT], [0.795, 0.795], color=BOXC, lw=0.35, zorder=3)

for k, (lab, sym) in enumerate([("estimator configuration", ""), ("computing environment", "$E$"),
                                ("scoring rule", "$S$")]):
    y = 0.545 - 0.172 * k
    hline(x0 + 0.09, x0 + w - 0.09, y + 0.086)
    text(x0 + 0.09, y, lab, ha="left")
    if sym:
        text(x0 + w - 0.11, y, sym, size=7.0, ha="right")

# ---- 2. repeated runs: K estimates from one start; one diverges, one stops ---------------------------------
x0, w = X0[1], WIDTHS[1]
t = np.linspace(0, 1, 500)


def gt(u):
    """A figure-eight path with some irregularity (top view), unit scale; one and a quarter laps."""
    th = 2 * np.pi * 1.25 * u - 0.9
    lap = 1 - 0.14 * u
    return (lap * (0.60 * np.sin(th) + 0.045 * np.sin(3 * th + 0.7)),
            lap * (0.36 * np.sin(2 * th + 0.12) + 0.05 * np.cos(th + 0.4)))


gx, gy = gt(t)
cx, cy, sc = x0 + w / 2, 0.92, 0.76
for k in range(6):
    a, ph = 0.03 + 0.016 * k, 1.1 * k + 0.4
    dx = a * t ** 1.3 * np.cos(ph) + 0.012 * np.sin(2 * np.pi * (1.5 + 0.3 * k) * t + k)
    dy = a * t ** 1.3 * np.sin(ph) + 0.012 * np.cos(2 * np.pi * (1.2 + 0.4 * k) * t + 2 * k)
    ax.plot(cx + sc * (gx + dx), cy + sc * (gy + dy), color=NAT_BLUE, lw=0.7, alpha=0.75, zorder=4,
            solid_capstyle="round")
s = np.clip((t - 0.42) / 0.58, 0, None)                            # a run that diverges: a growing offset
dvx = gx + 0.62 * s ** 1.7 * np.cos(3.7 - 2.5 * s)
dvy = gy + 0.62 * s ** 1.7 * np.sin(3.7 - 2.5 * s)
div, = ax.plot(cx + sc * dvx, cy + sc * dvy, color=NAT_ORANGE, lw=0.9, zorder=5, solid_capstyle="round")
div.set_clip_path(CLIP[1])
text(cx + sc * dvx[-1] - 0.12, cy + sc * dvy[-1] + 0.012, "diverged", size=5.2, color=GREY, ha="right")
m = t < 0.30                                                       # a run that stops with a numerical failure
fx, fy = cx + sc * (gx[m] + 0.05 * t[m]), cy + sc * (gy[m] - 0.10 * t[m])
ax.plot(fx, fy, color=NAT_RED, lw=0.9, zorder=5)
ax.scatter([fx[-1]], [fy[-1]], marker="x", s=20, c=NAT_RED, lw=1.2, zorder=6)
ax.scatter([cx + sc * gx[0]], [cy + sc * gy[0]], s=11, c=INK, lw=0, zorder=7)
text(x0 + w / 2, 0.17, "identical input in every run", color=GREY)

# ---- 3. record and classify: one row per run; the check of the recorded settings, then the outcome. A run whose
# record does not match its request is excluded before counting; the K numbered runs form the denominator. -----
x0, w = X0[2], WIDTHS[2]
CR, CE, CI, CM, CO = x0 + 0.14, x0 + 0.32, x0 + 0.56, x0 + 0.79, x0 + 0.855
text(CR, 1.575, "run", color=GREY)
text(CE, 1.575, "$E$", size=6.4, color=GREY)
text(CI, 1.575, "input", color=GREY)
text(CM - 0.03, 1.575, "outcome", color=GREY, ha="left")
hline(x0 + 0.07, x0 + w - 0.07, 1.495, color=FRAME, lw=0.5)
ROWS = [("1", True, "completed"), ("2", True, "completed"), ("3", True, "crash"), ("–", False, "excl."),
        ("4", True, "?"), None, ("$K$", True, "completed")]          # the excluded run has no number: it is not one of the K
for k, row in enumerate(ROWS):
    y = 1.40 - 0.168 * k
    if row is None:
        for xc in (CR, CE, CI, CM):
            ax.scatter([xc] * 3, [y + 0.035, y, y - 0.035], s=0.9, c=GREY, lw=0, zorder=5)
        continue
    run, ok, outcome = row
    ink = INK if ok else "#9AA2AB"
    text(CR, y, run, color=ink)
    if ok:
        check(CE, y)
    else:
        text(CE, y, "≠", size=7.0, fontweight="bold")
    text(CI, y, "9c2e…", size=5.0, color=ink if ok else "#9AA2AB", family=MONO)
    if outcome == "completed":
        ax.scatter([CM], [y], s=10, c=INK, lw=0, zorder=5)
        text(CO, y, outcome, ha="left")
    elif outcome == "crash":
        ax.scatter([CM], [y], marker="x", s=14, c=NAT_RED, lw=1.1, zorder=5)
        text(CO, y, outcome, ha="left")
    elif outcome == "?":
        text(CM, y, "?", size=6.6, fontweight="bold")
        text(CO, y, "undetermined", size=5.4, ha="left")
    else:
        text(CO, y, outcome, ha="left", color=ink)
    if k < len(ROWS) - 1 and ROWS[k + 1] is not None:
        hline(x0 + 0.07, x0 + w - 0.07, y - 0.084, lw=0.35)
hline(x0 + 0.07, x0 + w - 0.07, 1.40 - 0.168 * 6 - 0.09, color=FRAME, lw=0.5)
check(x0 + 0.13, 0.155, a=0.024)
text(x0 + 0.19, 0.155, "as requested", size=5.3, color=GREY, ha="left")
text(x0 + 0.76, 0.157, "≠", size=6.4, color=GREY, fontweight="bold")
text(x0 + 0.82, 0.155, "mismatch", size=5.3, color=GREY, ha="left")

# ---- 4. score: a completed run after a rigid alignment; position errors ------------------------------------
x0, w = X0[3], WIDTHS[3]
cx, cy, sc = x0 + w / 2, 0.80, 0.78
ex_ = gx + 0.075 * np.sin(2 * np.pi * 1.5 * t + 0.5) * (0.25 + t) + 0.012
ey_ = gy + 0.070 * np.cos(2 * np.pi * 1.1 * t + 0.2) * (0.25 + t) - 0.010
for k in range(10, t.size, 24):
    ax.plot([cx + sc * gx[k], cx + sc * ex_[k]], [cy + sc * gy[k], cy + sc * ey_[k]], color=NAT_RED, lw=0.7,
            zorder=4)
ax.plot(cx + sc * gx, cy + sc * gy, color=INK, lw=0.9, zorder=5)
ax.plot(cx + sc * ex_, cy + sc * ey_, color=NAT_BLUE, lw=0.9, ls=(0, (3.0, 1.4)), zorder=5)
for k, (lab, kw) in enumerate([("ground truth", dict(color=INK, lw=0.9)),
                               ("estimate, aligned", dict(color=NAT_BLUE, lw=0.9, ls=(0, (3.0, 1.4)))),
                               ("position error", dict(color=NAT_RED, lw=0.7))]):
    y = 1.585 - 0.105 * k
    ax.plot([x0 + 0.10, x0 + 0.26], [y, y], zorder=5, **kw)
    text(x0 + 0.31, y, lab, size=5.4, ha="left")
text(x0 + w / 2, 0.17, "rigid alignment, no scale", color=GREY)

# ---- 5. reliability: share of all K runs that complete within the tolerance --------------------------------
x0, w = X0[4], WIDTHS[4]
px0, py0, pw, ph = x0 + 0.27, 0.40, w - 0.27 - 0.13, 1.13
K, done = 20, 16
xs = np.sort(np.clip(rng.lognormal(-1.25, 0.55, done), 0.05, 0.86))
sxp = lambda v: px0 + v * pw                                     # noqa: E731
syp = lambda r: py0 + r * ph * 0.88                              # noqa: E731
for r in (0.25, 0.5, 0.75):
    hline(px0, px0 + pw, syp(r), color="#E4E7EB", lw=0.4)
stx = [sxp(0)] + [sxp(v) for v in xs for _ in (0, 1)] + [sxp(1.0)]
sty = [syp(0)] + [syp(r) for k in range(done) for r in (k / K, (k + 1) / K)] + [syp(done / K)]
ax.fill_between(stx, py0, sty, color=NAT_BLUE, alpha=0.10, lw=0, zorder=3)
ax.plot(stx, sty, color=NAT_BLUE, lw=1.1, zorder=5, solid_joinstyle="miter")
ax.plot([px0, px0 + pw], [syp(1), syp(1)], color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=4)
ax.plot([px0, px0 + pw], [py0, py0], color=INK, lw=0.6, zorder=6)
ax.plot([px0, px0], [py0, py0 + ph], color=INK, lw=0.6, zorder=6)
for r, lab in ((0, "0"), (1, "1")):
    ax.plot([px0 - 0.025, px0], [syp(r), syp(r)], color=INK, lw=0.6, zorder=6)
    text(px0 - 0.045, syp(r), lab, size=6.0, ha="right")
ax.annotate("", xy=(px0 + pw - 0.045, syp(done / K)), xytext=(px0 + pw - 0.045, syp(1)), zorder=7,
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=NAT_RED, shrinkA=0, shrinkB=0, mutation_scale=5))
k_star = 10
es = 0.5 * (xs[k_star - 1] + xs[k_star])
ax.plot([sxp(es)] * 2, [py0, syp(k_star / K)], color=INK, lw=0.5, ls=(0, (1.5, 1.5)), zorder=4)
ax.plot([px0, sxp(es)], [syp(k_star / K)] * 2, color=INK, lw=0.5, ls=(0, (1.5, 1.5)), zorder=4)
ax.scatter([sxp(es)], [syp(k_star / K)], s=9, c=INK, lw=0, zorder=7)
ax.plot([sxp(es)] * 2, [py0 - 0.025, py0], color=INK, lw=0.6, zorder=6)
text(sxp(es) - 0.008, py0 - 0.078, r"$\varepsilon$", size=6.4)
text(sxp(es) + 0.034, py0 - 0.058, "*", size=5.6)
text(px0 + pw - 0.07, syp(0.5 * (1 + done / K)), "crash or undetermined", size=5.2, ha="right")
text(px0 + pw, 0.17, r"tolerance $\varepsilon$", ha="right", color=GREY)
text(x0 + 0.09, py0 + ph * 0.5, r"$R_c(\varepsilon)$", size=6.4, rotation=90)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".pdf", bbox_inches=None, pad_inches=0)
fig.savefig(OUT + ".png", dpi=300, bbox_inches=None, pad_inches=0)
print("wrote", _layout.rel(OUT + ".pdf"))
