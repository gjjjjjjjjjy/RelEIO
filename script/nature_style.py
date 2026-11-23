"""Unified figure style for the paper (single shared module; every figure imports it).

Design goal: ONE consistent visual language across all figures so the whole paper
reads as a single set, using the classic Nature/Science blue-orange-green palette
(muted, print-grade saturation --- lively but not garish).

    blue   = valid / normal / primary series   (the calm main color)
    orange = catastrophic completion / second series (warm highlight)
    red    = numerical crash / failure emphasis (the alarm color)
    green  = third series

Categorical distinctions are carried by these few well-separated hues plus marker
shape. Heatmaps use a white->blue sequential ramp in the same blue family, so a
heatmap, a bar chart and a scatter all look like they belong together.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Semantic palette (color carries meaning, not just decoration) -----------
# valid/stable=blue, catastrophic=amber, crash/failure=dark red,
# protocol/uncertainty=deep green-grey. Muted, print-grade saturation.
INK        = "#23272b"   # near-black for text/frame (softer than pure black)
NAT_BLUE   = "#2F5E9E"   # valid / stable / primary
NAT_ORANGE = "#D9822B"   # catastrophic completion
NAT_RED    = "#B23A35"   # numerical crash / failure emphasis
NAT_GREEN  = "#4F7C68"   # protocol / uncertainty / third series
NAT_PURPLE = "#8E6FA8"   # spare 5th series
GREY_FAINT = "#e8eaed"   # gridlines / faint fills

# Ordered series colors. Blue first; the warm colors carry "worse".
SERIES_CYCLE = [NAT_BLUE, NAT_ORANGE, NAT_GREEN, NAT_RED, NAT_PURPLE, INK]


def apply_nature_style():
    """Quiet IEEE-style rcParams: light frame, readable type, generous whitespace.

    (Function name kept for backward compatibility with the figure scripts.)
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        # Everything thin and quiet.
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.4,
        "grid.linewidth": 0.4,
        # Text / frame in soft ink, not pure black.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#444b52",   # dark-grey frame
        "xtick.color": "#444b52",
        "ytick.color": "#444b52",
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        # Keep top/right spines (IEEE convention; reads more "finished").
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        # Very faint horizontal grid only (set per-axes where wanted).
        "axes.grid": False,
        "grid.color": GREY_FAINT,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES_CYCLE),
        # Legend: frameless, airy.
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.borderpad": 0.3,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        # Output.
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --- Semantic colors for the three-class outcome (used in EVERY figure) ------
# blue = valid (normal); orange = catastrophic; red = numerical crash.
C_VALID = NAT_BLUE      # valid run               -> blue circle
C_CATAS = NAT_ORANGE    # catastrophic completion -> orange triangle
C_CRASH = NAT_RED       # numerical crash         -> red x marker

# --- Generic categorical series for bar charts -------------------------------
SERIES = {
    "a": NAT_BLUE,      # primary / baseline
    "b": NAT_ORANGE,    # second category
    "c": NAT_GREEN,     # third category
    "d": NAT_RED,       # alarm / fourth
}

# --- Failure heatmap: white -> amber -> dark red (warm = risk) ----------------
# A failure-rate heatmap should read as "risk", so it uses a warm ramp rather
# than the calm blue family.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "risk_warm", ["#fbf6ef", "#f5d9a8", NAT_ORANGE, NAT_RED, "#7d211d"], N=256)
