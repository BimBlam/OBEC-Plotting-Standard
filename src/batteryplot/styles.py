"""
Matplotlib style configuration for publication-quality battery data plots.

Design principles:
- Vector-first: all output in SVG (and optionally PDF)
- Colorblind-safe palette (Wong 2011 8-color palette)
- Consistent typography: default font size 8pt, figure width ~3.5in (single column) or ~7in (double column)
- No chartjunk: no top/right spines, minimal gridlines
- Editable SVG: use Type 3 / TrueType fonts; set svg.fonttype = 'none' so text is real text
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple

# Wong colorblind-safe palette (2011)
WONG_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
]

SINGLE_COL_WIDTH_IN = 3.5
DOUBLE_COL_WIDTH_IN = 7.0
DEFAULT_HEIGHT_IN = 2.8
FONT_SIZE_BASE = 8
LINE_WIDTH = 1.0
MARKER_SIZE = 4


def apply_style(theme: str = "publication") -> None:
    """Apply matplotlib rcParams for publication-quality output."""
    params = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_BASE - 1,
        "ytick.labelsize": FONT_SIZE_BASE - 1,
        "legend.fontsize": FONT_SIZE_BASE - 1,
        "axes.titlesize": FONT_SIZE_BASE,
        # Lines
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Grid
        "axes.grid": False,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # SVG — use actual text, not outlines, for editability
        "svg.fonttype": "none",
        # PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Layout
        "figure.constrained_layout.use": True,
        # Color cycle
        "axes.prop_cycle": matplotlib.cycler(color=WONG_PALETTE),
    }
    if theme == "dark":
        params.update({
            "figure.facecolor": "#1e1e1e",
            "axes.facecolor": "#1e1e1e",
            "axes.edgecolor": "#cccccc",
            "text.color": "#cccccc",
            "xtick.color": "#cccccc",
            "ytick.color": "#cccccc",
            "axes.labelcolor": "#cccccc",
        })
    plt.rcParams.update(params)


def get_fig_ax(ncols=1, nrows=1, width="single", height=None, **kwargs):
    """Create a figure with standard publication dimensions."""
    w = SINGLE_COL_WIDTH_IN if width == "single" else DOUBLE_COL_WIDTH_IN
    h = height if height is not None else DEFAULT_HEIGHT_IN * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, axes


def add_panel_label(ax, label: str, x=-0.15, y=1.05) -> None:
    """Add a panel label (a), (b), etc. in the standard position."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FONT_SIZE_BASE,
            fontweight="bold", va="top", ha="right")


def save_figure(fig, output_dir, stem: str, formats=("svg", "pdf")) -> list:
    """Save figure in requested formats. Returns list of saved paths."""
    import pathlib
    saved = []
    for fmt in formats:
        p = pathlib.Path(output_dir) / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved
