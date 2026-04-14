"""
Generate placeholder SVG/PDF figures for plots where required data are absent.

A placeholder contains:
- The intended plot title
- "Data Absent" message
- List of missing required columns
- Optional note (e.g. if derivation failed)

The placeholder is a real matplotlib figure saved as SVG so it is vector-editable.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional
from batteryplot.styles import apply_style, SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN


def make_placeholder(
    title: str,
    missing_columns: List[str],
    output_dir: Path,
    stem: str,
    formats: tuple = ("svg", "pdf"),
    note: Optional[str] = None,
) -> List[Path]:
    """
    Generate a placeholder figure indicating data absence.

    Parameters
    ----------
    title : str
        The intended plot title (shown at top of placeholder).
    missing_columns : list of str
        Canonical column names that were missing from the input data.
    output_dir : Path
        Directory where the placeholder file(s) will be saved.
    stem : str
        File name stem (without extension).
    formats : tuple of str
        Output formats, e.g. ("svg", "pdf").
    note : str, optional
        Additional explanation, e.g. a derivation failure message.

    Returns
    -------
    list of Path
        Paths to all saved placeholder files.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, title, ha="center", va="top", fontsize=9,
            fontweight="bold", transform=ax.transAxes, wrap=True)

    # "Data Absent" banner
    ax.text(0.5, 0.72, "Data Absent", ha="center", va="top", fontsize=11,
            color="#CC4444", fontweight="bold", transform=ax.transAxes)

    # Missing columns
    if missing_columns:
        missing_str = "Missing required columns:\n" + "\n".join(f"  \u2022 {c}" for c in missing_columns)
    else:
        missing_str = "Required columns not found in this file."
    ax.text(0.5, 0.55, missing_str, ha="center", va="top", fontsize=7,
            color="#666666", transform=ax.transAxes, family="monospace")

    if note:
        ax.text(0.5, 0.12, f"Note: {note}", ha="center", va="bottom", fontsize=6,
                color="#888888", transform=ax.transAxes, style="italic")

    # Subtle border
    rect = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                    boxstyle="round,pad=0.02",
                                    linewidth=0.8, edgecolor="#cccccc",
                                    facecolor="#fafafa", transform=ax.transAxes)
    ax.add_patch(rect)

    saved = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = output_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved
