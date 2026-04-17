"""
Generate placeholder SVG/PDF figures for plots where required data are absent.

A placeholder contains four sections:

1. The intended plot title.
2. A "Data Absent" banner.
3. A column status table split into three categories:
   - PRESENT AND POPULATED  — column exists and has non-trivial values
   - PRESENT BUT EMPTY      — column exists but is all-zero, all-NaN, or
                              all-identical (e.g. the test never reached that
                              measurement point)
   - ABSENT FROM FILE       — column was not found in the raw CSV at all
4. An optional explanatory note (derivation failure, insufficient cycles, etc.)

The placeholder is a real matplotlib figure saved as SVG so it is
vector-editable in Inkscape / Illustrator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from batteryplot.styles import apply_style, DEFAULT_HEIGHT_IN, SINGLE_COL_WIDTH_IN

logger = logging.getLogger("batteryplot.placeholders")


# ---------------------------------------------------------------------------
# Column diagnostic dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColumnDiagnostic:
    """
    Categorised status of the canonical columns relevant to a single plot.

    Attributes
    ----------
    present_populated : list of str
        Columns that exist in the DataFrame and contain at least some
        non-zero, non-NaN values.
    present_empty : list of (str, str)
        Columns that exist but are uninformative: ``(column_name, reason)``
        where *reason* is a short human-readable explanation such as
        ``"all-zero"`` or ``"all-NaN"`` or ``"constant value"``.
    absent : list of str
        Columns that were not found in the raw CSV at all.
    note : str, optional
        Free-text explanation appended at the bottom of the placeholder.
    """
    present_populated: List[str] = field(default_factory=list)
    present_empty: List[Tuple[str, str]] = field(default_factory=list)
    absent: List[str] = field(default_factory=list)
    note: Optional[str] = None


def diagnose_columns(
    df,                          # pd.DataFrame — the analysis dataframe
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> ColumnDiagnostic:
    """
    Categorise *required* (and *optional*) columns into three buckets.

    Parameters
    ----------
    df : pd.DataFrame
        The canonical analysis DataFrame.
    required : sequence of str
        Column names that the plot needs to function.
    optional : sequence of str
        Column names that improve the plot if present.

    Returns
    -------
    ColumnDiagnostic
    """
    import pandas as pd
    import numpy as np

    diag = ColumnDiagnostic()
    all_cols = list(required) + list(optional)

    for col in all_cols:
        if col not in df.columns:
            diag.absent.append(col)
            continue

        series = df[col]

        # Try numeric coercion
        numeric = pd.to_numeric(series, errors="coerce")
        n_valid = int(numeric.notna().sum())
        n_total = len(series)

        if n_valid == 0:
            # All NaN after coercion — or non-numeric object column with no values
            if series.isna().all() or (series.astype(str).str.strip().isin(["", "N/A", "nan", "NaN"])).all():
                diag.present_empty.append((col, "all-NaN / N/A"))
            else:
                # Non-numeric column with some values (e.g. step_type strings)
                diag.present_populated.append(col)
            continue

        # Numeric column with some valid values
        valid_vals = numeric.dropna()
        if (valid_vals == 0).all():
            diag.present_empty.append((col, "all-zero (not measured / sensor not connected)"))
        elif valid_vals.nunique() == 1:
            diag.present_empty.append(
                (col, f"constant value ({valid_vals.iloc[0]:.4g}) — no variation recorded")
            )
        elif n_valid < 0.05 * n_total:
            pct = 100.0 * n_valid / n_total
            diag.present_empty.append(
                (col, f"mostly empty ({pct:.1f}% populated — test may not have reached this step)")
            )
        else:
            diag.present_populated.append(col)

    return diag


# ---------------------------------------------------------------------------
# Placeholder figure
# ---------------------------------------------------------------------------

def make_placeholder(
    title: str,
    missing_columns: List[str],
    output_dir: Path,
    stem: str,
    formats: tuple = ("svg", "pdf"),
    note: Optional[str] = None,
    diagnostic: Optional[ColumnDiagnostic] = None,
) -> List[Path]:
    """
    Generate a placeholder figure indicating that a plot could not be produced.

    If a :class:`ColumnDiagnostic` is supplied it is used to render the
    detailed column-status table.  Otherwise the function falls back to the
    simple ``missing_columns`` list so that callers that have not yet been
    updated continue to work correctly.

    Parameters
    ----------
    title : str
        Intended plot title.
    missing_columns : list of str
        Required columns that are absent.  Used only when *diagnostic* is
        ``None``.
    output_dir : Path
        Directory to save files in.
    stem : str
        File stem (no extension).
    formats : tuple of str
        Output formats, e.g. ``("svg", "pdf")``.
    note : str, optional
        Extra explanation appended at the bottom.
    diagnostic : ColumnDiagnostic, optional
        Rich column-status object.  When provided, overrides *missing_columns*.

    Returns
    -------
    list of Path
    """
    apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide height: richer content needs more vertical space
    if diagnostic is not None:
        n_rows = (
            len(diagnostic.present_populated)
            + len(diagnostic.present_empty)
            + len(diagnostic.absent)
        )
        height = max(DEFAULT_HEIGHT_IN, DEFAULT_HEIGHT_IN + 0.18 * max(0, n_rows - 4))
    else:
        n_rows = len(missing_columns)
        height = max(DEFAULT_HEIGHT_IN, DEFAULT_HEIGHT_IN + 0.18 * max(0, n_rows - 4))

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN * 1.3, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Subtle background border ---
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.015",
        linewidth=0.8, edgecolor="#cccccc",
        facecolor="#fafafa",
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(rect)

    # ---- Layout cursor (top → bottom in axes-fraction units) ----
    y = 0.95

    # Title
    ax.text(
        0.5, y, title,
        ha="center", va="top",
        fontsize=9, fontweight="bold",
        transform=ax.transAxes,
    )
    y -= 0.10

    # "Data Absent" banner
    ax.text(
        0.5, y, "DATA ABSENT",
        ha="center", va="top",
        fontsize=10, fontweight="bold", color="#CC4444",
        transform=ax.transAxes,
    )
    y -= 0.10

    if diagnostic is not None:
        y = _render_diagnostic(ax, diagnostic, y)
    else:
        # Legacy fallback: just list missing columns
        if missing_columns:
            lines = ["Missing required columns:"] + [f"  \u2022 {c}" for c in missing_columns]
        else:
            lines = ["Required columns not found in this file."]
        ax.text(
            0.5, y, "\n".join(lines),
            ha="center", va="top",
            fontsize=7, color="#555555",
            family="monospace",
            transform=ax.transAxes,
        )
        y -= 0.06 * len(lines)

    # Optional note
    effective_note = note or (diagnostic.note if diagnostic else None)
    if effective_note:
        y = max(y - 0.04, 0.04)
        ax.text(
            0.5, y,
            f"(i) {effective_note}",
            ha="center", va="top",
            fontsize=6.5, color="#777777", style="italic",
            transform=ax.transAxes,
            wrap=True,
        )

    saved: List[Path] = []
    for fmt in formats:
        p = output_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Internal rendering helper
# ---------------------------------------------------------------------------

_GREEN  = "#2a7a3b"
_ORANGE = "#b86e00"
_RED    = "#a02020"
_GRAY   = "#666666"

_BULLET_PRESENT  = "[+]"
_BULLET_EMPTY    = "[ ]"
_BULLET_ABSENT   = "[x]"


def _render_diagnostic(ax, diag: ColumnDiagnostic, y_start: float) -> float:
    """
    Render the three-section column status table onto *ax*.

    Returns the y position after the last rendered element.
    """
    LINE_H  = 0.072   # vertical step per line (axes fraction)
    INDENT  = 0.07    # left margin for bullet text
    COL_X   = 0.09    # x for bullet symbol
    TEXT_X  = 0.14    # x for column name / reason text

    y = y_start

    def _section_header(label: str, color: str) -> None:
        nonlocal y
        ax.text(
            INDENT, y, label,
            ha="left", va="top",
            fontsize=7, fontweight="bold", color=color,
            transform=ax.transAxes,
        )
        y -= LINE_H * 0.7

    def _row(bullet: str, col_name: str, reason: str, col_color: str) -> None:
        nonlocal y
        ax.text(
            COL_X, y, bullet,
            ha="left", va="top",
            fontsize=7, color=col_color,
            transform=ax.transAxes,
        )
        label = col_name if not reason else f"{col_name}  —  {reason}"
        ax.text(
            TEXT_X, y, label,
            ha="left", va="top",
            fontsize=6.5, color="#333333",
            family="monospace",
            transform=ax.transAxes,
        )
        y -= LINE_H

    # ---- Section 1: present and populated ----
    if diag.present_populated:
        _section_header(f"Present & populated ({len(diag.present_populated)})", _GREEN)
        for col in diag.present_populated:
            _row(_BULLET_PRESENT, col, "", _GREEN)
        y -= LINE_H * 0.3

    # ---- Section 2: present but empty ----
    if diag.present_empty:
        _section_header(
            f"Present but uninformative ({len(diag.present_empty)})", _ORANGE
        )
        for col, reason in diag.present_empty:
            _row(_BULLET_EMPTY, col, reason, _ORANGE)
        y -= LINE_H * 0.3

    # ---- Section 3: absent ----
    if diag.absent:
        _section_header(f"Absent from file ({len(diag.absent)})", _RED)
        for col in diag.absent:
            _row(_BULLET_ABSENT, col, "column not found in CSV", _RED)
        y -= LINE_H * 0.3

    if not diag.present_populated and not diag.present_empty and not diag.absent:
        ax.text(
            0.5, y, "No column information available.",
            ha="center", va="top", fontsize=7, color=_GRAY,
            transform=ax.transAxes,
        )
        y -= LINE_H

    return y
