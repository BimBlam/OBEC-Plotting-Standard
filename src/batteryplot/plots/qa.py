"""
Quality-assurance (QA) plots for the batteryplot package.

Provides:
- plot_temperature_vs_time: temperature (and optionally humidity) over test time.
- plot_current_voltage_overview: full-test current and voltage overview.
- plot_data_availability: horizontal bar chart of column completeness.

Scientific assumptions
----------------------
Temperature / humidity
~~~~~~~~~~~~~~~~~~~~~~
Rows where temperature_c == 0 AND humidity_pct == 0 are assumed to be
"sensor not connected" artefacts and are dropped before plotting.  This
heuristic is applied only when both columns are present.  If only one sensor
is available the zero-filter is applied to that column only.

Current / voltage overview
~~~~~~~~~~~~~~~~~~~~~~~~~~
This is a full-test overview intended for QA—it is not a per-cycle detail
view.  Thin lines (linewidth 0.5) are used because the dataset is typically
very large.  No downsampling is applied by default; the caller is expected to
pass an appropriately thinned df if performance is a concern.

Data-availability chart
~~~~~~~~~~~~~~~~~~~~~~~
All canonical column names defined in the registry are checked against both
df and cycle_summary.  The completeness score for each column is:
  completeness = (non-null row count) / (total row count) × 100  [%]
Colour coding:
  green  (#4CAF50): present and ≥ 80 % complete
  orange (#FF9800): present but < 80 % complete
  gray   (#BDBDBD): absent (not in df or cycle_summary)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from batteryplot.styles import (
    apply_style,
    WONG_PALETTE,
    SINGLE_COL_WIDTH_IN,
    DOUBLE_COL_WIDTH_IN,
    DEFAULT_HEIGHT_IN,
    save_figure,
    add_assumption_warning,
)
from batteryplot.placeholders import (
    make_placeholder,
    diagnose_columns,
    ColumnDiagnostic,
)

logger = logging.getLogger("batteryplot.plots.qa")

# ---------------------------------------------------------------------------
# Canonical column list (used by data_availability plot)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "elapsed_time_s", "step_time_s", "cycle_index", "step_index", "step_type",
    "current_a", "voltage_v", "capacity_ah", "energy_wh", "power_w",
    "ac_impedance_ohm", "dcir_ohm", "resistance_ohm", "conductivity_s_cm",
    "temperature_c", "humidity_pct", "specific_capacity_ah_g",
    "charge_capacity_ah", "discharge_capacity_ah",
    "charge_energy_wh", "discharge_energy_wh",
    "timestamp_dt", "record_index", "loop_index",
    # Cycle-summary-only columns
    "coulombic_efficiency_pct", "energy_efficiency_pct", "capacity_retention_pct",
    "mean_dcir_ohm", "mean_ac_impedance_ohm",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _get_formats(config) -> tuple:
    return tuple(getattr(config, "output_formats", ("svg", "pdf")))


def _get_theme(config) -> str:
    return getattr(config, "theme", "publication")


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_temperature_vs_time(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot temperature versus elapsed test time, with optional humidity overlay.

    Rows where temperature_c == 0 AND humidity_pct == 0 simultaneously are
    treated as sensor-absent artefacts and excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.  Must contain temperature_c and elapsed_time_s.
    cycle_summary, pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["temperature_c", "elapsed_time_s"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning("plot_temperature_vs_time: missing columns %s", missing)
        diag = diagnose_columns(df, required, optional=["humidity_pct"])
        diag.note = ("temperature_c all-zero usually means the thermocouple "
                     "was not connected or the cycler channel has no sensor.")
        return make_placeholder(
            title="Temperature vs. Time",
            missing_columns=missing,
            output_dir=output_dir,
            stem="temperature_vs_time",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    has_humidity = "humidity_pct" in df.columns
    plot_df = df.copy()

    # Drop sensor-absent rows
    if has_humidity:
        mask_absent = (plot_df["temperature_c"] == 0) & (plot_df["humidity_pct"] == 0)
    else:
        mask_absent = plot_df["temperature_c"] == 0
    plot_df = plot_df[~mask_absent]

    if plot_df.empty:
        return make_placeholder(
            title="Temperature vs. Time",
            missing_columns=[],
            output_dir=output_dir,
            stem="temperature_vs_time",
            formats=_get_formats(config),
            note="All temperature values are zero — sensor likely not connected.",
        )

    time_h = plot_df["elapsed_time_s"] / 3600.0

    if has_humidity:
        fig, (ax_t, ax_h) = plt.subplots(
            2, 1,
            figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN * 1.5),
            sharex=True,
        )
        ax_t.plot(time_h, plot_df["temperature_c"], color=WONG_PALETTE[6], linewidth=0.8)
        ax_t.set_ylabel("Temperature (\u00b0C)")
        ax_t.set_title("Temperature and Humidity vs. Time")

        ax_h.plot(time_h, plot_df["humidity_pct"], color=WONG_PALETTE[3], linewidth=0.8)
        ax_h.set_ylabel("Humidity (%)")
        ax_h.set_xlabel("Time (h)")
    else:
        fig, ax_t = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))
        ax_t.plot(time_h, plot_df["temperature_c"], color=WONG_PALETTE[6], linewidth=0.8)
        ax_t.set_xlabel("Time (h)")
        ax_t.set_ylabel("Temperature (\u00b0C)")
        ax_t.set_title("Temperature vs. Time")

    return save_figure(fig, output_dir, "temperature_vs_time", formats=_get_formats(config))


def plot_current_voltage_overview(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Full-test current and voltage overview (two-panel figure).

    Top panel: voltage_v vs. time (h).
    Bottom panel: current_a vs. time (h).

    Thin lines (linewidth 0.5) are used to keep large datasets legible.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.  Must contain current_a, voltage_v, elapsed_time_s.
    cycle_summary, pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["current_a", "voltage_v", "elapsed_time_s"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning("plot_current_voltage_overview: missing columns %s", missing)
        diag = diagnose_columns(df, required)
        return make_placeholder(
            title="Current and Voltage Overview",
            missing_columns=missing,
            output_dir=output_dir,
            stem="current_voltage_overview",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    time_h = df["elapsed_time_s"] / 3600.0

    fig, (ax_v, ax_i) = plt.subplots(
        2, 1,
        figsize=(DOUBLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN * 1.4),
        sharex=True,
    )

    ax_v.plot(time_h, df["voltage_v"], color=WONG_PALETTE[5], linewidth=0.5, rasterized=True)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_title("Current and Voltage Overview")

    ax_i.plot(time_h, df["current_a"], color=WONG_PALETTE[6], linewidth=0.5, rasterized=True)
    ax_i.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax_i.set_ylabel("Current (A)")
    ax_i.set_xlabel("Time (h)")

    return save_figure(fig, output_dir, "current_voltage_overview", formats=_get_formats(config))


def plot_data_availability(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Horizontal bar chart showing data completeness for all canonical columns.

    Each bar represents one canonical column.  Its length is the completeness
    percentage (non-null rows / total rows × 100).  Colour encoding:
    - Green  (#4CAF50): present and ≥ 80 % complete
    - Orange (#FF9800): present but < 80 % complete (i.e. mostly null)
    - Gray   (#BDBDBD): absent from both df and cycle_summary

    Columns from cycle_summary are checked against that DataFrame; all others
    are checked against df.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.
    cycle_summary : pd.DataFrame
        Per-cycle summary DataFrame.
    pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This plot always succeeds (it shows what IS available)
    cycle_summary_cols = {
        "coulombic_efficiency_pct", "energy_efficiency_pct", "capacity_retention_pct",
        "mean_dcir_ohm", "mean_ac_impedance_ohm",
        "charge_capacity_ah", "discharge_capacity_ah",
        "charge_energy_wh", "discharge_energy_wh",
    }

    completeness: dict = {}
    for col in CANONICAL_COLUMNS:
        # Determine which DataFrame to check
        if col in cycle_summary_cols:
            source = cycle_summary if (cycle_summary is not None and isinstance(cycle_summary, pd.DataFrame)) else None
        else:
            source = df if (df is not None and isinstance(df, pd.DataFrame)) else None

        if source is None or source.empty:
            completeness[col] = -1.0  # absent / no data
        elif col not in source.columns:
            completeness[col] = -1.0  # absent
        else:
            n_total = len(source)
            n_valid = source[col].notna().sum()
            completeness[col] = (n_valid / n_total * 100.0) if n_total > 0 else 0.0

    # Sort: present first (by completeness desc), then absent
    sorted_items = sorted(completeness.items(), key=lambda kv: kv[1], reverse=True)
    cols = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]

    # Color coding
    GREEN = "#4CAF50"
    ORANGE = "#FF9800"
    GRAY = "#BDBDBD"

    colors = []
    display_vals = []
    for v in vals:
        if v < 0:
            colors.append(GRAY)
            display_vals.append(0.0)
        elif v >= 80:
            colors.append(GREEN)
            display_vals.append(v)
        else:
            colors.append(ORANGE)
            display_vals.append(v)

    n_cols = len(cols)
    fig_height = max(DEFAULT_HEIGHT_IN, n_cols * 0.28 + 0.6)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, fig_height))

    y_pos = np.arange(n_cols)
    bars = ax.barh(y_pos, display_vals, color=colors, height=0.7, edgecolor="none")

    # Mark truly absent columns with a thin vertical dashed line at 0
    for i, v in enumerate(vals):
        if v < 0:
            ax.text(1.0, i, "absent", va="center", ha="left", fontsize=5.5,
                    color="#999999", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols, fontsize=6, family="monospace")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Completeness (%)")
    ax.set_title("Data Availability")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=GREEN, label="\u2265 80 % complete"),
        mpatches.Patch(facecolor=ORANGE, label="< 80 % complete"),
        mpatches.Patch(facecolor=GRAY, label="Absent"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="lower right")

    return save_figure(fig, output_dir, "data_availability", formats=_get_formats(config))
