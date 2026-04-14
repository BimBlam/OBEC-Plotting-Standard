"""
Voltage profile plots for the batteryplot package.

Provides:
- plot_voltage_vs_capacity: charge/discharge voltage profiles at representative cycles
- plot_voltage_vs_time: voltage (and optionally current) over elapsed test time

Scientific assumptions
----------------------
- Capacity columns may be cumulative (monotonically increasing across the full test).
  Within each cycle we subtract the cycle minimum so both charge and discharge arcs
  start near 0 Ah.
- Charge half-cycle: current_a > 0 (conventional current flowing into the cell).
- Discharge half-cycle: current_a < 0.
- If current_a is absent all data in the cycle is treated as a single arc.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batteryplot.styles import (
    apply_style,
    WONG_PALETTE,
    SINGLE_COL_WIDTH_IN,
    DOUBLE_COL_WIDTH_IN,
    DEFAULT_HEIGHT_IN,
    save_figure,
)
from batteryplot.placeholders import make_placeholder

logger = logging.getLogger("batteryplot.plots.voltage_profiles")


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _representative_cycles(cycle_summary: pd.DataFrame, config) -> List[int]:
    rep = getattr(config, "representative_cycles", None)
    if rep:
        return list(rep)
    if cycle_summary is None or "cycle_index" not in cycle_summary.columns or cycle_summary.empty:
        return []
    cycles = sorted(cycle_summary["cycle_index"].dropna().unique().tolist())
    if len(cycles) == 0:
        return []
    if len(cycles) <= 2:
        return cycles
    mid = cycles[len(cycles) // 2]
    return [cycles[0], mid, cycles[-1]]


def _reset_capacity_within_cycle(cap_series: pd.Series) -> pd.Series:
    return cap_series - cap_series.min()


def plot_voltage_vs_capacity(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot charge/discharge voltage profiles for representative cycles.

    Each representative cycle is drawn with two line styles:
    - Solid line: discharge half-cycle (current_a < 0)
    - Dashed line: charge half-cycle (current_a > 0)

    Colour is consistent per cycle across both half-cycles, drawn from the
    Wong (2011) colorblind-safe palette.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned timeseries DataFrame with canonical column names.
    cycle_summary : pd.DataFrame
        One row per cycle; used to select representative cycles.
    pulse_df : pd.DataFrame
        Pulse events DataFrame (not used here, included for uniform signature).
    config : BatteryPlotConfig
        Pipeline configuration object.
    output_dir : Path
        Directory where output files are saved.

    Returns
    -------
    list of Path
        Saved file paths.
    """
    theme = getattr(config, "theme", "publication")
    apply_style(theme)

    required = ["voltage_v", "capacity_ah"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning("plot_voltage_vs_capacity: missing columns %s — generating placeholder", missing)
        return make_placeholder(
            title="Voltage vs. Capacity",
            missing_columns=missing,
            output_dir=output_dir,
            stem="voltage_vs_capacity",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_cycles = _representative_cycles(cycle_summary, config)

    if not rep_cycles and "cycle_index" in df.columns:
        all_cycles = sorted(df["cycle_index"].dropna().unique().tolist())
        if len(all_cycles) == 0:
            rep_cycles = []
        elif len(all_cycles) <= 3:
            rep_cycles = all_cycles
        else:
            mid = all_cycles[len(all_cycles) // 2]
            rep_cycles = [all_cycles[0], mid, all_cycles[-1]]

    has_cycle = "cycle_index" in df.columns and len(rep_cycles) > 0
    has_current = "current_a" in df.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    plotted_any = False

    if has_cycle:
        for i, cyc in enumerate(rep_cycles):
            color = WONG_PALETTE[i % len(WONG_PALETTE)]
            cyc_df = df[df["cycle_index"] == cyc].copy()
            if len(cyc_df) < 2:
                logger.warning("Cycle %s has fewer than 2 data points — skipping.", cyc)
                continue

            cyc_df["capacity_reset"] = _reset_capacity_within_cycle(cyc_df["capacity_ah"])

            if has_current:
                chg = cyc_df[cyc_df["current_a"] > 0]
                dchg = cyc_df[cyc_df["current_a"] < 0]
            else:
                chg = pd.DataFrame()
                dchg = cyc_df

            label_shown = False
            if len(dchg) >= 2:
                ax.plot(
                    dchg["capacity_reset"],
                    dchg["voltage_v"],
                    color=color,
                    linestyle="-",
                    label=f"Cycle {int(cyc)}" if not label_shown else "_nolegend_",
                )
                label_shown = True
                plotted_any = True
            if len(chg) >= 2:
                ax.plot(
                    chg["capacity_reset"],
                    chg["voltage_v"],
                    color=color,
                    linestyle="--",
                    label=f"Cycle {int(cyc)} (chg)" if not label_shown else "_nolegend_",
                )
                label_shown = True
                plotted_any = True
    else:
        cap_reset = _reset_capacity_within_cycle(df["capacity_ah"])
        ax.plot(cap_reset, df["voltage_v"], color=WONG_PALETTE[5], linestyle="-")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return make_placeholder(
            title="Voltage vs. Capacity",
            missing_columns=[],
            output_dir=output_dir,
            stem="voltage_vs_capacity",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
            note="No representative cycles had sufficient data points (>=2).",
        )

    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs. Capacity")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if has_current:
            from matplotlib.lines import Line2D
            solid_patch = Line2D([0], [0], color="gray", linestyle="-", label="Discharge")
            dashed_patch = Line2D([0], [0], color="gray", linestyle="--", label="Charge")
            ax.legend(handles=handles + [solid_patch, dashed_patch],
                      labels=labels + ["Discharge", "Charge"],
                      fontsize=7, loc="best")
        else:
            ax.legend(fontsize=7, loc="best")

    formats = tuple(getattr(config, "output_formats", ("svg", "pdf")))
    return save_figure(fig, output_dir, "voltage_vs_capacity", formats=formats)


def plot_voltage_vs_time(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot voltage (and optionally current) versus elapsed test time.

    If current_a is present in df, a two-panel figure is created with voltage
    on top and current on the bottom, sharing the same x-axis.  Otherwise a
    single voltage panel is drawn.

    The time axis is converted from seconds to hours (elapsed_time_s / 3600).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned timeseries DataFrame with canonical column names.
    cycle_summary : pd.DataFrame
        One row per cycle (not used here, included for uniform signature).
    pulse_df : pd.DataFrame
        Pulse events DataFrame (not used here).
    config : BatteryPlotConfig
        Pipeline configuration object.
    output_dir : Path
        Directory where output files are saved.

    Returns
    -------
    list of Path
        Saved file paths.
    """
    theme = getattr(config, "theme", "publication")
    apply_style(theme)

    required = ["voltage_v", "elapsed_time_s"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning("plot_voltage_vs_time: missing columns %s — generating placeholder", missing)
        return make_placeholder(
            title="Voltage and Current vs. Time",
            missing_columns=missing,
            output_dir=output_dir,
            stem="voltage_vs_time",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_h = df["elapsed_time_s"] / 3600.0
    has_current = "current_a" in df.columns

    if has_current:
        fig, (ax_v, ax_i) = plt.subplots(
            2, 1,
            figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN * 1.6),
            sharex=True,
        )
        ax_v.plot(time_h, df["voltage_v"], color=WONG_PALETTE[5], linewidth=0.8)
        ax_v.set_ylabel("Voltage (V)")
        ax_v.set_title("Voltage and Current vs. Time")

        ax_i.plot(time_h, df["current_a"], color=WONG_PALETTE[6], linewidth=0.8)
        ax_i.set_ylabel("Current (A)")
        ax_i.set_xlabel("Time (h)")
        ax_i.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    else:
        fig, ax_v = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))
        ax_v.plot(time_h, df["voltage_v"], color=WONG_PALETTE[5], linewidth=0.8)
        ax_v.set_xlabel("Time (h)")
        ax_v.set_ylabel("Voltage (V)")
        ax_v.set_title("Voltage vs. Time")

    formats = tuple(getattr(config, "output_formats", ("svg", "pdf")))
    return save_figure(fig, output_dir, "voltage_vs_time", formats=formats)
