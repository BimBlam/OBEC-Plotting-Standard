"""
Voltage profile plots for the batteryplot package.

Provides
--------
plot_voltage_vs_capacity
    Charge and discharge arcs overlaid on the same axes for each representative
    cycle.  Charge runs left→right (0 → Q_chg); discharge runs right→left
    (Q_dis → 0), so the two arcs form the classic closed "butterfly" loop used
    in electrochemistry literature.

plot_voltage_vs_time
    Voltage (and optionally current) versus elapsed test time.

Scientific assumptions
----------------------
- The Arbin ``Capacity (AHr)`` column is a **within-step accumulator**: it
  resets to 0 at the start of every new step.  So for a charge step it goes
  0 → Q_chg; for a discharge step it goes 0 → Q_dis.  We therefore do NOT
  subtract a cycle minimum — we use the raw step-accumulated value directly.
- Charge half-cycle  : current_a > CURRENT_THRESHOLD  (default 1e-4 A)
- Discharge half-cycle: current_a < -CURRENT_THRESHOLD
- If current_a is absent every data point in the cycle is treated as a single
  arc (no charge/discharge separation).
- For the overlaid plot the discharge capacity axis is reflected so that both
  arcs share a common "absolute capacity" x-axis running left (0 Ah) to right
  (max capacity of that cycle).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

logger = logging.getLogger("batteryplot.plots.voltage_profiles")

# Current magnitude below this is treated as "rest" (neither charge nor discharge)
CURRENT_THRESHOLD = 1e-4  # A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """Return list of required column names absent from df."""
    return [c for c in required if c not in df.columns]


def _representative_cycles(cycle_summary: pd.DataFrame, config) -> List[int]:
    """
    Return the list of cycle indices to plot.

    Uses config.representative_cycles when set; otherwise picks
    [first, middle, last] from cycle_summary, preferring cycles in the
    "cycling" test region when test_region is available.
    """
    rep = getattr(config, "representative_cycles", None)
    if rep:
        return [int(c) for c in rep]
    if cycle_summary is None or "cycle_index" not in cycle_summary.columns or cycle_summary.empty:
        return []

    # Prefer "cycling" region when test_region column is available
    cs = cycle_summary
    if "test_region" in cs.columns:
        cycling_cs = cs[cs["test_region"] == "cycling"]
        if not cycling_cs.empty:
            cs = cycling_cs

    cycles = sorted(cs["cycle_index"].dropna().unique().tolist())
    if len(cycles) == 0:
        return []
    if len(cycles) <= 3:
        return [int(c) for c in cycles]
    mid = cycles[len(cycles) // 2]
    return [int(cycles[0]), int(mid), int(cycles[-1])]


def _split_segments(cyc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single-cycle DataFrame into charge and discharge sub-frames.

    Returns (charge_df, discharge_df).  Each sub-frame may be empty.

    Uses the ``segment`` column (set by :func:`label_charge_discharge`) when
    available, because it is derived from the cycler's step-type code (MD
    column) and correctly handles Maccor-style exports where current is always
    positive.  Falls back to current sign only when ``segment`` is absent.

    A defensive voltage-based sanity check swaps the two frames if the
    "charge" sub-frame has a *lower* mean voltage than the "discharge"
    sub-frame — physically, charge half-cycles always span a higher mean
    voltage than discharge half-cycles for any normal battery chemistry.
    """
    if "segment" in cyc_df.columns:
        chg  = cyc_df[cyc_df["segment"] == "charge"].copy()
        dchg = cyc_df[cyc_df["segment"] == "discharge"].copy()
    elif "current_a" in cyc_df.columns:
        chg  = cyc_df[cyc_df["current_a"] >  CURRENT_THRESHOLD].copy()
        dchg = cyc_df[cyc_df["current_a"] < -CURRENT_THRESHOLD].copy()
    else:
        chg  = pd.DataFrame()
        dchg = cyc_df.copy()

    # Voltage sanity check: charge mean V should be >= discharge mean V.
    if (
        len(chg) >= 2
        and len(dchg) >= 2
        and "voltage_v" in chg.columns
    ):
        v_chg = pd.to_numeric(chg["voltage_v"], errors="coerce").mean()
        v_dchg = pd.to_numeric(dchg["voltage_v"], errors="coerce").mean()
        if not (pd.isna(v_chg) or pd.isna(v_dchg)) and v_chg < v_dchg:
            logger.info(
                "Voltage sanity swap: charge mean V (%.3f) < discharge mean V (%.3f); "
                "swapping charge ↔ discharge.",
                v_chg, v_dchg,
            )
            chg, dchg = dchg, chg

    return chg, dchg


def _arc_capacity(segment_df: pd.DataFrame, mirror: bool = False) -> pd.Series:
    """
    Return the capacity x-axis for one half-cycle arc.

    The Arbin ``capacity_ah`` column accumulates from 0 within each step, so
    it already represents "charge passed since step start".  We use it directly.

    For the discharge arc we mirror the axis so that capacity decreases from
    left to right (i.e. discharge runs right → left on the shared axis), giving
    the closed-loop butterfly shape.  Specifically:

        x_discharge = Q_dis_max − capacity_ah

    so the arc starts at Q_dis_max (full) and ends at 0 (empty), matching the
    charge arc direction.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Half-cycle rows with a ``capacity_ah`` column.
    mirror : bool
        If True, mirror the axis (use for discharge).

    Returns
    -------
    pd.Series
        Capacity values (Ah) for plotting.
    """
    cap = segment_df["capacity_ah"].copy()
    cap = cap.abs()  # discharge accumulator may appear negative in some exports
    # Reset to zero-based within this arc, but only if the first value is
    # significantly above zero (> 1% of max).  When it already starts near 0,
    # subtracting again would produce negative values due to float noise.
    if len(cap) > 0:
        cap_max = cap.max()
        if cap_max > 0 and cap.iloc[0] > 0.01 * cap_max:
            cap = cap - cap.iloc[0]
            cap = cap.abs()  # ensure no negatives after subtraction
    if mirror:
        cap = cap.max() - cap
    return cap


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_voltage_vs_capacity(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Charge/discharge voltage profiles for representative cycles, overlaid.

    Each representative cycle contributes two arcs on the same axes:
    - **Charge** (dashed line): voltage rises from V_min → V_max as capacity
      increases left → right.
    - **Discharge** (solid line): voltage falls from V_max → V_min; the
      capacity axis is mirrored so the discharge arc runs right → left,
      closing the loop with the charge arc at both ends.

    Cycles are colour-coded using the Wong (2011) colorblind-safe palette.
    A shared style legend (solid = discharge, dashed = charge) is added once.

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
        Saved file paths (SVG and/or PDF).
    """
    theme = getattr(config, "theme", "publication")
    apply_style(theme)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Column guard ---
    required = ["voltage_v", "capacity_ah"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning(
            "plot_voltage_vs_capacity: missing columns %s — generating placeholder", missing
        )
        diag = diagnose_columns(df, required, optional=["cycle_index", "current_a"])
        return make_placeholder(
            title="Voltage vs. Capacity",
            missing_columns=missing,
            output_dir=output_dir,
            stem="voltage_vs_capacity",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
            diagnostic=diag,
        )

    has_current = "current_a" in df.columns
    has_cycle   = "cycle_index" in df.columns

    # --- Select representative cycles ---
    if has_cycle:
        rep_cycles = _representative_cycles(cycle_summary, config)
        if not rep_cycles:
            all_cycles = sorted(df["cycle_index"].dropna().unique().tolist())
            if len(all_cycles) <= 3:
                rep_cycles = [int(c) for c in all_cycles]
            elif all_cycles:
                mid = all_cycles[len(all_cycles) // 2]
                rep_cycles = [int(all_cycles[0]), int(mid), int(all_cycles[-1])]
    else:
        rep_cycles = []

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))
    plotted_any = False
    cycle_handles = []   # one handle per cycle for the cycle-colour legend

    if has_cycle and rep_cycles:
        for i, cyc in enumerate(rep_cycles):
            color = WONG_PALETTE[i % len(WONG_PALETTE)]
            cyc_df = df[df["cycle_index"] == cyc].copy()

            if len(cyc_df) < 2:
                logger.warning("Cycle %s: fewer than 2 data points — skipping.", cyc)
                continue

            if has_current or "segment" in cyc_df.columns:
                chg, dchg = _split_segments(cyc_df)
            else:
                chg  = pd.DataFrame()
                dchg = cyc_df

            # --- Discharge arc (solid) ---
            if len(dchg) >= 2:
                x_dchg = _arc_capacity(dchg, mirror=False)
                # Mirror so discharge runs from Q_max → 0 (right to left)
                x_dchg = x_dchg.max() - x_dchg
                ax.plot(
                    x_dchg,
                    dchg["voltage_v"].values,
                    color=color,
                    linestyle="-",
                    linewidth=1.0,
                    label="_nolegend_",
                )
                plotted_any = True

            # --- Charge arc (dashed) ---
            if len(chg) >= 2:
                x_chg = _arc_capacity(chg, mirror=False)
                ax.plot(
                    x_chg,
                    chg["voltage_v"].values,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    label="_nolegend_",
                )
                plotted_any = True

            # One proxy handle per cycle for the colour legend
            if len(dchg) >= 2 or len(chg) >= 2:
                proxy = mlines.Line2D([], [], color=color, linestyle="-",
                                      label=f"Cycle {cyc}")
                cycle_handles.append(proxy)

    else:
        # No cycle column — plot all data as one arc
        cap = (df["capacity_ah"] - df["capacity_ah"].iloc[0]).abs()
        ax.plot(cap, df["voltage_v"], color=WONG_PALETTE[5], linestyle="-")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        diag2 = diagnose_columns(df, required, optional=["cycle_index", "current_a"])
        diag2.note = "No representative cycles had sufficient data points (≥ 2). Check cycle_index mapping."
        return make_placeholder(
            title="Voltage vs. Capacity",
            missing_columns=[],
            output_dir=output_dir,
            stem="voltage_vs_capacity",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
            diagnostic=diag2,
        )

    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs. Capacity by Cycle")

    # --- Legend: style legend + cycle-colour legend ---
    has_segment_info = has_current or "segment" in df.columns
    style_handles = []
    if has_segment_info:
        style_handles = [
            mlines.Line2D([], [], color="gray", linestyle="-",  label="Discharge"),
            mlines.Line2D([], [], color="gray", linestyle="--", label="Charge"),
        ]
    all_handles = cycle_handles + style_handles
    if all_handles:
        ax.legend(handles=all_handles, fontsize=7, loc="best",
                  frameon=False, handlelength=1.8)

    formats = tuple(getattr(config, "output_formats", ("svg", "pdf")))
    # Assumption warnings — flag anything that required a default to be assumed
    _warnings: list[str] = []
    if not has_segment_info:
        _warnings.append("current_a absent: charge/discharge separation not possible; all data shown as single arc")
    if not has_cycle:
        _warnings.append("cycle_index absent: all data treated as one arc")
    add_assumption_warning(fig, _warnings)
    return save_figure(fig, output_dir, "voltage_vs_capacity", formats=formats)


def plot_voltage_vs_time(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Voltage (and optionally current) versus elapsed test time.

    If current_a is present a two-panel figure is produced: voltage on top,
    current on the bottom, sharing the same x-axis.  Otherwise a single
    voltage panel is drawn.

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["voltage_v", "elapsed_time_s"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning(
            "plot_voltage_vs_time: missing columns %s — generating placeholder", missing
        )
        diag = diagnose_columns(df, required, optional=["current_a", "cycle_index"])
        return make_placeholder(
            title="Voltage and Current vs. Time",
            missing_columns=missing,
            output_dir=output_dir,
            stem="voltage_vs_time",
            formats=tuple(getattr(config, "output_formats", ("svg", "pdf"))),
            diagnostic=diag,
        )

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
    _vt_warnings: list[str] = []
    if not has_current:
        _vt_warnings.append("current_a absent: current panel omitted; cannot verify charge/discharge direction")
    add_assumption_warning(fig, _vt_warnings)
    return save_figure(fig, output_dir, "voltage_vs_time", formats=formats)
