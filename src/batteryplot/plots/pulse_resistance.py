"""
Pulse and resistance plots for the batteryplot package.

Provides:
- plot_dcir_vs_current: DCIR versus applied current (or current density).
- plot_pulse_analysis: Pulse polarisation decomposition into ohmic and kinetic
  contributions.

Scientific assumptions
----------------------
DCIR labelling
~~~~~~~~~~~~~~
Two distinct quantities are handled:
1. Measured DCIR (dcir_ohm in df): directly recorded by cycler firmware.
   Labeled "Measured DCIR".
2. Estimated DCIR (estimated_dcir in pulse_df, or derived from dV/I):
   calculated post-hoc from a voltage step / current step.
   Labeled "Estimated DCIR (dV/I)".
These must never be conflated.

Current density
~~~~~~~~~~~~~~~
Current density [mA/cm2] = |current_a| * 1000 / electrode_area_cm2.
Used only when config.electrode_area_cm2 is set and > 0.

Pulse decomposition
~~~~~~~~~~~~~~~~~~~
Immediate voltage drop at pulse onset (dV0 = V_before - V_first) approximates
ohmic resistance: R_ohm = dV0 / I.
Total drop at end of pulse: R_total = dV_total / I.
Kinetic: R_kinetic = R_total - R_ohm.
Only pulses with step_time < 200 s and |I| > 0.01 A are included.
At least 3 valid pulse events are required.
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
    DEFAULT_HEIGHT_IN,
    save_figure,
    add_assumption_warning,
)
from batteryplot.placeholders import (
    make_placeholder,
    diagnose_columns,
    ColumnDiagnostic,
)

logger = logging.getLogger("batteryplot.plots.pulse_resistance")


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _get_formats(config) -> tuple:
    return tuple(getattr(config, "output_formats", ("svg", "pdf")))


def _get_theme(config) -> str:
    return getattr(config, "theme", "publication")


def _is_valid_df(df) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


def plot_dcir_vs_current(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Scatter plot of DCIR versus applied current (or current density).

    Measured DCIR (dcir_ohm) from df is shown as filled circles, colour-coded
    by cycle_index.  Estimated DCIR from pulse_df (estimated_dcir column) is
    overlaid as open diamonds, if present.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.  Must contain dcir_ohm.
    cycle_summary : pd.DataFrame
        Not used; included for uniform signature.
    pulse_df : pd.DataFrame
        Pulse events DataFrame.  If it contains estimated_dcir and current_a,
        these are overlaid.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["dcir_ohm"]
    missing = _check_columns(df, required)
    if missing:
        logger.warning("plot_dcir_vs_current: missing df columns %s", missing)
        diag = diagnose_columns(df, required, optional=["current_a", "cycle_index"])
        diag.note = ("DCIR (Ohms) must be present and non-zero. "
                     "If all-zero, the cycler did not perform a pulse measurement. "
                     "Current density axis requires electrode_area_cm2 in config.")
        return make_placeholder(
            title="DCIR vs. Current Density",
            missing_columns=missing,
            output_dir=output_dir,
            stem="dcir_vs_current",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    plot_df = df[df["dcir_ohm"].notna() & (df["dcir_ohm"] != 0)].copy()
    if plot_df.empty:
        return make_placeholder(
            title="DCIR vs. Current Density",
            missing_columns=[],
            output_dir=output_dir,
            stem="dcir_vs_current",
            formats=_get_formats(config),
            note="dcir_ohm column is present but all values are zero or NaN.",
        )

    area = getattr(config, "electrode_area_cm2", None)
    use_density = area is not None and area > 0
    has_current = "current_a" in plot_df.columns

    if has_current:
        if use_density:
            plot_df = plot_df.copy()
            plot_df["x_val"] = plot_df["current_a"].abs() * 1000.0 / area
            x_label = "Current Density (mA cm\u207b\u00b2)"
        else:
            plot_df = plot_df.copy()
            plot_df["x_val"] = plot_df["current_a"].abs()
            x_label = "|Current| (A)"
    else:
        plot_df = plot_df.copy()
        plot_df["x_val"] = np.arange(len(plot_df), dtype=float)
        x_label = "Record Index (no current_a)"

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    has_cycle = "cycle_index" in plot_df.columns
    if has_cycle:
        sc = ax.scatter(
            plot_df["x_val"], plot_df["dcir_ohm"],
            c=plot_df["cycle_index"], cmap="viridis",
            s=10, alpha=0.7, marker="o", linewidths=0,
            label="Measured DCIR",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Cycle Index", fontsize=7)
    else:
        ax.scatter(
            plot_df["x_val"], plot_df["dcir_ohm"],
            color=WONG_PALETTE[5], s=10, alpha=0.7, marker="o",
            label="Measured DCIR",
        )

    if _is_valid_df(pulse_df) and "estimated_dcir" in pulse_df.columns and "current_a" in pulse_df.columns:
        pdf = pulse_df.dropna(subset=["estimated_dcir", "current_a"]).copy()
        if not pdf.empty:
            if use_density:
                px = pdf["current_a"].abs() * 1000.0 / area
            else:
                px = pdf["current_a"].abs()
            ax.scatter(
                px, pdf["estimated_dcir"],
                color=WONG_PALETTE[6], s=18, alpha=0.85, marker="D",
                facecolors="none", linewidths=0.8,
                label="Estimated DCIR (\u0394V/I)",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("DCIR (\u03a9)")
    ax.set_title("DCIR vs. Current Density")
    ax.legend(fontsize=7, loc="best")

    _dcir_warnings: list[str] = []
    if getattr(config, "electrode_area_cm2", None) is None:
        _dcir_warnings.append(
            "electrode_area_cm2 not set: x-axis shows |current| (A), not current density"
        )
    add_assumption_warning(fig, _dcir_warnings)
    return save_figure(fig, output_dir, "dcir_vs_current", formats=_get_formats(config))


def plot_pulse_analysis(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Pulse resistance decomposition: ohmic vs. kinetic contributions.

    Pulse events are extracted from df using the following criteria:
    - step_time_s < 200 s  (short pulses only)
    - |current_a| > 0.01 A  (exclude rest and near-zero steps)

    For each qualifying pulse segment the immediate voltage drop dV0 and the
    end-of-pulse voltage drop dV_total are computed.

    R_ohmic   = dV0 / |I|
    R_kinetic = (dV_total - dV0) / |I|

    Results are shown as a stacked bar chart per pulse event.  At least 3
    valid pulses are required; otherwise a placeholder is returned.

    Scientific note: This is an approximation; it does not resolve contributions
    separated by EIS.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.  Requires current_a, voltage_v, elapsed_time_s.
    cycle_summary : pd.DataFrame
        Not used; included for uniform signature.
    pulse_df : pd.DataFrame
        Pre-computed pulse events.  If valid, used before re-extracting from df.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_ts = ["current_a", "voltage_v", "elapsed_time_s"]
    missing = _check_columns(df, required_ts)
    if missing:
        logger.warning("plot_pulse_analysis: missing df columns %s", missing)
        diag = diagnose_columns(df, required,
                                optional=["dcir_ohm", "step_time_s", "procedure_step"])
        diag.note = ("Pulse detection: step_time_s < 200 s and |current| > 0.01 A "
                     "followed by a rest. Minimum 3 pulses required.")
        return make_placeholder(
            title="Pulse Resistance Decomposition",
            missing_columns=missing,
            output_dir=output_dir,
            stem="pulse_analysis",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    pulse_records = []

    if _is_valid_df(pulse_df):
        for col_r_ohm in ("r_ohmic_ohm", "estimated_dcir", "dcir_ohm_measured"):
            if col_r_ohm in pulse_df.columns and "current_a" in pulse_df.columns:
                tmp = pulse_df.dropna(subset=[col_r_ohm, "current_a"]).copy()
                for _, row in tmp.iterrows():
                    r_ohm = abs(row[col_r_ohm])
                    r_kin = abs(row["r_kinetic_ohm"]) if "r_kinetic_ohm" in tmp.columns else 0.0
                    pulse_records.append({
                        "i_abs": abs(row["current_a"]),
                        "r_ohmic": r_ohm,
                        "r_kinetic": r_kin,
                    })
                break

    if len(pulse_records) < 3:
        pulse_records = _extract_pulses_from_timeseries(df)

    if len(pulse_records) < 3:
        return make_placeholder(
            title="Pulse Resistance Decomposition",
            missing_columns=[],
            output_dir=output_dir,
            stem="pulse_analysis",
            formats=_get_formats(config),
            note=(
                f"Only {len(pulse_records)} valid pulse event(s) found (need >= 3). "
                "Check that short constant-current pulses (|I| > 0.01 A, t < 200 s) are present."
            ),
        )

    pulse_df_plot = pd.DataFrame(pulse_records)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    x = np.arange(len(pulse_df_plot))
    ax.bar(x, pulse_df_plot["r_ohmic"], color=WONG_PALETTE[5], label="Ohmic (R\u2126)")
    ax.bar(x, pulse_df_plot["r_kinetic"], bottom=pulse_df_plot["r_ohmic"],
           color=WONG_PALETTE[1], label="Kinetic (R\u2081)")

    ax.set_xlabel("Pulse Index")
    ax.set_ylabel("Resistance (\u03a9)")
    ax.set_title("Pulse Resistance Decomposition")
    ax.legend(fontsize=7, loc="best")

    ax.text(
        0.98, 0.05,
        "Estimated from \u0394V/I \u2014 not EIS-resolved",
        transform=ax.transAxes, fontsize=5.5, color="#888888",
        ha="right", va="bottom", style="italic",
    )

    return save_figure(fig, output_dir, "pulse_analysis", formats=_get_formats(config))


def _extract_pulses_from_timeseries(df: pd.DataFrame) -> List[dict]:
    """Extract pulse events from timeseries, returning list of dicts with r_ohmic and r_kinetic."""
    records: List[dict] = []

    work = df.copy()
    has_step_time = "step_time_s" in work.columns

    mask_i = work["current_a"].abs() > 0.01
    if has_step_time:
        mask_i = mask_i & (work["step_time_s"] < 200)

    work["_pulse"] = mask_i.astype(int)
    work["_block"] = (work["_pulse"].diff() != 0).cumsum()

    for block_id, block in work[work["_pulse"] == 1].groupby("_block"):
        if len(block) < 2:
            continue
        signs = np.sign(block["current_a"])
        if signs.nunique() > 1:
            continue

        i_mean = block["current_a"].abs().mean()
        if i_mean < 0.01:
            continue

        first_idx = block.index[0]
        try:
            iloc_pos = work.index.get_loc(first_idx)
        except KeyError:
            continue
        if iloc_pos == 0:
            continue

        v_before = work.iloc[iloc_pos - 1]["voltage_v"]
        v_start = block.iloc[0]["voltage_v"]
        v_end = block.iloc[-1]["voltage_v"]

        dv0 = abs(v_before - v_start)
        dv_total = abs(v_before - v_end)

        r_ohmic = dv0 / i_mean
        r_kinetic = max(0.0, (dv_total - dv0) / i_mean)

        records.append({
            "i_abs": i_mean,
            "r_ohmic": r_ohmic,
            "r_kinetic": r_kinetic,
        })

    return records
