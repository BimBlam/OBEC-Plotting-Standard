"""
Cycle-summary plots for the batteryplot package.

Provides:
- plot_capacity_retention: discharge (and charge) capacity vs. cycle number,
  with optional capacity-retention % on a right y-axis.
- plot_coulombic_efficiency: coulombic (and optionally energy) efficiency vs. cycle.
- plot_dcir_vs_cycle: DCIR (and optionally AC impedance) trend vs. cycle.

Scientific assumptions
----------------------
- capacity_retention_pct is defined as:
      discharge_capacity_ah / nominal_capacity_ah x 100
  and is expected to already exist in cycle_summary; if absent, it is computed
  here only if config.nominal_capacity_ah is set (non-None and > 0).
- coulombic_efficiency_pct is defined as:
      discharge_capacity_ah / charge_capacity_ah x 100  [%]
  A horizontal reference line is drawn at 100 %.
- energy_efficiency_pct is defined as:
      discharge_energy_wh / charge_energy_wh x 100  [%]
  (column name "energy_efficiency_pct" if present in cycle_summary).
- DCIR (mean_dcir_ohm) is the per-cycle average of the cycler-measured DC
  internal resistance.  AC impedance (mean_ac_impedance_ohm) typically
  represents the real part of impedance at a fixed frequency (e.g. 1 kHz).
  Both are plotted on the same y-axis with clear legend labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

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

logger = logging.getLogger("batteryplot.plots.cycle_summary")


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _get_formats(config) -> tuple:
    return tuple(getattr(config, "output_formats", ("svg", "pdf")))


def _get_theme(config) -> str:
    return getattr(config, "theme", "publication")


def plot_capacity_retention(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot discharge (and charge) capacity versus cycle number.

    If capacity_retention_pct is available (or can be derived from
    nominal_capacity_ah), it is overlaid on a secondary right y-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame (not used here; included for uniform signature).
    cycle_summary : pd.DataFrame
        One row per cycle with at minimum cycle_index and discharge_capacity_ah.
    pulse_df : pd.DataFrame
        Pulse events DataFrame (not used here).
    config : BatteryPlotConfig
        Pipeline configuration.
    output_dir : Path
        Directory where output files are saved.

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["cycle_index", "discharge_capacity_ah"]
    missing = _check_columns(cycle_summary, required)
    if missing:
        logger.warning("plot_capacity_retention: missing columns %s — placeholder", missing)
        diag = diagnose_columns(cycle_summary, required,
                                optional=["charge_capacity_ah", "capacity_retention_pct"])
        diag.note = ("capacity_retention_pct requires nominal_capacity_ah in config. "
                     "coulombic_efficiency requires both charge and discharge columns.")
        return make_placeholder(
            title="Capacity Retention vs. Cycle Number",
            missing_columns=missing,
            output_dir=output_dir,
            stem="capacity_retention",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    cs = cycle_summary.sort_values("cycle_index").copy()

    # Prefer "cycling" region when test_region is available
    region_filtered = False
    if "test_region" in cs.columns:
        cycling_cs = cs[cs["test_region"] == "cycling"]
        if not cycling_cs.empty:
            logger.info("plot_capacity_retention: using %d cycling cycles (of %d total).", len(cycling_cs), len(cs))
            cs = cycling_cs
            region_filtered = True
        else:
            logger.info("plot_capacity_retention: no cycling cycles found; using all %d cycles.", len(cs))

    has_retention = "capacity_retention_pct" in cs.columns
    if not has_retention:
        nom = getattr(config, "nominal_capacity_ah", None)
        if nom and nom > 0:
            cs["capacity_retention_pct"] = cs["discharge_capacity_ah"] / nom * 100.0
            has_retention = True

    has_charge = "charge_capacity_ah" in cs.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    ax.plot(
        cs["cycle_index"], cs["discharge_capacity_ah"],
        marker="o", markersize=4, linestyle="-",
        color=WONG_PALETTE[5], label="Discharge capacity",
    )
    if has_charge:
        ax.plot(
            cs["cycle_index"], cs["charge_capacity_ah"],
            marker="^", markersize=4, linestyle="--",
            color=WONG_PALETTE[6], label="Charge capacity",
        )

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity Retention vs. Cycle Number")

    if has_retention:
        ax2 = ax.twinx()
        ax2.plot(
            cs["cycle_index"], cs["capacity_retention_pct"],
            marker="s", markersize=3, linestyle=":",
            color=WONG_PALETTE[1], alpha=0.7, label="Retention (%)",
        )
        ax2.set_ylabel("Capacity Retention (%)")
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="best")
    else:
        ax.legend(fontsize=7, loc="best")

    _cr_warnings: list[str] = []
    if region_filtered:
        cmin, cmax = int(cs["cycle_index"].min()), int(cs["cycle_index"].max())
        _cr_warnings.append(f"Showing cycling region: cycles {cmin}\u2013{cmax}")
    if has_retention and getattr(config, "nominal_capacity_ah", None) is None:
        _cr_warnings.append(
            "nominal_capacity_ah not set — retention % computed from first-cycle "
            "discharge capacity (relative, not absolute)"
        )
    add_assumption_warning(fig, _cr_warnings)
    return save_figure(fig, output_dir, "capacity_retention", formats=_get_formats(config))


def plot_coulombic_efficiency(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot coulombic efficiency (and optionally energy efficiency) vs. cycle.

    A dashed horizontal reference line is drawn at 100 %.  The y-axis is
    constrained to 80-105 % by default and auto-extended if data fall outside.

    Scientific note: coulombic efficiency (CE) = Q_discharge / Q_charge x 100.
    CE < 100 % indicates irreversible capacity losses (e.g. SEI formation,
    lithium plating, electrolyte decomposition).  First-cycle CE is often low
    and is plotted without special treatment.

    Parameters
    ----------
    df, pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    cycle_summary : pd.DataFrame
        One row per cycle with cycle_index and coulombic_efficiency.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["cycle_index", "coulombic_efficiency_pct"]
    missing = _check_columns(cycle_summary, required)
    if missing:
        logger.warning("plot_coulombic_efficiency: missing columns %s — placeholder", missing)
        diag = diagnose_columns(cycle_summary, required,
                                optional=["energy_efficiency_pct", "charge_capacity_ah",
                                          "discharge_capacity_ah"])
        diag.note = ("coulombic_efficiency_pct = Q_discharge / Q_charge × 100. "
                     "Requires a cycle that contains both a charge and a discharge step.")
        return make_placeholder(
            title="Coulombic and Energy Efficiency vs. Cycle",
            missing_columns=missing,
            output_dir=output_dir,
            stem="coulombic_efficiency",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    cs = cycle_summary.sort_values("cycle_index").copy()

    # Prefer "cycling" region when test_region is available
    region_filtered = False
    if "test_region" in cs.columns:
        cycling_cs = cs[cs["test_region"] == "cycling"]
        if not cycling_cs.empty:
            logger.info("plot_coulombic_efficiency: using %d cycling cycles (of %d total).", len(cycling_cs), len(cs))
            cs = cycling_cs
            region_filtered = True
        else:
            logger.info("plot_coulombic_efficiency: no cycling cycles found; using all %d cycles.", len(cs))

    has_energy_eff = "energy_efficiency_pct" in cs.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    ax.plot(
        cs["cycle_index"], cs["coulombic_efficiency_pct"],
        marker="o", markersize=4, linestyle="-",
        color=WONG_PALETTE[5], label="Coulombic efficiency",
    )
    if has_energy_eff:
        ax.plot(
            cs["cycle_index"], cs["energy_efficiency_pct"],
            marker="s", markersize=4, linestyle="--",
            color=WONG_PALETTE[1], label="Energy efficiency",
        )

    ax.axhline(100.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    all_vals = cs["coulombic_efficiency_pct"].dropna().tolist()
    if has_energy_eff:
        all_vals += cs["energy_efficiency_pct"].dropna().tolist()
    ymin = min(80.0, min(all_vals) - 1.0) if all_vals else 80.0
    ymax = max(105.0, max(all_vals) + 1.0) if all_vals else 105.0
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Coulombic and Energy Efficiency vs. Cycle")
    ax.legend(fontsize=7, loc="best")

    _ce_warnings: list[str] = []
    if region_filtered:
        cmin, cmax = int(cs["cycle_index"].min()), int(cs["cycle_index"].max())
        _ce_warnings.append(f"Showing cycling region: cycles {cmin}\u2013{cmax}")
    add_assumption_warning(fig, _ce_warnings)
    return save_figure(fig, output_dir, "coulombic_efficiency", formats=_get_formats(config))


def plot_dcir_vs_cycle(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Plot mean per-cycle DCIR (and optionally AC impedance) vs. cycle number.

    Scientific note: DCIR (DC internal resistance) is estimated from a
    voltage step divided by the current step, typically during a short pulse.
    Rising DCIR over cycling indicates increasing cell impedance, commonly
    caused by SEI growth, lithium plating, or contact degradation.

    AC impedance (real part at 1 kHz) is plotted if mean_ac_impedance_ohm is
    present in cycle_summary.  Both quantities share the same y-axis (Ohm) and
    are clearly distinguished in the legend.

    Parameters
    ----------
    df, pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    cycle_summary : pd.DataFrame
        One row per cycle with cycle_index and mean_dcir_ohm.
    config : BatteryPlotConfig
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["cycle_index", "mean_dcir_ohm"]
    missing = _check_columns(cycle_summary, required)
    if missing:
        logger.warning("plot_dcir_vs_cycle: missing columns %s — placeholder", missing)
        diag = diagnose_columns(cycle_summary, required,
                                optional=["mean_ac_impedance_ohm", "dcir_ohm"])
        diag.note = ("mean_dcir_ohm is the per-cycle average of the cycler's DCIR column. "
                     "If the DCIR column was all-zero, the cycler may not have "
                     "performed a pulse measurement.")
        return make_placeholder(
            title="DCIR vs. Cycle Number",
            missing_columns=missing,
            output_dir=output_dir,
            stem="dcir_vs_cycle",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    cs = cycle_summary.sort_values("cycle_index").copy()
    has_eis = "mean_ac_impedance_ohm" in cs.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    ax.plot(
        cs["cycle_index"], cs["mean_dcir_ohm"],
        marker="o", markersize=4, linestyle="-",
        color=WONG_PALETTE[5], label="DCIR (measured)",
    )
    if has_eis:
        ax.plot(
            cs["cycle_index"], cs["mean_ac_impedance_ohm"],
            marker="s", markersize=4, linestyle="--",
            color=WONG_PALETTE[1], label="AC impedance (|Z| @ 1 kHz)",
        )

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Resistance (\u03a9)")
    ax.set_title("DCIR vs. Cycle Number")
    ax.legend(fontsize=7, loc="best")

    return save_figure(fig, output_dir, "dcir_vs_cycle", formats=_get_formats(config))
