"""
Rate capability plots for the batteryplot package.

Provides:
- plot_rate_capability: capacity vs. C-rate (or absolute current) per cycle.
- plot_rate_voltage_profiles: voltage-capacity curves for cycles at different rates.

Scientific assumptions
----------------------
- True C-rate = |mean current| / nominal_capacity_ah.  The x-axis is labelled
  "C-rate" ONLY when config.nominal_capacity_ah is set.  Without it, the x-axis
  label is "|Current| (A)".
- Mean absolute current per cycle is computed from the timeseries df (group by
  cycle_index, take mean of |current_a|).  If current_a is absent from df but
  the cycle_summary has a c_rate column, that is used instead.
- A log scale is applied to the x-axis when the C-rate range spans more than
  one decade.
- plot_rate_voltage_profiles is only generated if at least 2 distinct current
  levels are detected.  Cycles are grouped by their mean |current_a| rounded to
  2 significant figures; one representative cycle (most data points) per group.
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
)
from batteryplot.placeholders import make_placeholder

logger = logging.getLogger("batteryplot.plots.rate_capability")


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _get_formats(config) -> tuple:
    return tuple(getattr(config, "output_formats", ("svg", "pdf")))


def _get_theme(config) -> str:
    return getattr(config, "theme", "publication")


def _sig_round(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    factor = 10 ** (sig - int(np.floor(np.log10(abs(x)))) - 1)
    return round(x * factor) / factor


def _mean_current_per_cycle(df: pd.DataFrame) -> Optional[pd.Series]:
    if "cycle_index" not in df.columns or "current_a" not in df.columns:
        return None
    return df.groupby("cycle_index")["current_a"].apply(lambda s: s.abs().mean())


def plot_rate_capability(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Rate capability: discharge (and charge) capacity vs. C-rate or |current|.

    Scientific note: True C-rate requires nominal_capacity_ah. Without it, we
    plot vs. mean absolute current per cycle (A), clearly labeled.  Never label
    the x-axis as C-rate unless nominal_capacity_ah is set.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame (used to compute mean current per cycle).
    cycle_summary : pd.DataFrame
        One row per cycle with cycle_index and discharge_capacity_ah.
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

    required_cs = ["cycle_index", "discharge_capacity_ah"]
    missing = _check_columns(cycle_summary, required_cs)
    if missing:
        logger.warning("plot_rate_capability: missing cycle_summary columns %s", missing)
        return make_placeholder(
            title="Rate Capability: Capacity vs. C-rate",
            missing_columns=missing,
            output_dir=output_dir,
            stem="rate_capability",
            formats=_get_formats(config),
        )

    cs = cycle_summary.sort_values("cycle_index").copy()
    nom_cap = getattr(config, "nominal_capacity_ah", None)
    use_c_rate = (nom_cap is not None and nom_cap > 0)

    mean_i = _mean_current_per_cycle(df)
    if mean_i is not None:
        cs = cs.join(mean_i.rename("mean_abs_i"), on="cycle_index", how="left")
        if use_c_rate:
            cs["x_val"] = cs["mean_abs_i"] / nom_cap
            x_label = "C-rate"
        else:
            cs["x_val"] = cs["mean_abs_i"]
            x_label = "|Current| (A)"
    elif "c_rate" in cs.columns and use_c_rate:
        cs["x_val"] = cs["c_rate"]
        x_label = "C-rate"
    else:
        cs["x_val"] = cs["cycle_index"]
        x_label = "Cycle Number (no current data)"
        use_c_rate = False

    cs = cs.dropna(subset=["x_val", "discharge_capacity_ah"])
    cs = cs[cs["x_val"] > 0]

    if cs.empty:
        return make_placeholder(
            title="Rate Capability: Capacity vs. C-rate",
            missing_columns=[],
            output_dir=output_dir,
            stem="rate_capability",
            formats=_get_formats(config),
            note="No valid data after filtering zero/NaN x-values.",
        )

    has_charge = "charge_capacity_ah" in cs.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    ax.plot(
        cs["x_val"], cs["discharge_capacity_ah"],
        marker="o", markersize=4, linestyle="-",
        color=WONG_PALETTE[5], label="Discharge",
    )
    if has_charge:
        ax.plot(
            cs["x_val"], cs["charge_capacity_ah"],
            marker="o", markersize=4, linestyle="--",
            markerfacecolor="none", color=WONG_PALETTE[6], label="Charge",
        )

    if use_c_rate and cs["x_val"].max() / cs["x_val"].min() > 10:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Rate Capability: Capacity vs. C-rate")
    ax.legend(fontsize=7, loc="best")

    return save_figure(fig, output_dir, "rate_capability", formats=_get_formats(config))


def plot_rate_voltage_profiles(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Voltage-capacity profiles at representative cycles covering different rates.

    Only generated if at least 2 distinct current levels are detected.  For each
    distinct mean |current_a| level (rounded to 2 sig. figs.) the cycle with the
    most data points is selected as the representative.

    Scientific note: These curves show how the voltage profile changes shape as
    the applied current increases--higher rates typically cause larger polarisation,
    shifting the average discharge voltage downward and narrowing the usable
    capacity window.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame with voltage_v, capacity_ah, and cycle_index.
    cycle_summary : pd.DataFrame
        One row per cycle (used to verify required columns are present).
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

    required_ts = ["voltage_v", "capacity_ah", "cycle_index"]
    missing = _check_columns(df, required_ts)
    if missing:
        logger.warning("plot_rate_voltage_profiles: missing df columns %s", missing)
        return make_placeholder(
            title="Voltage Profiles at Representative C-rates",
            missing_columns=missing,
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
        )

    has_current = "current_a" in df.columns
    if not has_current:
        return make_placeholder(
            title="Voltage Profiles at Representative C-rates",
            missing_columns=["current_a"],
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
            note="current_a is required to distinguish current levels.",
        )

    per_cycle = (
        df.groupby("cycle_index")
        .agg(
            mean_abs_i=("current_a", lambda s: s.abs().mean()),
            n_points=("voltage_v", "count"),
        )
        .reset_index()
    )
    per_cycle["i_rounded"] = per_cycle["mean_abs_i"].apply(lambda x: _sig_round(x, 2))
    per_cycle = per_cycle[per_cycle["i_rounded"] > 0]

    distinct_levels = sorted(per_cycle["i_rounded"].unique())
    if len(distinct_levels) < 2:
        return make_placeholder(
            title="Voltage Profiles at Representative C-rates",
            missing_columns=[],
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
            note=f"Only {len(distinct_levels)} distinct current level(s) found; need >= 2.",
        )

    nom_cap = getattr(config, "nominal_capacity_ah", None)
    use_c_rate = (nom_cap is not None and nom_cap > 0)

    rep_cycles = {}
    for lvl in distinct_levels:
        subset = per_cycle[per_cycle["i_rounded"] == lvl]
        best = subset.loc[subset["n_points"].idxmax(), "cycle_index"]
        rep_cycles[lvl] = int(best)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    for idx, (lvl, cyc) in enumerate(sorted(rep_cycles.items())):
        color = WONG_PALETTE[idx % len(WONG_PALETTE)]
        cyc_df = df[df["cycle_index"] == cyc].copy()
        if len(cyc_df) < 2:
            continue

        cyc_df = cyc_df.copy()
        cyc_df["cap_reset"] = cyc_df["capacity_ah"] - cyc_df["capacity_ah"].min()

        if use_c_rate:
            rate_label = f"{lvl / nom_cap:.2g}C"
        else:
            rate_label = f"{lvl:.3g} A"

        ax.plot(
            cyc_df["cap_reset"],
            cyc_df["voltage_v"],
            color=color,
            linewidth=0.9,
            label=rate_label,
        )

    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage Profiles at Representative C-rates")
    ax.legend(fontsize=7, loc="best", title="Rate" if use_c_rate else "|Current|")

    return save_figure(fig, output_dir, "rate_voltage_profiles", formats=_get_formats(config))
