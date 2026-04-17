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
    add_assumption_warning,
)
from batteryplot.placeholders import (
    make_placeholder,
    diagnose_columns,
    ColumnDiagnostic,
)

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


def _format_current(current_a: float) -> str:
    """Format a current value with appropriate unit (A, mA, or uA) and 3 sig figs."""
    abs_i = abs(current_a)
    if abs_i == 0:
        return "0 A"
    if abs_i >= 0.1:
        return f"{abs_i:.3g} A"
    ma = abs_i * 1e3
    if ma >= 0.1:
        return f"{ma:.3g} mA"
    return f"{abs_i * 1e6:.3g} \u00b5A"


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
        diag = diagnose_columns(cycle_summary, required_cs,
                                optional=["charge_capacity_ah", "c_rate"])
        diag.note = ("C-rate axis requires nominal_capacity_ah in config. "
                     "Without it, |current| (A) is used as x-axis.")
        return make_placeholder(
            title="Rate Capability: Capacity vs. C-rate",
            missing_columns=missing,
            output_dir=output_dir,
            stem="rate_capability",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    # Filter to rate_test region if available
    cs = cycle_summary.sort_values("cycle_index").copy()
    if "test_region" in cs.columns:
        rate_cs = cs[cs["test_region"] == "rate_test"]
        if not rate_cs.empty:
            logger.info("plot_rate_capability: using %d rate_test cycles (of %d total).", len(rate_cs), len(cs))
            cs = rate_cs
        else:
            logger.info("plot_rate_capability: no rate_test cycles found; using all %d cycles.", len(cs))

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

    _rc_warnings: list[str] = []
    if getattr(config, "nominal_capacity_ah", None) is None:
        _rc_warnings.append(
            "nominal_capacity_ah not set: x-axis shows |current| (A), not C-rate"
        )
    add_assumption_warning(fig, _rc_warnings)
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
        diag = diagnose_columns(df, required_ts, optional=["current_a"])
        diag.note = ("Requires at least 2 distinct current magnitudes across "
                     "cycles to show rate-dependent voltage profiles.")
        return make_placeholder(
            title="Voltage vs. Capacity by Rate",
            missing_columns=missing,
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
            diagnostic=diag,
        )

    has_current = "current_a" in df.columns
    if not has_current:
        return make_placeholder(
            title="Voltage vs. Capacity by Rate",
            missing_columns=["current_a"],
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
            note="current_a is required to distinguish current levels.",
        )

    # Filter to rate_test cycles if test_region column is available
    plot_df = df
    if "test_region" in df.columns:
        rate_df = df[df["test_region"] == "rate_test"]
        if not rate_df.empty:
            logger.info("plot_rate_voltage_profiles: using %d rate_test rows.", len(rate_df))
            plot_df = rate_df

    per_cycle = (
        plot_df.groupby("cycle_index")
        .agg(
            mean_abs_i=("current_a", lambda s: s.abs().mean()),
            n_points=("voltage_v", "count"),
        )
        .reset_index()
    )
    # Round to 1 sig fig for grouping (to merge near-identical rates that differ
    # only due to float noise, e.g. 0.00097 A vs 0.00099 A both → 0.001 A).
    # Keep the full 2-sig-fig mean for the axis label (computed below per group).
    per_cycle["i_group"] = per_cycle["mean_abs_i"].apply(lambda x: _sig_round(x, 1))
    per_cycle = per_cycle[per_cycle["i_group"] > 0]

    distinct_levels = sorted(per_cycle["i_group"].unique())
    if len(distinct_levels) < 2:
        return make_placeholder(
            title="Voltage vs. Capacity by Rate",
            missing_columns=[],
            output_dir=output_dir,
            stem="rate_voltage_profiles",
            formats=_get_formats(config),
            note=f"Only {len(distinct_levels)} distinct current level(s) found; need >= 2.",
        )

    nom_cap = getattr(config, "nominal_capacity_ah", None)
    use_c_rate = (nom_cap is not None and nom_cap > 0)

    # For each 1-sig-fig group: pick the representative cycle (most data points)
    # and compute the 2-sig-fig mean |current| for the label.
    rep_cycles: dict = {}   # group_value -> (cycle_id, label_current_A)
    for lvl in distinct_levels:
        subset = per_cycle[per_cycle["i_group"] == lvl]
        best_row = subset.loc[subset["n_points"].idxmax()]
        best_cyc = int(best_row["cycle_index"])
        # Use 2-sig-fig mean of all cycles in this group for a stable label
        mean_i_label = _sig_round(subset["mean_abs_i"].mean(), 2)
        rep_cycles[lvl] = (best_cyc, mean_i_label)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    legend_handles: list = []
    legend_labels: list = []

    for idx, (lvl, (cyc, label_i)) in enumerate(sorted(rep_cycles.items())):
        color = WONG_PALETTE[idx % len(WONG_PALETTE)]
        cyc_df = plot_df[plot_df["cycle_index"] == cyc].copy()
        if len(cyc_df) < 2:
            continue

        if use_c_rate:
            rate_label = f"{label_i / nom_cap:.2g}C"
        else:
            rate_label = _format_current(label_i)

        # Split into charge and discharge arcs so each is plotted independently.
        # This prevents the artefact where the end of the charge arc is connected
        # directly to the start of the discharge arc (both accumulators reset to 0
        # at the start of each step in Maccor/Arbin exports).
        plotted = False
        if "segment" in cyc_df.columns:
            chg = cyc_df[cyc_df["segment"] == "charge"]
            dis = cyc_df[cyc_df["segment"] == "discharge"]
        else:
            # Fallback: infer segment from current sign
            if "current_a" in cyc_df.columns:
                cur = pd.to_numeric(cyc_df["current_a"], errors="coerce")
                chg = cyc_df[cur > 0]
                dis = cyc_df[cur < 0]
            else:
                chg = pd.DataFrame()
                dis = cyc_df  # plot everything as discharge

        # --- Discharge arc (solid line) ---
        if len(dis) >= 2:
            dis_cap = pd.to_numeric(dis["capacity_ah"], errors="coerce")
            dis_v   = pd.to_numeric(dis["voltage_v"],   errors="coerce")
            # Reset accumulator: starts at 0, increases monotonically.
            # Use absolute value — Maccor may record as positive even on discharge.
            cap_reset = dis_cap.abs() - dis_cap.abs().min()
            valid = dis_v.notna() & cap_reset.notna()
            if valid.sum() >= 2:
                line, = ax.plot(
                    cap_reset[valid], dis_v[valid],
                    color=color, linewidth=0.9, linestyle="-",
                    label=rate_label,
                )
                legend_handles.append(line)
                legend_labels.append(rate_label)
                plotted = True

        # --- Charge arc (dashed line, same colour, no extra legend entry) ---
        if len(chg) >= 2:
            chg_cap = pd.to_numeric(chg["capacity_ah"], errors="coerce")
            chg_v   = pd.to_numeric(chg["voltage_v"],   errors="coerce")
            cap_reset = chg_cap.abs() - chg_cap.abs().min()
            valid = chg_v.notna() & cap_reset.notna()
            if valid.sum() >= 2:
                if not plotted:
                    # If discharge was absent, use charge as the legend entry
                    line, = ax.plot(
                        cap_reset[valid], chg_v[valid],
                        color=color, linewidth=0.9, linestyle="--",
                        label=rate_label,
                    )
                    legend_handles.append(line)
                    legend_labels.append(rate_label)
                else:
                    ax.plot(
                        cap_reset[valid], chg_v[valid],
                        color=color, linewidth=0.9, linestyle="--",
                        alpha=0.55,
                    )

    ax.set_xlabel("Capacity (Ah)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs. Capacity by Rate")

    # Build legend with rate labels only (one entry per rate, from discharge arc).
    # Add a style note for charge vs discharge if any charge arcs were plotted.
    rate_title = "Rate"
    if legend_handles:
        ax.legend(legend_handles, legend_labels,
                  fontsize=7, loc="best", title=rate_title)
    else:
        ax.legend(fontsize=7, loc="best", title=rate_title)

    # Solid = discharge, dashed = charge
    add_assumption_warning(fig, ["Solid lines: discharge arcs. Dashed lines (faded): charge arcs."])

    return save_figure(fig, output_dir, "rate_voltage_profiles", formats=_get_formats(config))
