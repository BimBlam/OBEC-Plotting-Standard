"""
Energy-power (Ragone) plot for the batteryplot package.

Scientific assumptions
----------------------
True Ragone analysis requires constant-power discharge data, which cycler
data rarely provides directly.  Here we estimate:

  Discharge energy  E_d  from discharge_energy_wh column per cycle [Wh]
  Discharge time    T_d  = discharge_capacity_ah / mean_discharge_|I|  [h]
  Discharge power   P_d  = E_d / T_d  [W]

Normalisation basis
~~~~~~~~~~~~~~~~~~~
- If config.active_mass_g is set (> 0): gravimetric basis [Wh/g and W/g].
- If config.active_mass_g AND config.density_g_cm3 are both set:
  volumetric basis [Wh/L and W/L] is also shown.
- Otherwise: absolute basis [Wh and W].

Minimum data requirement
~~~~~~~~~~~~~~~~~~~~~~~~
The plot is NOT generated if fewer than 3 cycles have valid (non-NaN, > 0)
discharge_energy_wh and discharge_capacity_ah values.
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

logger = logging.getLogger("batteryplot.plots.ragone")


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _get_formats(config) -> tuple:
    return tuple(getattr(config, "output_formats", ("svg", "pdf")))


def _get_theme(config) -> str:
    return getattr(config, "theme", "publication")


def _mean_discharge_i_per_cycle(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or "cycle_index" not in df.columns or "current_a" not in df.columns:
        return None
    dchg = df[df["current_a"] < 0].copy()
    if dchg.empty:
        return None
    result = dchg.groupby("cycle_index")["current_a"].apply(lambda s: s.abs().mean())
    return result if not result.empty else None


def plot_ragone(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config,
    output_dir: Path,
) -> List[Path]:
    """
    Energy-power (Ragone) plot.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries DataFrame.  Used to compute mean discharge current per cycle.
    cycle_summary : pd.DataFrame
        One row per cycle.  Must contain discharge_energy_wh and
        discharge_capacity_ah.
    pulse_df : pd.DataFrame
        Not used; included for uniform signature.
    config : BatteryPlotConfig
        Relevant attributes: active_mass_g, density_g_cm3, output_formats, theme.
    output_dir : Path

    Returns
    -------
    list of Path
    """
    apply_style(_get_theme(config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cs = ["discharge_energy_wh", "discharge_capacity_ah"]
    missing = _check_columns(cycle_summary, required_cs)
    if missing:
        logger.warning("plot_ragone: missing cycle_summary columns %s", missing)
        return make_placeholder(
            title="Energy\u2013Power (Ragone) Plot",
            missing_columns=missing,
            output_dir=output_dir,
            stem="ragone",
            formats=_get_formats(config),
        )

    cs = cycle_summary.copy()
    cs = cs.dropna(subset=["discharge_energy_wh", "discharge_capacity_ah"])
    cs = cs[(cs["discharge_energy_wh"] > 0) & (cs["discharge_capacity_ah"] > 0)]

    if len(cs) < 3:
        return make_placeholder(
            title="Energy\u2013Power (Ragone) Plot",
            missing_columns=[],
            output_dir=output_dir,
            stem="ragone",
            formats=_get_formats(config),
            note=f"Only {len(cs)} cycle(s) with valid energy + capacity data (need >= 3).",
        )

    mean_i = _mean_discharge_i_per_cycle(df)
    used_heuristic = False

    if mean_i is not None and "cycle_index" in cs.columns:
        cs = cs.join(mean_i.rename("mean_dchg_i"), on="cycle_index", how="left")
        valid = cs["mean_dchg_i"].notna() & (cs["mean_dchg_i"] > 0)
        if valid.any():
            cs.loc[valid, "t_discharge_h"] = (
                cs.loc[valid, "discharge_capacity_ah"] / cs.loc[valid, "mean_dchg_i"]
            )
        if (~valid).any():
            avg_v = (cs.loc[valid, "discharge_energy_wh"] / cs.loc[valid, "discharge_capacity_ah"]).mean()
            avg_v_global = avg_v if not np.isnan(avg_v) else 3.7
            cs.loc[~valid, "t_discharge_h"] = (
                cs.loc[~valid, "discharge_capacity_ah"] ** 2
                / cs.loc[~valid, "discharge_energy_wh"]
            )
            used_heuristic = True
    else:
        cs["t_discharge_h"] = cs["discharge_capacity_ah"] ** 2 / cs["discharge_energy_wh"]
        used_heuristic = True

    cs["power_w"] = cs["discharge_energy_wh"] / cs["t_discharge_h"]
    cs = cs.dropna(subset=["power_w"])
    cs = cs[cs["power_w"] > 0]

    if len(cs) < 3:
        return make_placeholder(
            title="Energy\u2013Power (Ragone) Plot",
            missing_columns=[],
            output_dir=output_dir,
            stem="ragone",
            formats=_get_formats(config),
            note="Fewer than 3 cycles remained after computing power. Check current data.",
        )

    active_mass_g = getattr(config, "active_mass_g", None)
    density_g_cm3 = getattr(config, "density_g_cm3", None)

    if active_mass_g and active_mass_g > 0:
        energy_col = cs["discharge_energy_wh"] / active_mass_g
        power_col = cs["power_w"] / active_mass_g
        x_label = "Power (W g\u207b\u00b9)"
        y_label = "Energy (Wh g\u207b\u00b9)"
        basis_note = "Gravimetric basis"
    else:
        energy_col = cs["discharge_energy_wh"]
        power_col = cs["power_w"]
        x_label = "Power (W)"
        y_label = "Energy (Wh)"
        basis_note = "Absolute basis (no active mass provided)"

    has_cycle = "cycle_index" in cs.columns

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, DEFAULT_HEIGHT_IN))

    if has_cycle:
        sc = ax.scatter(
            power_col, energy_col,
            c=cs["cycle_index"], cmap="plasma",
            s=18, alpha=0.85, zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Cycle Index", fontsize=7)
    else:
        ax.scatter(power_col, energy_col, color=WONG_PALETTE[5], s=18, alpha=0.85, zorder=3)

    if has_cycle:
        cs_sorted = cs.sort_values("cycle_index")
        ax.plot(
            power_col.loc[cs_sorted.index],
            energy_col.loc[cs_sorted.index],
            color="gray", linewidth=0.4, alpha=0.4, zorder=2,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Energy\u2013Power (Ragone) Plot")

    x_range = power_col.max() / (power_col.min() + 1e-30)
    y_range = energy_col.max() / (energy_col.min() + 1e-30)
    if x_range > 5:
        ax.set_xscale("log")
    if y_range > 5:
        ax.set_yscale("log")

    note_parts = [basis_note, "P estimated as E/T_d (approximation)"]
    if used_heuristic:
        note_parts.append("Current data unavailable; T_d estimated from Q\u00b2/E")
    ax.text(
        0.98, 0.98,
        "\n".join(note_parts),
        transform=ax.transAxes, fontsize=5.5, color="#888888",
        ha="right", va="top", style="italic",
    )

    if active_mass_g and active_mass_g > 0 and density_g_cm3 and density_g_cm3 > 0:
        vol_l = active_mass_g / density_g_cm3 * 1e-3
        energy_vol = cs["discharge_energy_wh"] / vol_l
        power_vol = cs["power_w"] / vol_l
        ax2 = ax.twiny()
        ax2.scatter(power_vol, energy_col, color=WONG_PALETTE[1], s=8,
                    alpha=0.5, marker="s", zorder=2, label="Volumetric")
        ax2.set_xlabel("Power (W L\u207b\u00b9)", color=WONG_PALETTE[1], fontsize=7)
        ax2.tick_params(axis="x", colors=WONG_PALETTE[1])
        ax2.spines["top"].set_visible(True)
        ax2.spines["top"].set_edgecolor(WONG_PALETTE[1])
        ax.text(0.02, 0.98, "Squares: volumetric basis",
                transform=ax.transAxes, fontsize=5.5, color=WONG_PALETTE[1],
                ha="left", va="top")

    return save_figure(fig, output_dir, "ragone", formats=_get_formats(config))
