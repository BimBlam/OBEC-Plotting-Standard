"""
batteryplot.summaries
=====================
High-level summary aggregation for the batteryplot pipeline.

This module collects diagnostics, dataset statistics, and per-plot-family
availability into structured objects that downstream reporters and the CLI
can consume directly.

The two main public functions are:

- :func:`build_full_summary` — overall dataset health report as a ``dict``.
- :func:`build_plot_availability` — per-plot-family availability as a
  ``pd.DataFrame``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from batteryplot.config import BatteryPlotConfig

logger = logging.getLogger("batteryplot")

# ---------------------------------------------------------------------------
# Per-plot-family column requirements
#
# Each entry: (plot_name, required_columns, optional_note)
# A plot is available if ALL required_columns are present in the DataFrame.
# ---------------------------------------------------------------------------

_PLOT_REQUIREMENTS: List[Tuple[str, str, List[str], str]] = [
    # family, plot_name, required_cols, note
    (
        "voltage_profiles",
        "Voltage vs. Capacity",
        ["voltage_v", "capacity_ah", "cycle_index", "segment"],
        "Requires at least one charge and one discharge cycle.",
    ),
    (
        "voltage_profiles",
        "Voltage vs. Time",
        ["voltage_v", "elapsed_time_s", "cycle_index"],
        "",
    ),
    (
        "voltage_profiles",
        "dQ/dV (Differential Capacity)",
        ["voltage_v", "capacity_ah", "cycle_index", "segment"],
        "Numerical derivative; requires sufficient data density.",
    ),
    (
        "cycle_summary",
        "Capacity vs. Cycle",
        ["cycle_index", "discharge_capacity_ah"],
        "",
    ),
    (
        "cycle_summary",
        "Coulombic Efficiency vs. Cycle",
        ["cycle_index", "coulombic_efficiency_pct"],
        "Requires both charge and discharge capacity per cycle.",
    ),
    (
        "cycle_summary",
        "Energy Efficiency vs. Cycle",
        ["cycle_index", "energy_efficiency_pct"],
        "Requires both charge and discharge energy per cycle.",
    ),
    (
        "cycle_summary",
        "Capacity Retention vs. Cycle",
        ["cycle_index", "capacity_retention_pct"],
        "Requires nominal_capacity_ah in config.",
    ),
    (
        "cycle_summary",
        "DCIR vs. Cycle",
        ["cycle_index", "mean_dcir_ohm"],
        "",
    ),
    (
        "rate_capability",
        "Rate Capability",
        ["cycle_index", "discharge_capacity_ah"],
        "Requires nominal_capacity_ah in config for C-rate labelling.",
    ),
    (
        "pulse_resistance",
        "Pulse DCIR vs. Cycle",
        ["cycle_index", "dcir_ohm_measured"],
        "Requires dcir_ohm column in raw data OR estimated from voltage steps.",
    ),
    (
        "pulse_resistance",
        "Pulse Voltage Response",
        ["cycle_index", "voltage_before", "voltage_after", "current_a"],
        "",
    ),
    (
        "ragone",
        "Ragone Plot",
        ["energy_axis", "power_axis"],
        "Uses gravimetric values if active_mass_g is set, otherwise absolute.",
    ),
    (
        "qa",
        "Data Coverage Heatmap",
        ["cycle_index", "elapsed_time_s"],
        "",
    ),
    (
        "qa",
        "Temperature vs. Cycle",
        ["cycle_index", "mean_temperature_c"],
        "",
    ),
    (
        "qa",
        "Impedance Trend",
        ["cycle_index", "mean_ac_impedance_ohm"],
        "",
    ),
]


# ---------------------------------------------------------------------------
# 1. Full summary dict
# ---------------------------------------------------------------------------


def build_full_summary(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config: BatteryPlotConfig,
) -> Dict[str, Any]:
    """
    Collect an overall health and statistics report for a single cell dataset.

    The returned dictionary is intended to be serialised (e.g. to JSON or
    YAML) or displayed in a console report.  All values are plain Python
    scalars or lists so that downstream code does not need to import
    pandas.

    Parameters
    ----------
    df:
        Canonical analysis DataFrame (from
        :func:`batteryplot.parsing.build_analysis_df`).
    cycle_summary:
        Per-cycle summary DataFrame (from
        :func:`batteryplot.transforms.compute_cycle_summary`).
    pulse_df:
        Pulse-event DataFrame (from
        :func:`batteryplot.transforms.detect_pulse_segments`).
    config:
        Pipeline configuration.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary with the following keys (all may be ``None`` if
        the underlying data is absent):

        ``n_data_points``
            Total number of rows in *df*.
        ``n_cycles``
            Number of unique cycles in *cycle_summary* (or *df* if summary
            is empty).
        ``available_canonical_columns``
            Sorted list of canonical column names present in *df*.
        ``first_cycle_discharge_cap_ah``
            Discharge capacity in the first cycle (Ah).
        ``last_cycle_discharge_cap_ah``
            Discharge capacity in the last cycle (Ah).
        ``capacity_fade_pct``
            ``(1 − Q_last / Q_first) × 100 %``.
        ``mean_coulombic_efficiency_pct``
            Mean Coulombic efficiency across all cycles.
        ``n_pulse_events``
            Number of detected pulse events.
        ``config_warnings``
            List of config-validation warnings (missing optional parameters).
        ``issues``
            List of data-quality issue strings detected during summarisation.
    """
    issues: List[str] = []
    summary: Dict[str, Any] = {}

    # --- Basic counts ---
    summary["n_data_points"] = int(len(df))

    if not cycle_summary.empty and "cycle_index" in cycle_summary.columns:
        summary["n_cycles"] = int(cycle_summary["cycle_index"].nunique())
    elif "cycle_index" in df.columns:
        summary["n_cycles"] = int(df["cycle_index"].nunique())
        issues.append(
            "cycle_summary DataFrame is empty; n_cycles derived from raw data."
        )
    else:
        summary["n_cycles"] = None
        issues.append("'cycle_index' column not found; cycle count unavailable.")

    # --- Available columns ---
    summary["available_canonical_columns"] = sorted(df.columns.tolist())

    # --- Capacity metrics ---
    if not cycle_summary.empty and "discharge_capacity_ah" in cycle_summary.columns:
        cap_col = pd.to_numeric(cycle_summary["discharge_capacity_ah"], errors="coerce")
        cap_valid = cap_col.dropna()

        first_cap = float(cap_valid.iloc[0]) if not cap_valid.empty else None
        last_cap = float(cap_valid.iloc[-1]) if not cap_valid.empty else None
        summary["first_cycle_discharge_cap_ah"] = first_cap
        summary["last_cycle_discharge_cap_ah"] = last_cap

        if first_cap and last_cap and first_cap > 0:
            fade = (1.0 - last_cap / first_cap) * 100.0
            summary["capacity_fade_pct"] = round(fade, 3)
        else:
            summary["capacity_fade_pct"] = None
    else:
        summary["first_cycle_discharge_cap_ah"] = None
        summary["last_cycle_discharge_cap_ah"] = None
        summary["capacity_fade_pct"] = None
        if not cycle_summary.empty:
            issues.append(
                "'discharge_capacity_ah' not in cycle_summary; "
                "capacity fade cannot be computed."
            )

    # --- Coulombic efficiency ---
    if (
        not cycle_summary.empty
        and "coulombic_efficiency_pct" in cycle_summary.columns
    ):
        ce = pd.to_numeric(
            cycle_summary["coulombic_efficiency_pct"], errors="coerce"
        )
        valid_ce = ce.dropna()
        summary["mean_coulombic_efficiency_pct"] = (
            round(float(valid_ce.mean()), 3) if not valid_ce.empty else None
        )
    else:
        summary["mean_coulombic_efficiency_pct"] = None

    # --- Pulse events ---
    summary["n_pulse_events"] = int(len(pulse_df)) if pulse_df is not None else 0

    # --- Config warnings ---
    from batteryplot.utils.validation import validate_config
    summary["config_warnings"] = validate_config(config)

    # --- Data quality checks ---
    if "voltage_v" in df.columns:
        v = pd.to_numeric(df["voltage_v"], errors="coerce")
        if v.notna().any():
            v_min, v_max = float(v.min()), float(v.max())
            if v_min < 0:
                issues.append(
                    f"Negative voltage values detected (min = {v_min:.4f} V). "
                    "Check for sensor offset or data corruption."
                )
            if v_max > 5.0:
                issues.append(
                    f"Unusually high voltage detected (max = {v_max:.4f} V > 5 V). "
                    "Verify cell chemistry and measurement range."
                )

    if "current_a" in df.columns:
        i = pd.to_numeric(df["current_a"], errors="coerce")
        if i.notna().any():
            i_max_abs = float(i.abs().max())
            if i_max_abs > 1000:
                issues.append(
                    f"Very large current detected (|I|_max = {i_max_abs:.1f} A). "
                    "Confirm units are amperes."
                )

    if not issues:
        issues.append("No data quality issues detected.")

    summary["issues"] = issues

    logger.info(
        "Full summary: %d data points, %d cycles, %d pulse events.",
        summary["n_data_points"],
        summary.get("n_cycles") or 0,
        summary["n_pulse_events"],
    )
    return summary


# ---------------------------------------------------------------------------
# 2. Plot availability
# ---------------------------------------------------------------------------


def build_plot_availability(
    df: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    pulse_df: pd.DataFrame,
    config: BatteryPlotConfig,
) -> pd.DataFrame:
    """
    Determine which plots can be generated given the available data columns.

    For each plot listed in the internal requirements table the function
    checks:

    - Whether the plot's **family** is in ``config.selected_plot_families``.
    - Whether all **required columns** are present in the relevant DataFrame
      (main *df* for raw-data plots, *cycle_summary* for cycle-level plots,
      *pulse_df* for pulse plots, a special combined frame for Ragone).

    Parameters
    ----------
    df:
        Canonical analysis DataFrame.
    cycle_summary:
        Per-cycle summary DataFrame.
    pulse_df:
        Pulse-event DataFrame.
    config:
        Pipeline configuration; ``selected_plot_families`` controls which
        families are requested.

    Returns
    -------
    pd.DataFrame
        One row per plot.  Columns:

        - ``plot_family``: one of the six families.
        - ``plot_name``: human-readable plot name.
        - ``available``: bool — True iff the family is selected *and* all
          required columns are present.
        - ``missing_columns``: list of missing column names (empty if
          available).
        - ``note``: descriptive note from the requirements table.
        - ``family_selected``: bool — True iff the family appears in
          ``config.selected_plot_families``.
    """
    selected_families = set(config.selected_plot_families)

    # Build a unified "merged" view for Ragone: needs columns from both
    # cycle_summary and possibly ragone_df (treated as cycle_summary here)
    ragone_df_cols = set(cycle_summary.columns) if not cycle_summary.empty else set()
    # Ragone plot availability is checked against the cycle_summary frame
    # after ragone points are computed — we check for the post-transform cols.
    # Since ragone_df is not passed in, we check the pre-conditions in cycle_summary.

    # Decide which dataframe to check per family
    def _source_df(family: str) -> pd.DataFrame:
        if family in ("pulse_resistance",):
            return pulse_df if pulse_df is not None else pd.DataFrame()
        if family in ("ragone",):
            # Ragone is computed from cycle_summary; check its precursor columns
            return cycle_summary if not cycle_summary.empty else pd.DataFrame()
        if family in ("cycle_summary", "rate_capability", "qa"):
            return cycle_summary if not cycle_summary.empty else df
        return df

    rows = []
    for family, plot_name, req_cols, note in _PLOT_REQUIREMENTS:
        family_selected = family in selected_families
        source = _source_df(family)
        present_cols = set(source.columns) if source is not None else set()

        # For qa plots also check in main df
        if family == "qa":
            present_cols = present_cols | set(df.columns)

        missing = [c for c in req_cols if c not in present_cols]
        available = family_selected and len(missing) == 0

        rows.append(
            {
                "plot_family": family,
                "plot_name": plot_name,
                "available": available,
                "missing_columns": missing,
                "note": note,
                "family_selected": family_selected,
            }
        )

    result = pd.DataFrame(rows)
    n_avail = result["available"].sum()
    logger.info(
        "Plot availability: %d / %d plots available.",
        n_avail,
        len(result),
    )
    return result
