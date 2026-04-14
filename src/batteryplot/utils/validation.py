"""
batteryplot.utils.validation
=============================
Light-weight validation helpers for DataFrames and configuration objects.

These functions produce *warnings* (strings) rather than raising exceptions,
so the pipeline can continue gracefully when optional data or parameters are
absent.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import pandas as pd

logger = logging.getLogger("batteryplot")


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str = "",
) -> Tuple[bool, List[str]]:
    """
    Check that a DataFrame contains every column in *required_cols*.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required_cols:
        Canonical column names that must be present.
    context:
        Optional descriptive label used in log messages (e.g.
        ``"cycle_summary"`` or ``"voltage_profile"``).

    Returns
    -------
    Tuple[bool, List[str]]
        ``(ok, missing_cols)`` where *ok* is ``True`` iff no columns are
        missing and *missing_cols* is the list of absent column names.

    Examples
    --------
    >>> ok, missing = validate_dataframe(df, ["voltage_v", "current_a"])
    >>> if not ok:
    ...     print(f"Cannot produce plot; missing: {missing}")
    """
    prefix = f"[{context}] " if context else ""
    missing = [col for col in required_cols if col not in df.columns]
    ok = len(missing) == 0
    if not ok:
        logger.debug(
            "%sMissing required column(s): %s", prefix, missing
        )
    return ok, missing


def validate_config(config: Any) -> List[str]:
    """
    Check a ``BatteryPlotConfig`` for common omissions that limit analysis.

    Returns a list of human-readable warning strings.  An empty list means
    no issues were detected.  Warnings are informational: the pipeline will
    still run but certain calculations or plots will be skipped.

    Parameters
    ----------
    config:
        A ``BatteryPlotConfig`` instance (accepted as ``Any`` to avoid
        circular imports).

    Returns
    -------
    List[str]
        Zero or more warning messages describing missing optional parameters.
    """
    warnings: List[str] = []

    if getattr(config, "nominal_capacity_ah", None) is None:
        warnings.append(
            "nominal_capacity_ah not set; C-rate will not be computed and "
            "capacity-retention plots will use absolute capacity (Ah)."
        )

    if getattr(config, "active_mass_g", None) is None:
        warnings.append(
            "active_mass_g not set; gravimetric specific-capacity (mAh g⁻¹) "
            "and Ragone plots will use absolute (non-normalised) values."
        )

    if getattr(config, "electrode_area_cm2", None) is None:
        warnings.append(
            "electrode_area_cm2 not set; areal capacity (mAh cm⁻²) "
            "normalisation will be skipped."
        )

    if getattr(config, "density_g_cm3", None) is None:
        warnings.append(
            "density_g_cm3 not set; volumetric capacity normalisation "
            "will be skipped."
        )

    selected = getattr(config, "selected_plot_families", [])
    if not selected:
        warnings.append(
            "selected_plot_families is empty; no plots will be generated."
        )

    for warning in warnings:
        logger.warning(warning)

    return warnings
