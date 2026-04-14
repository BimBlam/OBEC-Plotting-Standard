"""
batteryplot.transforms
======================
Data-transformation and derived-metrics layer for the batteryplot pipeline.

All functions operate on DataFrames whose columns use the canonical naming
convention defined in :mod:`batteryplot.aliases` (e.g. ``current_a``,
``voltage_v``).  Input DataFrames are **not** mutated; copies or new
DataFrames are returned instead.

Scientific notes
----------------
*Charge / discharge labelling* relies on the sign of the measured current.
By convention (IEC 61960):

- Positive current  → charging the cell (current flows into the cell).
- Negative current  → discharging the cell (current flows out).

When a ``step_type`` column is present in the Arbin data its coded values
take precedence over the current-sign heuristic because the cycler firmware
tracks the intended step mode unambiguously.

*C-rate* is defined as ``|I| / C_n`` where ``C_n`` is the nominal
(1 C) capacity in ampere-hours.  A C-rate of 1 C means the cell is
charged or discharged in one hour at constant current.

*Specific capacity* is expressed in mAh g⁻¹ to match the electrochemical
literature convention: ``Q_sp = |Q| × 1000 / m_active``.

*Pulse resistance* (DCIR) can be measured directly by the cycler instrument
or estimated from a voltage step: ``R = ΔV / ΔI``.  The estimate is less
reliable than the instrument measurement and is flagged accordingly.

*Ragone plot* points relate energy density to power density.  Without
active-mass data the absolute values (Wh and W) are used; with mass the
gravimetric values (Wh g⁻¹ and W g⁻¹) are reported.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from batteryplot.config import BatteryPlotConfig

logger = logging.getLogger("batteryplot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default threshold for identifying rest segments (A).
#: Current magnitudes below this value are treated as rest (open-circuit).
CURRENT_REST_THRESHOLD_A: float = 1e-4

#: Maximum step duration for a segment to qualify as a *pulse* (seconds).
PULSE_MAX_DURATION_S: float = 200.0

#: Minimum current magnitude to flag a segment as a pulse (A).
PULSE_MIN_CURRENT_A: float = 0.01

# Arbin-style step-type codes (vary by firmware version)
_CHARGE_CODES = {"c", "cc", "chg", "charge", "cccv", "cv", "1", "2", "3"}
_DISCHARGE_CODES = {"d", "dc", "dis", "discharge", "dcv", "4", "5", "6"}
_REST_CODES = {"r", "rest", "ocp", "ocv", "0"}


# ---------------------------------------------------------------------------
# 1. Charge / discharge segment labelling
# ---------------------------------------------------------------------------


def label_charge_discharge(
    df: pd.DataFrame,
    current_threshold_a: float = CURRENT_REST_THRESHOLD_A,
) -> pd.DataFrame:
    """
    Add a ``segment`` column to *df* classifying each row as one of:
    ``"charge"``, ``"discharge"``, ``"rest"``, or ``"unknown"``.

    Labelling strategy
    ------------------
    1. If ``step_type`` is present **and** its values map to known charge /
       discharge / rest codes, those codes are used directly.
    2. Otherwise, the sign of ``current_a`` is used:

       - ``current_a > +threshold``  → ``"charge"``
       - ``current_a < -threshold``  → ``"discharge"``
       - ``|current_a| ≤ threshold`` → ``"rest"``
       - ``current_a`` is NaN        → ``"unknown"``

    Parameters
    ----------
    df:
        Canonical DataFrame from :func:`batteryplot.parsing.build_analysis_df`.
    current_threshold_a:
        Magnitude threshold below which current is considered zero / rest.
        Defaults to 100 µA (1 × 10⁻⁴ A) to handle noise at rest.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``segment`` column.
    """
    df = df.copy()

    # Initialise to "unknown"
    segment = pd.Series("unknown", index=df.index, dtype=object)

    # Try step_type column first
    if "step_type" in df.columns:
        st = df["step_type"].fillna("").astype(str).str.strip().str.lower()
        charge_mask = st.isin(_CHARGE_CODES)
        discharge_mask = st.isin(_DISCHARGE_CODES)
        rest_mask = st.isin(_REST_CODES)

        n_from_step = (charge_mask | discharge_mask | rest_mask).sum()
        if n_from_step > 0:
            logger.info(
                "Segment labelling: using step_type column for %d / %d rows.",
                n_from_step,
                len(df),
            )
            segment[charge_mask] = "charge"
            segment[discharge_mask] = "discharge"
            segment[rest_mask] = "rest"

            # Rows not covered by step_type codes fall through to current heuristic
            uncovered = ~(charge_mask | discharge_mask | rest_mask)
        else:
            logger.debug(
                "step_type column present but no recognised codes found; "
                "falling back to current-sign heuristic."
            )
            uncovered = pd.Series(True, index=df.index)
    else:
        uncovered = pd.Series(True, index=df.index)

    # Current-sign heuristic for uncovered rows
    if "current_a" in df.columns and uncovered.any():
        i = pd.to_numeric(df.loc[uncovered, "current_a"], errors="coerce")
        segment.loc[uncovered & i.notna() & (i > current_threshold_a)] = "charge"
        segment.loc[uncovered & i.notna() & (i < -current_threshold_a)] = "discharge"
        segment.loc[uncovered & i.notna() & (i.abs() <= current_threshold_a)] = "rest"
        logger.debug(
            "Segment labelling: current-sign heuristic applied to %d rows "
            "(threshold = %.2e A).",
            uncovered.sum(),
            current_threshold_a,
        )
    elif "current_a" not in df.columns:
        logger.warning(
            "Segment labelling: 'current_a' column not found; "
            "all uncovered rows labelled 'unknown'."
        )

    counts = segment.value_counts().to_dict()
    logger.info("Segment counts: %s", counts)

    df["segment"] = segment
    return df


# ---------------------------------------------------------------------------
# 2. Cycle summary
# ---------------------------------------------------------------------------


def compute_cycle_summary(
    df: pd.DataFrame,
    config: BatteryPlotConfig,
) -> pd.DataFrame:
    """
    Compute per-cycle electrochemical summary statistics.

    For each unique value in ``cycle_index`` the following quantities are
    extracted or calculated:

    Capacity
        *Preferred*: dedicated waveform columns ``charge_capacity_ah`` /
        ``discharge_capacity_ah`` (pre-integrated by the Arbin firmware).
        *Fallback*: the maximum value of ``capacity_ah`` within the charge
        or discharge segment of that cycle.  The maximum is used because
        Arbin resets the running ``capacity_ah`` accumulator at the start of
        each step, so the peak value equals the total accumulated capacity.

    Energy
        Analogous logic with ``charge_energy_wh`` / ``discharge_energy_wh``
        and ``energy_wh``.

    Coulombic efficiency
        ``η_CE = Q_dis / Q_chg × 100 %``

    Energy efficiency
        ``η_EE = E_dis / E_chg × 100 %``

    Capacity retention
        ``CR = Q_dis / Q_nominal × 100 %`` (requires ``nominal_capacity_ah``
        in *config*).

    Parameters
    ----------
    df:
        Canonical DataFrame with a ``segment`` column (added by
        :func:`label_charge_discharge`).
    config:
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        One row per cycle.  The ``cycle_index`` column is the group key.
        All other columns are ``NaN`` when insufficient data are available.
    """
    if "cycle_index" not in df.columns:
        logger.warning("compute_cycle_summary: 'cycle_index' not found; returning empty DataFrame.")
        return pd.DataFrame()

    if "segment" not in df.columns:
        logger.info("compute_cycle_summary: 'segment' not found; calling label_charge_discharge.")
        df = label_charge_discharge(df)

    rows = []
    for cycle_id, cycle_df in df.groupby("cycle_index", sort=True):
        chg = cycle_df[cycle_df["segment"] == "charge"]
        dis = cycle_df[cycle_df["segment"] == "discharge"]

        row: dict = {"cycle_index": cycle_id}

        # --- Charge capacity ---
        row["charge_capacity_ah"] = _extract_capacity(
            cycle_df, chg, "charge_capacity_ah", "capacity_ah"
        )
        # --- Discharge capacity ---
        row["discharge_capacity_ah"] = _extract_capacity(
            cycle_df, dis, "discharge_capacity_ah", "capacity_ah"
        )

        # --- Charge energy ---
        row["charge_energy_wh"] = _extract_capacity(
            cycle_df, chg, "charge_energy_wh", "energy_wh"
        )
        # --- Discharge energy ---
        row["discharge_energy_wh"] = _extract_capacity(
            cycle_df, dis, "discharge_energy_wh", "energy_wh"
        )

        # --- Efficiencies ---
        q_chg = row["charge_capacity_ah"]
        q_dis = row["discharge_capacity_ah"]
        e_chg = row["charge_energy_wh"]
        e_dis = row["discharge_energy_wh"]

        if _both_positive(q_chg, q_dis):
            row["coulombic_efficiency_pct"] = q_dis / q_chg * 100.0
        else:
            row["coulombic_efficiency_pct"] = float("nan")

        if _both_positive(e_chg, e_dis):
            row["energy_efficiency_pct"] = e_dis / e_chg * 100.0
        else:
            row["energy_efficiency_pct"] = float("nan")

        # --- Impedance / resistance means ---
        for col, out in [
            ("dcir_ohm", "mean_dcir_ohm"),
            ("ac_impedance_ohm", "mean_ac_impedance_ohm"),
            ("temperature_c", "mean_temperature_c"),
        ]:
            if col in cycle_df.columns:
                numeric = pd.to_numeric(cycle_df[col], errors="coerce")
                row[out] = numeric.mean() if numeric.notna().any() else float("nan")
            else:
                row[out] = float("nan")

        # --- Capacity retention ---
        nom_cap = config.nominal_capacity_ah
        if nom_cap and not pd.isna(q_dis):
            row["capacity_retention_pct"] = q_dis / nom_cap * 100.0
        else:
            row["capacity_retention_pct"] = float("nan")

        rows.append(row)

    summary = pd.DataFrame(rows)
    logger.info(
        "Cycle summary computed: %d cycles, columns: %s",
        len(summary),
        list(summary.columns),
    )
    return summary


# ---------------------------------------------------------------------------
# 3. C-rate
# ---------------------------------------------------------------------------


def compute_crate(
    df: pd.DataFrame,
    nominal_capacity_ah: float,
) -> pd.Series:
    """
    Compute the C-rate for every row in *df*.

    C-rate is defined as ``|I| / C_n`` where *C_n* is the nominal 1-hour
    discharge capacity of the cell.  A C-rate of 1 C means the cell is fully
    discharged (or charged) in exactly one hour at constant current.

    Parameters
    ----------
    df:
        Canonical DataFrame containing ``current_a``.
    nominal_capacity_ah:
        Nominal cell capacity in ampere-hours (> 0).

    Returns
    -------
    pd.Series
        C-rate series aligned to *df*.index.  ``NaN`` where ``current_a`` is
        missing or non-numeric.

    Raises
    ------
    ValueError
        If *nominal_capacity_ah* ≤ 0.
    """
    if nominal_capacity_ah <= 0:
        raise ValueError(
            f"nominal_capacity_ah must be positive; got {nominal_capacity_ah}"
        )
    if "current_a" not in df.columns:
        logger.warning("compute_crate: 'current_a' not found; returning NaN series.")
        return pd.Series(float("nan"), index=df.index, name="crate_c")

    current = pd.to_numeric(df["current_a"], errors="coerce")
    crate = current.abs() / nominal_capacity_ah
    crate.name = "crate_c"
    return crate


# ---------------------------------------------------------------------------
# 4. Specific capacity
# ---------------------------------------------------------------------------


def compute_specific_capacity(
    df: pd.DataFrame,
    active_mass_g: float,
) -> pd.Series:
    """
    Convert absolute capacity (Ah) to gravimetric specific capacity (mAh g⁻¹).

    Specific capacity is the standard figure of merit for electrode materials
    in the electrochemical literature.  It is calculated as::

        Q_sp [mAh g⁻¹] = |Q [Ah]| × 1000 / m_active [g]

    Parameters
    ----------
    df:
        Canonical DataFrame containing ``capacity_ah``.
    active_mass_g:
        Mass of the active electrode material in grams (> 0).

    Returns
    -------
    pd.Series
        Specific capacity in mAh g⁻¹, aligned to *df*.index.

    Raises
    ------
    ValueError
        If *active_mass_g* ≤ 0.
    """
    if active_mass_g <= 0:
        raise ValueError(f"active_mass_g must be positive; got {active_mass_g}")
    if "capacity_ah" not in df.columns:
        logger.warning(
            "compute_specific_capacity: 'capacity_ah' not found; returning NaN series."
        )
        return pd.Series(float("nan"), index=df.index, name="specific_capacity_mah_g")

    cap = pd.to_numeric(df["capacity_ah"], errors="coerce")
    sp = cap.abs() * 1000.0 / active_mass_g
    sp.name = "specific_capacity_mah_g"
    return sp


# ---------------------------------------------------------------------------
# 5. Pulse detection
# ---------------------------------------------------------------------------


def detect_pulse_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify short constant-current *pulse* segments in the dataset.

    A pulse is operationally defined as a step that satisfies **all** of:

    1. ``step_time_s < PULSE_MAX_DURATION_S`` (default 200 s) — the step is
       short relative to typical charge/discharge steps (which last minutes to
       hours).
    2. ``|current_a| > PULSE_MIN_CURRENT_A`` (default 10 mA) — a non-zero
       current is applied.
    3. The step is immediately followed by a rest step (``segment == "rest"``).

    Resistance estimation
    ---------------------
    If the ``dcir_ohm`` column is present, that measurement is used directly
    (labelled ``"measured_dcir"``).

    Otherwise, DCIR is estimated from the immediate voltage step at pulse onset:

    .. math::

        R_{est} = \\frac{\\Delta V}{\\Delta I}

    where ``ΔV`` is the voltage difference between the last rest row before
    the pulse and the first row of the pulse, and ``ΔI`` is the corresponding
    change in current.  This estimate is labelled ``"estimated_dcir"`` and
    is inherently less reliable than an instrument measurement.

    Parameters
    ----------
    df:
        Canonical DataFrame.  Must have had :func:`label_charge_discharge`
        called (so ``segment`` column exists).  Requires at minimum
        ``current_a``, ``step_time_s``, ``voltage_v``.

    Returns
    -------
    pd.DataFrame
        One row per detected pulse event.  Columns:

        - ``pulse_index``: sequential integer starting at 0
        - ``cycle_index``: cycle in which the pulse occurred (NaN if absent)
        - ``current_a``: mean current during the pulse step
        - ``dcir_ohm_measured``: from ``dcir_ohm`` column (NaN if absent)
        - ``voltage_before``: last voltage value before pulse onset
        - ``voltage_after``: first voltage value at pulse onset (≈ t=0⁺)
        - ``delta_v_immediate``: ``voltage_after − voltage_before``
        - ``estimated_dcir_v_over_i``: ``|ΔV / ΔI|`` (NaN if ΔI ≈ 0)
        - ``dcir_source``: ``"measured_dcir"`` or ``"estimated_dcir"``
        - ``step_duration_s``: total step duration in seconds
    """
    required = {"current_a", "step_time_s", "voltage_v"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "detect_pulse_segments: missing columns %s; returning empty DataFrame.",
            missing,
        )
        return pd.DataFrame()

    if "segment" not in df.columns:
        logger.info("detect_pulse_segments: calling label_charge_discharge.")
        df = label_charge_discharge(df)

    pulses = []
    pulse_idx = 0

    # Group by step boundaries using step_index if available, otherwise
    # detect boundaries by sign changes in current
    if "step_index" in df.columns:
        step_col = "step_index"
    else:
        # Synthetic step groups based on segment transitions
        step_col = None

    if step_col is not None:
        groups = df.groupby([col for col in ["cycle_index", step_col] if col in df.columns],
                            sort=False)
    else:
        # Fall back: detect contiguous blocks with same segment label
        df = df.copy()
        df["_step_block"] = (df["segment"] != df["segment"].shift()).cumsum()
        groups = df.groupby(
            [col for col in ["cycle_index", "_step_block"] if col in df.columns],
            sort=False,
        )

    prev_segment_last_row: Optional[pd.Series] = None

    for _keys, step_df in groups:
        if step_df.empty:
            prev_segment_last_row = None
            continue

        current = pd.to_numeric(step_df["current_a"], errors="coerce")
        step_time = pd.to_numeric(step_df["step_time_s"], errors="coerce")
        voltage = pd.to_numeric(step_df["voltage_v"], errors="coerce")
        segment = step_df["segment"].iloc[0]

        mean_current = current.abs().mean()
        max_step_time = step_time.max() if step_time.notna().any() else float("nan")

        is_pulse = (
            not pd.isna(max_step_time)
            and max_step_time < PULSE_MAX_DURATION_S
            and not pd.isna(mean_current)
            and mean_current > PULSE_MIN_CURRENT_A
            and segment in ("charge", "discharge")
        )

        if is_pulse:
            # Voltage before = last row of previous segment
            if prev_segment_last_row is not None:
                v_before = pd.to_numeric(prev_segment_last_row.get("voltage_v", float("nan")),
                                         errors="coerce")
                i_before = pd.to_numeric(prev_segment_last_row.get("current_a", 0.0),
                                         errors="coerce")
            else:
                v_before = float("nan")
                i_before = 0.0

            v_after = voltage.dropna().iloc[0] if voltage.notna().any() else float("nan")
            delta_v = v_after - v_before if not pd.isna(v_before) else float("nan")
            i_pulse = current.dropna().iloc[0] if current.notna().any() else float("nan")
            delta_i = i_pulse - i_before

            # Estimated DCIR
            if abs(delta_i) > 1e-9:
                est_dcir = abs(delta_v / delta_i) if not pd.isna(delta_v) else float("nan")
            else:
                est_dcir = float("nan")

            # Measured DCIR (prefer this)
            if "dcir_ohm" in step_df.columns:
                dcir_meas = pd.to_numeric(step_df["dcir_ohm"], errors="coerce").mean()
                dcir_source = "measured_dcir" if not pd.isna(dcir_meas) else "estimated_dcir"
            else:
                dcir_meas = float("nan")
                dcir_source = "estimated_dcir"

            cycle_id = (
                step_df["cycle_index"].iloc[0]
                if "cycle_index" in step_df.columns
                else float("nan")
            )

            pulses.append(
                {
                    "pulse_index": pulse_idx,
                    "cycle_index": cycle_id,
                    "current_a": current.mean(),
                    "dcir_ohm_measured": dcir_meas,
                    "voltage_before": v_before,
                    "voltage_after": v_after,
                    "delta_v_immediate": delta_v,
                    "estimated_dcir_v_over_i": est_dcir,
                    "dcir_source": dcir_source,
                    "step_duration_s": max_step_time,
                }
            )
            pulse_idx += 1

        prev_segment_last_row = step_df.iloc[-1]

    result = pd.DataFrame(pulses)
    logger.info("Detected %d pulse segments.", len(result))
    return result


# ---------------------------------------------------------------------------
# 6. Ragone data
# ---------------------------------------------------------------------------


def compute_ragone_points(
    cycle_summary: pd.DataFrame,
    active_mass_g: Optional[float],
) -> pd.DataFrame:
    """
    Compute Ragone-plot (energy density vs. power density) data points.

    For each cycle in *cycle_summary* the following are calculated:

    .. math::

        E = E_{dis} / m_{active}  \\quad \\text{[Wh g⁻¹]} \\quad
        \\text{(or } E_{dis} \\text{ [Wh] if no mass)}

    .. math::

        P \\approx E / t_{dis} \\quad \\text{where} \\quad
        t_{dis} = E_{dis} / Q_{dis}

    The approximate discharge time (in hours) is derived from the ratio of
    discharge energy to discharge capacity, which equals the mean discharge
    voltage:

    .. math::

        t_{dis} \\approx E_{dis} [Wh] / Q_{dis} [Ah]  \\quad \\text{[h]}

    This approximation is exact for constant-voltage discharge and a good
    approximation for CCCV profiles.

    Parameters
    ----------
    cycle_summary:
        DataFrame from :func:`compute_cycle_summary`.
    active_mass_g:
        Active electrode mass in grams.  If ``None``, absolute energy (Wh)
        and power (W) are reported.

    Returns
    -------
    pd.DataFrame
        Columns: ``cycle_index``, ``energy_axis``, ``power_axis``, ``basis``.
        ``basis`` is ``"gravimetric [Wh/g, W/g]"`` or ``"absolute [Wh, W]"``.
        Returns an empty DataFrame if required columns are absent or all values
        are non-positive.
    """
    required = {"discharge_energy_wh", "discharge_capacity_ah"}
    if not required.issubset(set(cycle_summary.columns)):
        missing = required - set(cycle_summary.columns)
        logger.warning(
            "compute_ragone_points: missing columns %s; returning empty DataFrame.",
            missing,
        )
        return pd.DataFrame()

    e_dis = pd.to_numeric(cycle_summary["discharge_energy_wh"], errors="coerce")
    q_dis = pd.to_numeric(cycle_summary["discharge_capacity_ah"], errors="coerce")

    valid = e_dis.notna() & q_dis.notna() & (e_dis > 0) & (q_dis > 0)
    if not valid.any():
        logger.warning(
            "compute_ragone_points: no valid (positive) energy/capacity data; "
            "returning empty DataFrame."
        )
        return pd.DataFrame()

    # Approximate mean discharge voltage (V) → determines discharge time
    # t_dis [h] = E [Wh] / Q [Ah]  → mean voltage
    t_dis_h = e_dis / q_dis  # hours (= mean V in a simple ohmic model)
    # Power [W] = Energy [Wh] / time [h]
    power_abs = e_dis / t_dis_h  # W

    if active_mass_g and active_mass_g > 0:
        basis = "gravimetric [Wh/g, W/g]"
        energy_axis = e_dis / active_mass_g
        power_axis = power_abs / active_mass_g
    else:
        basis = "absolute [Wh, W]"
        energy_axis = e_dis
        power_axis = power_abs

    result = pd.DataFrame(
        {
            "cycle_index": cycle_summary["cycle_index"].values
            if "cycle_index" in cycle_summary.columns
            else range(len(cycle_summary)),
            "energy_axis": energy_axis.values,
            "power_axis": power_axis.values,
            "basis": basis,
        }
    )
    result = result[valid.values].reset_index(drop=True)
    logger.info(
        "Ragone data computed: %d points, basis = '%s'.", len(result), basis
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_capacity(
    cycle_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    waveform_col: str,
    running_col: str,
) -> float:
    """
    Extract a capacity (or energy) value preferring the waveform column.

    Priority:
    1. Non-NaN value in ``waveform_col`` of *cycle_df* (pre-integrated by cycler).
    2. Maximum value of ``running_col`` in *segment_df* (peak of running total).

    Returns ``NaN`` if neither source yields a valid positive value.
    """
    # Preferred: waveform (pre-integrated) column on the full cycle
    if waveform_col in cycle_df.columns:
        wf_vals = pd.to_numeric(cycle_df[waveform_col], errors="coerce")
        wf_valid = wf_vals[wf_vals > 0]
        if not wf_valid.empty:
            return float(wf_valid.max())

    # Fallback: peak of running accumulator in segment
    if running_col in segment_df.columns and not segment_df.empty:
        run_vals = pd.to_numeric(segment_df[running_col], errors="coerce")
        run_valid = run_vals[run_vals > 0]
        if not run_valid.empty:
            return float(run_valid.max())

    return float("nan")


def _both_positive(a: float, b: float) -> bool:
    """Return True if both *a* and *b* are finite and positive."""
    try:
        return (
            not pd.isna(a) and not pd.isna(b)
            and float(a) > 0 and float(b) > 0
        )
    except (TypeError, ValueError):
        return False
