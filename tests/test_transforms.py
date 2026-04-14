"""Tests for batteryplot.transforms."""
import numpy as np
import pandas as pd
import pytest

from batteryplot.config import BatteryPlotConfig, default_config
from batteryplot.transforms import (
    label_charge_discharge,
    compute_cycle_summary,
    compute_crate,
    compute_specific_capacity,
    detect_pulse_segments,
    compute_ragone_points,
    CURRENT_REST_THRESHOLD_A,
)


def _make_df(n: int = 30) -> pd.DataFrame:
    """Create a minimal canonical DataFrame for testing."""
    half = n // 2
    current = [0.5] * half + [-0.5] * half  # charge then discharge
    voltage = [3.8 + 0.1 * (i / half) for i in range(half)] + \
              [3.9 - 0.1 * (i / half) for i in range(half)]
    capacity = [0.0 + 0.001 * i for i in range(half)] + \
               [0.015 - 0.001 * i for i in range(half)]
    cycle = [1] * n
    step_time = [10.0 * i for i in range(n)]

    return pd.DataFrame({
        "current_a": current,
        "voltage_v": voltage,
        "capacity_ah": capacity,
        "cycle_index": cycle,
        "elapsed_time_s": [10.0 * i for i in range(n)],
        "step_time_s": step_time,
    })


# ---------------------------------------------------------------------------
# label_charge_discharge
# ---------------------------------------------------------------------------


def test_label_charge_discharge_adds_segment_column():
    df = _make_df()
    result = label_charge_discharge(df)
    assert "segment" in result.columns


def test_label_charge_discharge_values():
    df = _make_df()
    result = label_charge_discharge(df)
    assert (result["segment"].iloc[:15] == "charge").all()
    assert (result["segment"].iloc[15:] == "discharge").all()


def test_label_charge_discharge_rest():
    df = pd.DataFrame({
        "current_a": [0.0, 1e-6, 0.0],
        "voltage_v": [3.8, 3.8, 3.8],
    })
    result = label_charge_discharge(df)
    assert (result["segment"] == "rest").all()


def test_label_charge_discharge_does_not_mutate():
    df = _make_df()
    _ = label_charge_discharge(df)
    assert "segment" not in df.columns


def test_label_charge_discharge_step_type_preferred():
    df = pd.DataFrame({
        "current_a": [0.5, 0.5, 0.5],
        "voltage_v": [3.8, 3.85, 3.9],
        "step_type": ["C", "C", "D"],  # last row marked D regardless of current
    })
    result = label_charge_discharge(df)
    assert result["segment"].iloc[0] == "charge"
    assert result["segment"].iloc[2] == "discharge"


# ---------------------------------------------------------------------------
# compute_cycle_summary
# ---------------------------------------------------------------------------


def test_compute_cycle_summary_returns_dataframe():
    df = _make_df()
    df = label_charge_discharge(df)
    cfg = default_config()
    summary = compute_cycle_summary(df, cfg)
    assert isinstance(summary, pd.DataFrame)


def test_compute_cycle_summary_one_row_per_cycle():
    df = _make_df()
    df = label_charge_discharge(df)
    cfg = default_config()
    summary = compute_cycle_summary(df, cfg)
    assert len(summary) == 1
    assert summary["cycle_index"].iloc[0] == 1


def test_compute_cycle_summary_coulombic_efficiency():
    df = _make_df()
    df = label_charge_discharge(df)
    cfg = default_config()
    summary = compute_cycle_summary(df, cfg)
    # Coulombic efficiency should be finite and <= 100%
    ce = summary["coulombic_efficiency_pct"].iloc[0]
    assert not np.isnan(ce)
    assert 0 < ce <= 115  # allow overshoot for synthetic test data (discharge > charge due to construction)


def test_compute_cycle_summary_capacity_retention():
    df = _make_df()
    df = label_charge_discharge(df)
    cfg = BatteryPlotConfig(nominal_capacity_ah=0.015)
    summary = compute_cycle_summary(df, cfg)
    cr = summary["capacity_retention_pct"].iloc[0]
    assert not np.isnan(cr)
    assert cr > 0


# ---------------------------------------------------------------------------
# compute_crate
# ---------------------------------------------------------------------------


def test_compute_crate_basic():
    df = pd.DataFrame({"current_a": [1.5, -3.0, 0.0]})
    cr = compute_crate(df, nominal_capacity_ah=3.0)
    assert cr[0] == pytest.approx(0.5)
    assert cr[1] == pytest.approx(1.0)
    assert cr[2] == pytest.approx(0.0)


def test_compute_crate_invalid_nominal_raises():
    df = pd.DataFrame({"current_a": [1.0]})
    with pytest.raises(ValueError):
        compute_crate(df, nominal_capacity_ah=0.0)


def test_compute_crate_missing_current_col():
    df = pd.DataFrame({"voltage_v": [3.8]})
    cr = compute_crate(df, nominal_capacity_ah=1.0)
    assert np.isnan(cr.iloc[0])


# ---------------------------------------------------------------------------
# compute_specific_capacity
# ---------------------------------------------------------------------------


def test_compute_specific_capacity_basic():
    df = pd.DataFrame({"capacity_ah": [1.0, 2.0]})
    sp = compute_specific_capacity(df, active_mass_g=2.0)
    assert sp[0] == pytest.approx(500.0)  # 1 Ah * 1000 / 2 g = 500 mAh/g
    assert sp[1] == pytest.approx(1000.0)


def test_compute_specific_capacity_zero_mass_raises():
    df = pd.DataFrame({"capacity_ah": [1.0]})
    with pytest.raises(ValueError):
        compute_specific_capacity(df, active_mass_g=0.0)


# ---------------------------------------------------------------------------
# compute_ragone_points
# ---------------------------------------------------------------------------


def test_compute_ragone_empty_if_no_data():
    result = compute_ragone_points(pd.DataFrame(), active_mass_g=None)
    assert result.empty


def test_compute_ragone_absolute():
    cs = pd.DataFrame({
        "cycle_index": [1, 2, 3],
        "discharge_energy_wh": [1.5, 1.4, 1.3],
        "discharge_capacity_ah": [0.5, 0.5, 0.5],
    })
    result = compute_ragone_points(cs, active_mass_g=None)
    assert "energy_axis" in result.columns
    assert "power_axis" in result.columns
    assert len(result) == 3
    assert "absolute" in result["basis"].iloc[0]


def test_compute_ragone_gravimetric():
    cs = pd.DataFrame({
        "cycle_index": [1],
        "discharge_energy_wh": [2.0],
        "discharge_capacity_ah": [0.5],
    })
    result = compute_ragone_points(cs, active_mass_g=10.0)
    assert "gravimetric" in result["basis"].iloc[0]
    assert result["energy_axis"].iloc[0] == pytest.approx(0.2)  # 2 Wh / 10 g
