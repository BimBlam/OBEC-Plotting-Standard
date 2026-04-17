"""Tests for the fuzzy-guess test-region classifier (classify_test_regions)."""
import numpy as np
import pandas as pd
import pytest

from batteryplot.transforms import classify_test_regions, filter_cycles_by_region


def _make_cycle_summary(
    n_cycles: int,
    discharge_currents: list[float],
    charge_currents: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a minimal cycle_summary with the specified per-cycle currents.

    Parameters
    ----------
    n_cycles : int
        Total number of cycles.
    discharge_currents : list[float]
        Mean |discharge current| for each cycle (length must equal n_cycles).
    charge_currents : list[float] | None
        Mean |charge current| per cycle.  Defaults to same as discharge.
    """
    if charge_currents is None:
        charge_currents = discharge_currents
    assert len(discharge_currents) == n_cycles
    return pd.DataFrame({
        "cycle_index": list(range(1, n_cycles + 1)),
        "discharge_capacity_ah": [1.0] * n_cycles,
        "mean_discharge_current_a": discharge_currents,
        "mean_charge_current_a": charge_currents,
    })


class TestClassifyTestRegions:
    """classify_test_regions should label cycles as formation / rate_test / cycling / unknown."""

    def test_classify_rate_then_cycling(self):
        """5 rate-test cycles (alternating rates) + 50 cycling cycles at fixed rate."""
        # 5 cycles alternating between 0.5 A and 1.0 A → rate_test
        rate_currents = [0.5, 1.0, 0.5, 1.0, 0.5]
        # 50 cycles at constant 0.5 A → cycling
        cycling_currents = [0.5] * 50
        all_currents = rate_currents + cycling_currents
        cs = _make_cycle_summary(55, all_currents)

        result = classify_test_regions(cs)

        assert "test_region" in result.columns
        # The 50-cycle constant block should be tagged "cycling"
        cycling_rows = result[result["test_region"] == "cycling"]
        assert len(cycling_rows) >= 50
        # The short alternating block should be tagged "rate_test" or "formation"
        non_cycling = result[result["test_region"].isin(["rate_test", "formation"])]
        assert len(non_cycling) >= 1

    def test_classify_cycling_only(self):
        """100 cycles at the same rate → all 'cycling'."""
        cs = _make_cycle_summary(100, [0.5] * 100)

        result = classify_test_regions(cs)

        assert "test_region" in result.columns
        cycling_rows = result[result["test_region"] == "cycling"]
        assert len(cycling_rows) == 100

    def test_classify_rate_only(self):
        """20 cycles alternating among 5 rates → all 'rate_test'."""
        rates = [0.1, 0.2, 0.5, 1.0, 2.0]
        # 4 cycles at each rate, interleaved: 0.1,0.1,0.1,0.1, 0.2,0.2,...
        currents = []
        for r in rates:
            currents.extend([r] * 4)
        cs = _make_cycle_summary(20, currents)

        result = classify_test_regions(cs)

        assert "test_region" in result.columns
        # No block is >= 10 cycles → none should be "cycling"
        cycling_rows = result[result["test_region"] == "cycling"]
        assert len(cycling_rows) == 0
        # All should be rate_test or formation
        assert set(result["test_region"].unique()) <= {"rate_test", "formation"}

    def test_classify_no_current(self):
        """No current columns → all cycles tagged 'unknown'."""
        cs = pd.DataFrame({
            "cycle_index": list(range(1, 21)),
            "discharge_capacity_ah": [1.0] * 20,
        })

        result = classify_test_regions(cs)

        assert "test_region" in result.columns
        assert (result["test_region"] == "unknown").all()

    def test_filter_cycles_by_region_fallback(self):
        """filter_cycles_by_region returns all rows when preferred region is absent."""
        cs = pd.DataFrame({
            "cycle_index": list(range(1, 11)),
            "discharge_capacity_ah": [1.0] * 10,
            "test_region": ["rate_test"] * 10,
        })

        result = filter_cycles_by_region(cs, "cycling")

        # No "cycling" rows exist, so all rows should be returned as fallback
        assert len(result) == 10
        assert (result["test_region"] == "rate_test").all()
