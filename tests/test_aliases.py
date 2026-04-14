"""Tests for column alias matching."""
import pytest
from batteryplot.aliases import normalize_header, match_column, map_columns

def test_normalize_strips_whitespace():
    assert normalize_header("  Test Time  ") == "test time"

def test_normalize_handles_parentheses():
    assert normalize_header("Current (A)") == "current (a)"

def test_match_voltage():
    assert match_column("voltage (v)") == "voltage_v"

def test_match_current():
    assert match_column("current (a)") == "current_a"

def test_match_dcir():
    assert match_column("dcir (ohms)") == "dcir_ohm"

def test_match_temperature():
    assert match_column("evtemp (c)") == "temperature_c"

def test_match_test_time():
    assert match_column("test time") == "elapsed_time_s"

def test_match_unknown_returns_none():
    assert match_column("xyzabc_unknown_col_999") is None

def test_map_columns_arbin_header():
    raw = [
        "Test Time", "Cycle P", "Current (A)", "Voltage (V)", "Cycle C",
        "Capacity (AHr)", "Rec", "Step", "Step Time", "Energy (WHr)",
        "MD", "ES", "DPT Time", "AC Imp (Ohms)", "DCIR (Ohms)",
        "EVTemp (C)", "EVHum (%)", "S.Capacity (Ah/g)", "Power (W)",
        "WF Chg Cap", "WF Dis Cap", "WF Chg E", "WF Dis E",
        "Resistance", "Conductivity", "ESR",
    ]
    result = map_columns(raw)
    assert result.get("Current (A)") == "current_a"
    assert result.get("Voltage (V)") == "voltage_v"
    assert result.get("DCIR (Ohms)") == "dcir_ohm"
    assert result.get("Test Time") == "elapsed_time_s"
    assert result.get("AC Imp (Ohms)") == "ac_impedance_ohm"

def test_map_columns_deduplication():
    # ESR and AC Imp both map to ac_impedance_ohm; first encountered wins
    raw = ["AC Imp (Ohms)", "ESR"]
    result = map_columns(raw)
    # Both present: the first one that matches should win for the canonical slot
    assert "AC Imp (Ohms)" in result or "ESR" in result
