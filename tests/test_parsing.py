"""Tests for batteryplot.parsing."""
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batteryplot.config import default_config
from batteryplot.parsing import (
    detect_header_row,
    parse_time_column,
    parse_datetime_column,
    load_csv,
    build_analysis_df,
)

# ---------------------------------------------------------------------------
# Sample Arbin-style CSV content
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
Today's Date ,4/14/2026
Date of Test:,4/10/2026
Test Time,Cycle P,Current (A),Voltage (V),Cycle C,Capacity (AHr),Rec,Step,Step Time,Energy (WHr),MD
  0d 00:00:00,1,0.5,3.800,1,0.000,1,1,  0d 00:00:00,0.000,C
  0d 00:00:10,1,0.5,3.850,1,0.001,2,1,  0d 00:00:10,0.004,C
  0d 00:00:20,1,0.5,3.900,1,0.003,3,1,  0d 00:00:20,0.012,C
  0d 00:00:30,1,-0.5,3.800,2,0.003,4,2,  0d 00:00:00,0.012,D
  0d 00:00:40,1,-0.5,3.750,2,0.001,5,2,  0d 00:00:10,0.004,D
  0d 00:00:50,2,0.5,3.800,1,0.000,6,1,  0d 00:00:00,0.000,C
"""


def _write_sample(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text(SAMPLE_CSV)
    return p


# ---------------------------------------------------------------------------
# detect_header_row
# ---------------------------------------------------------------------------


def test_detect_header_row(tmp_path):
    p = _write_sample(tmp_path)
    idx, cols = detect_header_row(p)
    assert idx == 2
    assert "Test Time" in cols
    assert "Voltage (V)" in cols


def test_detect_header_row_returns_correct_column_count(tmp_path):
    p = _write_sample(tmp_path)
    _, cols = detect_header_row(p)
    assert len(cols) == 11  # SAMPLE_CSV has 11 columns


# ---------------------------------------------------------------------------
# parse_time_column
# ---------------------------------------------------------------------------


def test_parse_time_column_day_format():
    s = pd.Series(["  0d 00:00:00", "  1d 02:03:04", "  0d 01:00:00"])
    result = parse_time_column(s)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(86400 + 7384.0)
    assert result[2] == pytest.approx(3600.0)


def test_parse_time_column_na_handling():
    s = pd.Series(["N/A", "", "  0d 00:00:10"])
    result = parse_time_column(s)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(10.0)


def test_parse_time_column_numeric_fallback():
    s = pd.Series(["3600", "120.5", "0"])
    result = parse_time_column(s)
    assert result[0] == pytest.approx(3600.0)
    assert result[1] == pytest.approx(120.5)
    assert result[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_datetime_column
# ---------------------------------------------------------------------------


def test_parse_datetime_column_arbin_format():
    s = pd.Series(["4/10/2026 3:02:24 AM", "4/10/2026 11:59:59 PM"])
    result = parse_datetime_column(s)
    assert result[0] == pd.Timestamp("2026-04-10 03:02:24")
    assert result[1] == pd.Timestamp("2026-04-10 23:59:59")


def test_parse_datetime_column_coerces_bad_values():
    s = pd.Series(["not-a-date", "4/10/2026 3:02:24 AM"])
    result = parse_datetime_column(s)
    assert pd.isna(result[0])
    assert not pd.isna(result[1])


# ---------------------------------------------------------------------------
# load_csv + build_analysis_df (integration)
# ---------------------------------------------------------------------------


def test_load_csv_returns_tuple(tmp_path):
    p = _write_sample(tmp_path)
    cfg = default_config()
    raw_df, col_map, meta = load_csv(p, cfg)
    assert isinstance(raw_df, pd.DataFrame)
    assert isinstance(col_map, dict)
    assert isinstance(meta, dict)


def test_load_csv_metadata(tmp_path):
    p = _write_sample(tmp_path)
    cfg = default_config()
    _, _, meta = load_csv(p, cfg)
    assert "Today's Date" in meta
    assert "4/14/2026" in meta["Today's Date"]


def test_load_csv_column_map(tmp_path):
    p = _write_sample(tmp_path)
    cfg = default_config()
    _, col_map, _ = load_csv(p, cfg)
    assert "Voltage (V)" in col_map
    assert col_map["Voltage (V)"] == "voltage_v"
    assert col_map["Current (A)"] == "current_a"


def test_build_analysis_df_canonical_columns(tmp_path):
    p = _write_sample(tmp_path)
    cfg = default_config()
    raw_df, col_map, _ = load_csv(p, cfg)
    df = build_analysis_df(raw_df, col_map)
    assert "voltage_v" in df.columns
    assert "current_a" in df.columns
    assert "capacity_ah" in df.columns


def test_build_analysis_df_numeric_types(tmp_path):
    p = _write_sample(tmp_path)
    cfg = default_config()
    raw_df, col_map, _ = load_csv(p, cfg)
    df = build_analysis_df(raw_df, col_map)
    assert df["voltage_v"].dtype in (np.float64, np.float32, float)
