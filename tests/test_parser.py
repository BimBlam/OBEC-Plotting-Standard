"""Tests for CSV parser."""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import textwrap
from batteryplot.parsing import detect_header_row, parse_time_column

def make_arbin_style_csv(tmp_path) -> Path:
    content = textwrap.dedent("""
        Today's Date ,4/14/2026
        Date of Test:,4/10/2026 3:02:23 AM
        Test Time,Cycle P,Current (A),Voltage (V),Cycle C,Capacity (AHr),Rec,Step,Step Time,Energy (WHr),MD
          0d 00:00:00,0,0,0.156,1,0,1,1,  0d 00:00:00,0,R
          0d 00:00:10,0,0,0.157,1,0,2,1,  0d 00:00:10,0,R
          0d 00:00:20,0,0.1,0.158,1,0.001,3,2,  0d 00:00:10,0.000016,C
    """).lstrip()
    p = tmp_path / "test_cell.csv"
    p.write_text(content)
    return p

def test_detect_header_row(tmp_path):
    p = make_arbin_style_csv(tmp_path)
    row_idx, cols = detect_header_row(p)
    assert row_idx == 2  # 0-indexed, so row 3 in file (0-based = 2)
    assert "Test Time" in cols
    assert "Current (A)" in cols
    assert "Voltage (V)" in cols

def test_parse_time_column_ddays_format():
    s = pd.Series(["  0d 00:00:00", "  0d 00:00:10", "  0d 01:00:00", "  1d 00:00:00"])
    result = parse_time_column(s)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(3600.0)
    assert result.iloc[3] == pytest.approx(86400.0)

def test_parse_time_column_na():
    s = pd.Series(["N/A", "  0d 00:00:10"])
    result = parse_time_column(s)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(10.0)

def test_detect_header_handles_many_metadata_rows(tmp_path):
    # File with 5 metadata rows before real header
    lines = [f"meta_{i},value_{i}\n" for i in range(5)]
    lines.append("Current (A),Voltage (V),Capacity (AHr),Cycle P\n")
    lines.append("0.1,3.8,0.001,1\n")
    lines.append("0.1,3.81,0.002,1\n")
    p = tmp_path / "multi_meta.csv"
    p.write_text("".join(lines))
    row_idx, cols = detect_header_row(p, max_scan=10)
    assert "Current (A)" in cols
    assert "Voltage (V)" in cols
