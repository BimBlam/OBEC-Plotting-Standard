"""Tests for placeholder figure generation."""
import pytest
from pathlib import Path
from batteryplot.placeholders import make_placeholder

def test_placeholder_creates_svg(tmp_path):
    paths = make_placeholder(
        title="Test Plot",
        missing_columns=["voltage_v", "current_a"],
        output_dir=tmp_path,
        stem="test_plot",
        formats=("svg",),
    )
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".svg"

def test_placeholder_creates_pdf(tmp_path):
    paths = make_placeholder(
        title="Another Plot",
        missing_columns=["dcir_ohm"],
        output_dir=tmp_path,
        stem="another_plot",
        formats=("svg", "pdf"),
    )
    assert len(paths) == 2
    assert any(p.suffix == ".pdf" for p in paths)

def test_placeholder_svg_contains_text(tmp_path):
    paths = make_placeholder(
        title="Missing Data Test",
        missing_columns=["some_column"],
        output_dir=tmp_path,
        stem="missing",
        formats=("svg",),
        note="This is a test note.",
    )
    content = paths[0].read_text(encoding="utf-8")
    # SVG should contain the title text
    assert "Missing Data Test" in content or "Data Absent" in content

def test_placeholder_with_empty_missing_columns(tmp_path):
    paths = make_placeholder(
        title="No Missing Cols",
        missing_columns=[],
        output_dir=tmp_path,
        stem="no_missing",
        formats=("svg",),
    )
    assert paths[0].exists()
