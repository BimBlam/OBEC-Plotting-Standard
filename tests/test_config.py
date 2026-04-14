"""Tests for batteryplot.config."""
import tempfile
from pathlib import Path
import pytest
from batteryplot.config import (
    BatteryPlotConfig,
    load_config,
    default_config,
    save_default_config,
)


def test_default_config_returns_instance():
    cfg = default_config()
    assert isinstance(cfg, BatteryPlotConfig)


def test_default_config_input_dir():
    cfg = default_config()
    assert cfg.input_dir == Path("input")


def test_default_config_output_dir():
    cfg = default_config()
    assert cfg.output_dir == Path("output")


def test_default_config_nominal_cap_is_none():
    cfg = default_config()
    assert cfg.nominal_capacity_ah is None


def test_default_config_theme():
    cfg = default_config()
    assert cfg.theme == "publication"


def test_invalid_theme_raises():
    with pytest.raises(Exception):
        BatteryPlotConfig(theme="neon")


def test_invalid_log_level_raises():
    with pytest.raises(Exception):
        BatteryPlotConfig(log_level="VERBOSE")


def test_invalid_plot_family_raises():
    with pytest.raises(Exception):
        BatteryPlotConfig(selected_plot_families=["nonexistent_family"])


def test_invalid_output_format_raises():
    with pytest.raises(Exception):
        BatteryPlotConfig(output_formats=["bmp"])


def test_save_default_config_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.example.yaml"
        save_default_config(path)
        assert path.exists()
        content = path.read_text()
        assert "nominal_capacity_ah" in content
        assert "batteryplot" in content


def test_load_config_yaml(tmp_path):
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(
        "theme: dark\nlog_level: DEBUG\nnominal_capacity_ah: 3.0\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.theme == "dark"
    assert cfg.log_level == "DEBUG"
    assert cfg.nominal_capacity_ah == pytest.approx(3.0)


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
