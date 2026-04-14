"""Tests for the plot registry."""
import pytest
from batteryplot.plots.registry import PLOT_REGISTRY, REGISTRY_BY_KEY, REGISTRY_BY_FAMILY

def test_registry_not_empty():
    assert len(PLOT_REGISTRY) > 0

def test_all_families_represented():
    families = {spec.family for spec in PLOT_REGISTRY}
    expected = {"voltage_profiles", "cycle_summary", "rate_capability", "pulse_resistance", "ragone", "qa"}
    assert expected.issubset(families)

def test_registry_by_key_lookup():
    assert "voltage_vs_capacity" in REGISTRY_BY_KEY
    assert "capacity_retention" in REGISTRY_BY_KEY
    assert "ragone" in REGISTRY_BY_KEY

def test_all_specs_have_required_fields():
    for spec in PLOT_REGISTRY:
        assert spec.key, f"Spec missing key: {spec}"
        assert spec.title, f"Spec {spec.key} missing title"
        assert spec.family, f"Spec {spec.key} missing family"
        assert isinstance(spec.required_columns, list)
        assert isinstance(spec.optional_columns, list)

def test_registry_by_family():
    assert "voltage_profiles" in REGISTRY_BY_FAMILY
    for family, specs in REGISTRY_BY_FAMILY.items():
        assert len(specs) >= 1
