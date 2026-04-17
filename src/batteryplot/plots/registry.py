"""
Plot registry: defines all expected plots, their required/optional columns,
derivation rules, and placeholder fallback metadata.

Each entry in the registry defines:
- key: unique identifier (also used as file stem prefix)
- title: human-readable title
- family: one of the six families
- required_columns: list of canonical column names that must exist in df or cycle_summary
- optional_columns: columns that enhance the plot if present
- data_source: "timeseries" | "cycle_summary" | "pulse" | "any"
- description: scientific rationale
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlotSpec:
    key: str
    title: str
    family: str
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    data_source: str = "timeseries"
    description: str = ""


PLOT_REGISTRY: List[PlotSpec] = [
    # --- Voltage profiles ---
    PlotSpec(
        key="voltage_vs_capacity",
        title="Voltage vs. Capacity",
        family="voltage_profiles",
        required_columns=["voltage_v", "capacity_ah"],
        optional_columns=["cycle_index", "current_a"],
        data_source="timeseries",
        description="Charge/discharge voltage profiles. Representative cycles shown. "
                    "Requires voltage and capacity columns.",
    ),
    PlotSpec(
        key="voltage_vs_time",
        title="Voltage and Current vs. Time",
        family="voltage_profiles",
        required_columns=["voltage_v", "elapsed_time_s"],
        optional_columns=["current_a", "cycle_index"],
        data_source="timeseries",
        description="Overview of voltage (and optionally current) versus elapsed test time.",
    ),
    # --- Cycle summary ---
    PlotSpec(
        key="capacity_retention",
        title="Capacity Retention vs. Cycle Number",
        family="cycle_summary",
        required_columns=["cycle_index", "discharge_capacity_ah"],
        optional_columns=["charge_capacity_ah", "capacity_retention_pct"],
        data_source="cycle_summary",
        description="Discharge (and charge) capacity versus cycle index. "
                    "Retention % requires nominal_capacity_ah in config.",
    ),
    PlotSpec(
        key="coulombic_efficiency",
        title="Coulombic and Energy Efficiency vs. Cycle",
        family="cycle_summary",
        required_columns=["cycle_index", "coulombic_efficiency_pct"],
        optional_columns=["energy_efficiency_pct"],
        data_source="cycle_summary",
        description="Ratio of discharge to charge capacity per cycle. "
                    "Values near 100% indicate good cell health.",
    ),
    PlotSpec(
        key="dcir_vs_cycle",
        title="DCIR vs. Cycle Number",
        family="cycle_summary",
        required_columns=["cycle_index", "mean_dcir_ohm"],
        optional_columns=["mean_ac_impedance_ohm"],
        data_source="cycle_summary",
        description="Mean per-cycle DCIR (and optionally AC impedance) trend. "
                    "Rising DCIR indicates increasing internal resistance.",
    ),
    # --- Rate capability ---
    PlotSpec(
        key="rate_capability",
        title="Rate Capability: Capacity vs. C-rate",
        family="rate_capability",
        required_columns=["cycle_index", "discharge_capacity_ah"],
        optional_columns=["charge_capacity_ah", "c_rate"],
        data_source="cycle_summary",
        description="Discharge and charge capacity as a function of C-rate. "
                    "Requires nominal_capacity_ah in config for C-rate axis. "
                    "Without it, current (A) is used as the x-axis.",
    ),
    PlotSpec(
        key="rate_voltage_profiles",
        title="Voltage vs. Capacity by Rate",
        family="rate_capability",
        required_columns=["voltage_v", "capacity_ah", "cycle_index"],
        optional_columns=["current_a"],
        data_source="timeseries",
        description="Voltage-capacity curves at selected cycles representing different rates.",
    ),
    # --- Pulse / resistance ---
    PlotSpec(
        key="dcir_vs_current",
        title="DCIR vs. Current Density",
        family="pulse_resistance",
        required_columns=["dcir_ohm"],
        optional_columns=["current_a", "cycle_index"],
        data_source="timeseries",
        description="DC internal resistance as a function of applied current. "
                    "DCIR measured directly by cycler (labeled as measured_dcir). "
                    "Current density requires electrode_area_cm2 in config.",
    ),
    PlotSpec(
        key="pulse_analysis",
        title="Pulse Resistance Decomposition",
        family="pulse_resistance",
        required_columns=["current_a", "voltage_v", "elapsed_time_s"],
        optional_columns=["dcir_ohm", "step_time_s"],
        data_source="timeseries",
        description="Pulse polarization analysis. Immediate (ohmic) and longer-term "
                    "(kinetic) voltage drops are estimated from constant-current pulses. "
                    "Only segments with step_time < 200 s and |I| > 0.01 A are used. "
                    "Estimated DCIR = ΔV/I (labeled as estimated, not measured).",
    ),
    # --- Ragone ---
    PlotSpec(
        key="ragone",
        title="Energy\u2013Power (Ragone) Plot",
        family="ragone",
        required_columns=["discharge_energy_wh", "discharge_capacity_ah"],
        optional_columns=["charge_energy_wh"],
        data_source="cycle_summary",
        description="Discharge energy vs. approximate discharge power. "
                    "If active_mass_g is in config, axes are gravimetric (Wh/g, W/g). "
                    "Otherwise absolute (Wh, W). "
                    "Power estimated as energy / discharge_time; discharge_time = capacity / |I_avg|. "
                    "Do not use volumetric basis unless density_g_cm3 and active_mass_g are both set.",
    ),
    # --- QA ---
    PlotSpec(
        key="temperature_vs_time",
        title="Temperature vs. Time",
        family="qa",
        required_columns=["temperature_c", "elapsed_time_s"],
        optional_columns=["humidity_pct"],
        data_source="timeseries",
        description="Cell temperature (and optionally humidity) over the test duration.",
    ),
    PlotSpec(
        key="current_voltage_overview",
        title="Current and Voltage Overview",
        family="qa",
        required_columns=["current_a", "voltage_v", "elapsed_time_s"],
        optional_columns=[],
        data_source="timeseries",
        description="Full-test overview of current and voltage signals. "
                    "Useful for QA: detecting step anomalies, rest periods, protocol issues.",
    ),
    PlotSpec(
        key="data_availability",
        title="Data Availability Summary",
        family="qa",
        required_columns=[],
        optional_columns=[],
        data_source="any",
        description="Visual summary of which canonical columns are present and their completeness.",
    ),
]

REGISTRY_BY_KEY = {spec.key: spec for spec in PLOT_REGISTRY}
REGISTRY_BY_FAMILY: dict = {}
for spec in PLOT_REGISTRY:
    REGISTRY_BY_FAMILY.setdefault(spec.family, []).append(spec)
