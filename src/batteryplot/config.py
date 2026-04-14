"""
batteryplot.config
==================
Pydantic v2 configuration model for the batteryplot pipeline.

All user-facing settings are centralised here so that every downstream
module can import a single, validated ``BatteryPlotConfig`` object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger("batteryplot")

# ---------------------------------------------------------------------------
# Supported plot families (used for validation & availability reporting)
# ---------------------------------------------------------------------------
PLOT_FAMILIES: List[str] = [
    "voltage_profiles",
    "cycle_summary",
    "rate_capability",
    "pulse_resistance",
    "ragone",
    "qa",
]

OUTPUT_FORMATS: List[str] = ["svg", "pdf", "png"]
THEMES: List[str] = ["publication", "dark"]


# ---------------------------------------------------------------------------
# Main config model
# ---------------------------------------------------------------------------


class BatteryPlotConfig(BaseModel):
    """
    Top-level configuration for the batteryplot pipeline.

    All path fields default to relative paths so that the package can be
    used from any working directory.  Override by supplying an absolute
    path or a YAML/TOML config file.

    Scientific fields
    -----------------
    nominal_capacity_ah
        Rated capacity of the cell in ampere-hours.  Required for
        C-rate computation and capacity-retention normalisation.
    active_mass_g
        Mass of the electrode active material in grams.  Required for
        gravimetric specific-capacity (mAh g⁻¹) and Ragone plots.
    electrode_area_cm2
        Geometric electrode area in cm².  Required for areal-capacity
        normalisation (mAh cm⁻²).
    density_g_cm3
        Active-material density in g cm⁻³.  Used only for volumetric
        normalisation; optional.
    """

    # --- I/O ---------------------------------------------------------------
    input_dir: Path = Field(Path("input"), description="Directory containing raw CSV files.")
    output_dir: Path = Field(Path("output"), description="Root directory for all output artefacts.")

    # --- Cell parameters ---------------------------------------------------
    nominal_capacity_ah: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Nominal (rated) cell capacity in ampere-hours. "
            "Required for C-rate and capacity-retention calculations."
        ),
    )
    active_mass_g: Optional[float] = Field(
        None,
        gt=0,
        description="Active electrode mass in grams. Required for gravimetric normalisation.",
    )
    electrode_area_cm2: Optional[float] = Field(
        None,
        gt=0,
        description="Geometric electrode area in cm². Required for areal normalisation.",
    )
    density_g_cm3: Optional[float] = Field(
        None,
        gt=0,
        description="Active-material bulk density in g cm⁻³. Used for volumetric normalisation.",
    )

    # --- Plot control ------------------------------------------------------
    representative_cycles: Optional[List[int]] = Field(
        None,
        description=(
            "Explicit cycle indices to use for voltage-profile overlays. "
            "If None, the pipeline auto-selects the first, middle, and last cycles."
        ),
    )
    selected_plot_families: List[str] = Field(
        default_factory=lambda: list(PLOT_FAMILIES),
        description="Subset of plot families to generate.  Defaults to all families.",
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["svg", "pdf"],
        description="List of output file formats for every plot.",
    )
    theme: str = Field(
        "publication",
        description="Matplotlib style theme.  One of: 'publication', 'dark'.",
    )

    # --- Pipeline behaviour ------------------------------------------------
    overwrite: bool = Field(
        True,
        description="If True, overwrite existing output files.  If False, skip existing outputs.",
    )

    # --- Header detection --------------------------------------------------
    header_search_rows: int = Field(
        10,
        ge=2,
        description=(
            "Number of rows to scan from the top of the CSV when searching for the real "
            "column-header row.  Arbin files typically have 2 metadata rows before the "
            "header, so the default of 10 is conservative."
        ),
    )
    min_numeric_fraction: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of columns that must parse as numeric for a row to be "
            "considered a data row (rather than a header row).  Used by detect_header_row."
        ),
    )

    # --- Logging -----------------------------------------------------------
    log_level: str = Field(
        "INFO",
        description="Python logging level.  One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )

    # --- Validators --------------------------------------------------------

    @field_validator("selected_plot_families")
    @classmethod
    def _validate_plot_families(cls, v: List[str]) -> List[str]:
        unknown = set(v) - set(PLOT_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown plot families: {unknown}. Valid choices: {PLOT_FAMILIES}"
            )
        return v

    @field_validator("output_formats")
    @classmethod
    def _validate_output_formats(cls, v: List[str]) -> List[str]:
        unknown = set(v) - set(OUTPUT_FORMATS)
        if unknown:
            raise ValueError(
                f"Unknown output formats: {unknown}. Valid choices: {OUTPUT_FORMATS}"
            )
        return v

    @field_validator("theme")
    @classmethod
    def _validate_theme(cls, v: str) -> str:
        if v not in THEMES:
            raise ValueError(f"Unknown theme '{v}'. Valid choices: {THEMES}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"Unknown log level '{v}'. Valid choices: {valid}")
        return v_upper

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Config I/O helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> BatteryPlotConfig:
    """
    Load a ``BatteryPlotConfig`` from a YAML or TOML file.

    The function first tries to parse the file as YAML (using PyYAML if
    available).  If YAML parsing raises an error *or* PyYAML is not installed,
    it falls back to TOML parsing (using the standard-library ``tomllib`` on
    Python ≥ 3.11, otherwise ``tomli``).

    Parameters
    ----------
    path:
        Filesystem path to the config file.  Both ``.yaml`` / ``.yml`` and
        ``.toml`` suffixes are accepted (but the content, not the extension,
        determines which parser is tried first).

    Returns
    -------
    BatteryPlotConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as either YAML or TOML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: Dict[str, Any] = {}

    # --- Try YAML first ---
    try:
        import yaml  # type: ignore[import]

        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.debug("Loaded config from YAML: %s", path)
    except ImportError:
        logger.debug("PyYAML not installed; falling back to TOML parser.")
        raw = _load_toml(path)
    except Exception as yaml_err:
        logger.debug("YAML parsing failed (%s); trying TOML.", yaml_err)
        try:
            raw = _load_toml(path)
        except Exception as toml_err:
            raise ValueError(
                f"Could not parse '{path}' as YAML ({yaml_err}) or TOML ({toml_err})."
            ) from toml_err

    # Convert any string paths to Path objects before validation
    for key in ("input_dir", "output_dir"):
        if key in raw and isinstance(raw[key], str):
            raw[key] = Path(raw[key])

    return BatteryPlotConfig(**raw)


def _load_toml(path: Path) -> Dict[str, Any]:
    """Load a TOML file, returning its contents as a dict."""
    try:
        import tomllib  # Python ≥ 3.11
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Neither 'tomllib' (Python ≥ 3.11) nor 'tomli' is available. "
                "Install 'tomli' to read TOML config files."
            ) from exc

    with path.open("rb") as fh:
        return tomllib.load(fh)


def default_config() -> BatteryPlotConfig:
    """
    Return a ``BatteryPlotConfig`` instance with all default values.

    Useful for programmatic access to the canonical defaults without reading
    any file.
    """
    return BatteryPlotConfig()


def save_default_config(path: Path) -> None:
    """
    Write a commented YAML example config file to *path*.

    The generated file includes all supported fields with their default values
    and descriptive comments.  It is intended as a starting-point template for
    users.

    Parameters
    ----------
    path:
        Destination path for the example YAML file (e.g. ``config.example.yaml``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = """\
# batteryplot example configuration
# Generated by batteryplot.config.save_default_config()
#
# All fields are optional; the pipeline runs with defaults if no config is supplied.

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# Directory containing raw CSV files from the battery cycler.
input_dir: "input"

# Root directory for all generated plots and tables.
output_dir: "output"

# ---------------------------------------------------------------------------
# Cell / electrode parameters
# ---------------------------------------------------------------------------

# Rated (nominal) cell capacity in ampere-hours.
# Required for C-rate and capacity-retention calculations.
# nominal_capacity_ah: 3.0

# Mass of active electrode material in grams.
# Required for gravimetric specific-capacity (mAh g⁻¹) and Ragone plots.
# active_mass_g: 2.5

# Geometric electrode area in cm².
# Required for areal-capacity normalisation.
# electrode_area_cm2: 1.2695

# Bulk density of the active material in g cm⁻³.
# Used for volumetric normalisation only.
# density_g_cm3: 2.1

# ---------------------------------------------------------------------------
# Plot control
# ---------------------------------------------------------------------------

# Explicit cycle indices to include in voltage-profile overlay plots.
# If omitted, the pipeline auto-selects the first, middle, and last cycles.
# representative_cycles: [1, 50, 100]

# Plot families to generate.
# Available families: voltage_profiles, cycle_summary, rate_capability,
#                     pulse_resistance, ragone, qa
selected_plot_families:
  - voltage_profiles
  - cycle_summary
  - rate_capability
  - pulse_resistance
  - ragone
  - qa

# Output file formats for every generated figure.
# Supported: svg, pdf, png
output_formats:
  - svg
  - pdf

# Visual theme: "publication" (clean, paper-ready) or "dark"
theme: "publication"

# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

# Overwrite existing output files (true) or skip them (false).
overwrite: true

# ---------------------------------------------------------------------------
# CSV header detection
# ---------------------------------------------------------------------------

# Number of rows to scan from the top of the CSV for the header row.
header_search_rows: 10

# Minimum fraction of columns that must be numeric for a row to be classified
# as a data row (rather than the header row).
min_numeric_fraction: 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Python logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level: "INFO"
"""
    path.write_text(content, encoding="utf-8")
    logger.info("Saved example config to: %s", path)
