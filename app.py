"""
BatteryPlot Streamlit GUI
=========================
Cross-platform web interface for battery test data plotting and analysis.

Usage
-----
    streamlit run app.py

The app assumes the ``batteryplot`` package is importable (install with
``pip install -e .`` from the repo root).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Ensure local src/ is on path (fallback when package is not pip-installed)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.resolve()
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from batteryplot.config import (
        PLOT_FAMILIES,
        OUTPUT_FORMATS,
        THEMES,
        BatteryPlotConfig,
        load_config,
    )
    from batteryplot.io import discover_input_files, run_batch
except ImportError as exc:  # pragma: no cover
    st.error(
        f"Cannot import ``batteryplot``. Please install the package first:\n\n"
        f"    pip install -e .\n\n({exc})"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BatteryPlot", layout="wide")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "input_dir": "input",
    "output_dir": "output",
    "nominal_capacity_ah": 0.0,
    "active_mass_g": 0.0,
    "electrode_area_cm2": 0.0,
    "density_g_cm3": 0.0,
    "selected_plot_families": list(PLOT_FAMILIES),
    "output_formats": ["svg", "pdf"],
    "theme": "publication",
    "overwrite": True,
    "header_search_rows": 10,
    "min_numeric_fraction": 0.5,
    "log_level": "INFO",
    "batch_df": None,
    "last_config": None,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _apply_config_to_state(cfg: BatteryPlotConfig) -> None:
    """Copy a ``BatteryPlotConfig`` into Streamlit session state."""
    st.session_state.input_dir = str(cfg.input_dir)
    st.session_state.output_dir = str(cfg.output_dir)
    st.session_state.nominal_capacity_ah = cfg.nominal_capacity_ah or 0.0
    st.session_state.active_mass_g = cfg.active_mass_g or 0.0
    st.session_state.electrode_area_cm2 = cfg.electrode_area_cm2 or 0.0
    st.session_state.density_g_cm3 = cfg.density_g_cm3 or 0.0
    st.session_state.selected_plot_families = list(cfg.selected_plot_families)
    st.session_state.output_formats = list(cfg.output_formats)
    st.session_state.theme = cfg.theme
    st.session_state.overwrite = cfg.overwrite
    st.session_state.header_search_rows = cfg.header_search_rows
    st.session_state.min_numeric_fraction = cfg.min_numeric_fraction
    st.session_state.log_level = cfg.log_level


def _build_config_from_state() -> BatteryPlotConfig:
    """Build a ``BatteryPlotConfig`` from the current session state."""
    kwargs: dict = {
        "input_dir": Path(st.session_state.input_dir),
        "output_dir": Path(st.session_state.output_dir),
        "selected_plot_families": st.session_state.selected_plot_families,
        "output_formats": st.session_state.output_formats,
        "theme": st.session_state.theme,
        "overwrite": st.session_state.overwrite,
        "header_search_rows": st.session_state.header_search_rows,
        "min_numeric_fraction": st.session_state.min_numeric_fraction,
        "log_level": st.session_state.log_level,
    }

    if st.session_state.nominal_capacity_ah > 0.0:
        kwargs["nominal_capacity_ah"] = st.session_state.nominal_capacity_ah
    if st.session_state.active_mass_g > 0.0:
        kwargs["active_mass_g"] = st.session_state.active_mass_g
    if st.session_state.electrode_area_cm2 > 0.0:
        kwargs["electrode_area_cm2"] = st.session_state.electrode_area_cm2
    if st.session_state.density_g_cm3 > 0.0:
        kwargs["density_g_cm3"] = st.session_state.density_g_cm3

    return BatteryPlotConfig(**kwargs)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔋 BatteryPlot")
st.caption("Cross-platform GUI for battery test data plotting and analysis")

with st.sidebar:
    st.header("Directories")
    st.text_input("Input Directory", key="input_dir", help="Folder containing raw CSV / Excel files")
    st.text_input("Output Directory", key="output_dir", help="Folder where results are written")

    st.divider()

    _col_load, _col_save = st.columns(2)
    with _col_load:
        if st.button("Load config.yaml", use_container_width=True):
            _cfg_path = Path("config.yaml")
            if _cfg_path.exists():
                try:
                    _loaded = load_config(_cfg_path)
                    _apply_config_to_state(_loaded)
                    st.success("Loaded config.yaml")
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Load failed: {_exc}")
            else:
                st.warning("config.yaml not found")

    with _col_save:
        if st.button("Save config.yaml", use_container_width=True):
            try:
                _cfg = _build_config_from_state()
                _data = _cfg.model_dump()
                # Convert Path objects → str for clean YAML
                for _k in ("input_dir", "output_dir"):
                    _data[_k] = str(_data[_k])
                _cfg_path = Path("config.yaml")
                with _cfg_path.open("w", encoding="utf-8") as _fh:
                    yaml.dump(_data, _fh, default_flow_style=False, sort_keys=False)
                st.success("Saved config.yaml")
            except Exception as _exc:
                st.error(f"Save failed: {_exc}")

    st.divider()

    st.header("Cell Parameters")
    st.number_input(
        "Nominal Capacity (Ah)",
        key="nominal_capacity_ah",
        min_value=0.0,
        format="%.6f",
        help="Required for C-rate calculation. Set to 0 to disable.",
    )
    st.number_input(
        "Active Mass (g)",
        key="active_mass_g",
        min_value=0.0,
        format="%.6f",
        help="Required for gravimetric normalisation. Set to 0 to disable.",
    )
    st.number_input(
        "Electrode Area (cm²)",
        key="electrode_area_cm2",
        min_value=0.0,
        format="%.6f",
        help="Required for areal normalisation. Set to 0 to disable.",
    )
    st.number_input(
        "Density (g/cm³)",
        key="density_g_cm3",
        min_value=0.0,
        format="%.6f",
        help="Required for volumetric Ragone. Set to 0 to disable.",
    )

    st.divider()

    st.header("Plot Control")
    st.multiselect("Plot Families", options=PLOT_FAMILIES, key="selected_plot_families")
    st.multiselect("Output Formats", options=OUTPUT_FORMATS, key="output_formats")
    st.selectbox("Theme", options=THEMES, key="theme")

    st.divider()

    st.header("Behaviour")
    st.toggle("Overwrite existing outputs", key="overwrite")
    st.number_input(
        "Header Search Rows",
        key="header_search_rows",
        min_value=2,
        max_value=100,
        help="Rows to scan when auto-detecting the CSV header",
    )
    st.slider(
        "Min Numeric Fraction",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="min_numeric_fraction",
        help="Minimum fraction of numeric columns for a row to be considered data",
    )
    st.selectbox("Log Level", options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], key="log_level")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
cfg = _build_config_from_state()

st.header("Input Files")
if st.button("🔄 Refresh File List", type="secondary"):
    st.rerun()

try:
    _files = discover_input_files(cfg.input_dir)
    if _files:
        st.write(f"Found **{len(_files)}** supported file(s) in ``{cfg.input_dir}``:")
        _file_df = pd.DataFrame(
            {"Filename": [f.name for f in _files], "Extension": [f.suffix for f in _files]}
        )
        st.dataframe(_file_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            f"No supported files found in ``{cfg.input_dir}``. "
            "Add .csv, .txt, .xls, or .xlsx files to process."
        )
except FileNotFoundError:
    st.error(f"Input directory not found: ``{cfg.input_dir}``")

st.divider()

# Run button
if st.button("▶️ Run BatteryPlot", type="primary", use_container_width=True):
    _progress = st.progress(0, text="Starting batch processing...")
    try:
        # run_batch handles discovery, processing, and summary export
        batch_df = run_batch(cfg)
        st.session_state.batch_df = batch_df
        st.session_state.last_config = cfg
        _progress.empty()
        st.success("Batch processing complete!")
        st.rerun()
    except Exception as exc:
        _progress.empty()
        st.error(f"Batch processing failed: {exc}")
        st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.batch_df is not None:
    df = st.session_state.batch_df

    if df.empty:
        st.info("No files were processed. Check the input directory and try again.")
    else:
        st.divider()
        st.header("Batch Summary")
        st.dataframe(df, use_container_width=True, hide_index=True)

        _total_cells = len(df)
        _total_real = int(df["plots_generated"].sum()) if "plots_generated" in df.columns else 0
        _total_ph = int(df["plots_placeholder"].sum()) if "plots_placeholder" in df.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Cells Processed", _total_cells)
        c2.metric("Real Plots", _total_real)
        c3.metric("Placeholders", _total_ph)

        st.header("Per-Cell Results")
        for _, row in df.iterrows():
            cell_name = row.get("cell_name", "Unknown")
            status = row.get("status", "unknown")
            icon = "✅" if status == "ok" else "❌" if status == "error" else "⚠️"

            with st.expander(f"{icon} {cell_name} — {status.upper()}"):
                col_left, col_right = st.columns([1, 3])

                with col_left:
                    st.write(f"**Cycles:** {row.get('n_cycles', 'N/A')}")
                    st.write(f"**Data Points:** {row.get('n_data_points', 'N/A')}")
                    st.write(f"**Real Plots:** {row.get('plots_generated', 0)}")
                    st.write(f"**Placeholders:** {row.get('plots_placeholder', 0)}")

                    excel_path = row.get("excel_path")
                    if excel_path and Path(excel_path).exists():
                        st.download_button(
                            label="📊 Download Excel",
                            data=Path(excel_path).read_bytes(),
                            file_name=Path(excel_path).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"excel_{cell_name}",
                        )

                    log_path = row.get("log_path")
                    if log_path and Path(log_path).exists():
                        st.download_button(
                            label="📄 Download Log",
                            data=Path(log_path).read_bytes(),
                            file_name=Path(log_path).name,
                            mime="text/plain",
                            key=f"log_{cell_name}",
                        )

                with col_right:
                    out_cfg = st.session_state.last_config
                    if out_cfg is not None:
                        plots_dir = out_cfg.output_dir / cell_name / "plots"
                        if plots_dir.exists():
                            svgs = sorted(plots_dir.glob("*.svg"))
                            if svgs:
                                for i in range(0, len(svgs), 3):
                                    pc = st.columns(3)
                                    for j, svg in enumerate(svgs[i : i + 3]):
                                        with pc[j]:
                                            st.caption(svg.stem)
                                            st.image(str(svg), use_container_width=True)
                            else:
                                st.info("No plot files found.")
                        else:
                            st.info("Plot directory not found.")

                    warnings = row.get("warnings", "")
                    if warnings:
                        st.warning(f"Warnings: {warnings}")
