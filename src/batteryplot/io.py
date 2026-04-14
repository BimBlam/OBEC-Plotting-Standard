"""
batteryplot.io
==============
High-level file discovery and per-cell processing orchestration.

This module provides:
- discover_csv_files: scan a directory for CSV files
- process_cell: full pipeline for a single CSV file
- run_batch: process all (or selected) CSV files and return a summary DataFrame
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd

from batteryplot.utils.paths import cell_output_dir, ensure_dir

logger = logging.getLogger("batteryplot")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_csv_files(input_dir: Path) -> List[Path]:
    """
    Scan *input_dir* for CSV files, ignoring hidden and temp files.

    Rules:
    - Filename must end with ``.csv`` (case-insensitive via glob).
    - Filenames starting with ``.`` or ``~`` are ignored.
    - Results are sorted alphabetically by filename.

    Parameters
    ----------
    input_dir : Path
        Directory to scan.

    Returns
    -------
    List[Path]
        Sorted list of discovered CSV paths.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_csv = list(input_dir.glob("*.csv"))
    # Also try case-insensitive (some systems may have .CSV)
    all_csv += [p for p in input_dir.glob("*.CSV") if p not in all_csv]

    # Filter out hidden/temp files
    filtered = [
        p for p in all_csv
        if not p.name.startswith(".") and not p.name.startswith("~")
    ]

    # Sort alphabetically
    filtered.sort(key=lambda p: p.name.lower())

    logger.info("Found %d CSV file(s) in %s", len(filtered), input_dir)
    return filtered


# ---------------------------------------------------------------------------
# Single-cell processing
# ---------------------------------------------------------------------------


def _setup_file_log_handler(log_dir: Path, cell_name: str) -> tuple:
    """
    Add a FileHandler to the root batteryplot logger for the current cell.

    Returns (handler, log_path) so the caller can remove the handler afterward.
    """
    log_path = log_dir / f"{cell_name}_processing.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.getLogger("batteryplot").addHandler(handler)
    return handler, log_path


def process_cell(
    csv_path: Path,
    config,  # BatteryPlotConfig
    output_base: Path,
    force_overwrite: bool = True,
) -> dict:
    """
    Full pipeline for a single cell CSV.

    Steps:
    1.  Create output directory structure (plots/, data/, logs/).
    2.  Set up per-file log handler.
    3.  Load and parse CSV → raw_df, column_map, metadata.
    4.  Build analysis DataFrame.
    5.  Label charge/discharge segments.
    6.  Compute cycle summary.
    7.  Detect pulse segments.
    8.  Compute derived quantities (c_rate, specific_capacity if configured).
    9.  Build plot availability table.
    10. Generate all plots (real or placeholder).
    11. Export Excel workbook.
    12. Write per-file processing log summary.
    13. Return summary dict.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV file.
    config : BatteryPlotConfig
        Configuration object with all pipeline settings.
    output_base : Path
        Root output directory; per-cell subdirectories are created below this.
    force_overwrite : bool
        If True, overwrite existing output files.

    Returns
    -------
    dict
        Summary with keys: cell_name, n_cycles, n_data_points,
        plots_generated, plots_placeholder, excel_path, log_path, warnings.
    """
    # Lazy imports to avoid circular dependencies
    from batteryplot.parsing import load_csv, build_analysis_df
    from batteryplot.transforms import (
        label_charge_discharge,
        compute_cycle_summary,
        detect_pulse_segments,
        compute_crate,
        compute_specific_capacity,
    )
    from batteryplot.summaries import build_plot_availability
    from batteryplot.excel_export import export_excel
    from batteryplot.plots.registry import PLOT_REGISTRY, REGISTRY_BY_KEY
    from batteryplot.plots.voltage_profiles import plot_voltage_vs_capacity, plot_voltage_vs_time
    from batteryplot.plots.cycle_summary import (
        plot_capacity_retention,
        plot_coulombic_efficiency,
        plot_dcir_vs_cycle,
    )
    from batteryplot.plots.rate_capability import plot_rate_capability, plot_rate_voltage_profiles
    from batteryplot.plots.pulse_resistance import plot_dcir_vs_current, plot_pulse_analysis
    from batteryplot.plots.ragone import plot_ragone
    from batteryplot.plots.qa import (
        plot_temperature_vs_time,
        plot_current_voltage_overview,
        plot_data_availability,
    )

    csv_path = Path(csv_path)
    output_base = Path(output_base)
    warnings: List[str] = []

    # ------------------------------------------------------------------
    # 1. Output directory structure
    # ------------------------------------------------------------------
    cell_dir = cell_output_dir(output_base, csv_path)
    plots_dir = ensure_dir(cell_dir / "plots")
    data_dir = ensure_dir(cell_dir / "data")
    log_dir = ensure_dir(cell_dir / "logs")
    cell_name = cell_dir.name

    # ------------------------------------------------------------------
    # 2. Per-file log handler
    # ------------------------------------------------------------------
    log_handler, log_path = _setup_file_log_handler(log_dir, cell_name)
    logger.info("=" * 60)
    logger.info("Processing cell: %s  (%s)", cell_name, csv_path)
    logger.info("=" * 60)

    try:
        # ------------------------------------------------------------------
        # 3. Load CSV
        # ------------------------------------------------------------------
        raw_df, column_map, metadata = load_csv(
            csv_path,
            config,
        )

        # ------------------------------------------------------------------
        # 4. Build analysis DataFrame
        # ------------------------------------------------------------------
        analysis_df = build_analysis_df(raw_df, column_map)
        n_data_points = len(analysis_df)

        # ------------------------------------------------------------------
        # 5. Label charge / discharge
        # ------------------------------------------------------------------
        analysis_df = label_charge_discharge(analysis_df)

        # ------------------------------------------------------------------
        # 6. Cycle summary
        # ------------------------------------------------------------------
        cycle_summary = compute_cycle_summary(
            analysis_df,
            config,
        )
        n_cycles = len(cycle_summary)

        # ------------------------------------------------------------------
        # 7. Pulse segments
        # ------------------------------------------------------------------
        pulse_df = detect_pulse_segments(analysis_df)

        # ------------------------------------------------------------------
        # 8. Derived quantities
        # ------------------------------------------------------------------
        if config.nominal_capacity_ah:
            crate_series = compute_crate(analysis_df, config.nominal_capacity_ah)
            analysis_df = analysis_df.copy()
            analysis_df["c_rate"] = crate_series.values
            # Also add c_rate to cycle_summary if mean current available
            if "c_rate" in analysis_df.columns and "cycle_index" in analysis_df.columns:
                cr_per_cycle = (
                    analysis_df.dropna(subset=["c_rate", "cycle_index"])
                    .groupby("cycle_index")["c_rate"]
                    .apply(lambda x: x.abs().median())
                    .reset_index()
                    .rename(columns={"c_rate": "c_rate"})
                )
                if not cycle_summary.empty and "cycle_index" in cycle_summary.columns:
                    cycle_summary = cycle_summary.merge(cr_per_cycle, on="cycle_index", how="left")

        if config.active_mass_g:
            sp_series = compute_specific_capacity(analysis_df, config.active_mass_g)
            analysis_df = analysis_df.copy()
            analysis_df["specific_capacity_mah_g"] = sp_series.values

        # ------------------------------------------------------------------
        # 9. Plot availability
        # ------------------------------------------------------------------
        selected_families = config.selected_plot_families
        plot_availability = build_plot_availability(
            df=analysis_df,
            cycle_summary=cycle_summary,
            pulse_df=pulse_df,
            config=config,
        )

        # ------------------------------------------------------------------
        # 10. Generate all plots
        # ------------------------------------------------------------------
        formats = tuple(config.output_formats)
        rep_cycles = config.representative_cycles

        plots_generated = 0
        plots_placeholder = 0

        # Helper: track generated vs placeholder
        # plot_availability may use 'plot_name' column (existing summaries) or 'key'
        _avail_col = "key" if "key" in plot_availability.columns else "plot_name"
        _title_col = "title" if "title" in plot_availability.columns else "plot_name"

        def _count_results(paths: List[Path], spec_key: str) -> None:
            nonlocal plots_generated, plots_placeholder
            if not paths:
                return
            # Check if it's a real plot or placeholder by checking availability table
            avail_row = plot_availability[plot_availability[_avail_col] == spec_key]
            if avail_row.empty and "plot_name" in plot_availability.columns:
                # Try matching by plot name (partial)
                avail_row = plot_availability[
                    plot_availability["plot_name"].str.contains(
                        spec_key.replace("_", " "), case=False, na=False
                    )
                ]
            is_real = (
                avail_row["available"].any() if not avail_row.empty else False
            )
            if is_real:
                plots_generated += 1
            else:
                plots_placeholder += 1

        # Common positional call signature for all existing plot functions:
        # plot_fn(df, cycle_summary, pulse_df, config, output_dir) -> List[Path]
        _plot_args = (analysis_df, cycle_summary, pulse_df, config, plots_dir)

        # --- Voltage profiles ---
        if "voltage_profiles" in selected_families:
            paths = plot_voltage_vs_capacity(*_plot_args)
            _count_results(paths, "voltage_vs_capacity")

            paths = plot_voltage_vs_time(*_plot_args)
            _count_results(paths, "voltage_vs_time")

        # --- Cycle summary ---
        if "cycle_summary" in selected_families:
            paths = plot_capacity_retention(*_plot_args)
            _count_results(paths, "capacity_retention")

            paths = plot_coulombic_efficiency(*_plot_args)
            _count_results(paths, "coulombic_efficiency")

            paths = plot_dcir_vs_cycle(*_plot_args)
            _count_results(paths, "dcir_vs_cycle")

        # --- Rate capability ---
        if "rate_capability" in selected_families:
            paths = plot_rate_capability(*_plot_args)
            _count_results(paths, "rate_capability")

            paths = plot_rate_voltage_profiles(*_plot_args)
            _count_results(paths, "rate_voltage_profiles")

        # --- Pulse resistance ---
        if "pulse_resistance" in selected_families:
            paths = plot_dcir_vs_current(*_plot_args)
            _count_results(paths, "dcir_vs_current")

            paths = plot_pulse_analysis(*_plot_args)
            _count_results(paths, "pulse_analysis")

        # --- Ragone ---
        if "ragone" in selected_families:
            paths = plot_ragone(*_plot_args)
            _count_results(paths, "ragone")

        # --- QA ---
        if "qa" in selected_families:
            paths = plot_temperature_vs_time(*_plot_args)
            _count_results(paths, "temperature_vs_time")

            paths = plot_current_voltage_overview(*_plot_args)
            _count_results(paths, "current_voltage_overview")

            paths = plot_data_availability(*_plot_args)
            _count_results(paths, "data_availability")

        # ------------------------------------------------------------------
        # 11. Export Excel workbook
        # ------------------------------------------------------------------
        excel_path = data_dir / f"{cell_name}.xlsx"
        try:
            export_excel(
                output_path=excel_path,
                raw_df=raw_df,
                column_map=column_map,
                analysis_df=analysis_df,
                cycle_summary=cycle_summary,
                plot_availability=plot_availability,
                metadata=metadata,
                pulse_df=pulse_df if not pulse_df.empty else None,
            )
        except Exception as exc:
            msg = f"Excel export failed: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            excel_path = None

        # ------------------------------------------------------------------
        # 12. Save data files
        # ------------------------------------------------------------------
        try:
            ts_path = data_dir / "cleaned_timeseries.csv"
            analysis_df.to_csv(ts_path, index=False)
            logger.info("Saved cleaned_timeseries.csv (%d rows)", len(analysis_df))
        except Exception as exc:
            logger.warning("Could not save cleaned_timeseries.csv: %s", exc)

        try:
            if not cycle_summary.empty:
                cs_path = data_dir / "cycle_summary.csv"
                cycle_summary.to_csv(cs_path, index=False)
                logger.info("Saved cycle_summary.csv (%d cycles)", len(cycle_summary))
        except Exception as exc:
            logger.warning("Could not save cycle_summary.csv: %s", exc)

        # ------------------------------------------------------------------
        # 13. Summary dict
        # ------------------------------------------------------------------
        logger.info(
            "Cell %s: %d cycles, %d data points, %d real plots, %d placeholders.",
            cell_name, n_cycles, n_data_points, plots_generated, plots_placeholder,
        )

        return {
            "cell_name": cell_name,
            "n_cycles": n_cycles,
            "n_data_points": n_data_points,
            "plots_generated": plots_generated,
            "plots_placeholder": plots_placeholder,
            "excel_path": str(excel_path) if excel_path else None,
            "log_path": str(log_path),
            "warnings": "; ".join(warnings) if warnings else "",
            "status": "ok",
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Failed to process %s: %s\n%s", csv_path.name, exc, tb)
        return {
            "cell_name": cell_name,
            "n_cycles": 0,
            "n_data_points": 0,
            "plots_generated": 0,
            "plots_placeholder": 0,
            "excel_path": None,
            "log_path": str(log_path),
            "warnings": f"FAILED: {exc}",
            "status": "error",
        }

    finally:
        # Always remove the per-file log handler
        logging.getLogger("batteryplot").removeHandler(log_handler)
        log_handler.close()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def run_batch(
    config,
    specific_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process all (or selected) CSV files in config.input_dir.

    Parameters
    ----------
    config : BatteryPlotConfig
        Pipeline configuration.
    specific_files : list of str, optional
        If given, only process files whose names (without directory) match
        this list.  Matching is case-insensitive on the filename only.

    Returns
    -------
    pd.DataFrame
        Batch summary DataFrame; one row per cell with columns from
        :func:`process_cell`'s return dict.  Also saved as CSV and XLSX
        in config.output_dir.
    """
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)

    # Discover
    csv_files = discover_csv_files(input_dir)

    # Filter if specific files requested
    if specific_files:
        want = {f.lower() for f in specific_files}
        csv_files = [
            p for p in csv_files
            if p.name.lower() in want or p.stem.lower() in want
        ]
        logger.info(
            "Filtered to %d file(s) from specific_files list.", len(csv_files)
        )

    if not csv_files:
        logger.warning("No CSV files to process in %s.", input_dir)
        return pd.DataFrame()

    logger.info("Starting batch: %d cell(s) to process.", len(csv_files))

    summaries = []
    for i, csv_path in enumerate(csv_files, start=1):
        logger.info("--- [%d/%d] %s ---", i, len(csv_files), csv_path.name)
        summary = process_cell(
            csv_path=csv_path,
            config=config,
            output_base=output_dir,
            force_overwrite=config.overwrite,
        )
        summaries.append(summary)

    batch_df = pd.DataFrame(summaries)

    # Totals
    total_cells = len(batch_df)
    total_plots = int(batch_df["plots_generated"].sum()) if "plots_generated" in batch_df else 0
    total_placeholders = int(batch_df["plots_placeholder"].sum()) if "plots_placeholder" in batch_df else 0

    logger.info(
        "Batch complete: %d cells | %d real plots | %d placeholders.",
        total_cells, total_plots, total_placeholders,
    )

    # Save batch summary
    try:
        csv_out = output_dir / "batch_summary.csv"
        batch_df.to_csv(csv_out, index=False)
        logger.info("Saved batch_summary.csv to %s", csv_out)
    except Exception as exc:
        logger.warning("Could not save batch_summary.csv: %s", exc)

    try:
        xlsx_out = output_dir / "batch_summary.xlsx"
        batch_df.to_excel(xlsx_out, index=False, engine="openpyxl")
        logger.info("Saved batch_summary.xlsx to %s", xlsx_out)
    except Exception as exc:
        logger.warning("Could not save batch_summary.xlsx: %s", exc)

    return batch_df
