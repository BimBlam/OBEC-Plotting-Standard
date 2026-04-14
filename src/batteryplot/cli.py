"""
batteryplot.cli
===============
Command-line interface for batteryplot using typer.

Commands
--------
run          Process all (or selected) CSV files and generate plots + Excel.
inspect      Parse a single CSV and report detected columns/cycles without plotting.
init-config  Write a config.example.yaml to disk.
list-plots   Print a table of all registered plots.
validate     Discover CSVs and report column mapping quality.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

# Optional rich import — used for pretty tables if available
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None

app = typer.Typer(
    name="batteryplot",
    help="Battery test data plotting and analysis toolkit.",
    add_completion=False,
    no_args_is_help=True,
)

logger = logging.getLogger("batteryplot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    """Configure root batteryplot logger."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("batteryplot")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(numeric)


def _load_config(config_path: Optional[Path], input_dir: Optional[Path], output_dir: Optional[Path]):
    """
    Load config from file (or defaults) and apply CLI overrides.

    Priority: CLI flags > config file > defaults.
    """
    from batteryplot.config import load_config, default_config, BatteryPlotConfig

    if config_path is not None:
        cfg = load_config(config_path)
    else:
        # Look for config.yaml in CWD
        cwd_config = Path.cwd() / "config.yaml"
        if cwd_config.exists():
            cfg = load_config(cwd_config)
            typer.echo(f"Loaded config from: {cwd_config}", err=True)
        else:
            cfg = default_config()

    # Apply CLI overrides (rebuild with updated fields)
    overrides = {}
    if input_dir is not None:
        overrides["input_dir"] = input_dir
    if output_dir is not None:
        overrides["output_dir"] = output_dir

    if overrides:
        cfg = cfg.model_copy(update=overrides)

    return cfg


def _print_table_rich(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    """Print a rich table to stdout."""
    table = Table(title=title, show_lines=False, header_style="bold cyan")
    for h in headers:
        table.add_column(h, no_wrap=False)
    for row in rows:
        table.add_row(*[str(v) for v in row])
    _console.print(table)


def _print_table_plain(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    """Print a plain-text table to stdout."""
    if title:
        typer.echo(f"\n{title}")
        typer.echo("=" * len(title))
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    typer.echo(fmt.format(*headers))
    typer.echo("  ".join("-" * w for w in col_widths))
    for row in rows:
        typer.echo(fmt.format(*[str(v) for v in row]))


def _print_table(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    if _RICH:
        _print_table_rich(headers, rows, title)
    else:
        _print_table_plain(headers, rows, title)


# ---------------------------------------------------------------------------
# batteryplot run
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", "-i", help="Directory containing CSV files."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Root output directory."),
    files: Optional[List[str]] = typer.Option(None, "--files", "-f", help="Specific file names to process (repeatable)."),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite", help="Overwrite existing outputs."),
) -> None:
    """
    Process all (or selected) CSV files in the input directory.

    Generates plots, Excel workbooks, and a batch summary table.
    """
    cfg = _load_config(config, input_dir, output_dir)
    _setup_logging(cfg.log_level)

    if not overwrite:
        cfg = cfg.model_copy(update={"overwrite": False})

    typer.echo(f"Input:  {cfg.input_dir}")
    typer.echo(f"Output: {cfg.output_dir}")

    from batteryplot.io import run_batch

    batch_df = run_batch(cfg, specific_files=list(files) if files else None)

    if batch_df.empty:
        typer.echo("No files were processed.")
        raise typer.Exit(code=1)

    # Print summary table
    headers = ["Cell", "Cycles", "Data Points", "Real Plots", "Placeholders", "Status"]
    rows = [
        [
            row.get("cell_name", ""),
            row.get("n_cycles", ""),
            row.get("n_data_points", ""),
            row.get("plots_generated", ""),
            row.get("plots_placeholder", ""),
            row.get("status", ""),
        ]
        for _, row in batch_df.iterrows()
    ]
    _print_table(headers, rows, title="Batch Processing Summary")

    typer.echo(
        f"\nTotal: {len(batch_df)} cell(s) | "
        f"{batch_df['plots_generated'].sum()} real plots | "
        f"{batch_df['plots_placeholder'].sum()} placeholders"
    )
    typer.echo(f"Results saved to: {cfg.output_dir}")


# ---------------------------------------------------------------------------
# batteryplot inspect
# ---------------------------------------------------------------------------


@app.command("inspect")
def cmd_inspect(
    file: Path = typer.Argument(..., help="CSV file to inspect."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
) -> None:
    """
    Parse a single CSV file and report column mapping and data structure.

    Does NOT generate any plots.
    """
    cfg = _load_config(config, None, None)
    _setup_logging(cfg.log_level)

    if not file.exists():
        typer.echo(f"ERROR: File not found: {file}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nInspecting: {file}")
    typer.echo("=" * 60)

    from batteryplot.parsing import load_csv, build_analysis_df, detect_header_row

    try:
        header_row_idx, raw_cols = detect_header_row(
            file,
            max_scan=cfg.header_search_rows,
            min_numeric_fraction=cfg.min_numeric_fraction,
        )
        typer.echo(f"Detected header at row index: {header_row_idx}")
        typer.echo(f"Total columns in header:      {len(raw_cols)}")
    except (ValueError, Exception) as e:
        typer.echo(f"Header detection: {e}", err=True)
        header_row_idx = 0
        raw_cols = []

    raw_df, column_map, metadata = load_csv(
        file,
        cfg,
    )
    analysis_df = build_analysis_df(raw_df, column_map)

    typer.echo(f"Data rows:                    {len(raw_df)}")

    # Metadata
    if metadata:
        typer.echo("\n--- Metadata (from header rows) ---")
        for k, v in metadata.items():
            typer.echo(f"  {k}: {v}")

    # Column mapping
    mapped_rows = [[raw, canon] for raw, canon in column_map.items()]
    unmapped = [c for c in raw_df.columns if c not in column_map]

    typer.echo(f"\n--- Mapped columns ({len(mapped_rows)}) ---")
    _print_table(["Raw Column", "Canonical Name"], mapped_rows)

    if unmapped:
        typer.echo(f"\n--- Unmapped columns ({len(unmapped)}) ---")
        for c in unmapped:
            typer.echo(f"  {c}")

    # Cycle detection
    if "cycle_index" in analysis_df.columns:
        n_cycles = analysis_df["cycle_index"].nunique()
        typer.echo(f"\nCycles detected: {n_cycles}")
    else:
        typer.echo("\nCycles detected: N/A (no cycle_index column mapped)")

    # Available canonical columns
    canonical_cols = [c for c in analysis_df.columns if c in set(column_map.values())]
    typer.echo(f"\nAvailable canonical columns ({len(canonical_cols)}):")
    for col in canonical_cols:
        n_valid = analysis_df[col].notna().sum()
        pct = n_valid / max(len(analysis_df), 1) * 100
        typer.echo(f"  {col:<35} {n_valid:>8} rows ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# batteryplot init-config
# ---------------------------------------------------------------------------


@app.command("init-config")
def cmd_init_config(
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output", "-o",
        help="Destination path for the config file.",
    ),
) -> None:
    """
    Write a commented example configuration file (config.yaml).

    Edit the generated file to match your data and run `batteryplot run`.
    """
    from batteryplot.config import save_default_config

    # Use the full example config from the task spec rather than the
    # minimal default; save_default_config writes a fully commented YAML.
    save_default_config(output)
    typer.echo(f"Config written to {output}")


# ---------------------------------------------------------------------------
# batteryplot list-plots
# ---------------------------------------------------------------------------


@app.command("list-plots")
def cmd_list_plots() -> None:
    """
    Print a table of all registered plots with their families and required columns.
    """
    from batteryplot.plots.registry import PLOT_REGISTRY

    headers = ["Key", "Family", "Required Columns", "Description"]
    rows = []
    for spec in PLOT_REGISTRY:
        req_cols = ", ".join(spec.required_columns) if spec.required_columns else "(none)"
        desc = spec.description[:60] + "…" if len(spec.description) > 60 else spec.description
        rows.append([spec.key, spec.family, req_cols, desc])

    _print_table(headers, rows, title="Available Plots")
    typer.echo(f"\n{len(PLOT_REGISTRY)} plots in registry.")


# ---------------------------------------------------------------------------
# batteryplot validate
# ---------------------------------------------------------------------------


@app.command("validate")
def cmd_validate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml file."),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", "-i", help="Directory containing CSV files."),
) -> None:
    """
    Discover CSV files and report column mapping quality for each.

    Parses each file without generating plots; prints a summary table of
    mapped/unmapped columns and estimated available plots per file.
    """
    cfg = _load_config(config, input_dir, None)
    _setup_logging(cfg.log_level)

    from batteryplot.io import discover_csv_files
    from batteryplot.parsing import load_csv, build_analysis_df
    from batteryplot.transforms import compute_cycle_summary, detect_pulse_segments
    from batteryplot.summaries import build_plot_availability

    try:
        csv_files = discover_csv_files(cfg.input_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1)

    if not csv_files:
        typer.echo(f"No CSV files found in: {cfg.input_dir}")
        raise typer.Exit(code=0)

    typer.echo(f"Found {len(csv_files)} CSV file(s) in {cfg.input_dir}\n")

    summary_rows = []
    for csv_path in csv_files:
        try:
            raw_df, column_map, metadata = load_csv(
                csv_path,
                cfg,
            )
            analysis_df = build_analysis_df(raw_df, column_map)
            cycle_summary = compute_cycle_summary(analysis_df, cfg)
            pulse_df = detect_pulse_segments(analysis_df)
            availability = build_plot_availability(
                df=analysis_df,
                cycle_summary=cycle_summary,
                pulse_df=pulse_df,
                config=cfg,
            )
            n_mapped = len(column_map)
            n_unmapped = len(raw_df.columns) - n_mapped
            n_avail = int(availability["available"].sum()) if not availability.empty else 0
            n_total = len(availability)
            summary_rows.append([
                csv_path.name,
                str(n_mapped),
                str(n_unmapped),
                f"{n_avail}/{n_total}",
                "OK",
            ])
        except Exception as exc:
            summary_rows.append([
                csv_path.name,
                "N/A",
                "N/A",
                "N/A",
                f"ERROR: {exc}",
            ])

    headers = ["Filename", "Mapped Cols", "Unmapped Cols", "Available Plots", "Status"]
    _print_table(headers, summary_rows, title="Validation Summary")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
