"""
batteryplot.utils.paths
=======================
Path manipulation helpers used throughout the batteryplot pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_stem(name: str) -> str:
    """
    Convert a filename stem to a safe directory name.

    Replaces characters that are illegal on Windows (``< > : " / \\ | ? *``),
    control characters (0x00–0x1F), whitespace, and dots with underscores.
    Leading/trailing underscores are stripped.  Returns ``"unnamed_cell"`` for
    empty or all-special-character inputs.

    Parameters
    ----------
    name:
        The raw filename stem (i.e. the filename without its extension).

    Returns
    -------
    str
        A filesystem-safe directory name derived from *name*.

    Examples
    --------
    >>> sanitize_stem("Cell 01: NMC/Graphite (2024)")
    'Cell_01__NMC_Graphite_(2024)'
    >>> sanitize_stem("  ")
    'unnamed_cell'
    """
    # Replace forbidden characters and control characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    # Replace whitespace and dot runs with a single underscore
    name = re.sub(r"[\s.]+", "_", name)
    # Strip leading/trailing underscores
    name = name.strip("_")
    return name or "unnamed_cell"


def ensure_dir(path: Path) -> Path:
    """
    Create *path* (and all intermediate parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The same *path* object (for chaining convenience).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def cell_output_dir(output_dir: Path, csv_path: Path) -> Path:
    """
    Derive a per-cell output subdirectory from the CSV filename.

    The subdirectory name is the sanitised stem of *csv_path* placed directly
    under *output_dir*.  The directory is **not** created by this function;
    call :func:`ensure_dir` on the result if you need it to exist on disk.

    Parameters
    ----------
    output_dir:
        Root output directory (e.g. ``Path("output")``).
    csv_path:
        Path to the raw CSV file being processed.

    Returns
    -------
    Path
        ``output_dir / sanitize_stem(csv_path.stem)``

    Examples
    --------
    >>> cell_output_dir(Path("output"), Path("data/Cell_01.csv"))
    PosixPath('output/Cell_01')
    """
    stem = sanitize_stem(csv_path.stem)
    return output_dir / stem
