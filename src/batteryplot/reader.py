"""
batteryplot.reader
==================
Format-agnostic file ingestion for battery cycler exports.

Supported formats
-----------------
- ``.csv``  — comma-separated; Arbin, Maccor CSV exports
- ``.txt``  — tab- or comma-separated; Maccor .txt exports
             Header metadata may include a ``SCap:`` or ``Specific Capacity:``
             row that carries the active mass in g (used if ``active_mass_g``
             is not already set in config).
- ``.xls``  — Legacy Excel 97-2003 (Maccor .xls exports).
             Cells J1 and K1 often carry specific-capacity / mass metadata.
- ``.xlsx`` — Modern Excel.  Same metadata scan as .xls.

All loaders return the same three objects:
    raw_df   : pd.DataFrame  — raw data, all columns as strings
    metadata : dict[str,str] — key/value pairs extracted from file header
    mass_g   : float | None  — active mass in grams extracted from file,
                               or None if not found / not meaningful

The caller (``parsing.load_file``) then hands raw_df to the existing
column-normalisation and alias-matching pipeline unchanged.

Mass extraction and default-mass detection
------------------------------------------
Maccor exports sometimes embed the active mass (in grams) in the file so
that the cycler can compute specific capacity on-the-fly.  However, users
who never changed the default leave it at **1 g**, which is physically
unreasonable for the thin-film / coin cells used here.

Rules:
- If ``active_mass_g`` is already set in config to something other than 1,
  the file-embedded value is logged but not used (config takes precedence).
- If ``active_mass_g`` is 1 (the default), it is flagged as a likely
  default value; a warning is stored in metadata and the plot system will
  display an assumption warning.
- If ``active_mass_g`` is None and the file provides a value, that value
  is returned; the caller can then store it in a copy of config.
- The same 1-g threshold applies to file-embedded values: if the file says
  1 g, we flag it as a likely default rather than using it.

Default mass threshold: 1.0 g (configurable via config.default_mass_threshold_g).
"""
from __future__ import annotations

import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger("batteryplot.reader")

# File extensions this module will attempt to read (lowercase, with dot)
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".txt", ".xls", ".xlsx")

# Regex patterns for mass extraction from text/CSV metadata rows
# Matches lines like:  "SCap: 0.019"  or  "Specific Capacity: 19 mAh/g"
# or "Mass: 0.019 g"
_MASS_PATTERNS: list[re.Pattern] = [
    # Maccor .txt  →  "SCap:  0.019000"  (value in Ah/g → mass in g = capacity_ah / scap_ah_g)
    # We do NOT try to infer mass from SCap directly because we don't know nominal capacity;
    # instead we store SCap as a metadata field and let the user set active_mass_g.
    # However if the row is labeled "Mass" or "Active Mass" we take it directly.
    re.compile(r"^\s*(?:active[\s_-]*)?mass[\s:,=]+([0-9]+\.?[0-9]*)\s*(?:g|gram)?", re.IGNORECASE),
    # "Wt:  0.019"  (Maccor weight field)
    re.compile(r"^\s*wt[\s:,=]+([0-9]+\.?[0-9]*)\s*(?:g|gram)?", re.IGNORECASE),
]

# Regex for the Maccor .txt SCap line.
# Maccor writes:  "SCap:\t0.023331 g"  (active mass in grams, NOT Ah/g).
# We capture the numeric part; the trailing " g" unit is optional but expected.
_SCAP_PATTERN = re.compile(
    r"^\s*scap[\s:,=]+([0-9]+\.?[0-9e\-+]*)\s*(?:g|gram)?",
    re.IGNORECASE,
)

# How large a "default mass" value can be before we treat it as real.
# Any file-embedded or config mass value <= this threshold is flagged.
_DEFAULT_MASS_SENTINEL = 1.0  # grams


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_supported(path: Path) -> bool:
    """Return True if *path* has an extension this reader can handle."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(
    filepath: Path,
    config,              # BatteryPlotConfig
) -> Tuple[pd.DataFrame, Dict[str, str], Optional[float]]:
    """
    Load any supported battery-cycler export file into a raw DataFrame.

    The returned DataFrame has:
    - All column names stripped of leading/trailing whitespace.
    - All cell values as strings (dtype=str) — downstream alias matching
      and numeric coercion is handled by ``parsing.build_analysis_df``.

    Parameters
    ----------
    filepath : Path
        Path to the input file.  Must exist.
    config : BatteryPlotConfig
        Pipeline configuration.  Used for header-detection parameters and
        the default-mass threshold.

    Returns
    -------
    raw_df : pd.DataFrame
        Raw data, column names stripped.
    metadata : dict[str, str]
        Key/value pairs extracted from the file header.
    mass_g : float | None
        Active mass found in the file, or None.
        Already checked against the default-mass sentinel; callers should
        examine the ``"mass_warning"`` key in *metadata* if set.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".csv", ".txt"):
        raw_df, metadata = _load_text_file(filepath, config)
    elif ext in (".xls", ".xlsx"):
        raw_df, metadata = _load_excel_file(filepath, config)
    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'.  "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    mass_g = _resolve_mass(metadata, config)
    return raw_df, metadata, mass_g


# ---------------------------------------------------------------------------
# Text-based loader  (.csv / .txt)
# ---------------------------------------------------------------------------


def _load_text_file(
    filepath: Path,
    config,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a CSV or tab/comma-delimited TXT file.

    Maccor .txt exports are typically tab-separated with a short block of
    key:value metadata rows at the top before the real column header.  The
    same header-detection heuristic used for Arbin CSV files handles both.

    Steps
    -----
    1. Sniff the delimiter (tab vs comma) from the first few data rows.
    2. Detect the header row using the existing heuristic.
    3. Scan pre-header rows for mass / SCap metadata.
    4. Read the file with pandas using the detected delimiter and header row.
    """
    # --- 1. Sniff delimiter ---
    delimiter = _sniff_delimiter(filepath)
    logger.info("Text file '%s': delimiter detected as %r", filepath.name, delimiter)

    # --- 2. Detect header row and collect metadata ---
    header_idx, metadata = _detect_header_and_metadata_text(
        filepath,
        delimiter=delimiter,
        max_scan=config.header_search_rows,
        min_numeric_fraction=config.min_numeric_fraction,
    )
    logger.info(
        "Header row at line %d for '%s'.", header_idx, filepath.name
    )

    # --- 3. Read DataFrame ---
    raw_df = pd.read_csv(
        filepath,
        sep=delimiter,
        skiprows=header_idx,
        header=0,
        dtype=str,
        na_values=["N/A", "n/a", "NA", "", "---"],
        keep_default_na=True,
        # Note: low_memory is not supported by the Python engine; omitted here.
        # dtype=str already prevents mixed-type inference warnings.
        engine="python",   # python engine handles multi-char separators and is more lenient
    )
    raw_df.columns = [c.strip() for c in raw_df.columns]

    # Drop Unnamed columns (artifact of trailing delimiters)
    raw_df = raw_df.loc[:, ~raw_df.columns.str.startswith("Unnamed:")]

    logger.debug("Loaded text file '%s': shape %s", filepath.name, raw_df.shape)
    return raw_df, metadata


def _sniff_delimiter(filepath: Path, n_lines: int = 5) -> str:
    """
    Guess whether the file is tab- or comma-delimited.

    Reads the first *n_lines* non-empty lines and counts tab vs comma
    occurrences.  Returns ``'\\t'`` if tabs dominate, else ``','``.
    """
    tab_count = 0
    comma_count = 0
    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(n_lines):
                line = fh.readline()
                if not line:
                    break
                tab_count   += line.count("\t")
                comma_count += line.count(",")
    except OSError:
        return ","
    return "\t" if tab_count > comma_count else ","


def _detect_header_and_metadata_text(
    filepath: Path,
    delimiter: str,
    max_scan: int,
    min_numeric_fraction: float,
) -> Tuple[int, Dict[str, str]]:
    """
    Detect header row and extract metadata from a text file.

    Returns (header_row_index_0based, metadata_dict).
    """
    lines: list[str] = []
    with filepath.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(max_scan + 2):      # read a couple extra for confirmation
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))

    def _is_numeric(token: str) -> bool:
        t = token.strip()
        if t in ("", "N/A", "n/a", "NA", "nan", "NaN", "---"):
            return True
        try:
            float(t)
            return True
        except ValueError:
            return False

    def _numeric_frac(tokens: list[str]) -> float:
        return sum(_is_numeric(t) for t in tokens) / len(tokens) if tokens else 0.0

    def _non_numeric_count(tokens: list[str]) -> int:
        return sum(1 for t in tokens if not _is_numeric(t))

    header_idx = 0
    for i, line in enumerate(lines[:-1]):
        tokens = [t.strip() for t in line.split(delimiter)]
        if len(tokens) < 3:
            continue
        if _non_numeric_count(tokens) < 3:
            continue
        next_tokens = [t.strip() for t in lines[i + 1].split(delimiter)]
        if _numeric_frac(next_tokens) >= min_numeric_fraction:
            header_idx = i
            break
    else:
        # Fallback: last line with >=3 non-numeric tokens
        for i, line in enumerate(lines):
            tokens = [t.strip() for t in line.split(delimiter)]
            if len(tokens) >= 3 and _non_numeric_count(tokens) >= 3:
                header_idx = i
                break

    metadata = _parse_text_metadata(lines[:header_idx], delimiter)
    return header_idx, metadata


def _parse_text_metadata(lines: list[str], delimiter: str) -> Dict[str, str]:
    """
    Parse pre-header lines as key/value metadata.

    Also scans for SCap and mass fields specific to Maccor .txt exports.
    """
    metadata: Dict[str, str] = {}
    for line in lines:
        # Try delimiter-split  key, value
        parts = [p.strip() for p in line.split(delimiter, maxsplit=1)]
        if len(parts) == 2 and parts[0]:
            key = parts[0].rstrip(":").strip()
            val = parts[1].strip()
            if key:
                metadata[key] = val

        # Also try colon-split regardless of delimiter
        if ":" in line:
            colon_parts = [p.strip() for p in line.split(":", maxsplit=1)]
            key = colon_parts[0].strip()
            val = colon_parts[1].strip() if len(colon_parts) > 1 else ""
            if key and key not in metadata:
                metadata[key] = val

        # SCap row — in Maccor .txt the SCap value IS the active mass in grams.
        # e.g.  "SCap:\t0.023331 g"  →  active_mass_g_from_file = "0.023331"
        # Store it in both "SCap" (for reference) and "active_mass_g_from_file".
        m = _SCAP_PATTERN.match(line)
        if m:
            metadata["SCap"] = m.group(1)
            # Only take as mass if a more-specific "Mass:" row hasn't already set it.
            if "active_mass_g_from_file" not in metadata:
                metadata["active_mass_g_from_file"] = m.group(1)
            logger.info(
                "SCap metadata found (= active mass): %s g", m.group(1)
            )

        # Direct mass row ("Mass:", "Active Mass:", "Wt:") — takes precedence
        # over SCap if both appear.
        for pat in _MASS_PATTERNS:
            m = pat.match(line)
            if m:
                metadata["active_mass_g_from_file"] = m.group(1)
                logger.info("Mass metadata found in file: %s g", m.group(1))
                break

    return metadata


# ---------------------------------------------------------------------------
# Excel loader  (.xls / .xlsx)
# ---------------------------------------------------------------------------


def _load_excel_file(
    filepath: Path,
    config,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a Maccor-style .xls or .xlsx file.

    Maccor XLS exports typically have:
    - A header block in the first few rows (rows 0–N) with key/value pairs
      spread across columns A/B or in merged cells.
    - The real column header at row N+1.
    - Data starting at row N+2.

    The cells J1 and K1 (0-indexed: row 0, columns 9 and 10) sometimes
    contain a specific-capacity label and value respectively.

    We scan all sheets but default to the first sheet, or the sheet named
    "Channel_*" or "Data" if present.
    """
    logger.info("Loading Excel file: %s", filepath.name)

    # openpyxl for xlsx; xlrd for xls (fallback)
    try:
        engine = "openpyxl" if filepath.suffix.lower() == ".xlsx" else "xlrd"
        xl = pd.ExcelFile(filepath, engine=engine)
    except Exception as exc:
        # xlrd may not be installed; try openpyxl as fallback for .xls
        logger.warning("Could not open with primary engine: %s. Trying openpyxl.", exc)
        xl = pd.ExcelFile(filepath, engine="openpyxl")

    # Choose sheet
    sheet_name = _choose_sheet(xl.sheet_names)
    logger.info("Using sheet: %r", sheet_name)

    # Read entire sheet as strings first so we can scan the header block
    raw_all = xl.parse(sheet_name, header=None, dtype=str)
    raw_all = raw_all.fillna("")

    # --- Extract metadata from top rows ---
    metadata: Dict[str, str] = {}
    metadata.update(_scan_excel_metadata(raw_all))

    # --- Detect the real header row ---
    header_idx = _detect_excel_header_row(raw_all, config.min_numeric_fraction)
    logger.info("Excel header row at index %d", header_idx)

    # --- Re-read from header row ---
    raw_df = xl.parse(
        sheet_name,
        header=header_idx,
        dtype=str,
    )
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_df = raw_df.loc[:, ~raw_df.columns.str.startswith("Unnamed:")]
    raw_df = raw_df.fillna(pd.NA)

    logger.debug("Excel loaded: shape %s", raw_df.shape)
    return raw_df, metadata


def _choose_sheet(sheet_names: list[str]) -> str:
    """
    Pick the most likely data sheet from a list of sheet names.

    Priority:
    1. "Channel_*" (Maccor naming convention)
    2. "Data", "Test", "Record" (common names)
    3. First sheet
    """
    for name in sheet_names:
        nl = name.lower()
        if nl.startswith("channel"):
            return name
    for name in sheet_names:
        nl = name.lower()
        if nl in ("data", "test", "record", "results"):
            return name
    return sheet_names[0]


def _scan_excel_metadata(raw_all: pd.DataFrame) -> Dict[str, str]:
    """
    Scan the first 10 rows of an Excel sheet for key/value metadata pairs.

    Specifically looks for:
    - Column A (key) + Column B (value) pairs where A looks like a label.
    - Cells J1 and K1 (row 0, cols 9 and 10) which Maccor uses for
      specific-capacity-related metadata.

    Returns a flat dict of string key → string value.
    """
    metadata: Dict[str, str] = {}
    max_meta_rows = min(10, len(raw_all))

    for row_i in range(max_meta_rows):
        row = raw_all.iloc[row_i]
        # Column A (index 0) as key, column B (index 1) as value
        if len(row) >= 2:
            key = str(row.iloc[0]).strip().rstrip(":").strip()
            val = str(row.iloc[1]).strip()
            if key and key.lower() not in ("nan", "", "none") and val.lower() not in ("nan", "", "none"):
                metadata[key] = val

        # Scan for mass/SCap in any cell of this row
        for cell_val in row:
            cell_str = str(cell_val).strip()
            for pat in _MASS_PATTERNS:
                m = pat.match(cell_str)
                if m:
                    metadata["active_mass_g_from_file"] = m.group(1)
                    logger.info("Mass found in Excel cell: %s g", m.group(1))
            m = _SCAP_PATTERN.match(cell_str)
            if m:
                metadata.setdefault("SCap", m.group(1))

    # Specifically check J1 and K1 (0-indexed row 0, cols 9 and 10).
    # Maccor XLS layout: J1 = "SCap" (label), K1 = mass value in grams.
    # e.g.  J1="SCap", K1=1  means active mass = 1 g (likely the factory default).
    # e.g.  J1="SCap", K1=0.023331  means active mass = 0.023331 g.
    if len(raw_all) > 0:
        row0 = raw_all.iloc[0]
        if len(row0) > 10:
            j1 = str(row0.iloc[9]).strip()
            k1 = str(row0.iloc[10]).strip()
            if j1.lower() not in ("nan", "", "none") and k1.lower() not in ("nan", "", "none"):
                logger.info("Excel J1='%s', K1='%s' (potential mass/SCap metadata)", j1, k1)
                metadata["excel_J1"] = j1
                metadata["excel_K1"] = k1
                # J1 == "SCap" (exact Maccor label) → K1 is the active mass in grams.
                if j1.lower() == "scap":
                    try:
                        metadata["active_mass_g_from_file"] = str(float(k1))
                        metadata["SCap"] = str(float(k1))
                        logger.info(
                            "Excel SCap/mass from J1/K1: %s g", k1
                        )
                    except ValueError:
                        logger.warning(
                            "Could not parse Excel K1 value as float: %r", k1
                        )
                # Fallback: any mass/weight label in J1
                elif re.search(r"mass|wt|weight", j1, re.IGNORECASE):
                    try:
                        metadata["active_mass_g_from_file"] = str(float(k1))
                    except ValueError:
                        pass

    return metadata


def _detect_excel_header_row(
    raw_all: pd.DataFrame,
    min_numeric_fraction: float,
) -> int:
    """
    Find the row index in *raw_all* that is the real column header.

    Uses the same heuristic as the text-file detector: the header row has
    ≥3 non-numeric tokens and the following row is ≥ min_numeric_fraction
    numeric.

    Returns the 0-based row index.
    """
    def _is_numeric(v: str) -> bool:
        v = str(v).strip()
        if v in ("", "nan", "N/A", "NA", "---"):
            return True
        try:
            float(v)
            return True
        except ValueError:
            return False

    n_rows = len(raw_all)
    for i in range(min(n_rows - 1, 15)):
        row = [str(v).strip() for v in raw_all.iloc[i]]
        non_num = sum(1 for v in row if not _is_numeric(v))
        if non_num < 3:
            continue
        next_row = [str(v).strip() for v in raw_all.iloc[i + 1]]
        num_frac = sum(_is_numeric(v) for v in next_row) / max(len(next_row), 1)
        if num_frac >= min_numeric_fraction:
            return i
    # Fallback
    for i in range(min(n_rows, 15)):
        row = [str(v).strip() for v in raw_all.iloc[i]]
        if sum(1 for v in row if not _is_numeric(v)) >= 3:
            return i
    return 0


# ---------------------------------------------------------------------------
# Interactive mass prompt
# ---------------------------------------------------------------------------


def prompt_mass_if_default(
    filepath: Path,
    mass_g: Optional[float],
    config,
) -> Optional[float]:
    """
    Interactively prompt the user when the resolved mass equals the
    default-mass sentinel (1.0 g by default).

    Behaviour
    ---------
    - If *mass_g* is None, or *mass_g* != sentinel, returns *mass_g* unchanged.
    - If *mass_g* == sentinel **and** the process is running in an interactive
      terminal (``sys.stdin.isatty()``): prints a message and waits for input.
      - If the user types a valid positive float, that value is returned.
      - If the user presses Enter (blank), *mass_g* (sentinel) is returned.
    - If the process is **not** interactive (batch/pipe/redirect), the prompt is
      skipped and *mass_g* is returned unchanged (the existing warning in
      metadata still applies).

    Parameters
    ----------
    filepath : Path
        The data file being loaded (used in the prompt message).
    mass_g : float or None
        The mass resolved by ``_resolve_mass``.
    config : BatteryPlotConfig
        Used to read ``default_mass_threshold_g``.

    Returns
    -------
    float or None
        The (possibly user-updated) active mass in grams.
    """
    threshold = getattr(config, "default_mass_threshold_g", _DEFAULT_MASS_SENTINEL)
    if mass_g is None or mass_g != threshold:
        return mass_g

    # Only prompt in a real interactive terminal
    if not sys.stdin.isatty():
        logger.debug(
            "Non-interactive session: skipping mass prompt for '%s'.",
            filepath.name,
        )
        return mass_g

    prompt_text = (
        f"\nSCap/mass field is {mass_g:.4g} g (likely a cycler default) for file"
        f" '{filepath.name}'.\n"
        f"Enter mass in grams (e.g. 0.014), or press Enter to keep {mass_g:.4g} g: "
    )
    try:
        user_input = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print()  # newline after ^C
        return mass_g

    if not user_input:
        logger.info(
            "User kept default mass (%.4g g) for '%s'.", mass_g, filepath.name
        )
        return mass_g

    try:
        new_mass = float(user_input)
    except ValueError:
        logger.warning(
            "Could not parse '%s' as a float; keeping %.4g g for '%s'.",
            user_input, mass_g, filepath.name,
        )
        return mass_g

    if new_mass <= 0:
        logger.warning(
            "Entered mass %.4g g is not positive; keeping %.4g g for '%s'.",
            new_mass, mass_g, filepath.name,
        )
        return mass_g

    logger.info(
        "User updated mass for '%s': %.4g g → %.6g g.",
        filepath.name, mass_g, new_mass,
    )
    return new_mass


# ---------------------------------------------------------------------------
# Mass resolution and default-mass detection
# ---------------------------------------------------------------------------


def _resolve_mass(
    metadata: Dict[str, str],
    config,
) -> Optional[float]:
    """
    Determine the active mass to use and check for the 1-g default.

    Logic
    -----
    1. If ``config.active_mass_g`` is set:
       - If it equals the sentinel (≤ ``default_mass_threshold_g``), flag it
         as a likely default and add a ``"mass_warning"`` to *metadata*.
       - Otherwise, use it and ignore any file-embedded value.
       - Return ``config.active_mass_g`` in both cases (caller decides).
    2. If ``config.active_mass_g`` is None:
       - Check ``metadata["active_mass_g_from_file"]``.
       - If found and > sentinel, return it.
       - If found but ≤ sentinel, flag it and return it (with warning).
       - If not found, return None.

    The ``"mass_warning"`` metadata key is set to a human-readable string
    when the mass is suspected to be the cycler default.

    Parameters
    ----------
    metadata : dict
        Metadata dict (mutated in-place to add warnings).
    config : BatteryPlotConfig

    Returns
    -------
    float or None
        The resolved mass value (grams), or None if unavailable.
    """
    threshold = getattr(config, "default_mass_threshold_g", _DEFAULT_MASS_SENTINEL)

    config_mass = getattr(config, "active_mass_g", None)

    if config_mass is not None:
        try:
            m = float(config_mass)
        except (ValueError, TypeError):
            m = None

        if m is not None and m <= threshold:
            warn = (
                f"active_mass_g = {m:.4g} g in config is at or below the "
                f"default-mass sentinel ({threshold} g). This is likely the "
                f"cycler default value and not the actual electrode mass. "
                f"Specific capacity and gravimetric plots may be physically "
                f"meaningless. Update active_mass_g in config.yaml."
            )
            metadata["mass_warning"] = warn
            logger.warning(warn)
        return m

    # No config mass — try file-embedded value
    file_mass_str = metadata.get("active_mass_g_from_file")
    if file_mass_str is None:
        return None

    try:
        m = float(file_mass_str)
    except (ValueError, TypeError):
        logger.warning("Could not parse file-embedded mass: %r", file_mass_str)
        return None

    if m <= threshold:
        warn = (
            f"File-embedded active_mass_g = {m:.4g} g is at or below the "
            f"default-mass sentinel ({threshold} g). This is likely the "
            f"cycler default and not the actual electrode mass. "
            f"Gravimetric plots will be flagged. "
            f"Set active_mass_g explicitly in config.yaml to suppress this warning."
        )
        metadata["mass_warning"] = warn
        logger.warning(warn)
    else:
        logger.info(
            "Using file-embedded active mass: %.4g g (above sentinel %.4g g).",
            m, threshold,
        )
        metadata["mass_info"] = f"active_mass_g = {m:.6g} g (from file metadata)"

    return m
