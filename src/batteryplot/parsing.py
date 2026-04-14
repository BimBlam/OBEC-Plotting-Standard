"""
batteryplot.parsing
===================
Robust CSV parser for Arbin-style battery-cycler exports.

Arbin file structure
--------------------
Row 0 (line 1):  metadata, e.g. ``Today's Date ,4/14/2026``
Row 1 (line 2):  metadata, e.g. ``Date of Test:,...``
Row 2 (line 3):  **real header row** with all column names
Row 3+:          data rows

Because different firmware versions and export templates sometimes include
additional or fewer metadata rows, the header row is *detected* rather than
assumed to be at a fixed line index.

Time columns
------------
Arbin exports ``Test Time`` and ``Step Time`` in the format ``  Xd HH:MM:SS``
where X is the number of elapsed days.  These are converted to total seconds
(float) during loading.

``DPT Time`` is an absolute datetime string (e.g. ``4/10/2026 3:02:24 AM``)
and is parsed to ``pd.Timestamp``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from batteryplot.aliases import map_columns
from batteryplot.config import BatteryPlotConfig

logger = logging.getLogger("batteryplot")

# Canonical names for the two Arbin time columns so we can post-process them
_TIME_CANONICALS = {"elapsed_time_s", "step_time_s"}
_DATETIME_CANONICAL = "timestamp_dt"


# ---------------------------------------------------------------------------
# 1. Header detection
# ---------------------------------------------------------------------------


def detect_header_row(
    filepath: Path,
    max_scan: int = 10,
    min_numeric_fraction: float = 0.5,
) -> Tuple[int, List[str]]:
    """
    Scan the first *max_scan* lines of *filepath* and identify the real CSV
    header row.

    Algorithm
    ---------
    A candidate header row must satisfy **all** of:

    1. It contains at least 3 comma-separated tokens.
    2. At least 3 of those tokens are **non-numeric** strings (i.e. they look
       like column labels rather than data values).
    3. The **next** row (if present) has a ``min_numeric_fraction`` of its
       tokens parse as float — confirming that data rows follow immediately.

    This reliably identifies row 2 (0-indexed) in standard Arbin exports where
    rows 0–1 are short key-value metadata lines.

    Parameters
    ----------
    filepath:
        Path to the CSV file.
    max_scan:
        Maximum number of lines to read from the top of the file.
    min_numeric_fraction:
        Fraction of tokens in the row *after* the candidate header that must
        be numeric for the candidate to be confirmed.

    Returns
    -------
    Tuple[int, List[str]]
        ``(header_row_index_0based, list_of_column_names)``

    Raises
    ------
    ValueError
        If no suitable header row is found within the first *max_scan* lines.
    """
    filepath = Path(filepath)
    lines: List[str] = []
    with filepath.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(max_scan):
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))

    def _is_numeric(token: str) -> bool:
        t = token.strip()
        if t in ("", "N/A", "n/a", "NA", "na", "NaN", "nan"):
            return True  # treat N/A as numeric-row token
        try:
            float(t)
            return True
        except ValueError:
            return False

    def _numeric_fraction(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if _is_numeric(t)) / len(tokens)

    def _non_numeric_count(tokens: List[str]) -> int:
        return sum(1 for t in tokens if not _is_numeric(t))

    for i, line in enumerate(lines[:-1]):  # need a next line to confirm
        tokens = [t.strip() for t in line.split(",")]
        if len(tokens) < 3:
            continue
        non_num = _non_numeric_count(tokens)
        if non_num < 3:
            continue
        # Check that the following row is mostly numeric (= data row)
        next_tokens = [t.strip() for t in lines[i + 1].split(",")]
        if _numeric_fraction(next_tokens) >= min_numeric_fraction:
            logger.info(
                "Header row detected at line %d (0-indexed). %d columns found.",
                i,
                len(tokens),
            )
            return i, tokens

    # Fallback: try the last scanned line as a header even without confirmation
    for i, line in enumerate(lines):
        tokens = [t.strip() for t in line.split(",")]
        if len(tokens) >= 3 and _non_numeric_count(tokens) >= 3:
            logger.warning(
                "Header detection fallback: using line %d as header row.", i
            )
            return i, tokens

    raise ValueError(
        f"Could not detect a header row in the first {max_scan} lines of '{filepath}'. "
        "Check that the file is a valid Arbin CSV export."
    )


# ---------------------------------------------------------------------------
# 2. Time-format parsers
# ---------------------------------------------------------------------------


def parse_time_column(series: pd.Series) -> pd.Series:
    """
    Convert an Arbin-style elapsed-time column to total seconds as float.

    Arbin exports time in the format ``"  Xd HH:MM:SS"`` where *X* is the
    number of elapsed days.  For example ``"  2d 03:15:30"`` represents
    2 days, 3 hours, 15 minutes, and 30 seconds = 185 730 s.

    Fallback rules (tried in order):

    1. Pattern ``Xd HH:MM:SS`` or ``Xd HH:MM:SS.fff`` → convert to float seconds.
    2. Plain numeric string → cast directly to float (assumed seconds).
    3. ``"N/A"``, empty string, or unparseable → ``NaN``.

    Parameters
    ----------
    series:
        Raw string series from the CSV (dtype object).

    Returns
    -------
    pd.Series
        Float series of elapsed seconds, with ``NaN`` for unparseable entries.
    """
    _DAY_TIME_RE = re.compile(
        r"^\s*(\d+)d\s+(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$",
        re.IGNORECASE,
    )

    def _convert(raw: object) -> float:
        if pd.isna(raw):
            return float("nan")
        s = str(raw).strip()
        if s in ("", "N/A", "n/a", "NA", "nan"):
            return float("nan")
        m = _DAY_TIME_RE.match(s)
        if m:
            days = int(m.group(1))
            hours = int(m.group(2))
            minutes = int(m.group(3))
            seconds = int(m.group(4))
            frac_str = m.group(5)
            frac = float("0." + frac_str) if frac_str else 0.0
            total = days * 86400 + hours * 3600 + minutes * 60 + seconds + frac
            return total
        # Fallback: plain number
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return series.map(_convert).astype(float)


def parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Parse an Arbin ``DPT Time`` column to ``pd.Timestamp``.

    Arbin exports absolute timestamps in formats such as:
    ``"4/10/2026 3:02:24 AM"`` or ``"2026-04-10 03:02:24"``.

    ``pd.to_datetime`` with ``errors='coerce'`` is used so that malformed
    entries become ``NaT`` rather than raising.

    Parameters
    ----------
    series:
        Raw string series from the CSV (dtype object).

    Returns
    -------
    pd.Series
        Series of ``pd.Timestamp`` (with ``NaT`` for unparseable entries).
    """
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


# ---------------------------------------------------------------------------
# 3. Main CSV loader
# ---------------------------------------------------------------------------


def load_csv(
    filepath: Path,
    config: BatteryPlotConfig,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Load an Arbin-style CSV file into a raw DataFrame and extract metadata.

    Processing steps:

    1. Detect the real header row using :func:`detect_header_row`.
    2. Collect any key-value metadata rows that precede the header.
    3. Read the CSV starting at the detected header row.
    4. Map raw column names to canonical names using :func:`map_columns`.
    5. Post-process time columns (elapsed-time → seconds; DPT Time → Timestamp).
    6. Drop columns that are entirely zero or entirely ``N/A``/``NaN``
       (log each dropped column at DEBUG level).

    Parameters
    ----------
    filepath:
        Path to the raw CSV file.
    config:
        Pipeline configuration.  Uses ``header_search_rows`` and
        ``min_numeric_fraction`` for header detection.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]
        - ``raw_df``: DataFrame with the original (raw) column names,
          time columns partially parsed.
        - ``column_map``: ``{raw_name → canonical_name}`` for matched columns.
        - ``metadata``: ``{key → value}`` for metadata rows before the header.

    Notes
    -----
    The returned ``raw_df`` retains the original column names.  Call
    :func:`build_analysis_df` to rename columns to their canonical names
    and apply final cleaning.
    """
    filepath = Path(filepath)
    logger.info("Loading CSV: %s", filepath)

    # --- 3.1 Detect header ---
    header_idx, raw_columns = detect_header_row(
        filepath,
        max_scan=config.header_search_rows,
        min_numeric_fraction=config.min_numeric_fraction,
    )
    logger.info(
        "Header row at line %d; %d columns detected.", header_idx, len(raw_columns)
    )

    # --- 3.2 Collect metadata from rows before the header ---
    metadata: Dict[str, str] = _extract_metadata(filepath, header_idx)
    if metadata:
        logger.debug("Metadata extracted: %s", metadata)

    # --- 3.3 Read CSV starting at the header row ---
    raw_df = pd.read_csv(
        filepath,
        skiprows=header_idx,
        header=0,
        dtype=str,          # read everything as str; we parse types ourselves
        na_values=["N/A", "n/a", "NA", ""],
        keep_default_na=True,
        low_memory=False,
    )
    # Strip whitespace from column names
    raw_df.columns = [c.strip() for c in raw_df.columns]
    logger.debug("Raw DataFrame shape: %s", raw_df.shape)

    # --- 3.4 Map columns ---
    column_map = map_columns(list(raw_df.columns))
    n_mapped = len(column_map)
    n_total = len(raw_df.columns)
    n_unmapped = n_total - n_mapped
    logger.info(
        "%d columns mapped, %d unmapped (out of %d).",
        n_mapped,
        n_unmapped,
        n_total,
    )

    # --- 3.5 Parse time columns in-place (on raw_df) ---
    for raw_col, canonical in column_map.items():
        if canonical in _TIME_CANONICALS:
            logger.debug("Parsing time column '%s' → '%s'", raw_col, canonical)
            raw_df[raw_col] = parse_time_column(raw_df[raw_col])
        elif canonical == _DATETIME_CANONICAL:
            logger.debug("Parsing datetime column '%s' → '%s'", raw_col, canonical)
            raw_df[raw_col] = parse_datetime_column(raw_df[raw_col])

    # --- 3.6 Drop all-zero or all-NaN columns ---
    # Pass column_map so critical measurement columns are never silently removed.
    raw_df = _drop_empty_columns(raw_df, column_map=column_map)

    return raw_df, column_map, metadata


# ---------------------------------------------------------------------------
# 4. Analysis DataFrame builder
# ---------------------------------------------------------------------------


def build_analysis_df(
    raw_df: pd.DataFrame,
    column_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Rename raw columns to canonical names and apply final data cleaning.

    This function:

    1. Renames raw columns to their canonical equivalents using *column_map*.
    2. Keeps only canonical columns (drops raw columns without a mapping).
    3. Drops rows where **both** ``current_a`` and ``voltage_v`` are ``NaN``
       (these rows carry no electrochemical information).
    4. Coerces all remaining object-dtype columns to numeric where possible,
       leaving non-convertible values as ``NaN``.

    Parameters
    ----------
    raw_df:
        Raw DataFrame returned by :func:`load_csv`.
    column_map:
        ``{raw_name → canonical_name}`` mapping from :func:`load_csv`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with canonical column names.
    """
    # Only rename raw columns that are present in raw_df AND in column_map
    rename_map = {raw: canonical for raw, canonical in column_map.items()
                  if raw in raw_df.columns}
    df = raw_df[list(rename_map.keys())].rename(columns=rename_map)

    # Drop rows where both current_a and voltage_v are NaN
    has_current = "current_a" in df.columns
    has_voltage = "voltage_v" in df.columns
    if has_current and has_voltage:
        mask = df["current_a"].isna() & df["voltage_v"].isna()
        n_dropped = mask.sum()
        if n_dropped:
            logger.debug("Dropping %d rows where both current_a and voltage_v are NaN.", n_dropped)
        df = df[~mask].reset_index(drop=True)
    elif has_current:
        df = df[~df["current_a"].isna()].reset_index(drop=True)
    elif has_voltage:
        df = df[~df["voltage_v"].isna()].reset_index(drop=True)

    # Coerce object/string columns to numeric (skip datetime and step_type).
    # In pandas 2.x, dtype=str in read_csv produces StringDtype rather than
    # object dtype, so we check for both.
    _SKIP_COERCE = {_DATETIME_CANONICAL, "step_type"}
    for col in df.columns:
        if col in _SKIP_COERCE:
            continue
        # Accept object dtype OR any pd.StringDtype (pandas 2.x)
        dtype = df[col].dtype
        is_string_like = dtype == object or isinstance(dtype, pd.StringDtype)
        if is_string_like:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Analysis DataFrame: %d rows × %d canonical columns.",
        len(df),
        len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_metadata(filepath: Path, header_idx: int) -> Dict[str, str]:
    """Read lines 0…header_idx-1 and parse them as key=value metadata."""
    metadata: Dict[str, str] = {}
    if header_idx == 0:
        return metadata
    with filepath.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= header_idx:
                break
            parts = [p.strip() for p in line.rstrip("\n").split(",", maxsplit=1)]
            if len(parts) == 2 and parts[0]:
                key = parts[0].rstrip(":").strip()
                val = parts[1].strip()
                if key:
                    metadata[key] = val
    return metadata


# Canonical names that must never be dropped even if all-zero.
# A rest-only test segment will have all-zero current; a fresh cell may
# have all-zero capacity.  We must never silently lose these columns.
_NEVER_DROP_CANONICALS: set = {
    "current_a", "voltage_v", "capacity_ah", "energy_wh", "power_w",
    "cycle_index", "step_index", "charge_capacity_ah", "discharge_capacity_ah",
    "charge_energy_wh", "discharge_energy_wh", "dcir_ohm", "ac_impedance_ohm",
    "resistance_ohm", "specific_capacity_ah_g", "temperature_c", "humidity_pct",
}


def _drop_empty_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Drop columns that are entirely NaN or all-zero, but ONLY if they are
    not mapped to a scientifically important canonical name.

    Scientific rationale: many Arbin exports include placeholder columns
    (e.g. VAR1..VAR50, GlobFlag1..GlobFlag64) that are always zero for a
    given test.  Dropping them reduces memory usage.  However, core
    measurement columns (current, voltage, capacity, etc.) must never be
    dropped even when they are all-zero in a rest-only segment.
    """
    # Build reverse map: raw_column_name -> canonical_name
    rev: Dict[str, str] = column_map or {}

    cols_to_drop: List[str] = []

    for col in df.columns:
        # Never drop if mapped to a protected canonical
        canonical = rev.get(col)
        if canonical and canonical in _NEVER_DROP_CANONICALS:
            continue

        series = df[col]
        all_na = series.isna().all()
        if all_na:
            cols_to_drop.append(col)
            continue
        # Check if numeric and all zero
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any() and (numeric.fillna(0) == 0).all():
                cols_to_drop.append(col)
        except Exception:
            pass  # non-numeric column; keep it

    if cols_to_drop:
        logger.debug(
            "Dropping %d all-zero or all-NaN columns: %s",
            len(cols_to_drop),
            cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)

    return df
