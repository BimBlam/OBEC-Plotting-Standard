"""
batteryplot.aliases
===================
Canonical column-name mapping for battery-cycler CSV exports.

Design
------
Each physical quantity measured by the cycler is assigned a single
*canonical* snake-case name that includes the SI unit suffix (e.g.
``current_a``, ``voltage_v``).  Every raw column header that could
plausibly refer to that quantity is listed as an *alias*.

Alias matching is intentionally case-insensitive and whitespace-tolerant
so that minor formatting differences between cycler firmware versions or
export templates do not cause failures.

Ambiguity handling
------------------
If the same normalised alias string appears under more than one canonical
name the *first* canonical name in ALIAS_TABLE wins and a warning is
logged.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger("batteryplot")

# ---------------------------------------------------------------------------
# Alias table: canonical_name → [list of raw alias patterns]
#
# Aliases are stored in lower-case, stripped form to match the output of
# normalize_header().  The order within each list is arbitrary; the order
# of ALIAS_TABLE entries determines priority when two canonicals share an
# alias string.
# ---------------------------------------------------------------------------

ALIAS_TABLE: Dict[str, List[str]] = {
    # ------------------------------------------------------------------
    # Time columns
    # ------------------------------------------------------------------
    "elapsed_time_s": [
        "test time",
        "testtime",
        "test time (s)",
        "test time (sec)",
        "elapsed time",
        "elapsed time (s)",
        "elapsed time (sec)",
        "time (s)",
        "time(s)",
        "time (sec)",
        "time",
        "total time",
    ],
    "step_time_s": [
        "step time",
        "steptime",
        "step time (s)",
        "step time (sec)",
        "step_time",
    ],
    # ------------------------------------------------------------------
    # Cycle / step indexing
    #
    # Maccor Series 4000 column semantics (confirmed from BattETL + firmware):
    #
    #   Cycle C  — Charge/discharge cycle counter.  Increments each time the
    #              loop that contains a charge+discharge pair completes one
    #              pass.  This is the number most people mean by "cycle number"
    #              and is what we group by for per-cycle statistics.  Maps to
    #              the canonical  cycle_index.
    #
    #   Cycle P  — Procedure-loop counter.  Counts how many times the outer
    #              procedure loop (or the test procedure itself) has repeated.
    #              In a simple test Cycle P == Cycle C; in nested-loop tests
    #              (e.g. formation at C/10 then cycling at C/3) they differ.
    #              Kept as  procedure_cycle  (metadata only; not used for
    #              grouping).
    #
    #   Step     — Procedure step number within the test schedule (e.g.
    #              Step 1=Rest, Step 2=Charge, Step 3=Discharge, Step 4=Loop).
    #              This RESETS and LOOPS BACK to earlier numbers each cycle;
    #              it is NOT a monotonic step index.  Kept as
    #              procedure_step  (metadata only).
    #
    # Arbin uses different column names so there is no collision risk.
    # ------------------------------------------------------------------
    "cycle_index": [
        "cycle c",
        "cycle_c",
        "cyc#",
        "cyc #",
        "cycle",
        "cycle number",
        "cycle no",
        "cycle no.",
        "cycle index",
        "cyc",
    ],
    # Maccor Step: schedule step number, loops back each cycle — metadata only.
    # Arbin Step: same concept, same name.
    "procedure_step": [
        "step",
        "step no",
        "step no.",
        "step number",
        "stepno",
        "step index",
    ],
    # Maccor Cycle P: procedure-loop counter — metadata only.
    "procedure_cycle": [
        "cycle p",
        "cycle_p",
    ],
    "step_type": [
        "md",
        "mode",
        "step type",
        "steptype",
        "step mode",
        "charge/discharge",
        "chg/dis",
    ],
    # ------------------------------------------------------------------
    # Electrical measurements
    # ------------------------------------------------------------------
    # Maccor exports current in mA ("Current (mA)"); the mA variants are
    # mapped here to the same canonical name.  The unit-conversion step in
    # build_analysis_df divides by 1000 for columns where the raw header
    # contains "(ma)" or "(milliamp)".
    "current_a": [
        "current (a)",
        "current(a)",
        "current (ma)",
        "current(ma)",
        "current",
        "cur (a)",
        "cur(a)",
        "cur (ma)",
        "cur(ma)",
        "i (a)",
        "i(a)",
        "i (ma)",
        "i(ma)",
        "amps",
        "ampere",
    ],
    "voltage_v": [
        "voltage (v)",
        "voltage(v)",
        "voltage",
        "volt (v)",
        "volt(v)",
        "v (v)",
        "v(v)",
        "volts",
    ],
    # Maccor capacity in mAHr; same canonical, converted ÷1000 when raw
    # header contains "(mahr)" / "(mah)" / "(mahr)".
    "capacity_ah": [
        "capacity (ahr)",
        "capacity(ahr)",
        "capacity (ah)",
        "capacity(ah)",
        "capacity (mahr)",
        "capacity(mahr)",
        "capacity (mah)",
        "capacity(mah)",
        "capacity",
        "cap (ahr)",
        "cap(ahr)",
        "cap (ah)",
        "cap(ah)",
        "cap (mahr)",
        "cap(mahr)",
        "charge (ah)",
        "cap",
    ],
    "energy_wh": [
        "energy (whr)",
        "energy(whr)",
        "energy (wh)",
        "energy(wh)",
        "energy (mwhr)",
        "energy(mwhr)",
        "energy (mwh)",
        "energy(mwh)",
        "energy",
        "e (wh)",
        "e(wh)",
    ],
    "power_w": [
        "power (w)",
        "power(w)",
        "power",
        "p (w)",
        "p(w)",
    ],
    # ------------------------------------------------------------------
    # Impedance / resistance
    # ------------------------------------------------------------------
    "ac_impedance_ohm": [
        "ac imp (ohms)",
        "ac imp(ohms)",
        "ac impedance",
        "ac imp",
        "ac_imp",
        "esr",
        "ac impedance (ohm)",
        "ac impedance (ohms)",
        "ac (ohms)",
        "ac(ohms)",
        "ac (ohm)",
        "ac(ohm)",
    ],
    "dcir_ohm": [
        "dcir (ohms)",
        "dcir(ohms)",
        "dcir",
        "dc internal resistance",
        "dc ir",
        "dcir (ohm)",
    ],
    "resistance_ohm": [
        "resistance",
        "res",
        "r (ohm)",
        "r(ohm)",
        "resistance (ohm)",
        "resistance (ohms)",
    ],
    "conductivity_s_cm": [
        "conductivity",
        "conductance",
        "conductivity (s/cm)",
        "conductivity (s cm-1)",
    ],
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    "temperature_c": [
        "evtemp (c)",
        "evtemp(c)",
        "ev temp",
        "temperature",
        "temperature (c)",
        "temp (c)",
        "temp(c)",
        "temp",
        "t (c)",
        "t(c)",
        "cell temp",
    ],
    "humidity_pct": [
        "evhum (%)",
        "evhum(%)",
        "humidity",
        "hum (%)",
        "hum(%)",
        "rh (%)",
        "relative humidity",
    ],
    # ------------------------------------------------------------------
    # Gravimetric / areal quantities
    # ------------------------------------------------------------------
    # Maccor exports specific capacity in mAh/g as "S.Capacity (mAh/g)".
    # The canonical unit is Ah/g; mAh/g variants are divided by 1000 in
    # build_analysis_df alongside current and capacity.
    "specific_capacity_ah_g": [
        "s.capacity (ah/g)",
        "s.capacity(ah/g)",
        "s.capacity (mah/g)",
        "s.capacity(mah/g)",
        "s.cap",
        "specific capacity",
        "specific capacity (ah/g)",
        "specific capacity (mah/g)",
        "specific cap",
        "cap (ah/g)",
        "cap(ah/g)",
        "s.capacity",
    ],
    # ------------------------------------------------------------------
    # Waveform (per-cycle aggregated) columns
    # These are present in Arbin exports as pre-integrated quantities
    # ------------------------------------------------------------------
    "charge_capacity_ah": [
        "wf chg cap",
        "wf charge cap",
        "wf_chg_cap",
        "chg cap",
        "charge capacity",
        "charge capacity (ah)",
        "charge cap (ah)",
        "chg capacity",
    ],
    "discharge_capacity_ah": [
        "wf dis cap",
        "wf discharge cap",
        "wf_dis_cap",
        "dis cap",
        "discharge capacity",
        "discharge capacity (ah)",
        "discharge cap (ah)",
        "dis capacity",
    ],
    "charge_energy_wh": [
        "wf chg e",
        "wf charge e",
        "wf_chg_e",
        "chg energy",
        "charge energy",
        "charge energy (wh)",
    ],
    "discharge_energy_wh": [
        "wf dis e",
        "wf discharge e",
        "wf_dis_e",
        "dis energy",
        "discharge energy",
        "discharge energy (wh)",
    ],
    # ------------------------------------------------------------------
    # Timestamps / identifiers
    # ------------------------------------------------------------------
    "timestamp_dt": [
        "dpt time",
        "dpttime",
        "datetime",
        "timestamp",
        "date time",
        "date/time",
        "abs. time",
        "absolute time",
    ],
    "record_index": [
        "rec",
        "record",
        "rec no",
        "record no",
        "record index",
        "rec index",
        "pt",
        "point",
    ],
    # ------------------------------------------------------------------
    # Loop counter (inner protocol loop in Arbin step editor)
    # ------------------------------------------------------------------
    "loop_index": [
        "loop1",
        "loop 1",
        "loop",
        "loop index",
        "loop count",
        "loop counter",
    ],
}


# ---------------------------------------------------------------------------
# Reverse lookup: alias_string → canonical_name
#
# Built once at module import time.  Ambiguity (same alias under two canonicals)
# is detected here and the first canonical wins.
# ---------------------------------------------------------------------------

def _build_reverse_map(table: Dict[str, List[str]]) -> Dict[str, str]:
    """Build the alias-to-canonical reverse lookup, warning on collisions."""
    reverse: Dict[str, str] = {}
    for canonical, aliases in table.items():
        for alias in aliases:
            if alias in reverse:
                logger.warning(
                    "Alias '%s' is claimed by both '%s' and '%s'; '%s' wins.",
                    alias,
                    reverse[alias],
                    canonical,
                    reverse[alias],
                )
            else:
                reverse[alias] = canonical
    return reverse


_REVERSE_MAP: Dict[str, str] = _build_reverse_map(ALIAS_TABLE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_header(raw: str) -> str:
    """
    Convert a raw column-header string into a normalised form suitable for
    alias lookup.

    Transformations applied (in order):

    1. Strip leading/trailing whitespace.
    2. Convert to lower-case.
    3. Collapse internal whitespace runs to a single space.
    4. Remove leading and trailing non-alphanumeric characters (e.g. asterisks,
       dashes used as decorative padding in some cycler exports).

    Parameters
    ----------
    raw:
        The raw column header string as read from the CSV.

    Returns
    -------
    str
        Normalised header string ready for alias lookup.

    Examples
    --------
    >>> normalize_header("  Voltage (V)  ")
    'voltage (v)'
    >>> normalize_header("**Current (A)**")
    'current (a)'
    """
    s = raw.strip()
    s = s.lower()
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    # Remove leading/trailing non-word characters (but keep parens, slashes,
    # dots, percent signs that appear inside unit annotations)
    s = s.strip("*-=#~_+[]")
    return s


def match_column(normalized: str) -> Optional[str]:
    """
    Return the canonical column name for a normalised alias string, or None.

    The lookup is a simple dictionary hit against the pre-built reverse map
    derived from :data:`ALIAS_TABLE`.  No fuzzy matching is performed; the
    input must exactly equal one of the registered alias strings.

    Parameters
    ----------
    normalized:
        A normalised header string produced by :func:`normalize_header`.

    Returns
    -------
    Optional[str]
        The canonical column name (e.g. ``"voltage_v"``), or ``None`` if
        no alias matches.
    """
    return _REVERSE_MAP.get(normalized)


def map_columns(raw_columns: List[str]) -> Dict[str, str]:
    """
    Map a list of raw CSV column headers to canonical column names.

    Each raw header is normalised with :func:`normalize_header` and then
    looked up in the reverse alias map.  Successfully matched headers are
    returned as a ``{raw → canonical}`` dictionary.  Unmatched headers are
    logged at DEBUG level so the caller can decide whether to keep them as-is.

    If two raw columns resolve to the same canonical name (e.g. the CSV
    contains both "Current (A)" and "Cur (A)"), a warning is logged and only
    the *first* occurrence is kept.

    Parameters
    ----------
    raw_columns:
        Ordered list of raw column-header strings from the CSV header row.

    Returns
    -------
    Dict[str, str]
        Mapping of ``{raw_column_name → canonical_name}`` for every column
        that was successfully matched.

    Notes
    -----
    Columns that are not matched are intentionally excluded from the return
    value (rather than passing them through unchanged) so that downstream code
    can always rely on the canonical namespace.  The caller may choose to
    retain unmatched raw columns under their original names.
    """
    result: Dict[str, str] = {}
    seen_canonical: Dict[str, str] = {}  # canonical → first raw that claimed it

    matched = 0
    unmatched_names: List[str] = []

    for raw in raw_columns:
        norm = normalize_header(raw)
        canonical = match_column(norm)

        if canonical is None:
            unmatched_names.append(raw)
            continue

        if canonical in seen_canonical:
            logger.warning(
                "Duplicate canonical mapping: raw column '%s' (normalised: '%s') "
                "resolves to '%s', but this canonical name was already claimed by "
                "raw column '%s'.  Keeping the first occurrence.",
                raw,
                norm,
                canonical,
                seen_canonical[canonical],
            )
            continue

        seen_canonical[canonical] = raw
        result[raw] = canonical
        matched += 1
        logger.debug("Mapped '%s' → '%s'", raw, canonical)

    if unmatched_names:
        logger.debug(
            "%d column(s) did not match any alias: %s",
            len(unmatched_names),
            unmatched_names,
        )

    logger.info(
        "Column mapping: %d matched, %d unmatched out of %d total.",
        matched,
        len(unmatched_names),
        len(raw_columns),
    )

    return result
