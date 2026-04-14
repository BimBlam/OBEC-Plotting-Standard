"""
batteryplot.utils.logging_utils
================================
Logging configuration helpers for the batteryplot package.

All modules within batteryplot use the named logger ``"batteryplot"``
(or a child such as ``"batteryplot.parsing"``).  Call :func:`setup_logging`
once at the start of a script or pipeline entry point to configure handlers
and formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return the root ``"batteryplot"`` logger.

    Sets up a ``StreamHandler`` that writes to *stderr* at the requested
    level.  If *log_file* is provided an additional ``FileHandler`` is
    attached that writes to that path (overwriting any existing file).

    This function is idempotent: calling it multiple times replaces
    existing handlers rather than stacking duplicates.

    Parameters
    ----------
    log_level:
        Python logging level string.  One of: ``DEBUG``, ``INFO``,
        ``WARNING``, ``ERROR``, ``CRITICAL``.  Case-insensitive.
    log_file:
        Optional filesystem path for a log file.  The parent directory
        is created automatically if it does not exist.

    Returns
    -------
    logging.Logger
        The configured ``"batteryplot"`` logger instance.

    Examples
    --------
    >>> from batteryplot.utils.logging_utils import setup_logging
    >>> log = setup_logging("DEBUG", log_file=Path("run.log"))
    >>> log.info("Pipeline started")
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger("batteryplot")
    root_logger.setLevel(numeric_level)

    # Remove pre-existing handlers to avoid duplicates on re-calls
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.debug(
        "Logging initialised at level %s%s.",
        log_level.upper(),
        f" (file: {log_file})" if log_file else "",
    )
    return root_logger
