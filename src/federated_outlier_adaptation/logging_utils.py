"""
Project logger.

Console output keeps the previous behaviour (DEBUG records only).  The file sink
is written to ``LOG_DIR`` (``results/logs`` by default, overridable with
``FOA_LOG_DIR``) instead of the current working directory, and it is disabled
entirely when that directory is not writable, so importing the package never
fails on a read-only checkout.
"""

import sys

from loguru import logger as NistLogger

from federated_outlier_adaptation.config import LOG_DIR

NistLogger.remove()

# Console: only DEBUG
NistLogger.add(
    sys.stderr,
    level="DEBUG",
    filter=lambda record: record["level"].name == "DEBUG",
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | {message}",
)

# File: only DEBUG
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NistLogger.add(
        LOG_DIR / "debug_only.log",
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
        rotation="1 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    )
except OSError:  # pragma: no cover - read-only or missing output location
    pass

__all__ = ["NistLogger"]
