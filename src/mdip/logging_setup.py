"""Minimal, readable logging setup.

One helper, ``get_logger``, returns a configured stdlib logger. We use the standard
library rather than a structured-logging framework because for an analysis project
plain, human-readable log lines are exactly what you want to see while a pipeline
runs — and there is nothing to explain away in an interview.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger that writes clean, timestamped lines to stdout.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.
        level: Logging level, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True

    return logging.getLogger(name)
