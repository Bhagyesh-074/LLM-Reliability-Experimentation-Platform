"""
core/logger.py
--------------
Centralised Loguru logger configuration for the LLM Reliability Platform.

Call ``setup_logging()`` exactly once at application start-up (e.g. from
``dashboard/app.py``).  All other modules obtain a pre-configured logger via
the standard ``from loguru import logger`` import — no further setup needed.

Usage:
    # At app entry-point:
    from core.logger import setup_logging
    setup_logging()

    # Everywhere else:
    from loguru import logger
    logger.info("Ready.")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from core.config import settings


def _build_log_path(log_dir: str, filename: str = "platform.log") -> Path:
    """Resolve and create the log directory, then return the full log path.

    Args:
        log_dir: Directory path (relative or absolute) where logs are stored.
        filename: Log file name within ``log_dir``.

    Returns:
        Absolute ``Path`` to the log file.
    """
    path = Path(log_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def setup_logging() -> None:
    """Configure Loguru sinks for the platform.

    Removes any pre-existing handlers, then attaches:

    * **stderr sink** — coloured, human-readable output for the console.
    * **rotating file sink** — structured records persisted to disk with
      automatic rotation and retention as specified in ``configs/default.yaml``.

    This function is idempotent: calling it multiple times removes all
    previously registered handlers before re-adding them, preventing duplicate
    log lines during hot-reload scenarios (e.g. Streamlit dev mode).

    Returns:
        None
    """
    cfg = settings.logging

    # Remove all existing handlers (safe to call even with no handlers).
    logger.remove()

    # --- Console sink ---------------------------------------------------
    logger.add(
        sys.stderr,
        level=cfg.level,
        format=cfg.format,
        colorize=cfg.colorize,
        backtrace=True,
        diagnose=settings.app.debug,
        enqueue=False,
    )

    # --- Rotating file sink ---------------------------------------------
    log_file = _build_log_path(cfg.log_dir)
    logger.add(
        str(log_file),
        level=cfg.level,
        format=cfg.format,
        rotation=cfg.rotation,
        retention=cfg.retention,
        compression="zip",
        backtrace=True,
        diagnose=settings.app.debug,
        enqueue=True,   # thread-safe async writes
        encoding="utf-8",
    )

    logger.info(
        "Logging initialised | level={level} | file={file}",
        level=cfg.level,
        file=log_file,
    )


def get_logger(name: str) -> "logger":  # type: ignore[valid-type]
    """Return a Loguru logger bound with a module name context.

    This is a convenience wrapper for modules that prefer an explicit named
    logger over the global ``loguru.logger`` singleton.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A Loguru logger instance with the ``name`` field pre-bound.
    """
    return logger.bind(name=name)