"""Logging utilities for scientific experiment tracking.

Sets up standardized loggers that output formatted messages to both
the console and persistent log files in the results/logs directory.
"""

from typing import Optional
from pathlib import Path
from datetime import datetime
import logging
import sys


DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "Opportunity_HAR",
    log_dir: Optional[str | Path] = None,
    log_filename: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a standardized logger with console and optional file handlers.

    Args:
        name (str): Identifier name for the logger.
        log_dir (Optional[str | Path]): Directory where the log file should be saved.
            If None, logging is directed only to stdout unless a file handler already exists.
        log_filename (Optional[str]): Name of the log file. If not provided and log_dir
            is specified, a timestamped filename is generated.
        level (int): Logging level threshold (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if directory specified)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"run_{timestamp}.log"

        file_handler = logging.FileHandler(log_path / log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_experiment_logger(
    experiment_name: str,
    output_dir: str | Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """Helper to configure a dedicated logger for a specific research experiment run.

    Args:
        experiment_name (str): Name of the experiment (e.g., 'baseline_cnn', 'eda_run').
        output_dir (str | Path): Base results/logs directory.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured experiment logger.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    return get_logger(
        name=f"exp_{experiment_name}",
        log_dir=output_dir,
        log_filename=log_filename,
        level=level,
    )
