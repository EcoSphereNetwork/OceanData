"""
Centralized logging setup for OceanData.

This module provides a consistent logging interface for all OceanData components.
"""

import logging
import os
import sys
from typing import Optional, Union, Dict, Any


def setup_logger(
    name: str = "OceanData",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Sets up and returns a logger with the specified configuration.

    Args:
        name: The name of the logger.
        level: The logging level.
        log_file: Optional path to a log file. If provided, file logging is enabled.
        format_str: Optional custom format string for logs.

    Returns:
        A configured logger instance.
    """
    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"

    formatter = logging.Formatter(format_str)

    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "OceanData") -> logging.Logger:
    """
    Returns a logger with the given name, creating it if necessary.

    Args:
        name: The name of the logger to retrieve.

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)


# Create a default logger
default_logger = setup_logger("OceanData")
