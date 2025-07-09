"""Utilities to list available data source connectors."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Type

from oceandata.data_integration.base import DataConnector
from oceandata.data_integration.connectors.browser_connector import BrowserDataConnector
from oceandata.data_integration.connectors.calendar_connector import (
    CalendarDataConnector,  # type: ignore
)

# Smartwatch connector has a dash in the filename, load dynamically
SmartwatchDataConnector = import_module(
    "oceandata.data_integration.connectors.smartwatch-connector"
).SmartwatchDataConnector


def get_data_sources() -> Dict[str, Type[DataConnector]]:
    """Return available data connector classes keyed by identifier."""
    return {
        "browser": BrowserDataConnector,
        "calendar": CalendarDataConnector,
        "smartwatch": SmartwatchDataConnector,
    }
