"""Utilities to list available data source connectors."""
from __future__ import annotations

from typing import Dict, Type


class DataConnector:
    """Base class for data connectors."""


class BrowserDataConnector(DataConnector):
    """Dummy browser connector."""


class CalendarDataConnector(DataConnector):
    """Dummy calendar connector."""


class SmartwatchDataConnector(DataConnector):
    """Dummy smartwatch connector."""


def get_data_sources() -> Dict[str, Type[DataConnector]]:
    """Return available data connector classes keyed by identifier."""
    return {
        "browser": BrowserDataConnector,
        "calendar": CalendarDataConnector,
        "smartwatch": SmartwatchDataConnector,
    }
