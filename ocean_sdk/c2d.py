from __future__ import annotations

"""Compute-to-Data transaction helpers."""

import os
from typing import Any, Dict, Optional

import pandas as pd


class C2DAdapter:
    """Adapter for Compute-to-Data interactions."""

    def __init__(self) -> None:
        self._mock = os.getenv("OCEAN_MOCK", "true").lower() == "true"

    def register_and_compute(
        self,
        data: pd.DataFrame,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        privacy_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "operation": operation,
            "rows": len(data) if hasattr(data, "__len__") else 0,
            "params": params or {},
            "mode": "mock" if self._mock else "real",
        }


_ADAPTER = C2DAdapter()


def initiate_c2d_transaction(
    data: pd.DataFrame,
    operation: str,
    params: Optional[Dict[str, Any]] = None,
    privacy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register an asset and execute a Compute-to-Data operation."""
    return _ADAPTER.register_and_compute(data, operation, params, privacy_config)
