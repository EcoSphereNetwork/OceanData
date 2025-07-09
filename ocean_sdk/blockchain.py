from __future__ import annotations

"""Blockchain helpers using Ocean Protocol mock integration."""

import os
from typing import Any, Dict, List


class OceanProtocolAdapter:
    """Adapter encapsulating Ocean Protocol interactions."""

    def publish_asset(
        self,
        name: str,
        metadata: Dict[str, Any],
        price: float,
        files: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        # In a real implementation this would call the Ocean Protocol SDK
        return {
            "tx": "0xmock" if self._mock else "0xreal",
            "name": name,
            "price": price,
            "metadata": metadata,
            "files": files or [],
        }

    def check_access_token(self, token: str) -> bool:
        return bool(token)

    def trigger_compute(self, dataset_id: str, model_id: str) -> str:
        # Returns a job id
        return f"job_{dataset_id}_{model_id}"

    def __init__(self) -> None:
        self._mock = os.getenv("OCEAN_MOCK", "true").lower() == "true"


_ADAPTER = OceanProtocolAdapter()


def publish_dataset(
    name: str,
    metadata: Dict[str, Any],
    price: float,
    files: List[Dict[str, Any]] | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a datatoken and publish a dataset via Ocean Protocol."""
    return _ADAPTER.publish_asset(name, metadata, price, files)
