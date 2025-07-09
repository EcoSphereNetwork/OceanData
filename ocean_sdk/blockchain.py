"""Blockchain helpers using Ocean Protocol mock integration."""
from __future__ import annotations

from typing import Any, Dict, List



def publish_dataset(
    name: str,
    metadata: Dict[str, Any],
    price: float,
    files: List[Dict[str, Any]] | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a datatoken and publish a dataset via Ocean Protocol."""
    return {
        "success": True,
        "name": name,
        "metadata": metadata,
        "price": price,
        "files": files or [],
    }
