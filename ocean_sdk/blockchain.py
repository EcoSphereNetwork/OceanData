"""Blockchain helpers using Ocean Protocol mock integration."""
from __future__ import annotations

from typing import Any, Dict, List

from oceandata.blockchain.tokenization import OceanDataTokenizer


def publish_dataset(
    name: str,
    metadata: Dict[str, Any],
    price: float,
    files: List[Dict[str, Any]] | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a datatoken and publish a dataset via Ocean Protocol."""
    tokenizer = OceanDataTokenizer(config)
    tokenizer.connect()
    return tokenizer.tokenize_dataset(name, metadata, price, files)
