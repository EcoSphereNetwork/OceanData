"""Compute-to-Data transaction helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from oceandata.privacy.compute_to_data import ComputeToDataManager


def initiate_c2d_transaction(
    data: pd.DataFrame,
    operation: str,
    params: Optional[Dict[str, Any]] = None,
    privacy_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Register an asset and execute a Compute-to-Data operation."""
    manager = ComputeToDataManager(privacy_config=privacy_config)

    asset_info = manager.register_asset(data, metadata={"owner": "sdk"})
    if not asset_info.get("success"):
        return asset_info

    token_info = manager.create_access_token(asset_info["asset_id"], [operation])
    if not token_info.get("success"):
        return token_info

    result = manager.process_query_with_token(token_info["token"], operation, params or {})
    result["asset_id"] = asset_info["asset_id"]
    result["token_id"] = token_info["token_id"]
    return result
