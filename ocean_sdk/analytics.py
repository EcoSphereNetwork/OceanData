"""Analysis wrappers for OceanData."""
from __future__ import annotations

from typing import Any, Dict, List

import pkgutil
from importlib import import_module

import pandas as pd

from oceandata.core.ocandata_ai import OceanDataAI
from oceandata.privacy.compute_to_data import ComputeToDataManager


def run_analysis(
    data: pd.DataFrame,
    source_type: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run built-in analysis pipelines for a given data source."""
    ai = OceanDataAI(config)
    return ai.analyze_data_source(data, source_type)


def list_available_models() -> Dict[str, str]:
    """Return available ML model modules with short descriptions."""
    package = import_module("oceandata.analytics.models")
    models: Dict[str, str] = {}
    for _, name, _ in pkgutil.iter_modules(package.__path__):
        module = import_module(f"oceandata.analytics.models.{name}")
        doc = (module.__doc__ or "").strip().splitlines()[0] if module.__doc__ else ""
        models[name] = doc
    return models


def train_model(
    data_id: str,
    model: str = "default",
    privacy_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Trigger Compute-to-Data model training for a dataset."""
    manager = ComputeToDataManager(privacy_config=privacy_config)
    token_info = manager.create_access_token(data_id, ["custom_model"])
    if not token_info.get("success"):
        return token_info
    return manager.process_query_with_token(
        token_info["token"], "custom_model", {"action": "train", "model": model}
    )
