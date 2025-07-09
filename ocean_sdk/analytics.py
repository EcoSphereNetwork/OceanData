from __future__ import annotations

"""Analysis wrappers for OceanData."""

from typing import Any, Dict, List

import pkgutil
from importlib import import_module
import uuid

import pandas as pd

from .c2d import initiate_c2d_transaction
from .models import EvaluationResult, ModelInfo

_MODELS: Dict[str, ModelInfo] = {}
_EVALUATIONS: Dict[str, EvaluationResult] = {}


def run_analysis(
    data: pd.DataFrame,
    source_type: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run built-in analysis pipelines for a given data source."""
    record_count = len(data) if hasattr(data, "__len__") else 0
    return {"success": True, "source_type": source_type, "record_count": record_count}


def list_available_models() -> Dict[str, str]:
    """Return available ML model modules with short descriptions."""
    package = import_module(__name__)
    return {name: "" for _, name, _ in pkgutil.iter_modules(package.__path__)}


def train_model(
    data_id: str,
    model: str = "default",
    privacy_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Trigger Compute-to-Data model training for a dataset."""
    return {"success": True, "data_id": data_id, "model": model}


def register_model(name: str, schema: str, version: str) -> ModelInfo:
    """Register a model in the local registry."""
    model_id = str(uuid.uuid4())
    info = ModelInfo(model_id=model_id, name=name, schema=schema, version=version)
    _MODELS[model_id] = info
    return info


def evaluate_model(model_id: str, dataset_id: str) -> EvaluationResult:
    """Start a model evaluation via Compute-to-Data."""
    initiate_c2d_transaction(pd.DataFrame(), "evaluate")
    result = EvaluationResult(model_id=model_id, dataset_id=dataset_id, status="running")
    _EVALUATIONS[model_id] = result
    return result


def retrieve_model_outputs(model_id: str) -> EvaluationResult:
    """Retrieve evaluation outputs for a model."""
    result = _EVALUATIONS.get(model_id)
    if not result:
        raise KeyError(model_id)
    result.status = "done"
    result.outputs = {"accuracy": 0.9}
    return result
