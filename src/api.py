"""FastAPI wrapper exposing SDK functions."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ocean_sdk.analytics import (
    evaluate_model,
    register_model,
    retrieve_model_outputs,
    list_available_models,
)

app = FastAPI(title="OceanData SDK API")


class EvaluateRequest(BaseModel):
    model_id: str
    dataset_id: str


@app.get("/models")
def get_models() -> dict:
    """List available models."""
    return list_available_models()


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    """Trigger a model evaluation."""
    result = evaluate_model(req.model_id, req.dataset_id)
    return result.model_dump()


@app.get("/results/{model_id}")
def results(model_id: str) -> dict:
    """Get evaluation outputs."""
    res = retrieve_model_outputs(model_id)
    return res.model_dump()


def save_openapi(path: str = "openapi.json") -> None:
    """Write OpenAPI schema to a file."""
    import json

    schema = app.openapi()
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
