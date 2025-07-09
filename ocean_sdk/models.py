from __future__ import annotations

"""Pydantic models for the OceanData Python SDK."""

from pydantic import BaseModel, Field


class UserIdentity(BaseModel):
    """Representation of a user identity."""

    uid: str = Field(..., description="Unique user identifier")
    wallet: str | None = Field(None, description="Blockchain wallet address")


class ModelInfo(BaseModel):
    """Metadata about a registered model."""

    model_id: str = Field(..., description="Model identifier")
    name: str
    schema: str
    version: str


class EvaluationResult(BaseModel):
    """Result of a model evaluation."""

    model_id: str
    dataset_id: str
    status: str
    outputs: dict | None = None
