import os
import uuid
import requests
from typing import Any, Dict


class OceanClient:
    """Minimal client for Ocean Protocol provider API."""

    def __init__(self, base_url: str | None = None, mock: bool | None = None) -> None:
        self.base_url = base_url or os.getenv("OCEAN_PROVIDER_URL", "http://localhost:8030")
        self.mock = mock if mock is not None else os.getenv("OCEAN_MOCK", "true").lower() == "true"

    def publish_dataset(self, name: str, metadata: Dict[str, Any], price: float) -> Dict[str, Any]:
        """Publish dataset metadata. Uses mock response when enabled."""
        if self.mock:
            return {
                "success": True,
                "did": f"did:op:{uuid.uuid4().hex}",
                "name": name,
                "price": price,
                "metadata": metadata,
                "tx": f"0x{uuid.uuid4().hex}",
            }
        try:  # pragma: no cover - requires network
            payload = {"name": name, "metadata": metadata, "price": price}
            response = requests.post(f"{self.base_url}/publish", json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            data["success"] = True
            return data
        except Exception as exc:  # pragma: no cover - network issues
            return {"success": False, "error": str(exc)}
