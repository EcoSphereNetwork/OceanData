import os
import uuid
import requests
from typing import Any, Dict


class StreamrConnector:
    """Minimal connector to create and publish Streamr streams."""

    def __init__(self, base_url: str | None = None, mock: bool | None = None) -> None:
        self.base_url = base_url or os.getenv("STREAMR_URL", "https://core.api.streamr.network")
        self.mock = mock if mock is not None else os.getenv("STREAMR_MOCK", "true").lower() == "true"

    def create_stream(self, path: str) -> Dict[str, Any]:
        if self.mock:
            return {"success": True, "stream_id": f"{path}-{uuid.uuid4().hex[:6]}"}
        try:  # pragma: no cover - requires network
            response = requests.post(f"{self.base_url}/streams", json={"id": path}, timeout=10)
            response.raise_for_status()
            return {"success": True, **response.json()}
        except Exception as exc:  # pragma: no cover - network issues
            return {"success": False, "error": str(exc)}

    def publish(self, stream_id: str, payload: Any) -> Dict[str, Any]:
        if self.mock:
            return {"success": True, "stream_id": stream_id, "size": len(str(payload))}
        try:  # pragma: no cover - requires network
            url = f"{self.base_url}/streams/{stream_id}/data"
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return {"success": True}
        except Exception as exc:  # pragma: no cover - network issues
            return {"success": False, "error": str(exc)}
