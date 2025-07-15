import os
import uuid
from typing import Any, Dict, Iterable


class DataUnionService:
    """Simple manager for collective dataset publishing."""

    def __init__(self, mock: bool | None = None) -> None:
        self.mock = mock if mock is not None else os.getenv("OCEAN_MOCK", "true").lower() == "true"
        self._unions: Dict[str, Dict[str, Any]] = {}

    def create_union(self, name: str, contributors: Iterable[str]) -> Dict[str, Any]:
        union_id = str(uuid.uuid4())
        self._unions[union_id] = {"name": name, "contributors": list(contributors)}
        return {"success": True, "union_id": union_id}

    def distribute_revenue(self, union_id: str, amount: float) -> Dict[str, Any]:
        info = self._unions.get(union_id)
        if not info:
            return {"success": False, "error": "unknown union"}
        share = amount / max(len(info["contributors"]), 1)
        if self.mock:
            return {"success": True, "share": share}
        # In real mode this would call a smart contract
        return {"success": True, "share": share}
