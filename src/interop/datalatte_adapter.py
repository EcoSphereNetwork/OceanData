import os
import uuid
from typing import Any, Dict, Iterable


class DatalatteAdapter:
    """Helpers inspired by the Datalatte project."""

    def __init__(self, mock: bool | None = None) -> None:
        self.mock = mock if mock is not None else os.getenv("DATALATTE_MOCK", "true").lower() == "true"

    def tokenize(self, user_id: str, records: Iterable[Any]) -> Dict[str, Any]:
        count = len(list(records))
        if self.mock:
            return {
                "success": True,
                "token_id": f"latte-{uuid.uuid4().hex[:8]}",
                "records": count,
                "user": user_id,
            }
        # In a real setup this would call a meta transaction service
        return {
            "success": True,
            "token_id": f"latte-{uuid.uuid4().hex[:8]}",
            "records": count,
            "user": user_id,
        }
