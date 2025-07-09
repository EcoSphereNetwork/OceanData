"""Simple token based authentication manager."""

from __future__ import annotations

import os
from typing import Dict, Optional


class AuthManager:
    """Manage API access tokens."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or self.load_token()

    def load_token(self) -> Optional[str]:
        """Load token from environment."""
        return os.getenv("OCEAN_TOKEN")

    def validate_token(self) -> bool:
        """Validate that a token exists (mock validation)."""
        return bool(self.token)

    def inject_auth_header(self, headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Return headers with Authorization field added if token is present."""
        headers = headers or {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
