from __future__ import annotations

"""User identity utilities."""

import os
import uuid
from typing import Optional

from .models import UserIdentity


def get_user_identity() -> UserIdentity:
    """Return the current user identity based on environment variables."""
    uid = os.getenv("OCEAN_UID") or str(uuid.uuid4())
    wallet = os.getenv("OCEAN_WALLET")
    return UserIdentity(uid=uid, wallet=wallet)
