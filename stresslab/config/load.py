"""Load runtime profiles from packaged JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import RuntimeProfile

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def load_profile(name: str) -> RuntimeProfile:
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid profile name: {name}")

    profile_path = _PROFILES_DIR / f"{name}.json"
    if not profile_path.is_file():
        raise ValueError(f"Unknown profile: {name}")

    with profile_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return RuntimeProfile.model_validate(data)
