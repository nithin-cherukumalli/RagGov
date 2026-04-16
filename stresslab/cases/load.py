"""Load curated stress cases from local fixtures."""

from __future__ import annotations

from pathlib import Path

from .models import StressCase

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_case(name: str) -> StressCase:
    fixture_name = name if name.endswith(".json") else f"{name}.json"
    fixture_path = FIXTURE_DIR / fixture_name
    return StressCase.model_validate_json(fixture_path.read_text(encoding="utf-8"))


def list_cases() -> list[str]:
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))
