"""Small JSON artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, BaseModel):
        serializable = payload.model_dump(mode="json")
    else:
        serializable = payload

    target.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
