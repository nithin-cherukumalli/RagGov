"""Run the Day 1 baseline-freeze checkpoint workflow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stresslab.runners.freeze_day1_baseline import freeze_day1_baseline


if __name__ == "__main__":
    freeze_day1_baseline()
