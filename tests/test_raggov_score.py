"""Foundation guard for the canonical scorer (scripts/raggov_score.py).

Asserts the scorer runs and reproduces the LOCKED Calib number (23/45). It deliberately
does NOT pin the induced-probe accuracy — that number is meant to move as the engine
improves; pinning it would fight intended progress. We only assert the probe scores all
145 rows and stays in a sane range.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "raggov_score", ROOT / "scripts" / "raggov_score.py"
)
raggov_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(raggov_score)


def test_calib_reproduces_locked_number() -> None:
    report = raggov_score.score_file(
        raggov_score.CALIB, mode="default", splits={"train", "dev", "heldout"}
    )
    assert report["n"] == 45
    assert report["correct"] == 23  # locked Calib baseline


def test_probe_scores_all_rows_in_sane_range() -> None:
    report = raggov_score.score_file(raggov_score.PROBE, mode="default")
    assert report["n"] == 145
    assert 0.30 <= report["accuracy"] <= 0.95  # wide guard; not pinned


def test_build_run_normalizes_dict_citations() -> None:
    run = raggov_score.build_run(
        {
            "query": "q",
            "retrieved_chunks": [{"doc_id": "d1", "text": "t"}],
            "answer": "a",
            "citations": [{"doc_id": "d1"}, "d2"],
        }
    )
    assert run.cited_doc_ids == ["d1", "d2"]
    assert run.retrieved_chunks[0].source_doc_id == "d1"
