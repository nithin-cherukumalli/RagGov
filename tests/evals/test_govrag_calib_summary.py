from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import summarize_govrag_calib


DATASET = PROJECT_ROOT / "evals" / "govrag_calib" / "calib_150_seed.jsonl"


def test_summary_counts_current_seed_dataset_without_mutation():
    before = DATASET.read_text(encoding="utf-8")

    summary = summarize_govrag_calib.summarize_dataset(DATASET)

    assert summary.total_cases == 50
    assert sum(summary.by_failure_family.values()) == 50
    assert summary.by_calibration_split["unset"] == 50
    assert summary.missing_family_targets["retrieval"] == 17
    assert DATASET.read_text(encoding="utf-8") == before


def test_summary_output_does_not_imply_production_calibration():
    summary = summarize_govrag_calib.summarize_dataset(DATASET)
    output = summarize_govrag_calib.format_summary(summary)

    assert "calibration_status: not_calibrated" in output
    assert "production_gating_eligible: false" in output
    assert "does not run GovRAG" in output


def test_summary_counts_dimensions_from_small_jsonl(tmp_path):
    dataset = tmp_path / "mini.jsonl"
    records = [
        {
            "failure_family": "clean_pass",
            "label_status": "seed",
            "calibration_split": "unset",
            "source_suite": "manual",
            "security_relevant": False,
            "adversarial": False,
        },
        {
            "failure_family": "security_privacy",
            "label_status": "reviewed",
            "calibration_split": "unset",
            "source_suite": "synthetic",
            "security_relevant": True,
            "adversarial": True,
        },
    ]
    dataset.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    summary = summarize_govrag_calib.summarize_dataset(dataset)

    assert summary.total_cases == 2
    assert summary.by_failure_family["clean_pass"] == 1
    assert summary.by_failure_family["security_privacy"] == 1
    assert summary.by_label_status["seed"] == 1
    assert summary.by_label_status["reviewed"] == 1
    assert summary.by_source_suite["manual"] == 1
    assert summary.security_relevant_count == 1
    assert summary.adversarial_count == 1
