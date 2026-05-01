"""Validation tests for the locked sufficiency analyzer gold set and calibration artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import SufficiencyResult
from raggov.models.run import RAGRun


ROOT = Path(__file__).resolve().parents[1]
GOLD_SET_PATH = ROOT / "data" / "sufficiency_gold_set_v1.jsonl"
CALIBRATION_REPORT_PATH = ROOT / "data" / "calibration_report_v1.json"


CATEGORY_KEYWORDS = {
    "fully sufficient": ["fully sufficient", "all evidence"],
    "scope mismatch": ["scope mismatch", "wrong scope"],
    "exception missing": ["exception missing", "exception absent"],
    "noisy but answerable": ["noisy", "buried"],
    "contradictory context": ["contradictory", "conflicting"],
    "stale context": ["stale", "superseded", "outdated"],
    "multi-hop missing": ["multi-hop", "dependent document"],
    "numeric missing": ["numeric missing", "number absent"],
    "date missing": ["date missing", "effective date absent"],
    "authority missing": ["authority missing", "issuing authority absent"],
}


def _load_gold_set() -> list[dict]:
    lines = GOLD_SET_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_gold_set_has_minimum_15_examples() -> None:
    examples = _load_gold_set()
    count = len(examples)
    print(f"\nGold set example count: {count}")
    assert count >= 15, f"Expected >= 15 examples, got {count}"


def test_gold_set_covers_all_10_categories() -> None:
    examples = _load_gold_set()
    found: dict[str, list[str]] = {cat: [] for cat in CATEGORY_KEYWORDS}

    for ex in examples:
        notes = (ex.get("gold_notes") or "").lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in notes for kw in keywords):
                found[category].append(ex["example_id"])

    print("\nCategory coverage:")
    missing_categories = []
    for cat, ids in found.items():
        print(f"  {cat}: {len(ids)} example(s) — {ids}")
        if not ids:
            missing_categories.append(cat)

    assert not missing_categories, (
        f"Missing categories in gold set: {missing_categories}"
    )


def test_all_examples_have_required_fields() -> None:
    examples = _load_gold_set()
    required_fields = [
        "example_id",
        "query",
        "retrieved_chunks",
        "gold_sufficiency_label",
        "gold_missing_evidence",
        "gold_covered_evidence",
        "gold_relevant_chunk_ids",
        "gold_should_abstain",
        "gold_failure_stage",
        "gold_notes",
    ]

    for ex in examples:
        eid = ex.get("example_id", "<unknown>")
        for field in required_fields:
            assert field in ex, f"Example {eid} missing field '{field}'"

        chunks = ex["retrieved_chunks"]
        assert isinstance(chunks, list) and len(chunks) > 0, (
            f"Example {eid}: retrieved_chunks must be a non-empty list"
        )
        for chunk in chunks:
            assert "chunk_id" in chunk, f"Example {eid}: chunk missing 'chunk_id'"
            assert "text" in chunk, f"Example {eid}: chunk missing 'text'"


def test_calibration_report_exists_and_valid() -> None:
    assert CALIBRATION_REPORT_PATH.exists(), (
        f"Calibration report not found at {CALIBRATION_REPORT_PATH}"
    )

    report = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))

    assert "term_coverage" in report["modes"], "Report missing term_coverage mode"
    assert "requirement_aware" in report["modes"], "Report missing requirement_aware mode"
    assert report["gold_set_size"] >= 15, (
        f"Report gold_set_size={report['gold_set_size']} < 15"
    )

    for mode in ("term_coverage", "requirement_aware"):
        m = report["modes"][mode]
        fpr = m["false_pass_rate"]
        assert isinstance(fpr, float), f"{mode} false_pass_rate is not a float"
        assert 0.0 <= fpr <= 1.0, f"{mode} false_pass_rate={fpr} out of [0,1]"
        assert "confusion_matrix" in m, f"{mode} missing confusion_matrix"


def test_analyzer_readiness_returns_valid_dict() -> None:
    analyzer = SufficiencyAnalyzer()
    readiness = analyzer.check_readiness()

    required_keys = [
        "schema_locked",
        "harness_locked",
        "analyzer_locked",
        "calibration_status",
        "gold_set_size",
        "gold_set_path",
        "calibration_report_path",
        "term_coverage_false_pass_rate",
        "requirement_aware_false_pass_rate",
        "recommended_for_gating",
        "recommended_for_advisory",
        "limitations",
    ]
    for key in required_keys:
        assert key in readiness, f"check_readiness() missing key '{key}'"

    assert readiness["schema_locked"] is True
    assert readiness["analyzer_locked"] is True
    assert readiness["recommended_for_gating"] is False
    assert readiness["gold_set_size"] >= 15


def test_requirement_aware_mode_runs_on_all_gold_examples() -> None:
    examples = _load_gold_set()
    analyzer = SufficiencyAnalyzer({"sufficiency_mode": "requirement_aware"})
    success_count = 0

    for ex in examples:
        chunks = [
            RetrievedChunk(
                chunk_id=str(c["chunk_id"]),
                text=str(c["text"]),
                source_doc_id=str((c.get("metadata") or {}).get("source_doc_id", c["chunk_id"])),
                score=c.get("score"),
                metadata=c.get("metadata") or {},
            )
            for c in ex["retrieved_chunks"]
        ]
        run = RAGRun(
            query=str(ex["query"]),
            retrieved_chunks=chunks,
            final_answer=str(ex.get("generated_answer") or ""),
        )

        result = analyzer.analyze(run)

        assert result is not None, f"{ex['example_id']}: analyze() returned None"
        assert result.sufficiency_result is not None, (
            f"{ex['example_id']}: sufficiency_result is None"
        )
        assert isinstance(result.sufficiency_result, SufficiencyResult), (
            f"{ex['example_id']}: sufficiency_result is not SufficiencyResult"
        )
        assert "requirement" in result.sufficiency_result.method or "term_coverage" in result.sufficiency_result.method, (
            f"{ex['example_id']}: unexpected method '{result.sufficiency_result.method}'"
        )
        success_count += 1

    print(f"\nRequirement-aware mode ran successfully on {success_count}/{len(examples)} gold examples")
    assert success_count == len(examples)


def test_calibration_harness_runs_on_seed_gold_set(tmp_path: Path) -> None:
    output_path = tmp_path / "calibration_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "calibrate_sufficiency.py"),
            "--gold-set",
            str(GOLD_SET_PATH),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["gold_set_size"] >= 15
    assert "term_coverage" in report["modes"]
    assert "requirement_aware" in report["modes"]
    assert len(report["modes"]["term_coverage"]["confusion_matrix"]) == 4
    assert all(
        len(row) == 4
        for row in report["modes"]["term_coverage"]["confusion_matrix"]
    )
    sweep = report["modes"]["term_coverage"]["threshold_sweep"]["sweep_results"]
    assert sweep[0]["threshold"] == 0.1
    assert sweep[-1]["threshold"] == 0.9
    assert len(sweep) == 17
    assert len(report["per_example_results"]) >= 15


def test_calibration_with_mock_llm_requirement_aware_does_not_fallback(tmp_path: Path) -> None:
    output_path = tmp_path / "calib_mock.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "calibrate_sufficiency.py"),
            "--gold-set",
            str(GOLD_SET_PATH),
            "--output",
            str(output_path),
            "--llm",
            "mock",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    report = json.loads(output_path.read_text(encoding="utf-8"))
    requirement_methods = {
        row["requirement_aware_details"]["method"]
        for row in report["per_example_results"]
    }

    assert requirement_methods == {"requirement_extraction_llm_v0"}
    for method in requirement_methods:
        assert "fallback" not in method
        assert "term_coverage" not in method
        assert "requirement_extraction" in method

    assert (
        report["modes"]["requirement_aware"]["false_pass_rate"]
        != report["modes"]["term_coverage"]["false_pass_rate"]
    )
