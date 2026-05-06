"""Integration tests for diagnosis-native golden suite runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from stresslab.cases import list_diagnosis_golden_cases, load_diagnosis_golden_case
from stresslab.diagnosis_evaluation import evaluate_diagnosis_case
from stresslab.runners import (
    render_diagnosis_suite_markdown,
    run_diagnosis_suite,
    write_diagnosis_suite_markdown_report,
    write_diagnosis_suite_report,
)
from raggov import diagnose_file


def test_all_diagnosis_golden_cases_load() -> None:
    case_ids = list_diagnosis_golden_cases()

    assert "clean_pass" in case_ids
    assert "post_rationalized_citation" in case_ids

    for case_id in case_ids:
        case = load_diagnosis_golden_case(case_id)
        assert case.case_id == case_id
        assert case.run_fixture.endswith(".json")


def test_diagnosis_golden_suite_matches_baseline_cases() -> None:
    result = run_diagnosis_suite(
        ["clean_pass", "citation_mismatch", "parser_table_loss", "post_rationalized_citation"]
    )

    assert result.total_count == 4
    assert result.matched_count == 4
    assert result.match_rate == 1.0
    assert result.observed_primary_failures == {
        "CITATION_MISMATCH": 1,
        "CLEAN": 1,
        "UNSUPPORTED_CLAIM": 1,
        "TABLE_STRUCTURE_LOSS": 1,
    }


def test_diagnosis_golden_evaluation_matches_real_diagnosis() -> None:
    case = load_diagnosis_golden_case("parser_hierarchy_loss")
    diagnosis = diagnose_file(ROOT / case.run_fixture, config=case.engine_config)
    evaluation = evaluate_diagnosis_case(case, diagnosis)

    assert evaluation.case_id == "parser_hierarchy_loss"
    assert evaluation.matched_overall is True


def test_external_enhanced_subset_matches_without_a2p_or_ncv() -> None:
    provider_config = {
        "mode": "external-enhanced",
        "enabled_external_providers": [
            "ragas",
            "deepeval",
            "refchecker_claim",
            "refchecker_citation",
            "ragchecker",
        ],
        "enable_a2p": False,
        "enable_ncv": False,
        "retrieval_relevance_provider": "native",
    }

    for case_id in ["clean_pass", "prompt_injection", "stale_retrieval"]:
        case = load_diagnosis_golden_case(case_id)
        diagnosis = diagnose_file(ROOT / case.run_fixture, config=provider_config)
        evaluation = evaluate_diagnosis_case(case, diagnosis)
        assert evaluation.matched_overall is True, f"{case_id}: {evaluation.notes}"


def test_write_diagnosis_suite_reports(tmp_path: Path) -> None:
    result = run_diagnosis_suite(["clean_pass", "unsupported_claims"])

    json_path = write_diagnosis_suite_report(result, tmp_path / "diagnosis-suite.json")
    md_path = write_diagnosis_suite_markdown_report(result, tmp_path / "diagnosis-suite.md")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["total_count"] == 2
    assert payload["matched_count"] == 2
    assert "# Diagnosis Golden Suite Report" in md_path.read_text(encoding="utf-8")


def test_render_diagnosis_suite_markdown_includes_sections() -> None:
    result = run_diagnosis_suite(["clean_pass"])
    markdown = render_diagnosis_suite_markdown(result)

    assert "# Diagnosis Golden Suite Report" in markdown
    assert "## Observed Primary Failures" in markdown
    assert "## Mismatches" in markdown
