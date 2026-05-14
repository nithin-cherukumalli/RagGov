"""Tests for claim-level diagnostic evaluation harness v0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from stresslab.cases import ClaimDiagnosisGoldSet
from stresslab.cases import load_claim_diagnosis_gold_set
from stresslab.claim_diagnosis_evaluation import (
    EVALUATION_STATUS,
    render_claim_diagnosis_report,
    run_claim_diagnosis_harness,
)
from stresslab.runners.run_claim_diagnosis_harness import (
    write_claim_diagnosis_markdown_report,
    write_claim_diagnosis_report,
)


def _minimal_case() -> dict[str, object]:
    return {
        "case_id": "schema_case",
        "query": "What is the refund window?",
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "text": "Refunds are available within thirty days.",
                "source_doc_id": "doc1",
                "score": 0.9,
            }
        ],
        "final_answer": "Refunds are available within thirty days.",
        "expected_claims": [
            {
                "claim_text": "Refunds are available within thirty days.",
                "expected_claim_label": "entailed",
            }
        ],
        "expected_sufficient": True,
        "expected_primary_stage": "UNKNOWN",
        "expected_fix_category": "other",
    }


def test_loads_legacy_version_cases_schema() -> None:
    gold = ClaimDiagnosisGoldSet.model_validate(
        {
            "version": "legacy_version_cases",
            "cases": [_minimal_case()],
        }
    )

    assert gold.version == "legacy_version_cases"
    assert gold.evaluation_status == "legacy_version_cases"
    assert len(gold.cases) == 1
    assert gold.examples[0].case_id == "schema_case"


def test_loads_current_evaluation_status_examples_schema() -> None:
    gold = ClaimDiagnosisGoldSet.model_validate(
        {
            "evaluation_status": "current_evaluation_examples",
            "examples": [_minimal_case()],
        }
    )

    assert gold.version == "current_evaluation_examples"
    assert gold.evaluation_status == "current_evaluation_examples"
    assert len(gold.examples) == 1
    assert gold.cases[0].case_id == "schema_case"


def test_invalid_gold_missing_expected_fields_fails_loudly() -> None:
    invalid_case = _minimal_case()
    invalid_case.pop("expected_claims")

    with pytest.raises(ValidationError, match="schema_case.*expected_claims"):
        ClaimDiagnosisGoldSet.model_validate(
            {
                "evaluation_status": "invalid_missing_expected_fields",
                "examples": [invalid_case],
            }
        )


def test_claim_diagnosis_gold_set_loads() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    assert gold.evaluation_status == EVALUATION_STATUS
    assert len(gold.examples) == 10
    assert gold.examples[0].expected_claims


def test_claim_diagnosis_gold_set_defaults_to_v1() -> None:
    gold = load_claim_diagnosis_gold_set()
    assert gold.evaluation_status == "diagnostic_gold_v1_large_unvalidated"
    assert len(gold.examples) >= 50
    assert any(example.category == "security" for example in gold.examples)


def test_claim_diagnosis_harness_runs_and_produces_metrics() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold)

    assert result.evaluation_status == EVALUATION_STATUS
    assert result.a2p_mode == "v1"
    assert result.total_examples == 10
    assert 0.0 <= result.claim_label_accuracy <= 1.0
    assert 0.0 <= result.sufficiency_accuracy <= 1.0
    assert 0.0 <= result.a2p_primary_cause_accuracy <= 1.0
    assert 0.0 <= result.primary_stage_accuracy <= 1.0
    assert 0.0 <= result.fix_category_exact_accuracy <= 1.0
    assert 0.0 <= result.fix_category_partial_accuracy <= 1.0
    assert isinstance(result.false_clean_count, int)
    assert "unsupported" in result.claim_label_breakdown
    assert "contradicted" in result.claim_label_breakdown

    report = render_claim_diagnosis_report(result)
    assert "Claim-Level Diagnostic Evaluation Harness" in report
    assert "Per-example results:" in report
    assert "false_clean_count=" in report


def test_claim_diagnosis_harness_detects_intentional_mismatch() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    # Force an intentional mismatch in expected stage for one case.
    gold.examples[0].expected_primary_stage = "RETRIEVAL"

    result = run_claim_diagnosis_harness(gold)

    assert result.mismatches
    assert any("stage mismatch" in "; ".join(mismatch["notes"]) for mismatch in result.mismatches)


def test_mismatch_report_contains_case_id_expected_actual() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    gold.examples[0].expected_primary_stage = "RETRIEVAL"

    result = run_claim_diagnosis_harness(gold)

    mismatch = result.mismatches[0]
    assert mismatch["case_id"] == gold.examples[0].case_id
    assert "expected" in mismatch
    assert "actual" in mismatch
    assert mismatch["expected"]["primary_stage"] == "RETRIEVAL"
    assert "primary_stage" in mismatch["actual"]


def test_claim_diagnosis_harness_generates_json_report(tmp_path: Path) -> None:
    result = run_claim_diagnosis_harness(load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0"))
    output_path = tmp_path / "claim_diagnosis_report.json"

    write_claim_diagnosis_report(result, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["total_examples"] == 10
    assert payload["case_count"] == 10
    assert payload["aggregate_metrics"]["false_clean_count"] == result.false_clean_count
    assert "claim_label_breakdown" in payload["aggregate_metrics"]
    assert "mismatches" in payload


def test_claim_diagnosis_harness_generates_markdown_report(tmp_path: Path) -> None:
    result = run_claim_diagnosis_harness(load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0"))
    output_path = tmp_path / "claim_diagnosis_report.md"

    write_claim_diagnosis_markdown_report(result, output_path)

    markdown = output_path.read_text(encoding="utf-8")
    assert "# Claim-Level Diagnostic Evaluation Report" in markdown
    assert "false_clean_count" in markdown
    assert "claim_label_breakdown" in markdown


def test_claim_diagnosis_harness_observes_non_null_sufficiency_for_some_cases() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold)

    assert any(case.sufficiency_pass for case in result.per_example)


def test_claim_diagnosis_gold_loads_new_axis_fields() -> None:
    """Test that new axis fields load correctly from gold."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")

    # Check stale_source_case has expected_freshness_validity
    stale_case = next((ex for ex in gold.examples if ex.case_id == "stale_source_case"), None)
    assert stale_case is not None
    assert stale_case.expected_claims
    assert stale_case.expected_claims[0].expected_freshness_validity == "stale"

    # Check citation_mismatch_case has expected_citation_validity
    citation_case = next((ex for ex in gold.examples if ex.case_id == "citation_mismatch_case"), None)
    assert citation_case is not None
    assert citation_case.expected_claims
    assert citation_case.expected_claims[0].expected_citation_validity == "invalid"


def test_stale_source_case_separates_claim_support_from_freshness() -> None:
    """Test that stale source case can have claim_label and freshness_validity independent."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    stale_case = next((ex for ex in gold.examples if ex.case_id == "stale_source_case"), None)
    assert stale_case is not None

    # The case should have expected_freshness_validity = stale
    assert stale_case.expected_claims[0].expected_freshness_validity == "stale"
    # The claim_label should follow value-aware contradiction semantics.
    expected_label = stale_case.expected_claims[0].expected_claim_label or stale_case.expected_claims[0].expected_label
    assert expected_label == "contradicted"


def test_citation_mismatch_case_separates_claim_support_from_citation_validity() -> None:
    """Test that citation mismatch case can have claim_label and citation_validity independent."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    citation_case = next((ex for ex in gold.examples if ex.case_id == "citation_mismatch_case"), None)
    assert citation_case is not None

    # The case should have expected_citation_validity = invalid
    assert citation_case.expected_claims[0].expected_citation_validity == "invalid"
    # The claim_label should follow value-aware contradiction semantics.
    expected_label = citation_case.expected_claims[0].expected_claim_label or citation_case.expected_claims[0].expected_label
    assert expected_label == "contradicted"


def test_weak_ambiguous_case_uses_contradicted_for_explicit_value_conflict() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    case = next((ex for ex in gold.examples if ex.case_id == "weak_ambiguous_case"), None)
    assert case is not None
    expected_label = case.expected_claims[0].expected_claim_label or case.expected_claims[0].expected_label
    assert expected_label == "contradicted"
    assert case.expected_claims[0].expected_a2p_primary_cause == "generation_contradicted_retrieved_evidence"


def test_unsupported_missing_cases_remain_unsupported() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    for case_id in ("unsupported_missing_1", "unsupported_missing_2"):
        case = next((ex for ex in gold.examples if ex.case_id == case_id), None)
        assert case is not None
        expected_label = case.expected_claims[0].expected_claim_label or case.expected_claims[0].expected_label
        assert expected_label == "unsupported"


def test_evaluator_tracks_separate_axis_metrics() -> None:
    """Test that evaluator computes separate accuracy for each axis."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold)

    # Check that all axis metrics exist
    assert hasattr(result, "claim_label_accuracy")
    assert hasattr(result, "citation_validity_accuracy")
    assert hasattr(result, "freshness_validity_accuracy")
    assert hasattr(result, "sufficiency_accuracy")
    assert hasattr(result, "a2p_primary_cause_accuracy")

    # All should be floats between 0 and 1
    assert 0.0 <= result.claim_label_accuracy <= 1.0
    assert 0.0 <= result.citation_validity_accuracy <= 1.0
    assert 0.0 <= result.freshness_validity_accuracy <= 1.0
    assert 0.0 <= result.sufficiency_accuracy <= 1.0
    assert 0.0 <= result.a2p_primary_cause_accuracy <= 1.0


def test_markdown_report_includes_axis_sections() -> None:
    """Test that markdown report includes per-axis mismatch sections."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold)
    report = render_claim_diagnosis_report(result)

    # Check for axis labels in report
    assert "Axis A - Claim Support" in report
    assert "Axis B - Citation Validity" in report
    assert "Axis C - Freshness Validity" in report
    assert "Axis D - Context Sufficiency" in report
    assert "Axis E - A2P Root Cause" in report


def test_report_includes_category_metrics_for_v1() -> None:
    gold = load_claim_diagnosis_gold_set()
    result = run_claim_diagnosis_harness(gold)
    report = render_claim_diagnosis_report(result)

    assert "Category Metrics:" in report
    assert "a2p_mode=v1" in report
    assert "security" in result.category_metrics


def test_claim_diagnosis_harness_reports_v2_mode_when_enabled() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold, engine_config={"enable_a2p": True, "use_a2p_v2": True})

    assert result.a2p_mode == "v2"


def test_claim_diagnosis_harness_reports_disabled_mode_when_a2p_disabled() -> None:
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold, engine_config={"enable_a2p": False})

    assert result.a2p_mode == "disabled"


def test_per_case_result_includes_all_axis_fields() -> None:
    """Test that per-case results track all axes."""
    gold = load_claim_diagnosis_gold_set("claim_diagnosis_gold_v0")
    result = run_claim_diagnosis_harness(gold)

    for case in result.per_example:
        assert hasattr(case, "claim_label_pass")
        assert hasattr(case, "citation_validity_pass")
        assert hasattr(case, "freshness_validity_pass")
        assert hasattr(case, "sufficiency_pass")
        assert hasattr(case, "a2p_primary_cause_pass")
        assert isinstance(case.claim_label_pass, bool)
        assert isinstance(case.citation_validity_pass, bool)
        assert isinstance(case.freshness_validity_pass, bool)
