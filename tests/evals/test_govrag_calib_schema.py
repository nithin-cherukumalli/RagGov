"""
Tests for GovRAG-Calib schema (GovRAGCalibCase, ClaimLabel, CitationLabel, RetrievedChunk).

Covers:
  - test_govrag_calib_schema_accepts_valid_case
  - test_govrag_calib_rejects_missing_primary_failure
  - test_govrag_calib_rejects_invalid_stage
  - test_govrag_calib_validator_detects_duplicate_case_ids
  - test_claim_label_evidence_ids_must_reference_chunks
  - test_citation_label_ids_must_reference_docs_or_chunks
"""

from __future__ import annotations

import sys
from pathlib import Path
import json

import pytest

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "evals"), str(PROJECT_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from govrag_calib.schema import (
    CitationLabel,
    ClaimLabel,
    GovRAGCalibCase,
    RetrievedChunk,
)

import validate_govrag_calib


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_chunk(chunk_id: str = "chunk-1", doc_id: str = "doc-1", rank: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": "Sample text for testing.",
        "rank": rank,
        "score": 0.85,
    }


def _valid_case_data(**overrides) -> dict:
    base = {
        "case_id": "gc-test-001",
        "domain": "software",
        "source_type": "fixture",
        "query": "What is the default timeout?",
        "retrieved_chunks": [_minimal_chunk()],
        "answer": "The default timeout is 30 seconds.",
        "citations": ["doc-1"],
        "expected_primary_failure": "UNSUPPORTED_CLAIM",
        "expected_stage": "GROUNDING",
        "expected_first_failing_node": "ClaimGroundingAnalyzer",
        "expected_root_cause": "Timeout not in context.",
        "expected_secondary_failures": [],
        "expected_claim_labels": [],
        "expected_citation_labels": [],
        "expected_retrieval_issue": "none",
        "expected_sufficiency_issue": "none",
        "expected_version_issue": "none",
        "expected_answer_quality_issue": "none",
        "expected_security_issue": "none",
        "expected_fix_category": "add_citation_grounding",
        "expected_human_review_required": False,
        "label_source": "human",
        "label_confidence": "high",
        "split": "train",
        "notes": "Test case.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------

class TestSchemaAcceptsValidCase:
    def test_minimal_valid_case(self):
        data = _valid_case_data()
        case = GovRAGCalibCase(**data)
        assert case.case_id == "gc-test-001"
        assert case.expected_primary_failure == "UNSUPPORTED_CLAIM"
        assert case.expected_stage == "GROUNDING"
        assert case.label_source == "human"
        assert case.label_confidence == "high"
        assert case.split == "train"

    def test_clean_case_accepted(self):
        data = _valid_case_data(
            expected_primary_failure="CLEAN",
            expected_stage="UNKNOWN",
            expected_root_cause=None,
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_primary_failure == "CLEAN"

    def test_all_label_sources_accepted(self):
        for src in ["human", "synthetic_mutation", "public_dataset_mapped", "benchmark_migrated"]:
            data = _valid_case_data(label_source=src)
            case = GovRAGCalibCase(**data)
            assert case.label_source == src

    def test_all_splits_accepted(self):
        for split in ["train", "dev", "heldout", "unset"]:
            data = _valid_case_data(split=split)
            case = GovRAGCalibCase(**data)
            assert case.split == split

    def test_all_label_confidences_accepted(self):
        for conf in ["high", "medium", "low"]:
            data = _valid_case_data(label_confidence=conf)
            case = GovRAGCalibCase(**data)
            assert case.label_confidence == conf

    def test_case_with_claim_labels_accepted(self):
        data = _valid_case_data(
            expected_claim_labels=[{
                "claim_id": "cl-1",
                "claim_text": "The timeout is 30s.",
                "expected_label": "unsupported",
                "expected_evidence_chunk_ids": [],
                "expected_failure_reason": "Not in context.",
                "critical_entities": [],
                "critical_values": ["30s"],
                "critical_dates": [],
            }]
        )
        case = GovRAGCalibCase(**data)
        assert len(case.expected_claim_labels) == 1
        assert case.expected_claim_labels[0].claim_id == "cl-1"

    def test_case_with_citation_labels_accepted(self):
        data = _valid_case_data(
            expected_citation_labels=[{
                "citation_id": "cit-1",
                "cited_doc_id": "doc-1",
                "cited_chunk_id": None,
                "expected_label": "phantom",
                "expected_reason": "Not retrieved.",
            }]
        )
        case = GovRAGCalibCase(**data)
        assert len(case.expected_citation_labels) == 1

    def test_security_case_with_human_review_accepted(self):
        data = _valid_case_data(
            expected_primary_failure="PROMPT_INJECTION",
            expected_stage="SECURITY",
            expected_human_review_required=True,
            expected_security_issue="prompt_injection_chunk",
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_human_review_required is True

    def test_optional_fields_default_correctly(self):
        data = _valid_case_data()
        case = GovRAGCalibCase(**data)
        assert case.expected_secondary_failures == []
        assert case.expected_claim_labels == []
        assert case.expected_citation_labels == []
        assert case.notes == "Test case."

    def test_notes_none_accepted(self):
        data = _valid_case_data(notes=None)
        case = GovRAGCalibCase(**data)
        assert case.notes is None


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------

class TestSchemaRejectsMissingPrimaryFailure:
    def test_missing_primary_failure(self):
        data = _valid_case_data()
        del data["expected_primary_failure"]
        with pytest.raises(Exception):
            GovRAGCalibCase(**data)

    def test_invalid_primary_failure_value(self):
        data = _valid_case_data(expected_primary_failure="TOTALLY_MADE_UP_FAILURE")
        with pytest.raises(Exception):
            GovRAGCalibCase(**data)

    def test_empty_primary_failure(self):
        data = _valid_case_data(expected_primary_failure="")
        with pytest.raises(Exception):
            GovRAGCalibCase(**data)


class TestSchemaRejectsInvalidStage:
    def test_invalid_stage_value(self):
        data = _valid_case_data(expected_stage="NOT_A_REAL_STAGE")
        with pytest.raises(Exception):
            GovRAGCalibCase(**data)

    def test_missing_stage(self):
        data = _valid_case_data()
        del data["expected_stage"]
        with pytest.raises(Exception):
            GovRAGCalibCase(**data)

    def test_all_valid_stages_accepted(self):
        valid_stages = [
            "PARSING", "CHUNKING", "EMBEDDING", "RETRIEVAL", "RERANKING",
            "CONTEXT_ASSEMBLY", "GROUNDING", "SUFFICIENCY", "GENERATION",
            "CITATION", "SECURITY", "CONFIDENCE", "UNKNOWN",
        ]
        for stage in valid_stages:
            data = _valid_case_data(expected_stage=stage)
            case = GovRAGCalibCase(**data)
            assert case.expected_stage == stage


# ---------------------------------------------------------------------------
# Duplicate case ID detection (via validator script logic)
# ---------------------------------------------------------------------------

class TestValidatorDetectsDuplicateCaseIds:
    def test_duplicate_ids_detected(self):
        """The validator's validate_dataset function should catch duplicates."""
        # We test the logic directly by simulating the Counter check.
        from collections import Counter

        case_ids = ["gc-001", "gc-002", "gc-001", "gc-003"]
        counter = Counter(case_ids)
        duplicates = [cid for cid, cnt in counter.items() if cnt > 1]
        assert "gc-001" in duplicates
        assert "gc-002" not in duplicates

    def test_unique_ids_not_flagged(self):
        from collections import Counter

        case_ids = ["gc-001", "gc-002", "gc-003"]
        counter = Counter(case_ids)
        duplicates = [cid for cid, cnt in counter.items() if cnt > 1]
        assert len(duplicates) == 0

    def test_schema_accepts_two_cases_with_different_ids(self):
        """Two separate cases with different IDs should both validate."""
        c1 = GovRAGCalibCase(**_valid_case_data(case_id="gc-test-001"))
        c2 = GovRAGCalibCase(**_valid_case_data(case_id="gc-test-002"))
        assert c1.case_id != c2.case_id


# ---------------------------------------------------------------------------
# Claim label evidence ID must reference existing chunks
# ---------------------------------------------------------------------------

class TestClaimLabelEvidenceIdsMustReferenceChunks:
    def test_valid_evidence_chunk_ref_accepted(self):
        data = _valid_case_data(
            retrieved_chunks=[_minimal_chunk("chunk-1", "doc-1")],
            expected_claim_labels=[{
                "claim_id": "cl-1",
                "claim_text": "Some claim.",
                "expected_label": "supported",
                "expected_evidence_chunk_ids": ["chunk-1"],
                "expected_failure_reason": None,
                "critical_entities": [],
                "critical_values": [],
                "critical_dates": [],
            }]
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_claim_labels[0].expected_evidence_chunk_ids == ["chunk-1"]

    def test_invalid_evidence_chunk_ref_rejected(self):
        data = _valid_case_data(
            retrieved_chunks=[_minimal_chunk("chunk-1", "doc-1")],
            expected_claim_labels=[{
                "claim_id": "cl-1",
                "claim_text": "Some claim.",
                "expected_label": "supported",
                "expected_evidence_chunk_ids": ["chunk-DOES-NOT-EXIST"],
                "expected_failure_reason": None,
                "critical_entities": [],
                "critical_values": [],
                "critical_dates": [],
            }]
        )
        with pytest.raises(Exception, match="chunk_id"):
            GovRAGCalibCase(**data)

    def test_empty_evidence_ids_accepted(self):
        """Unsupported claims have empty evidence_chunk_ids — this is valid."""
        data = _valid_case_data(
            expected_claim_labels=[{
                "claim_id": "cl-1",
                "claim_text": "Fabricated claim.",
                "expected_label": "unsupported",
                "expected_evidence_chunk_ids": [],
                "expected_failure_reason": "Not in context.",
                "critical_entities": [],
                "critical_values": [],
                "critical_dates": [],
            }]
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_claim_labels[0].expected_evidence_chunk_ids == []


# ---------------------------------------------------------------------------
# Citation label chunk ID must reference existing chunks
# ---------------------------------------------------------------------------

class TestCitationLabelIdsMustReferenceDocsOrChunks:
    def test_valid_citation_with_chunk_ref_accepted(self):
        data = _valid_case_data(
            retrieved_chunks=[_minimal_chunk("chunk-1", "doc-1")],
            expected_citation_labels=[{
                "citation_id": "cit-1",
                "cited_doc_id": "doc-1",
                "cited_chunk_id": "chunk-1",
                "expected_label": "supports",
                "expected_reason": None,
            }]
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_citation_labels[0].cited_chunk_id == "chunk-1"

    def test_phantom_citation_without_chunk_ref_accepted(self):
        """Phantom citations reference docs not in the retrieved set — no chunk_id."""
        data = _valid_case_data(
            expected_citation_labels=[{
                "citation_id": "cit-1",
                "cited_doc_id": "phantom-doc-99",
                "cited_chunk_id": None,
                "expected_label": "phantom",
                "expected_reason": "Doc not retrieved.",
            }]
        )
        case = GovRAGCalibCase(**data)
        assert case.expected_citation_labels[0].cited_chunk_id is None

    def test_invalid_cited_chunk_id_rejected(self):
        """A cited_chunk_id that doesn't exist in retrieved_chunks must be rejected."""
        data = _valid_case_data(
            retrieved_chunks=[_minimal_chunk("chunk-1", "doc-1")],
            expected_citation_labels=[{
                "citation_id": "cit-1",
                "cited_doc_id": "doc-1",
                "cited_chunk_id": "chunk-DOES-NOT-EXIST",
                "expected_label": "supports",
                "expected_reason": None,
            }]
        )
        with pytest.raises(Exception, match="cited_chunk_id"):
            GovRAGCalibCase(**data)

    def test_multiple_citation_labels_accepted(self):
        data = _valid_case_data(
            expected_citation_labels=[
                {"citation_id": "cit-1", "cited_doc_id": "phantom-1", "cited_chunk_id": None, "expected_label": "phantom", "expected_reason": None},
                {"citation_id": "cit-2", "cited_doc_id": "phantom-2", "cited_chunk_id": None, "expected_label": "phantom", "expected_reason": None},
            ]
        )
        case = GovRAGCalibCase(**data)
        assert len(case.expected_citation_labels) == 2

    def test_all_citation_label_values_accepted(self):
        for label in ["supports", "does_not_support", "phantom", "post_rationalized", "missing_required", "contradicted"]:
            data = _valid_case_data(
                expected_citation_labels=[{
                    "citation_id": "cit-1",
                    "cited_doc_id": "any-doc",
                    "cited_chunk_id": None,
                    "expected_label": label,
                    "expected_reason": None,
                }]
            )
            case = GovRAGCalibCase(**data)
            assert case.expected_citation_labels[0].expected_label == label


# ---------------------------------------------------------------------------
# GovRAG-Calib-150 JSONL validator
# ---------------------------------------------------------------------------

CALIB_ROOT = PROJECT_ROOT / "evals" / "govrag_calib"


def _load_seed_records() -> list[dict]:
    return [
        json.loads(line)
        for line in (CALIB_ROOT / "calib_150_seed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestGovRAGCalib150Validator:
    def test_schema_json_exists(self):
        schema_path = CALIB_ROOT / "schema.json"
        assert schema_path.exists()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert "case_id" in schema["required"]
        assert "calibration_status" in schema["required"]

    def test_seed_dataset_has_at_least_ten_valid_cases(self):
        records = _load_seed_records()
        assert len(records) >= 10
        result = validate_govrag_calib.validate_dataset(CALIB_ROOT / "calib_150_seed.jsonl")
        assert result.ok, result.errors
        assert result.total_cases >= 10
        assert any("target is 150" in warning for warning in result.warnings)

    def test_validator_rejects_duplicate_case_id(self, tmp_path):
        records = _load_seed_records()[:2]
        records[1]["case_id"] = records[0]["case_id"]
        dataset = tmp_path / "dupe.jsonl"
        dataset.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert not result.ok
        assert any("duplicate case_id" in error for error in result.errors)

    def test_validator_rejects_unknown_affected_claim_id(self, tmp_path):
        record = _load_seed_records()[0]
        record["expected_affected_claim_ids"] = ["missing-claim"]
        dataset = tmp_path / "bad_claim_ref.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert not result.ok
        assert any("affected claim_id" in error for error in result.errors)

    def test_validator_rejects_invalid_alternative_failure(self, tmp_path):
        record = _load_seed_records()[0]
        record["acceptable_alternative_failures"] = ["NOT_A_FAILURE"]
        dataset = tmp_path / "bad_alt.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert not result.ok
        assert any("acceptable_alternative_failures" in error for error in result.errors)

    def test_validator_rejects_unlocked_heldout_case(self, tmp_path):
        record = _load_seed_records()[0]
        record["calibration_split"] = "heldout"
        record["calibration_status"] = "reviewed"
        dataset = tmp_path / "bad_heldout.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert not result.ok
        assert any("heldout cases require" in error for error in result.errors)

    def test_validator_rejects_production_gating_field(self, tmp_path):
        record = _load_seed_records()[0]
        record["metadata"] = {"production_gating_eligible": False}
        dataset = tmp_path / "bad_gating.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert not result.ok
        assert any("production_gating_eligible" in error for error in result.errors)

    def test_validator_warns_for_citation_family_without_citations(self, tmp_path):
        record = _load_seed_records()[0]
        record.update(
            {
                "case_id": "gc-warning-citation",
                "failure_family": "citation",
                "expected_primary_failure": "CITATION_MISMATCH",
                "expected_stage": "CITATION",
                "citations": [],
                "expected_affected_claim_ids": ["claim-1"],
                "expected_affected_doc_ids": ["doc1"],
            }
        )
        dataset = tmp_path / "citation_warning.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert result.ok, result.errors
        assert any("citation-family case has no citations" in warning for warning in result.warnings)

    def test_validator_warns_for_non_clean_empty_affected_ids(self, tmp_path):
        record = _load_seed_records()[0]
        record.update(
            {
                "case_id": "gc-warning-affected",
                "failure_family": "grounding",
                "expected_primary_failure": "UNSUPPORTED_CLAIM",
                "expected_stage": "GROUNDING",
                "expected_affected_claim_ids": [],
                "expected_affected_doc_ids": [],
            }
        )
        dataset = tmp_path / "affected_warning.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert result.ok, result.errors
        assert any("empty expected_affected_claim_ids" in warning for warning in result.warnings)
        assert any("empty expected_affected_doc_ids" in warning for warning in result.warnings)

    def test_validator_warns_without_implying_production_gating(self, tmp_path):
        record = _load_seed_records()[0]
        record.update(
            {
                "case_id": "gc-warning-security-review",
                "failure_family": "security_privacy",
                "expected_primary_failure": "PROMPT_INJECTION",
                "expected_stage": "SECURITY",
                "difficulty": "high",
                "adversarial": True,
                "security_relevant": True,
                "expected_human_review_required": False,
            }
        )
        dataset = tmp_path / "security_warning.jsonl"
        dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = validate_govrag_calib.validate_dataset(dataset)

        assert result.ok, result.errors
        assert any("does not require human review" in warning for warning in result.warnings)
        assert not any("production_gating_eligible" in warning for warning in result.warnings)
