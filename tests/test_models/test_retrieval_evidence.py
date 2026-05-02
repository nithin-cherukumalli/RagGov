"""
Tests for the retrieval evidence substrate models.
"""

import pytest
from pydantic import ValidationError

from raggov.models.retrieval_evidence import (
    CalibrationStatus,
    ChunkEvidenceProfile,
    CitationStatus,
    EvidenceRole,
    FreshnessStatus,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalEvidenceProfile,
    RetrievalMethodType,
)


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------

def test_query_relevance_label_values():
    assert QueryRelevanceLabel.RELEVANT == "relevant"
    assert QueryRelevanceLabel.PARTIAL == "partial"
    assert QueryRelevanceLabel.IRRELEVANT == "irrelevant"
    assert QueryRelevanceLabel.UNKNOWN == "unknown"


def test_relevance_method_values():
    assert RelevanceMethod.LEXICAL_OVERLAP == "lexical_overlap"
    assert RelevanceMethod.EMBEDDING_SIMILARITY == "embedding_similarity"
    assert RelevanceMethod.LLM_JUDGE == "llm_judge"
    assert RelevanceMethod.NLI == "nli"
    assert RelevanceMethod.UNAVAILABLE == "unavailable"


def test_citation_status_values():
    assert CitationStatus.CITED == "cited"
    assert CitationStatus.UNCITED == "uncited"
    assert CitationStatus.PHANTOM == "phantom"
    assert CitationStatus.UNSUPPORTED == "unsupported"
    assert CitationStatus.UNKNOWN == "unknown"


def test_freshness_status_values():
    assert FreshnessStatus.VALID == "valid"
    assert FreshnessStatus.STALE_BY_AGE == "stale_by_age"
    assert FreshnessStatus.SUPERSEDED == "superseded"
    assert FreshnessStatus.UNKNOWN == "unknown"


def test_evidence_role_values():
    assert EvidenceRole.NECESSARY_SUPPORT == "necessary_support"
    assert EvidenceRole.PARTIAL_SUPPORT == "partial_support"
    assert EvidenceRole.BACKGROUND == "background"
    assert EvidenceRole.NOISE == "noise"
    assert EvidenceRole.CONTRADICTION == "contradiction"
    assert EvidenceRole.UNKNOWN == "unknown"


def test_retrieval_method_type_values():
    assert RetrievalMethodType.HEURISTIC_BASELINE == "heuristic_baseline"
    assert RetrievalMethodType.PRACTICAL_APPROXIMATION == "practical_approximation"
    assert RetrievalMethodType.RESEARCH_FAITHFUL == "research_faithful"


def test_calibration_status_values():
    assert CalibrationStatus.UNCALIBRATED == "uncalibrated"
    assert CalibrationStatus.MOCK_CALIBRATED == "mock_calibrated"
    assert CalibrationStatus.CALIBRATED == "calibrated"


# ---------------------------------------------------------------------------
# ChunkEvidenceProfile creation and defaults
# ---------------------------------------------------------------------------

def test_chunk_evidence_profile_minimal():
    profile = ChunkEvidenceProfile(chunk_id="c1")
    assert profile.chunk_id == "c1"
    assert profile.source_doc_id is None
    assert profile.query_relevance_label == QueryRelevanceLabel.UNKNOWN
    assert profile.query_relevance_score is None
    assert profile.relevance_method == RelevanceMethod.UNAVAILABLE
    assert profile.supported_claim_ids == []
    assert profile.contradicted_claim_ids == []
    assert profile.neutral_claim_ids == []
    assert profile.citation_status == CitationStatus.UNKNOWN
    assert profile.freshness_status == FreshnessStatus.UNKNOWN
    assert profile.evidence_role == EvidenceRole.UNKNOWN
    assert profile.warnings == []


def test_chunk_evidence_profile_mutable_defaults_are_independent():
    p1 = ChunkEvidenceProfile(chunk_id="c1")
    p2 = ChunkEvidenceProfile(chunk_id="c2")
    p1.supported_claim_ids.append("claim-1")
    assert p2.supported_claim_ids == []


def test_chunk_evidence_profile_full():
    profile = ChunkEvidenceProfile(
        chunk_id="c42",
        source_doc_id="doc-99",
        query_relevance_label=QueryRelevanceLabel.RELEVANT,
        query_relevance_score=0.87,
        relevance_method=RelevanceMethod.LEXICAL_OVERLAP,
        supported_claim_ids=["claim-1", "claim-2"],
        contradicted_claim_ids=["claim-3"],
        neutral_claim_ids=["claim-4"],
        citation_status=CitationStatus.CITED,
        freshness_status=FreshnessStatus.VALID,
        evidence_role=EvidenceRole.NECESSARY_SUPPORT,
        warnings=["low overlap ratio"],
    )
    assert profile.chunk_id == "c42"
    assert profile.source_doc_id == "doc-99"
    assert profile.query_relevance_label == "relevant"
    assert profile.query_relevance_score == 0.87
    assert profile.relevance_method == "lexical_overlap"
    assert len(profile.supported_claim_ids) == 2
    assert profile.citation_status == "cited"
    assert profile.freshness_status == "valid"
    assert profile.evidence_role == "necessary_support"
    assert profile.warnings == ["low overlap ratio"]


def test_chunk_evidence_profile_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ChunkEvidenceProfile(chunk_id="c1", unknown_field="oops")


# ---------------------------------------------------------------------------
# RetrievalEvidenceProfile creation and defaults
# ---------------------------------------------------------------------------

def test_retrieval_evidence_profile_minimal():
    profile = RetrievalEvidenceProfile()
    assert profile.run_id is None
    assert profile.overall_retrieval_status == "unknown"
    assert profile.chunks == []
    assert profile.missing_evidence_claim_ids == []
    assert profile.noisy_chunk_ids == []
    assert profile.contradictory_pairs == []
    assert profile.phantom_citation_doc_ids == []
    assert profile.stale_doc_ids == []
    assert profile.method_type == RetrievalMethodType.HEURISTIC_BASELINE
    assert profile.calibration_status == CalibrationStatus.UNCALIBRATED
    assert profile.recommended_for_gating is False
    assert profile.limitations == []


def test_retrieval_evidence_profile_v0_defaults_are_heuristic_and_uncalibrated():
    profile = RetrievalEvidenceProfile()
    assert profile.method_type == RetrievalMethodType.HEURISTIC_BASELINE
    assert profile.calibration_status == CalibrationStatus.UNCALIBRATED
    assert profile.recommended_for_gating is False


def test_retrieval_evidence_profile_mutable_defaults_are_independent():
    p1 = RetrievalEvidenceProfile()
    p2 = RetrievalEvidenceProfile()
    p1.noisy_chunk_ids.append("c1")
    assert p2.noisy_chunk_ids == []


def test_retrieval_evidence_profile_with_chunks():
    chunk = ChunkEvidenceProfile(
        chunk_id="c1",
        query_relevance_label=QueryRelevanceLabel.PARTIAL,
        citation_status=CitationStatus.UNCITED,
    )
    profile = RetrievalEvidenceProfile(
        run_id="run-001",
        overall_retrieval_status="degraded",
        chunks=[chunk],
        noisy_chunk_ids=["c1"],
        limitations=["lexical overlap only, no semantic model"],
    )
    assert profile.run_id == "run-001"
    assert profile.overall_retrieval_status == "degraded"
    assert len(profile.chunks) == 1
    assert profile.chunks[0].chunk_id == "c1"
    assert profile.noisy_chunk_ids == ["c1"]
    assert len(profile.limitations) == 1


def test_retrieval_evidence_profile_contradictory_pairs():
    profile = RetrievalEvidenceProfile(
        contradictory_pairs=[("c1", "c2"), ("c3", "c4")],
    )
    assert len(profile.contradictory_pairs) == 2
    assert profile.contradictory_pairs[0] == ("c1", "c2")


def test_retrieval_evidence_profile_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RetrievalEvidenceProfile(nonexistent_field=True)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_chunk_evidence_profile_serialization():
    profile = ChunkEvidenceProfile(
        chunk_id="c5",
        query_relevance_label=QueryRelevanceLabel.IRRELEVANT,
        citation_status=CitationStatus.PHANTOM,
        evidence_role=EvidenceRole.NOISE,
    )
    data = profile.model_dump()
    assert data["chunk_id"] == "c5"
    assert data["query_relevance_label"] == "irrelevant"
    assert data["citation_status"] == "phantom"
    assert data["evidence_role"] == "noise"

    restored = ChunkEvidenceProfile(**data)
    assert restored.query_relevance_label == QueryRelevanceLabel.IRRELEVANT
    assert restored.citation_status == CitationStatus.PHANTOM


def test_retrieval_evidence_profile_serialization():
    chunk = ChunkEvidenceProfile(chunk_id="c1", freshness_status=FreshnessStatus.STALE_BY_AGE)
    profile = RetrievalEvidenceProfile(
        run_id="run-abc",
        chunks=[chunk],
        stale_doc_ids=["doc-old"],
        method_type=RetrievalMethodType.HEURISTIC_BASELINE,
        calibration_status=CalibrationStatus.UNCALIBRATED,
    )

    data = profile.model_dump()
    assert data["run_id"] == "run-abc"
    assert data["method_type"] == "heuristic_baseline"
    assert data["calibration_status"] == "uncalibrated"
    assert data["chunks"][0]["freshness_status"] == "stale_by_age"

    restored = RetrievalEvidenceProfile(**data)
    assert restored.method_type == RetrievalMethodType.HEURISTIC_BASELINE
    assert restored.chunks[0].freshness_status == FreshnessStatus.STALE_BY_AGE


def test_retrieval_evidence_profile_json_round_trip():
    profile = RetrievalEvidenceProfile(
        run_id="run-json",
        contradictory_pairs=[("c1", "c2")],
        limitations=["heuristic only"],
    )
    json_str = profile.model_dump_json()
    restored = RetrievalEvidenceProfile.model_validate_json(json_str)
    assert restored.run_id == "run-json"
    assert restored.contradictory_pairs == [("c1", "c2")]
    assert restored.limitations == ["heuristic only"]
