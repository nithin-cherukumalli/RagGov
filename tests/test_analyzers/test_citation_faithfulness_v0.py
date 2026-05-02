"""Tests for CitationFaithfulnessAnalyzerV0."""

from __future__ import annotations

from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import (
    CitationCalibrationStatus,
    CitationFaithfulnessRisk,
    CitationMethodType,
    CitationSupportLabel,
)
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun


def chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source_doc_id=doc_id,
        score=None,
    )


def claim_record(
    claim_id: str,
    *,
    label: ClaimVerificationLabel = ClaimVerificationLabel.ENTAILED,
    cited_doc_ids: list[str] | None = None,
    cited_chunk_ids: list[str] | None = None,
    supporting_chunk_ids: list[str] | None = None,
    contradicting_chunk_ids: list[str] | None = None,
    candidate_evidence_chunk_ids: list[str] | None = None,
) -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(
        claim_id=claim_id,
        claim_text=f"{claim_id} text",
        verification_label=label,
        cited_doc_ids=cited_doc_ids or [],
        cited_chunk_ids=cited_chunk_ids or [],
        supporting_chunk_ids=supporting_chunk_ids or [],
        contradicting_chunk_ids=contradicting_chunk_ids or [],
        candidate_evidence_chunk_ids=candidate_evidence_chunk_ids or [],
    )


def run_with_records(
    records: list[ClaimEvidenceRecord],
    *,
    cited_doc_ids: list[str] | None = None,
    chunks: list[RetrievedChunk] | None = None,
    retrieval_evidence_profile: RetrievalEvidenceProfile | None = None,
    final_answer: str = "Answer with claims.",
) -> RAGRun:
    return RAGRun(
        query="query",
        retrieved_chunks=chunks or [chunk("c1", "doc-1")],
        final_answer=final_answer,
        cited_doc_ids=cited_doc_ids or [],
        retrieval_evidence_profile=retrieval_evidence_profile,
        metadata={"claim_evidence_records": records},
    )


def analyze(run: RAGRun):
    result = CitationFaithfulnessAnalyzerV0().analyze(run)
    assert result.citation_faithfulness_report is not None
    return result, result.citation_faithfulness_report


def test_skips_when_no_claim_evidence_exists() -> None:
    run = RAGRun(
        query="query",
        retrieved_chunks=[chunk("c1", "doc-1")],
        final_answer="Answer.",
    )

    result = CitationFaithfulnessAnalyzerV0().analyze(run)

    assert result.status == "skip"
    assert result.evidence == [
        "no claim evidence available for citation faithfulness analysis"
    ]
    assert result.citation_faithfulness_report is None


def test_skips_when_no_generated_answer() -> None:
    run = run_with_records([claim_record("claim-1")], final_answer="")

    result = CitationFaithfulnessAnalyzerV0().analyze(run)

    assert result.status == "skip"
    assert result.evidence == [
        "no claim evidence available for citation faithfulness analysis"
    ]


def test_detects_missing_citation() -> None:
    result, report = analyze(run_with_records([claim_record("claim-1")]))

    record = report.records[0]
    assert result.status == "warn"
    assert record.citation_support_label == CitationSupportLabel.CITATION_MISSING
    assert record.faithfulness_risk == CitationFaithfulnessRisk.HIGH
    assert report.missing_citation_claim_ids == ["claim-1"]


def test_detects_phantom_citation() -> None:
    result, report = analyze(
        run_with_records(
            [claim_record("claim-1", cited_doc_ids=["doc-ghost"])],
            chunks=[chunk("c1", "doc-1")],
        )
    )

    record = report.records[0]
    assert result.status == "fail"
    assert record.citation_support_label == CitationSupportLabel.CITATION_PHANTOM
    assert record.faithfulness_risk == CitationFaithfulnessRisk.HIGH
    assert report.phantom_citation_doc_ids == ["doc-ghost"]


def test_detects_fully_supported_citation() -> None:
    result, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    cited_doc_ids=["doc-1"],
                    supporting_chunk_ids=["c1"],
                )
            ],
            chunks=[chunk("c1", "doc-1")],
        )
    )

    record = report.records[0]
    assert result.status == "pass"
    assert record.citation_support_label == CitationSupportLabel.FULLY_SUPPORTED
    assert record.faithfulness_risk == CitationFaithfulnessRisk.LOW


def test_detects_fully_supported_citation_from_retrieval_profile() -> None:
    profile = RetrievalEvidenceProfile(
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="c1",
                source_doc_id="doc-1",
                supported_claim_ids=["claim-1"],
            )
        ]
    )
    result, report = analyze(
        run_with_records(
            [claim_record("claim-1", cited_doc_ids=["doc-1"])],
            chunks=[chunk("c1", "doc-1")],
            retrieval_evidence_profile=profile,
        )
    )

    assert result.status == "pass"
    assert report.records[0].supporting_chunk_ids == ["c1"]
    assert report.records[0].citation_support_label == CitationSupportLabel.FULLY_SUPPORTED


def test_detects_partially_supported_citation() -> None:
    result, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    cited_doc_ids=["doc-1"],
                    supporting_chunk_ids=["c1", "c2"],
                )
            ],
            chunks=[chunk("c1", "doc-1"), chunk("c2", "doc-2")],
        )
    )

    record = report.records[0]
    assert result.status == "pass"
    assert record.citation_support_label == CitationSupportLabel.PARTIALLY_SUPPORTED
    assert record.faithfulness_risk == CitationFaithfulnessRisk.MEDIUM


def test_detects_unsupported_citation_from_claim_support_label() -> None:
    result, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    label=ClaimVerificationLabel.INSUFFICIENT,
                    cited_doc_ids=["doc-1"],
                )
            ],
            chunks=[chunk("c1", "doc-1")],
        )
    )

    assert result.status == "warn"
    assert report.records[0].citation_support_label == CitationSupportLabel.UNSUPPORTED
    assert report.records[0].faithfulness_risk == CitationFaithfulnessRisk.HIGH


def test_detects_unsupported_citation() -> None:
    result, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    cited_doc_ids=["doc-2"],
                    supporting_chunk_ids=["c1"],
                )
            ],
            chunks=[chunk("c1", "doc-1"), chunk("c2", "doc-2")],
        )
    )

    record = report.records[0]
    assert result.status == "warn"
    assert record.citation_support_label == CitationSupportLabel.UNSUPPORTED
    assert record.faithfulness_risk == CitationFaithfulnessRisk.HIGH
    assert report.unsupported_claim_ids == ["claim-1"]
    assert record.explanation == (
        "claim appears supported by retrieved context, but not by cited source"
    )


def test_detects_contradicted_citation_from_claim_evidence() -> None:
    result, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    cited_doc_ids=["doc-1"],
                    contradicting_chunk_ids=["c1"],
                )
            ],
            chunks=[chunk("c1", "doc-1")],
        )
    )

    record = report.records[0]
    assert result.status == "fail"
    assert record.citation_support_label == CitationSupportLabel.CONTRADICTED
    assert record.faithfulness_risk == CitationFaithfulnessRisk.HIGH
    assert report.contradicted_claim_ids == ["claim-1"]


def test_detects_contradicted_citation_from_retrieval_profile() -> None:
    profile = RetrievalEvidenceProfile(
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="c1",
                source_doc_id="doc-1",
                contradicted_claim_ids=["claim-1"],
            )
        ]
    )
    result, report = analyze(
        run_with_records(
            [claim_record("claim-1", cited_doc_ids=["doc-1"])],
            chunks=[chunk("c1", "doc-1")],
            retrieval_evidence_profile=profile,
        )
    )

    assert result.status == "fail"
    assert report.records[0].citation_support_label == CitationSupportLabel.CONTRADICTED


def test_report_metadata_says_practical_approximation_and_uncalibrated() -> None:
    _, report = analyze(
        run_with_records(
            [
                claim_record(
                    "claim-1",
                    cited_doc_ids=["doc-1"],
                    supporting_chunk_ids=["c1"],
                )
            ],
            chunks=[chunk("c1", "doc-1")],
        )
    )

    assert report.method_type == CitationMethodType.PRACTICAL_APPROXIMATION
    assert report.calibration_status == CitationCalibrationStatus.UNCALIBRATED
    assert report.recommended_for_gating is False
    assert "v0 checks citation support using existing claim grounding and retrieval evidence only" in report.limitations
