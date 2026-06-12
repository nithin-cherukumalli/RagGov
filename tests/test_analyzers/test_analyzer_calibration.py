"""Calibration tests for analyzer warn/fail separation."""

from __future__ import annotations

from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.analyzers.security.anomalies import RetrievalAnomalyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import CitationFaithfulnessReport
from raggov.models.diagnosis import AnalyzerResult, FailureType, SufficiencyResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    EvidenceRole,
    QueryRelevanceLabel,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun
from raggov.models.version_validity import VersionValidityReport


def _chunk(chunk_id: str, text: str, doc_id: str = "doc-1", score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_doc_id=doc_id,
        score=score,
    )


def _claim_record(
    *,
    cited_doc_ids: list[str] | None = None,
    supporting_chunk_ids: list[str] | None = None,
) -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(
        claim_id="claim-1",
        claim_text="The answer is supported by retrieved evidence.",
        verification_label=ClaimVerificationLabel.ENTAILED,
        cited_doc_ids=cited_doc_ids or [],
        supporting_chunk_ids=supporting_chunk_ids or ["c1"],
    )


def test_citation_faithfulness_keeps_clean_cited_answer_pass() -> None:
    run = RAGRun(
        query="What is the answer?",
        retrieved_chunks=[_chunk("c1", "The answer is supported by retrieved evidence.")],
        final_answer="The answer is supported by retrieved evidence. [doc-1]",
        cited_doc_ids=["doc-1"],
        metadata={"claim_evidence_records": [_claim_record(cited_doc_ids=["doc-1"])]},
    )

    result = CitationFaithfulnessAnalyzerV0().analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None


def test_citation_faithfulness_keeps_best_doc_legacy_partial_support_pass() -> None:
    run = RAGRun(
        query="What is the answer?",
        retrieved_chunks=[
            _chunk("c1", "The answer is supported by retrieved evidence.", "doc-1"),
            _chunk("c2", "Supplemental retrieved evidence also overlaps.", "doc-2"),
        ],
        final_answer="The answer is supported by retrieved evidence. [doc-1]",
        cited_doc_ids=["doc-1"],
        metadata={
            "claim_evidence_records": [
                _claim_record(supporting_chunk_ids=["c1", "c2"])
            ]
        },
    )

    result = CitationFaithfulnessAnalyzerV0().analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None


def test_citation_faithfulness_fails_supported_claim_with_no_citations() -> None:
    run = RAGRun(
        query="What is the answer?",
        retrieved_chunks=[_chunk("c1", "The answer is supported by retrieved evidence.")],
        final_answer="The answer is supported by retrieved evidence.",
        cited_doc_ids=[],
        metadata={"claim_evidence_records": [_claim_record()]},
    )

    result = CitationFaithfulnessAnalyzerV0().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.CITATION_MISMATCH


def test_retrieval_anomaly_fails_near_duplicate_chunks() -> None:
    run = RAGRun(
        query="What is the policy on overtime?",
        retrieved_chunks=[
            _chunk("c1", "Overtime requires approval.", score=0.9),
            _chunk("c2", "Overtime requires manager approval.", score=0.85),
        ],
        final_answer="Overtime requires approval.",
    )

    result = RetrievalAnomalyAnalyzer().analyze(run)

    assert result.status == "fail"
    assert result.failure_type == FailureType.RETRIEVAL_ANOMALY
    assert result.evidence == ["near duplicate chunks c1 and c2 overlap=1.00"]


def test_retrieval_diagnosis_passes_single_noisy_chunk_below_warn_minimum() -> None:
    run = RAGRun(
        query="Which records must be retained for seven years?",
        retrieved_chunks=[
            _chunk("c1", "Signed vendor contracts must be retained for seven years.", score=0.95),
            _chunk("c2", "Draft contracts may be discarded after supersession.", score=0.82),
        ],
        final_answer="Signed vendor contracts must be retained for seven years.",
        cited_doc_ids=["doc-1"],
        citation_faithfulness_report=CitationFaithfulnessReport(run_id="retrieval-clean"),
        version_validity_report=VersionValidityReport(run_id="retrieval-clean"),
        retrieval_evidence_profile=RetrievalEvidenceProfile(
            chunks=[
                ChunkEvidenceProfile(chunk_id="c1", evidence_role=EvidenceRole.NECESSARY_SUPPORT),
                ChunkEvidenceProfile(chunk_id="c2", evidence_role=EvidenceRole.NOISE),
            ],
            noisy_chunk_ids=["c2"],
        ),
    )

    result = RetrievalDiagnosisAnalyzerV0(
        {
            "prior_results": [
                AnalyzerResult(
                    analyzer_name="SufficiencyAnalyzer",
                    status="pass",
                    sufficiency_result=SufficiencyResult(sufficient=True, method="test"),
                ),
                AnalyzerResult(
                    analyzer_name="ClaimGroundingAnalyzer",
                    status="pass",
                    grounding_evidence_bundle={"claim_evidence_records": []},
                ),
            ]
        }
    ).analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None


def test_scope_violation_passes_when_relevant_chunks_cover_irrelevant_tail() -> None:
    run = RAGRun(
        query="Which records must be retained for seven years?",
        retrieved_chunks=[
            _chunk("c1", "Signed vendor contracts must be retained for seven years.", score=0.95),
            _chunk("c2", "Draft contracts may be discarded after supersession.", score=0.82),
        ],
        final_answer="Signed vendor contracts must be retained for seven years.",
        retrieval_evidence_profile=RetrievalEvidenceProfile(
            chunks=[
                ChunkEvidenceProfile(
                    chunk_id="c1",
                    query_relevance_label=QueryRelevanceLabel.RELEVANT,
                ),
                ChunkEvidenceProfile(
                    chunk_id="c2",
                    query_relevance_label=QueryRelevanceLabel.IRRELEVANT,
                ),
            ]
        ),
    )

    result = ScopeViolationAnalyzer().analyze(run)

    assert result.status == "pass"
    assert result.failure_type is None
