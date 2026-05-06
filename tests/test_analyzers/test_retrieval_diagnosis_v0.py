"""Tests for RetrievalDiagnosisAnalyzerV0."""

from __future__ import annotations

from raggov.analyzers.retrieval_diagnosis import RetrievalDiagnosisAnalyzerV0
from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import CitationFaithfulnessReport
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType, SufficiencyResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.retrieval_diagnosis import RetrievalFailureType
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    EvidenceRole,
    QueryRelevanceLabel,
    RelevanceMethod,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun
from raggov.models.version_validity import (
    DocumentValidityRecord,
    DocumentValidityRisk,
    DocumentValidityStatus,
    VersionValidityReport,
)


def chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {doc_id}",
        source_doc_id=doc_id,
        score=0.8,
    )


def run(
    *,
    chunks: list[RetrievedChunk] | None = None,
    cited_doc_ids: list[str] | None = None,
    retrieval_profile: RetrievalEvidenceProfile | None = None,
    citation_report: CitationFaithfulnessReport | None = None,
    version_report: VersionValidityReport | None = None,
) -> RAGRun:
    return RAGRun(
        run_id="run-retrieval-diagnosis",
        query="query",
        retrieved_chunks=chunks or [],
        final_answer="Answer.",
        cited_doc_ids=cited_doc_ids or [],
        retrieval_evidence_profile=retrieval_profile,
        citation_faithfulness_report=citation_report,
        version_validity_report=version_report,
    )


def analyze(test_run: RAGRun, prior_results: list[AnalyzerResult] | None = None):
    result = RetrievalDiagnosisAnalyzerV0({"prior_results": prior_results or []}).analyze(test_run)
    assert result.retrieval_diagnosis_report is not None
    return result, result.retrieval_diagnosis_report


def test_no_retrieved_chunks_fails_as_retrieval_miss() -> None:
    result, report = analyze(run())

    assert result.status == "fail"
    assert result.failure_type == FailureType.INSUFFICIENT_CONTEXT
    assert result.stage == FailureStage.RETRIEVAL
    assert report.primary_failure_type == RetrievalFailureType.RETRIEVAL_MISS
    assert report.evidence_signals[0].signal_name == "no_retrieved_chunks"


def test_invalid_cited_docs_fail_as_version_retrieval_failure() -> None:
    version_report = VersionValidityReport(
        run_id="run-retrieval-diagnosis",
        document_records=[
            DocumentValidityRecord(
                doc_id="doc-old",
                validity_status=DocumentValidityStatus.SUPERSEDED,
                validity_risk=DocumentValidityRisk.HIGH,
            )
        ],
        superseded_doc_ids=["doc-old"],
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-old", "doc-old")], cited_doc_ids=["doc-old"], version_report=version_report)
    )

    assert result.status == "fail"
    assert report.primary_failure_type == RetrievalFailureType.VERSION_RETRIEVAL_FAILURE
    assert report.invalid_cited_doc_ids == ["doc-old"]
    assert report.evidence_signals[0].source_report == "VersionValidityReport"


def test_invalid_retrieved_uncited_docs_warn() -> None:
    version_report = VersionValidityReport(
        run_id="run-retrieval-diagnosis",
        document_records=[
            DocumentValidityRecord(
                doc_id="doc-old",
                validity_status=DocumentValidityStatus.EXPIRED,
                validity_risk=DocumentValidityRisk.HIGH,
            )
        ],
        expired_doc_ids=["doc-old"],
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-old", "doc-old")], cited_doc_ids=[], version_report=version_report)
    )

    assert result.status == "warn"
    assert report.primary_failure_type == RetrievalFailureType.VERSION_RETRIEVAL_FAILURE
    assert report.invalid_retrieved_doc_ids == ["doc-old"]
    assert "answer did not cite them" in report.alternative_explanations[0]


def test_phantom_citation_fails_as_citation_retrieval_mismatch() -> None:
    citation_report = CitationFaithfulnessReport(
        run_id="run-retrieval-diagnosis",
        phantom_citation_doc_ids=["doc-ghost"],
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-1", "doc-1")], cited_doc_ids=["doc-ghost"], citation_report=citation_report)
    )

    assert result.status == "fail"
    assert result.failure_type == FailureType.CITATION_MISMATCH
    assert report.primary_failure_type == RetrievalFailureType.CITATION_RETRIEVAL_MISMATCH
    assert report.evidence_signals[0].source_ids == ["doc-ghost"]


def test_insufficient_context_and_unsupported_claims_fail_as_retrieval_miss() -> None:
    sufficiency = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="fail",
        sufficiency_result=SufficiencyResult(
            sufficient=False,
            sufficiency_label="insufficient",
            affected_claims=["claim-1"],
            missing_evidence=["missing support"],
            method="test",
        ),
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        grounding_evidence_bundle={
            "claim_evidence_records": [
                ClaimEvidenceRecord(
                    claim_id="claim-1",
                    claim_text="Unsupported claim.",
                    verification_label=ClaimVerificationLabel.INSUFFICIENT,
                    candidate_evidence_chunk_ids=["chunk-1"],
                )
            ]
        },
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-1", "doc-1")]),
        prior_results=[sufficiency, grounding],
    )

    assert result.status == "fail"
    assert report.primary_failure_type == RetrievalFailureType.RETRIEVAL_MISS
    assert report.affected_claim_ids == ["claim-1"]
    assert report.claim_records[0].claim_text == "Unsupported claim."


def test_retrieval_noise_uses_existing_profile_noisy_chunks() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-retrieval-diagnosis",
        chunks=[
            ChunkEvidenceProfile(chunk_id="chunk-1", evidence_role=EvidenceRole.NOISE),
            ChunkEvidenceProfile(chunk_id="chunk-2", evidence_role=EvidenceRole.NOISE),
        ],
        noisy_chunk_ids=["chunk-1", "chunk-2"],
    )

    result, report = analyze(
        run(
            chunks=[chunk("chunk-1", "doc-1"), chunk("chunk-2", "doc-2")],
            retrieval_profile=profile,
        )
    )

    assert result.status == "warn"
    assert report.primary_failure_type == RetrievalFailureType.RETRIEVAL_NOISE
    assert report.noisy_chunk_ids == ["chunk-1", "chunk-2"]


def test_candidate_evidence_without_rank_labels_warns_rank_failure_unknown() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-retrieval-diagnosis",
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="chunk-1",
                query_relevance_label=QueryRelevanceLabel.UNKNOWN,
                evidence_role=EvidenceRole.PARTIAL_SUPPORT,
            )
        ],
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="warn",
        grounding_evidence_bundle={
            "claim_evidence_records": [
                ClaimEvidenceRecord(
                    claim_id="claim-1",
                    claim_text="Weakly supported claim.",
                    verification_label=ClaimVerificationLabel.INSUFFICIENT,
                    candidate_evidence_chunk_ids=["chunk-1"],
                )
            ]
        },
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-1", "doc-1")], retrieval_profile=profile),
        prior_results=[grounding],
    )

    assert result.status == "warn"
    assert result.failure_type == FailureType.RERANKER_FAILURE
    assert report.primary_failure_type == RetrievalFailureType.RANK_FAILURE_UNKNOWN
    assert report.candidate_chunk_ids == ["chunk-1"]


def test_missing_upstream_reports_warn_without_silent_fallback() -> None:
    result, report = analyze(run(chunks=[chunk("chunk-1", "doc-1")]))

    assert result.status == "warn"
    assert report.primary_failure_type == RetrievalFailureType.INSUFFICIENT_EVIDENCE_TO_DIAGNOSE
    assert "retrieval_evidence_profile" in report.missing_reports
    assert "citation_faithfulness_report" in report.missing_reports
    assert "version_validity_report" in report.missing_reports
    assert report.fallback_heuristics_used == []


def test_no_claims_skip_does_not_require_grounding_or_citation_reports() -> None:
    profile = RetrievalEvidenceProfile(run_id="run-retrieval-diagnosis", chunks=[])
    version = VersionValidityReport(run_id="run-retrieval-diagnosis")
    grounding_skip = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="skip",
        evidence=["no claims extracted from final answer"],
    )
    sufficiency = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="pass",
        sufficiency_result=SufficiencyResult(sufficient=True, method="test"),
    )
    analyzer = RetrievalDiagnosisAnalyzerV0(
        {"mode": "external-enhanced", "prior_results": [grounding_skip, sufficiency]}
    )

    result = analyzer.analyze(
        run(
            chunks=[chunk("chunk-1", "doc-1")],
            retrieval_profile=profile,
            version_report=version,
        )
    )
    report = result.retrieval_diagnosis_report

    assert result.status == "pass"
    assert report is not None
    assert report.primary_failure_type == RetrievalFailureType.NO_CLEAR_RETRIEVAL_FAILURE


def test_default_engine_places_retrieval_diagnosis_after_source_validity() -> None:
    names = [analyzer.__class__.__name__ for analyzer in DiagnosisEngine(config={}).analyzers]

    assert names.index("CitationFaithfulnessAnalyzerV0") < names.index(
        "TemporalSourceValidityAnalyzerV1"
    )
    assert names.index("TemporalSourceValidityAnalyzerV1") < names.index(
        "RetrievalDiagnosisAnalyzerV0"
    )
    assert names.index("RetrievalDiagnosisAnalyzerV0") < names.index("Layer6TaxonomyClassifier")


def test_externally_irrelevant_signals_trigger_retrieval_miss() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-retrieval-diagnosis",
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="chunk-1",
                query_relevance_label=QueryRelevanceLabel.IRRELEVANT,
                relevance_method=RelevanceMethod.CROSS_ENCODER,
            )
        ],
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        grounding_evidence_bundle={
            "claim_evidence_records": [
                ClaimEvidenceRecord(
                    claim_id="claim-1",
                    claim_text="Unsupported claim.",
                    verification_label=ClaimVerificationLabel.INSUFFICIENT,
                    candidate_evidence_chunk_ids=["chunk-1"],
                )
            ]
        },
    )

    result, report = analyze(
        run(chunks=[chunk("chunk-1", "doc-1")], retrieval_profile=profile),
        prior_results=[grounding],
    )

    assert result.status == "fail"
    assert report.primary_failure_type == RetrievalFailureType.RETRIEVAL_MISS
    assert report.affected_claim_ids == ["claim-1"]


def test_cross_encoder_provenance_is_preserved_when_external_relevance_affects_diagnosis() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-retrieval-diagnosis",
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="chunk-1",
                query_relevance_label=QueryRelevanceLabel.IRRELEVANT,
                query_relevance_score=-2.0,
                relevance_method=RelevanceMethod.CROSS_ENCODER,
            )
        ],
        external_signals=[
            {
                "provider": "cross_encoder",
                "signal_type": "retrieval_relevance",
                "metric_name": "cross_encoder_relevance",
                "value": -2.0,
                "label": "irrelevant",
                "affected_chunk_ids": ["chunk-1"],
                "calibration_status": "uncalibrated_locally",
                "recommended_for_gating": False,
            }
        ],
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        grounding_evidence_bundle={
            "claim_evidence_records": [
                ClaimEvidenceRecord(
                    claim_id="claim-1",
                    claim_text="Unsupported claim.",
                    verification_label=ClaimVerificationLabel.INSUFFICIENT,
                    candidate_evidence_chunk_ids=["chunk-1"],
                )
            ]
        },
    )

    _, report = analyze(
        run(chunks=[chunk("chunk-1", "doc-1")], retrieval_profile=profile),
        prior_results=[grounding],
    )

    external_signals = [
        signal for signal in report.evidence_signals
        if signal.source_report == "ExternalEvaluationResult:cross_encoder"
    ]
    assert external_signals
    assert external_signals[0].signal_name == "cross_encoder_relevance"
    assert external_signals[0].source_ids == ["chunk-1"]


def test_missing_external_signal_warns_in_external_enhanced_mode() -> None:
    profile = RetrievalEvidenceProfile(
        run_id="run-retrieval-diagnosis",
        chunks=[
            ChunkEvidenceProfile(
                chunk_id="chunk-1",
                query_relevance_label=QueryRelevanceLabel.RELEVANT,
                relevance_method=RelevanceMethod.LEXICAL_OVERLAP,
            )
        ],
    )
    # Mock prior results to satisfy other checks
    sufficiency = AnalyzerResult(
        analyzer_name="SufficiencyAnalyzer",
        status="pass",
        sufficiency_result=SufficiencyResult(sufficient=True, method="test"),
    )
    grounding = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="pass",
        grounding_evidence_bundle={"claim_evidence_records": []},
    )
    citation = CitationFaithfulnessReport(run_id="run-retrieval-diagnosis")
    version = VersionValidityReport(run_id="run-retrieval-diagnosis")

    analyzer = RetrievalDiagnosisAnalyzerV0(
        {
            "mode": "external-enhanced",
            "prior_results": [sufficiency, grounding],
            "enabled_external_providers": ["cross_encoder_relevance"],
            "retrieval_relevance_provider": "cross_encoder",
        }
    )
    test_run = run(chunks=[chunk("chunk-1", "doc-1")], retrieval_profile=profile, citation_report=citation, version_report=version)
    
    result = analyzer.analyze(test_run)
    report = result.retrieval_diagnosis_report

    assert "external_retrieval_relevance_signal" in report.missing_reports
