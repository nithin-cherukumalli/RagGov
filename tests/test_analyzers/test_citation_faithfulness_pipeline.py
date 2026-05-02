"""Pipeline integration tests for CitationFaithfulnessAnalyzerV0."""

from __future__ import annotations

from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.engine import DiagnosisEngine, diagnose
from raggov.io.serialize import diagnosis_to_dict
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import CitationSupportLabel
from raggov.models.diagnosis import AnalyzerResult
from raggov.models.grounding import ClaimEvidenceRecord, ClaimVerificationLabel
from raggov.models.run import RAGRun


class StaticClaimGroundingAnalyzer(ClaimGroundingAnalyzer):
    """Test double that emits claim grounding evidence without invoking internals."""

    def __init__(self, record: ClaimEvidenceRecord) -> None:
        self.config = {}
        self.weight = 0.9
        self._record = record

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer_name="ClaimGroundingAnalyzer",
            status="pass",
            evidence=["static claim grounding"],
            grounding_evidence_bundle={
                "claim_evidence_records": [self._record.model_dump(mode="json")]
            },
        )


def run() -> RAGRun:
    return RAGRun(
        run_id="run-citation-faithfulness",
        query="What is the refund policy?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Refund policy covers returns.",
                source_doc_id="doc-1",
                score=0.9,
            )
        ],
        final_answer="Refund policy covers returns.",
        cited_doc_ids=["doc-1"],
    )


def supported_record() -> ClaimEvidenceRecord:
    return ClaimEvidenceRecord(
        claim_id="claim-1",
        claim_text="Refund policy covers returns.",
        verification_label=ClaimVerificationLabel.ENTAILED,
        supporting_chunk_ids=["c1"],
    )


def test_citation_faithfulness_analyzer_is_importable() -> None:
    assert CitationFaithfulnessAnalyzerV0.__name__ == "CitationFaithfulnessAnalyzerV0"


def test_default_pipeline_order_places_citation_faithfulness_after_grounding_and_retrieval() -> None:
    engine = DiagnosisEngine(config={})
    names = [analyzer.__class__.__name__ for analyzer in engine.analyzers]

    assert "CitationFaithfulnessAnalyzerV0" in names
    assert names.index("ClaimGroundingAnalyzer") < names.index("CitationMismatchAnalyzer")
    assert names.index("CitationMismatchAnalyzer") < names.index("CitationFaithfulnessAnalyzerV0")


def test_pipeline_runs_citation_faithfulness_with_prior_claim_and_retrieval_outputs() -> None:
    engine = DiagnosisEngine(
        analyzers=[
            StaticClaimGroundingAnalyzer(supported_record()),
            CitationMismatchAnalyzer(),
            CitationFaithfulnessAnalyzerV0(),
        ]
    )

    diagnosis = engine.diagnose(run())

    assert "CitationFaithfulnessAnalyzerV0" in diagnosis.checks_run
    assert diagnosis.citation_faithfulness_report is not None
    assert diagnosis.citation_faithfulness_report.claim_grounding_used is True
    assert diagnosis.citation_faithfulness_report.retrieval_evidence_profile_used is True
    assert diagnosis.citation_faithfulness_report.records[0].citation_support_label == (
        CitationSupportLabel.FULLY_SUPPORTED
    )
    assert diagnosis.citation_faithfulness_report.records[0].faithfulness_risk == "low"


def test_report_includes_citation_faithfulness_section() -> None:
    engine = DiagnosisEngine(
        analyzers=[
            StaticClaimGroundingAnalyzer(supported_record()),
            CitationMismatchAnalyzer(),
            CitationFaithfulnessAnalyzerV0(),
        ]
    )

    payload = diagnosis_to_dict(engine.diagnose(run()))

    section = payload["citation_faithfulness_report"]
    assert section["method_type"] == "practical_approximation"
    assert section["calibration_status"] == "uncalibrated"
    assert section["recommended_for_gating"] is False
    assert section["limitations"]
    assert section["unsupported_claim_ids"] == []
    assert section["phantom_citation_doc_ids"] == []
    assert section["missing_citation_claim_ids"] == []
    assert section["contradicted_claim_ids"] == []
    assert section["claim_grounding_used"] is True
    assert section["retrieval_evidence_profile_used"] is True
    assert section["legacy_citation_fallback_used"] is True
    record = section["records"][0]
    assert record["claim_id"] == "claim-1"
    assert record["claim_text"] == "Refund policy covers returns."
    assert record["cited_doc_ids"] == ["doc-1"]
    assert record["cited_chunk_ids"] == ["c1"]
    assert record["supporting_chunk_ids"] == ["c1"]
    assert record["citation_support_label"] == "fully_supported"
    assert record["faithfulness_risk"] == "low"
    assert record["evidence_source"] == "legacy_citation_ids"
    assert "explanation" in record
    assert record["warnings"] == []


def test_report_clearly_marks_unavailable_dependencies() -> None:
    test_run = run()
    test_run.metadata["claim_evidence_records"] = [
        supported_record().model_dump(mode="json")
    ]
    test_run.retrieval_evidence_profile = None

    result = CitationFaithfulnessAnalyzerV0().analyze(test_run)

    assert result.citation_faithfulness_report is not None
    report = result.citation_faithfulness_report
    assert report.claim_grounding_used is True
    assert report.retrieval_evidence_profile_used is False
    assert report.legacy_citation_fallback_used is True


def test_default_diagnose_serializes_citation_faithfulness_report() -> None:
    test_run = run()
    diagnosis = diagnose(test_run)
    payload = diagnosis_to_dict(diagnosis)

    assert "CitationFaithfulnessAnalyzerV0" in payload["checks_run"]
    assert "citation_faithfulness_report" in payload
