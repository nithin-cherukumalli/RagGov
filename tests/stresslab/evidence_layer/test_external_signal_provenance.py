from __future__ import annotations

import json

from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer
from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.citation_faithfulness import CitationSupportLabel
from raggov.models.grounding import ClaimEvidenceRecord
from raggov.models.run import RAGRun
from tests.stresslab.evidence_layer import diagnose_fixture, load_evidence_case


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat(self, prompt: str) -> str:
        return self.responses.pop(0)


def _candidate(chunk_id: str, text: str, source_doc_id: str = "doc-1") -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        chunk_text=text,
        chunk_text_preview=text,
        lexical_overlap_score=1.0,
        anchor_overlap_score=1.0,
        value_overlap_score=0.0,
        retrieval_score=None,
        rerank_score=None,
    )


def _chunk(chunk_id: str, doc_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_doc_id=doc_id, text=text, score=None)


def _claim_llm_json(**overrides: object) -> str:
    payload = {
        "claim_id": "claim_001",
        "label": "unsupported",
        "supporting_chunk_ids": [],
        "contradicting_chunk_ids": [],
        "missing_evidence": ["refund duration"],
        "reason": "No chunk states the duration.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _citation_llm_json(**overrides: object) -> str:
    payload = {
        "claim_id": "claim-1",
        "cited_doc_id": "doc-1",
        "cited_chunk_id": "c1",
        "label": "does_not_support",
        "reason": "The cited source does not support the claim.",
        "evidence_quote": "Citation text.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_claim_external_signal_provenance_is_preserved() -> None:
    client = FakeLLM([_claim_llm_json()])
    run = RAGRun(
        query="What is the refund period?",
        final_answer="The refund policy covers hardware returns for thirty days.",
        retrieved_chunks=[_chunk("c1", "doc-1", "Refunds may be available.")],
    )
    analyzer = ClaimGroundingAnalyzer(
        {
            "claim_verifier": "structured_llm",
            "llm_client": client,
            "candidate_mode": "retrieved_top_k",
        }
    )

    result = analyzer.analyze(run)

    assert result.grounding_evidence_bundle is not None
    bundle_record = result.grounding_evidence_bundle.external_signal_records[0]
    claim_record = result.grounding_evidence_bundle.claim_evidence_records[0]
    claim_signal = claim_record.external_signal_records[0]

    assert bundle_record["provider"] == ExternalEvaluatorProvider.structured_llm.value
    assert bundle_record["signal_type"] == ExternalSignalType.claim_support.value
    assert bundle_record["label"] == "unsupported"
    assert bundle_record["calibration_status"] == "uncalibrated_locally"
    assert bundle_record["recommended_for_gating"] is False
    assert claim_signal == bundle_record


def test_citation_external_signal_provenance_is_preserved() -> None:
    client = FakeLLM([_citation_llm_json(label="does_not_support")])
    record = ClaimEvidenceRecord(
        claim_id="claim-1",
        claim_text="Refunds are available for thirty days.",
        verification_label="entailed",
        cited_doc_ids=["doc-1"],
        cited_chunk_ids=["c1"],
        candidate_evidence_chunk_ids=["c1"],
    )
    run = RAGRun(
        query="query",
        final_answer="Refunds are available for thirty days.",
        retrieved_chunks=[_chunk("c1", "doc-1", "Refunds may be available.")],
        metadata={"claim_evidence_records": [record.model_dump(mode="json")]},
    )

    result = CitationFaithfulnessAnalyzerV0(
        {"citation_verifier": "structured_llm", "llm_client": client}
    ).analyze(run)

    assert result.citation_faithfulness_report is not None
    report_record = result.citation_faithfulness_report.records[0]
    assert report_record.external_signal_provider == ExternalEvaluatorProvider.structured_llm.value
    assert report_record.external_signal_label == "does_not_support"
    assert report_record.citation_support_label == CitationSupportLabel.UNSUPPORTED
    assert report_record.external_signal_raw_payload is not None
    assert report_record.external_signal_raw_payload["label"] == "does_not_support"


def test_retrieval_external_signal_provenance_is_preserved() -> None:
    case = load_evidence_case("clean_policy_native").model_copy(
        update={
            "mode": "external-enhanced",
            "enabled_external_providers": ["ragas", "deepeval", "ragchecker"],
            "mock_external_results": [
                ExternalEvaluationResult(
                    provider=ExternalEvaluatorProvider.ragas,
                    adapter_name="ragas",
                    succeeded=True,
                    signals=[
                        ExternalSignalRecord(
                            provider=ExternalEvaluatorProvider.ragas,
                            signal_type=ExternalSignalType.retrieval_context_precision,
                            metric_name="context_precision",
                            value=0.21,
                            label="low",
                            affected_chunk_ids=["clean-native-chunk-1"],
                            raw_payload={"metric": "context_precision"},
                        )
                    ],
                ).model_dump(mode="json"),
                ExternalEvaluationResult(
                    provider=ExternalEvaluatorProvider.deepeval,
                    adapter_name="deepeval",
                    succeeded=True,
                    signals=[
                        ExternalSignalRecord(
                            provider=ExternalEvaluatorProvider.deepeval,
                            signal_type=ExternalSignalType.retrieval_contextual_relevancy,
                            metric_name="contextual_relevancy",
                            value=0.19,
                            label="low",
                            affected_chunk_ids=["clean-native-chunk-1"],
                            raw_payload={"metric": "contextual_relevancy"},
                        )
                    ],
                ).model_dump(mode="json"),
                ExternalEvaluationResult(
                    provider=ExternalEvaluatorProvider.ragchecker,
                    adapter_name="ragchecker",
                    succeeded=True,
                    signals=[
                        ExternalSignalRecord(
                            provider=ExternalEvaluatorProvider.ragchecker,
                            signal_type=ExternalSignalType.context_utilization,
                            metric_name="context_precision",
                            value=0.33,
                            label="low",
                            affected_chunk_ids=["clean-native-chunk-1"],
                            raw_payload={"metric": "context_precision"},
                        )
                    ],
                ).model_dump(mode="json"),
            ],
        }
    )

    diagnosis = diagnose_fixture(case)

    profile = case.to_run().retrieval_evidence_profile
    assert diagnosis.retrieval_diagnosis_report is not None
    run = case.to_run()
    diagnosis = diagnose_fixture(case)
    profile = run.retrieval_evidence_profile
    assert profile is None
    # The authoritative profile lives on the diagnosed run, so inspect the report provenance instead.
    provider_sources = {
        signal.source_report
        for signal in diagnosis.retrieval_diagnosis_report.evidence_signals
        if signal.source_report is not None and signal.source_report.startswith("ExternalEvaluationResult:")
    }
    assert "ExternalEvaluationResult:ragas" in provider_sources
    assert "ExternalEvaluationResult:deepeval" in provider_sources
    assert "ExternalEvaluationResult:ragchecker" in provider_sources
    for signal in diagnosis.retrieval_diagnosis_report.evidence_signals:
        if not (signal.source_report or "").startswith("ExternalEvaluationResult:"):
            continue
        assert "uncalibrated locally" in (signal.limitation or "").lower()
