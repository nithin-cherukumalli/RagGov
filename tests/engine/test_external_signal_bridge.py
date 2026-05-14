"""Tests for the external signal diagnosis bridge."""

from __future__ import annotations

import pytest

from raggov.engine import DiagnosisEngine
from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
    ExternalSignalRecord,
    ExternalSignalType,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.grounding import ClaimEvidenceRecord, GroundingEvidenceBundle
from raggov.models.retrieval_evidence import (
    ChunkEvidenceProfile,
    QueryRelevanceLabel,
    RetrievalEvidenceProfile,
)
from raggov.models.run import RAGRun
from raggov.analyzers.base import BaseAnalyzer
from raggov.analyzers.retrieval.evidence_profile import RetrievalEvidenceProfilerV0
from raggov.external_signal_bridge import build_external_signal_diagnosis_probes


class MockAnalyzer(BaseAnalyzer):
    def __init__(self, result: AnalyzerResult):
        super().__init__()
        self.result = result

    def analyze(self, run: RAGRun) -> AnalyzerResult:
        return self.result


def _build_run(signals: list[ExternalSignalRecord] | None = None) -> RAGRun:
    run = RAGRun(
        run_id="test-run",
        query="query",
        retrieved_chunks=[RetrievedChunk(chunk_id="1", text="text", source_doc_id="doc1", score=0.9)],
        final_answer="answer",
    )
    if signals:
        run.metadata["external_evaluation_results"] = [
            ExternalEvaluationResult(
                provider=signals[0].provider,
                succeeded=True,
                signals=signals,
            ).model_dump()
        ]
    return run


def _clean_analyzer() -> BaseAnalyzer:
    return MockAnalyzer(
        AnalyzerResult(
            analyzer_name="CleanAnalyzer",
            status="pass",
            failure_type=FailureType.CLEAN,
            stage=FailureStage.UNKNOWN,
        )
    )


def _run_with_retrieval_profile(
    chunk_profile: ChunkEvidenceProfile,
) -> RAGRun:
    signal = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.ragas,
        signal_type=ExternalSignalType.retrieval_context_precision,
        metric_name="context_precision",
        value=0.2,
        label="low",
        affected_chunk_ids=[chunk_profile.chunk_id],
        affected_doc_ids=[chunk_profile.source_doc_id] if chunk_profile.source_doc_id else [],
    )
    run = _build_run([signal])
    run.retrieval_evidence_profile = RetrievalEvidenceProfile(
        run_id=run.run_id,
        chunks=[chunk_profile],
    )
    return run


def test_bridge_handles_query_relevance_label() -> None:
    run = _run_with_retrieval_profile(
        ChunkEvidenceProfile(
            chunk_id="c1",
            source_doc_id="doc1",
            query_relevance_label=QueryRelevanceLabel.IRRELEVANT,
        )
    )

    probes = build_external_signal_diagnosis_probes(run, [])

    assert len(probes) == 1
    probe = probes[0]
    assert probe.affected_chunk_ids == ["c1"]
    assert probe.affected_doc_ids == ["doc1"]
    assert any("relevance label=irrelevant" in item for item in probe.native_evidence_found)


def test_bridge_handles_native_relevance_label() -> None:
    run = _run_with_retrieval_profile(
        ChunkEvidenceProfile(
            chunk_id="c1",
            source_doc_id="doc1",
            query_relevance_label=QueryRelevanceLabel.UNKNOWN,
            native_relevance_label=QueryRelevanceLabel.IRRELEVANT,
        )
    )

    probes = build_external_signal_diagnosis_probes(run, [])

    assert len(probes) == 1
    assert any("relevance label=irrelevant" in item for item in probes[0].native_evidence_found)


def test_bridge_handles_missing_relevance_label_gracefully() -> None:
    run = _run_with_retrieval_profile(
        ChunkEvidenceProfile(
            chunk_id="c1",
            source_doc_id="doc1",
        )
    )

    probes = build_external_signal_diagnosis_probes(run, [])

    assert len(probes) == 1
    assert probes[0].native_evidence_found == []
    assert any("relevance label" in item.lower() for item in probes[0].native_evidence_missing)


def test_external_signal_provenance_real_profile_no_attribute_error() -> None:
    signal = ExternalSignalRecord(
        provider=ExternalEvaluatorProvider.ragas,
        signal_type=ExternalSignalType.retrieval_context_precision,
        metric_name="context_precision",
        value=0.2,
        label="low",
        affected_chunk_ids=["c1"],
        affected_doc_ids=["doc1"],
    )
    run = _build_run([signal])
    run.query = "refund period"
    run.retrieved_chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_doc_id="doc1",
            text="The cafeteria menu changes every Friday.",
            score=0.9,
        )
    ]
    run.retrieval_evidence_profile = RetrievalEvidenceProfilerV0().build(run)

    probes = build_external_signal_diagnosis_probes(run, [])

    assert len(probes) == 1
    probe = probes[0]
    assert probe.affected_chunk_ids == ["c1"]
    assert probe.affected_doc_ids == ["doc1"]
    assert probe.provider == ExternalEvaluatorProvider.ragas.value
    assert probe.metric_name == "context_precision"
    assert probe.signal_type == ExternalSignalType.retrieval_context_precision.value
    assert any("relevance label=irrelevant" in item for item in probe.native_evidence_found)
    assert probe.model_dump(mode="json")["affected_chunk_ids"] == ["c1"]


def test_bridge_handles_grounding_record_verification_label() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.faithfulness,
            metric_name="faithfulness",
            value=0.2,
        )
    ])
    grounding_result = AnalyzerResult(
        analyzer_name="ClaimGroundingAnalyzer",
        status="fail",
        failure_type=FailureType.UNSUPPORTED_CLAIM,
        stage=FailureStage.GROUNDING,
        grounding_evidence_bundle=GroundingEvidenceBundle(
            claim_evidence_records=[
                ClaimEvidenceRecord(
                    claim_id="claim-1",
                    claim_text="unsupported claim",
                    verification_label="unsupported",
                )
            ]
        ),
    )

    probes = build_external_signal_diagnosis_probes(run, [grounding_result])

    assert len(probes) == 1
    assert any("unsupported or contradicted claims" in item for item in probes[0].native_evidence_found)


def test_ragas_context_precision_low_triggers_retrieval_precision_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_precision,
            metric_name="context_precision",
            value=0.2,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "retrieval_precision"
    assert "RetrievalDiagnosisAnalyzerV0" in probe.native_analyzers_to_check


def test_ragas_context_recall_low_triggers_retrieval_coverage_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_recall,
            metric_name="context_recall",
            value=0.3,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    assert diagnosis.external_diagnosis_probes[0].suspected_pipeline_node == "retrieval_coverage"


def test_deepeval_contextual_relevancy_low_triggers_scope_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.deepeval,
            signal_type=ExternalSignalType.retrieval_contextual_relevancy,
            metric_name="contextual_relevancy",
            value=0.1,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "retrieval_precision"
    assert probe.suspected_failure_type == FailureType.SCOPE_VIOLATION.value


def test_deepeval_contextual_precision_low_triggers_reranking_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.deepeval,
            signal_type=ExternalSignalType.retrieval_contextual_precision,
            metric_name="contextual_precision",
            value=0.2,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "retrieval_precision"
    assert probe.suspected_failure_type == FailureType.RERANKER_FAILURE.value


def test_ragas_faithfulness_low_triggers_claim_support_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.faithfulness,
            metric_name="faithfulness",
            value=0.4,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "claim_support"
    assert probe.suspected_failure_type == FailureType.UNSUPPORTED_CLAIM.value


def test_ragchecker_hallucination_high_triggers_claim_grounding_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragchecker,
            signal_type=ExternalSignalType.hallucination,
            metric_name="hallucination",
            value=0.9,
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    assert diagnosis.external_diagnosis_probes[0].suspected_pipeline_node == "claim_support"


def test_refchecker_claim_contradicted_triggers_high_severity_claim_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.refchecker,
            signal_type=ExternalSignalType.claim_support,
            metric_name="claim_support",
            value="contradicted",
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.severity == "high"
    assert probe.suspected_failure_type == FailureType.CONTRADICTED_CLAIM.value


def test_refchecker_citation_does_not_support_triggers_citation_probe() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.refchecker,
            signal_type=ExternalSignalType.citation_support,
            metric_name="citation_support",
            value="does_not_support",
        )
    ])
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()]).diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    assert diagnosis.external_diagnosis_probes[0].suspected_pipeline_node == "citation_support"


def test_high_external_bad_signal_blocks_clean() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_recall,
            metric_name="context_recall",
            value=0.1,  # triggers high severity probe
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.LOW_CONFIDENCE
    # Ensure evidence indicates the block
    assert any("blocked pending re-check" in e for e in diagnosis.evidence)


def test_external_bad_signal_does_not_override_prompt_injection() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_recall,
            metric_name="context_recall",
            value=0.1,  # high severity probe
        )
    ])
    injection_analyzer = MockAnalyzer(
        AnalyzerResult(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
        )
    )
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[injection_analyzer]).diagnose(run)
    # The primary failure should remain PROMPT_INJECTION, it should not be blocked/overridden
    assert diagnosis.primary_failure == FailureType.PROMPT_INJECTION
    assert len(diagnosis.external_diagnosis_probes) == 1


def test_external_bad_signal_with_matching_native_failure_strengthens_explanation() -> None:
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_precision,
            metric_name="context_precision",
            value=0.1,
        )
    ])
    from raggov.models.retrieval_diagnosis import RetrievalDiagnosisReport, RetrievalFailureType
    diag_report_dict = {
        "run_id": "test",
        "primary_failure_type": RetrievalFailureType.RETRIEVAL_NOISE,
        "recommended_fix": "fix",
        "method_type": "heuristic_baseline",
        "calibration_status": "uncalibrated",
        "evidence_signals": []
    }
    from raggov.models.retrieval_diagnosis import RetrievalDiagnosisReport
    diag_report = RetrievalDiagnosisReport(**diag_report_dict)
    diag_analyzer = MockAnalyzer(
        AnalyzerResult.model_construct(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            retrieval_diagnosis_report=diag_report,
        )
    )
    diagnosis = DiagnosisEngine(config={"mode": "native"}, analyzers=[diag_analyzer]).diagnose(run)
    probe = diagnosis.external_diagnosis_probes[0]
    # Native evidence should report noisy_context_suspected
    assert any("noisy_context_suspected=True" in ev for ev in probe.native_evidence_found)


def test_missing_reference_input_does_not_block_clean() -> None:
    # A signal with no bad metric doesn't block clean, even if missing inputs
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.custom,
            metric_name="reference_missing",
            value=None,
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    assert diagnosis.primary_failure == FailureType.CLEAN


# ---------------------------------------------------------------------------
# Regression protection: taxonomy drift
# ---------------------------------------------------------------------------

def test_retrieval_noise_external_signal_does_not_drift_to_security() -> None:
    """Retrieval noise from RAGAS must never produce a security-class primary failure."""
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragas,
            signal_type=ExternalSignalType.retrieval_context_precision,
            metric_name="context_precision",
            value=0.1,
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    security_failures = {
        FailureType.PROMPT_INJECTION.value,
        FailureType.SUSPICIOUS_CHUNK.value,
        FailureType.PRIVACY_VIOLATION.value,
    }
    actual = diagnosis.primary_failure.value if hasattr(diagnosis.primary_failure, "value") else str(diagnosis.primary_failure)
    assert actual not in security_failures, (
        f"Retrieval noise external signal drifted to security failure: '{actual}'"
    )


def test_citation_external_signal_does_not_drift_to_security() -> None:
    """Citation does_not_support from RefChecker must never cause a security primary failure."""
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.refchecker,
            signal_type=ExternalSignalType.citation_support,
            metric_name="citation_support",
            value="does_not_support",
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    security_failures = {
        FailureType.PROMPT_INJECTION.value,
        FailureType.SUSPICIOUS_CHUNK.value,
        FailureType.PRIVACY_VIOLATION.value,
    }
    actual = diagnosis.primary_failure.value if hasattr(diagnosis.primary_failure, "value") else str(diagnosis.primary_failure)
    assert actual not in security_failures


def test_ragas_low_context_metric_does_not_become_adversarial() -> None:
    """RAGAS/DeepEval low context metrics must map to retrieval nodes, not adversarial_context."""
    for signal_type, metric_name, provider in [
        (ExternalSignalType.retrieval_context_precision, "context_precision", ExternalEvaluatorProvider.ragas),
        (ExternalSignalType.retrieval_contextual_relevancy, "contextual_relevancy", ExternalEvaluatorProvider.deepeval),
    ]:
        run = _build_run([
            ExternalSignalRecord(
                provider=provider,
                signal_type=signal_type,
                metric_name=metric_name,
                value=0.1,
            )
        ])
        engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
        engine._get_missing_critical_evidence = lambda results, run: []
        diagnosis = engine.diagnose(run)
        assert len(diagnosis.external_diagnosis_probes) == 1
        probe = diagnosis.external_diagnosis_probes[0]
        # Must be a retrieval node, not adversarial
        assert probe.suspected_pipeline_node in ("retrieval_precision", "retrieval_coverage"), (
            f"Expected retrieval node, got '{probe.suspected_pipeline_node}'"
        )
        # Probe failure type must not be security-adjacent
        assert probe.suspected_failure_type not in (
            FailureType.SUSPICIOUS_CHUNK.value,
            FailureType.PROMPT_INJECTION.value,
        ), f"Probe drifted to security type: '{probe.suspected_failure_type}'"


def test_refchecker_unsupported_claim_triggers_claim_support_not_insufficient_context() -> None:
    """RefChecker claim unsupported must probe claim_support, not retrieval/insufficient_context."""
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.refchecker,
            signal_type=ExternalSignalType.claim_support,
            metric_name="claim_support",
            value="unsupported",
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "claim_support", (
        f"Expected 'claim_support', got '{probe.suspected_pipeline_node}'"
    )
    assert probe.suspected_failure_type != FailureType.INSUFFICIENT_CONTEXT.value, (
        "RefChecker unsupported claim must not map to INSUFFICIENT_CONTEXT without native evidence."
    )


def test_ragchecker_hallucination_high_triggers_claim_not_retrieval_miss() -> None:
    """RAGChecker hallucination high must investigate claim_support, not retrieval_miss."""
    run = _build_run([
        ExternalSignalRecord(
            provider=ExternalEvaluatorProvider.ragchecker,
            signal_type=ExternalSignalType.hallucination,
            metric_name="hallucination",
            value=0.85,
        )
    ])
    engine = DiagnosisEngine(config={"mode": "native"}, analyzers=[_clean_analyzer()])
    engine._get_missing_critical_evidence = lambda results, run: []
    diagnosis = engine.diagnose(run)
    assert len(diagnosis.external_diagnosis_probes) == 1
    probe = diagnosis.external_diagnosis_probes[0]
    assert probe.suspected_pipeline_node == "claim_support", (
        f"RAGChecker hallucination should map to 'claim_support', got '{probe.suspected_pipeline_node}'"
    )
    assert probe.suspected_failure_type != FailureType.RETRIEVAL_ANOMALY.value, (
        "RAGChecker hallucination must not default to RETRIEVAL_ANOMALY."
    )
