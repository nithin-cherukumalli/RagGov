from __future__ import annotations

from raggov.models.diagnosis import AnalyzerResult
from raggov.models.signals import (
    EvidenceSignalMetadata,
    default_uncalibrated_heuristic_signal,
)


def test_default_uncalibrated_heuristic_signal() -> None:
    signal = default_uncalibrated_heuristic_signal(
        signal_name="chunk_relevance",
        source_analyzer="RetrievalEvidenceProfilerV0",
        method="lexical_overlap",
        notes="default downgrade",
    )

    assert signal.signal_name == "chunk_relevance"
    assert signal.source_analyzer == "RetrievalEvidenceProfilerV0"
    assert signal.method == "lexical_overlap"
    assert signal.method_status == "heuristic_baseline"
    assert signal.calibration_status == "uncalibrated"
    assert signal.evidence_strength == "weak"
    assert signal.evidence_tier == "heuristic"
    assert signal.evidence_ids == []
    assert signal.notes == "default downgrade"


def test_structured_deterministic_signal_serializes() -> None:
    signal = EvidenceSignalMetadata(
        signal_name="no_retrieved_chunks",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="empty_retrieval_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
        evidence_ids=["run-1"],
    )

    assert signal.model_dump(mode="json") == {
        "signal_name": "no_retrieved_chunks",
        "source_analyzer": "RetrievalDiagnosisAnalyzerV0",
        "method": "empty_retrieval_check",
        "method_status": "structured_deterministic",
        "calibration_status": "uncalibrated",
        "evidence_strength": "hard",
        "evidence_tier": "structured",
        "evidence_ids": ["run-1"],
        "notes": None,
    }


def test_external_advisory_signal_serializes() -> None:
    signal = EvidenceSignalMetadata(
        signal_name="context_precision",
        source_analyzer="ExternalSignalBridge",
        method="ragas",
        method_status="external_advisory",
        calibration_status="unknown",
        evidence_strength="advisory",
        evidence_tier="external",
        evidence_ids=["chunk-1"],
        notes="External signals do not select primary failures by default.",
    )

    dumped = signal.model_dump(mode="json")

    assert dumped["method_status"] == "external_advisory"
    assert dumped["calibration_status"] == "unknown"
    assert dumped["evidence_strength"] == "advisory"
    assert dumped["evidence_tier"] == "external"
    assert dumped["evidence_ids"] == ["chunk-1"]


def test_calibrated_statistical_signal_allowed_but_not_default() -> None:
    signal = EvidenceSignalMetadata(
        signal_name="semantic_entropy",
        source_analyzer="SemanticEntropyAnalyzer",
        method="semantic_entropy_sampling_v1",
        method_status="calibrated_statistical",
        calibration_status="calibrated_heldout",
        evidence_strength="strong",
        evidence_tier="calibrated",
    )
    default_signal = default_uncalibrated_heuristic_signal(
        signal_name="semantic_entropy",
        source_analyzer="SemanticEntropyAnalyzer",
    )

    assert signal.method_status == "calibrated_statistical"
    assert signal.calibration_status == "calibrated_heldout"
    assert default_signal.method_status == "heuristic_baseline"
    assert default_signal.calibration_status == "uncalibrated"


def test_analyzer_result_accepts_optional_signal_metadata_if_field_added() -> None:
    signal = default_uncalibrated_heuristic_signal(
        signal_name="chunk_relevance",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
    )

    result = AnalyzerResult(
        analyzer_name="ScopeViolationAnalyzer",
        status="warn",
        evidence=["[profile] c1 label=irrelevant method=lexical_overlap"],
        signal_metadata=[signal],
    )

    assert result.signal_metadata == [signal]
    assert result.model_dump(mode="json")["signal_metadata"][0]["evidence_strength"] == "weak"


def test_missing_signal_metadata_is_backward_compatible() -> None:
    result = AnalyzerResult.model_validate(
        {
            "analyzer_name": "LegacyAnalyzer",
            "status": "pass",
            "evidence": ["legacy payload without signal metadata"],
        }
    )

    assert result.signal_metadata == []
    assert result.model_dump(mode="json")["signal_metadata"] == []
