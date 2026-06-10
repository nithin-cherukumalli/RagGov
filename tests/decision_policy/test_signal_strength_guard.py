from __future__ import annotations

import pytest

from raggov.decision_policy import select_primary_failure_with_policy
from raggov.models.diagnosis import (
    AnalyzerResult,
    FailureStage,
    FailureType,
    SufficiencyResult,
)
from raggov.models.signals import EvidenceSignalMetadata
from raggov.taxonomy import FAILURE_PRIORITY


def _result(
    *,
    analyzer_name: str,
    status: str,
    failure_type: FailureType,
    stage: FailureStage,
    evidence: list[str] | None = None,
    signal_metadata: list[EvidenceSignalMetadata] | None = None,
) -> AnalyzerResult:
    return AnalyzerResult(
        analyzer_name=analyzer_name,
        status=status,
        failure_type=failure_type,
        stage=stage,
        evidence=evidence or [f"{analyzer_name}:{failure_type.value}"],
        signal_metadata=signal_metadata or [],
    )


# 1. lexical_overlap_scope_signal_cannot_override_structured_insufficient_context
def test_lexical_overlap_scope_signal_cannot_override_structured_insufficient_context() -> None:
    # Lexical overlap metadata representing weak uncalibrated signal
    lexical_metadata = EvidenceSignalMetadata(
        signal_name="scope_lexical_irrelevant",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    results = [
        _result(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[lexical_metadata],
        ),
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["[sufficiency:missing_critical_requirement] missing entity coverage"],
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                sufficiency_label="insufficient",
                structured_failure_reason="missing_critical_requirement",
                recommended_fix_category="COVERAGE_EXPANSION",
                evidence_markers=["[sufficiency:missing_critical_requirement]"],
                method="structured_heuristic_v1",
            ),
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ScopeViolationAnalyzer": 0.99, "SufficiencyAnalyzer": 0.50},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT
    # Verify the trace has recorded the signal-strength suppression!
    assert any(
        c["analyzer_name"] == "ScopeViolationAnalyzer"
        and c.get("suppressed_reason") == "weak_uncalibrated_signal_cannot_override_structured_evidence"
        for c in trace["suppressed_candidates"]
    )


# 2. test_lexical_overlap_retrieval_irrelevant_is_weak_uncalibrated
def test_lexical_overlap_retrieval_irrelevant_is_weak_uncalibrated() -> None:
    from raggov.decision_policy_support import _is_weak_uncalibrated_signal, build_candidates
    
    lexical_metadata = EvidenceSignalMetadata(
        signal_name="lexical_overlap_relevance",
        source_analyzer="RetrievalEvidenceProfilerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    results = [
        _result(
            analyzer_name="RetrievalEvidenceProfilerV0",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[lexical_metadata],
        )
    ]
    candidates = build_candidates(results, {"RetrievalEvidenceProfilerV0": 1.0})
    assert _is_weak_uncalibrated_signal(candidates[0]) is True


# 3. test_external_advisory_retrieval_signal_cannot_select_primary_by_default
def test_external_advisory_retrieval_signal_cannot_select_primary_by_default() -> None:
    external_metadata = EvidenceSignalMetadata(
        signal_name="external_relevance",
        source_analyzer="RetrievalEvidenceProfilerV0",
        method="cross_encoder",
        method_status="external_advisory",
        calibration_status="unknown",
        evidence_strength="advisory",
        evidence_tier="external",
    )

    results = [
        _result(
            analyzer_name="RetrievalEvidenceProfilerV0",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[external_metadata],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalEvidenceProfilerV0": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CLEAN
    assert any(
        c["analyzer_name"] == "RetrievalEvidenceProfilerV0"
        and c.get("suppressed_reason") == "external_advisory_retrieval_signal_cannot_select_primary_by_default"
        for c in trace["suppressed_candidates"]
    )


# 4. test_heuristic_retrieval_anomaly_does_not_become_security
def test_heuristic_retrieval_anomaly_does_not_become_security() -> None:
    anomaly_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    # Retrieval anomaly based only on heuristic/noise proxy
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,  # Attempt to elevate to SECURITY
            signal_metadata=[anomaly_metadata],
        )
    ]

    selected, primary_result, _ = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.RETRIEVAL_ANOMALY
    assert primary_result is not None
    # Verify stage is demoted to RETRIEVAL
    assert primary_result.stage == FailureStage.RETRIEVAL


# 5. test_structured_no_chunks_retrieval_miss_remains_hard
def test_structured_no_chunks_retrieval_miss_remains_hard() -> None:
    miss_metadata = EvidenceSignalMetadata(
        signal_name="no_retrieved_chunks",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="no_retrieved_chunks_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[miss_metadata],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT
    assert trace["selected_primary_failure"] == FailureType.INSUFFICIENT_CONTEXT.value


# 6. test_structured_phantom_citation_lookup_remains_hard
def test_structured_phantom_citation_lookup_remains_hard() -> None:
    phantom_metadata = EvidenceSignalMetadata(
        signal_name="phantom_citation",
        source_analyzer="RetrievalEvidenceProfilerV0",
        method="cited_doc_id_lookup",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="RetrievalEvidenceProfilerV0",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[phantom_metadata],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalEvidenceProfilerV0": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CITATION_MISMATCH
    assert trace["selected_primary_failure"] == FailureType.CITATION_MISMATCH.value


# 7. test_quoted_entity_missing_scope_signal_remains_strong
def test_quoted_entity_missing_scope_signal_remains_strong() -> None:
    quoted_metadata = EvidenceSignalMetadata(
        signal_name="quoted_entity_missing",
        source_analyzer="ScopeViolationAnalyzer",
        method="quoted_entity_match",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[quoted_metadata],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ScopeViolationAnalyzer": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.SCOPE_VIOLATION
    assert trace["selected_primary_failure"] == FailureType.SCOPE_VIOLATION.value


# 8. test_missing_signal_metadata_preserves_legacy_behavior
def test_missing_signal_metadata_preserves_legacy_behavior() -> None:
    # No signal metadata is attached to ScopeViolationAnalyzer result
    results = [
        _result(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            evidence=["[profile] c1 label=irrelevant; wrong domain for query scope"],
            signal_metadata=[],
        ),
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["[sufficiency:term_coverage_heuristic_v0] low query term coverage"],
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                sufficiency_label="insufficient",
                method="term_coverage_heuristic_v0",
            ),
        ),
    ]

    # Without signal metadata, lexical overlap check doesn't suppress,
    # so ScopeViolationAnalyzer wins due to higher weight!
    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ScopeViolationAnalyzer": 0.99, "SufficiencyAnalyzer": 0.50},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.SCOPE_VIOLATION
    # Verifies Rule 5 warning trace is recorded
    assert any(
        "Signal metadata was missing for candidate ScopeViolationAnalyzer" in w
        for w in trace["warnings"]
    )


# 9. test_decision_trace_records_signal_strength_suppression
def test_decision_trace_records_signal_strength_suppression() -> None:
    lexical_metadata = EvidenceSignalMetadata(
        signal_name="scope_lexical_irrelevant",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    results = [
        _result(
            analyzer_name="ScopeViolationAnalyzer",
            status="fail",
            failure_type=FailureType.SCOPE_VIOLATION,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[lexical_metadata],
        ),
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["[sufficiency:missing_critical_requirement] missing entity coverage"],
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                sufficiency_label="insufficient",
                structured_failure_reason="missing_critical_requirement",
                recommended_fix_category="COVERAGE_EXPANSION",
                evidence_markers=["[sufficiency:missing_critical_requirement]"],
                method="structured_heuristic_v1",
            ),
        ),
    ]

    _, _, trace = select_primary_failure_with_policy(
        results,
        {"ScopeViolationAnalyzer": 0.99, "SufficiencyAnalyzer": 0.50},
        FAILURE_PRIORITY,
    )

    suppressed = trace["suppressed_candidates"][0]
    assert suppressed["suppressed_reason"] == "weak_uncalibrated_signal_cannot_override_structured_evidence"
    assert suppressed["method_status"] == "heuristic_baseline"
    assert suppressed["calibration_status"] == "uncalibrated"
    assert suppressed["evidence_strength"] == "weak"
    assert suppressed["evidence_tier_name"] == "heuristic"
