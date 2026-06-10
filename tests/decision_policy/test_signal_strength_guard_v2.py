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


# 1. term_coverage_sufficiency_cannot_override_structured_grounding_unsupported
def test_term_coverage_sufficiency_cannot_override_structured_grounding_unsupported() -> None:
    term_cov_metadata = EvidenceSignalMetadata(
        signal_name="term_coverage_fallback",
        source_analyzer="SufficiencyAnalyzer",
        method="term_coverage",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    unsupported_metadata = EvidenceSignalMetadata(
        signal_name="candidate_backed_unsupported_claim",
        source_analyzer="ClaimGroundingAnalyzer",
        method="claim_grounding_support_contract",
        method_status="practical_approximation",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            signal_metadata=[term_cov_metadata],
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[unsupported_metadata],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"SufficiencyAnalyzer": 0.9, "ClaimGroundingAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM


# 2. citation_missing_advisory_cannot_override_structured_grounding
def test_citation_missing_advisory_cannot_override_structured_grounding() -> None:
    missing_citation_metadata = EvidenceSignalMetadata(
        signal_name="citation_missing",
        source_analyzer="CitationFaithfulnessAnalyzerV0",
        method="citation_lookup",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    unsupported_metadata = EvidenceSignalMetadata(
        signal_name="candidate_backed_unsupported_claim",
        source_analyzer="ClaimGroundingAnalyzer",
        method="claim_grounding_support_contract",
        method_status="practical_approximation",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="CitationFaithfulnessAnalyzerV0",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.GROUNDING,
            evidence=["[citation:citation_missing] missing citation"],
            signal_metadata=[missing_citation_metadata],
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[unsupported_metadata],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"CitationFaithfulnessAnalyzerV0": 0.9, "ClaimGroundingAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM


# 3. weak_heuristic_scope_cannot_override_structured_sufficiency
def test_weak_heuristic_scope_cannot_override_structured_sufficiency() -> None:
    scope_metadata = EvidenceSignalMetadata(
        signal_name="scope_lexical_irrelevant",
        source_analyzer="ScopeViolationAnalyzer",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    sufficiency_metadata = EvidenceSignalMetadata(
        signal_name="missing_critical_requirement",
        source_analyzer="SufficiencyAnalyzer",
        method="structured_requirement_check",
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
            signal_metadata=[scope_metadata],
        ),
        _result(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            signal_metadata=[sufficiency_metadata],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ScopeViolationAnalyzer": 0.9, "SufficiencyAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT


# 4. structured_version_invalidity_can_override_downstream_citation_symptom
def test_structured_version_invalidity_can_override_downstream_citation_symptom() -> None:
    version_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    citation_metadata = EvidenceSignalMetadata(
        signal_name="related_non_supporting_citation",
        source_analyzer="CitationFaithfulnessAnalyzerV0",
        method="citation_entailment",
        method_status="practical_approximation",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="TemporalSourceValidityAnalyzerV1",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.GROUNDING,
            signal_metadata=[version_metadata],
        ),
        _result(
            analyzer_name="CitationFaithfulnessAnalyzerV0",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.GROUNDING,
            signal_metadata=[citation_metadata],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"TemporalSourceValidityAnalyzerV1": 0.8, "CitationFaithfulnessAnalyzerV0": 0.9},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.STALE_RETRIEVAL


# 5. structured_grounding_contradiction_can_override_generic_insufficient_context
def test_structured_grounding_contradiction_can_override_generic_insufficient_context() -> None:
    grounding_metadata = EvidenceSignalMetadata(
        signal_name="grounding_contradiction",
        source_analyzer="ClaimGroundingAnalyzer",
        method="contradiction_contract_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    sufficiency_metadata = EvidenceSignalMetadata(
        signal_name="term_coverage_fallback",
        source_analyzer="SufficiencyAnalyzer",
        method="term_coverage",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )

    results = [
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[grounding_metadata],
        ),
        _result(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            signal_metadata=[sufficiency_metadata],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ClaimGroundingAnalyzer": 0.8, "SufficiencyAnalyzer": 0.9},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CONTRADICTED_CLAIM


# 6. external_advisory_signal_cannot_select_primary_by_default
def test_external_advisory_signal_cannot_select_primary_by_default() -> None:
    external_metadata = EvidenceSignalMetadata(
        signal_name="external_relevance",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="cross_encoder",
        method_status="external_advisory",
        calibration_status="unknown",
        evidence_strength="advisory",
        evidence_tier="external",
    )

    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[external_metadata],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CLEAN
    assert any(
        c["analyzer_name"] == "RetrievalDiagnosisAnalyzerV0"
        and "suppressed_reason" in c
        for c in trace["suppressed_candidates"]
    )


# 7. missing_metadata_preserves_legacy_behavior_with_trace_warning
def test_missing_metadata_preserves_legacy_behavior_with_trace_warning() -> None:
    results = [
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[],
        )
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"ClaimGroundingAnalyzer": 1.0},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM
    assert any(
        "Signal metadata was missing for core candidate ClaimGroundingAnalyzer" in w
        for w in trace["warnings"]
    )


# 8. decision_trace_records_strength_comparison
def test_decision_trace_records_strength_comparison() -> None:
    grounding_metadata = EvidenceSignalMetadata(
        signal_name="grounding_contradiction",
        source_analyzer="ClaimGroundingAnalyzer",
        method="contradiction_contract_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[grounding_metadata],
        )
    ]

    _, _, trace = select_primary_failure_with_policy(
        results,
        {"ClaimGroundingAnalyzer": 1.0},
        FAILURE_PRIORITY,
    )

    assert "selected strength=hard" in trace["selection_reason"]


# 9. structured_source_validity_remains_hard_signal
def test_structured_source_validity_remains_hard_signal() -> None:
    from raggov.decision_policy_support import _is_strong_structured_signal, build_candidates

    version_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )

    results = [
        _result(
            analyzer_name="TemporalSourceValidityAnalyzerV1",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.GROUNDING,
            signal_metadata=[version_metadata],
        )
    ]

    candidates = build_candidates(results, {"TemporalSourceValidityAnalyzerV1": 1.0})
    assert _is_strong_structured_signal(candidates[0]) is True


# 10. no_signal_guard_regression_on_current_common_benchmark
def test_no_signal_guard_regression_on_current_common_benchmark() -> None:
    # Ensuring basic flow runs completely without exception
    results = [
        _result(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
        )
    ]
    selected, _, _ = select_primary_failure_with_policy(
        results,
        {"SufficiencyAnalyzer": 1.0},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.INSUFFICIENT_CONTEXT


# 11. signal_strength_guard_does_not_create_false_incomplete
def test_signal_strength_guard_does_not_create_false_incomplete() -> None:
    mixed_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    strong_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[mixed_metadata, strong_metadata],
        )
    ]
    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.RETRIEVAL_ANOMALY
    assert "signal_strength_guard_preserved_legacy_to_avoid_false_incomplete" in trace["selection_reason"]


# 12. guard_selects_strongest_remaining_candidate_after_suppression
def test_guard_selects_strongest_remaining_candidate_after_suppression() -> None:
    weak_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    strong_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[weak_metadata],
        ),
        _result(
            analyzer_name="TemporalSourceValidityAnalyzerV1",
            status="fail",
            failure_type=FailureType.STALE_RETRIEVAL,
            stage=FailureStage.GROUNDING,
            signal_metadata=[strong_metadata],
        ),
    ]
    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0, "TemporalSourceValidityAnalyzerV1": 0.8},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.STALE_RETRIEVAL
    assert "signal_strength_guard_selected_stronger_candidate" in trace["selection_reason"]


# 13. guard_preserves_legacy_candidate_when_no_structured_alternative_exists
def test_guard_preserves_legacy_candidate_when_no_structured_alternative_exists() -> None:
    mixed_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    strong_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[mixed_metadata, strong_metadata],
        )
    ]
    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.RETRIEVAL_ANOMALY
    assert "signal_strength_guard_preserved_legacy_to_avoid_false_incomplete" in trace["selection_reason"]


# 14. guard_trace_records_legacy_preservation
def test_guard_trace_records_legacy_preservation() -> None:
    mixed_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    strong_metadata = EvidenceSignalMetadata(
        signal_name="withdrawn_source",
        source_analyzer="TemporalSourceValidityAnalyzerV1",
        method="source_lifecycle_check",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="hard",
        evidence_tier="structured",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[mixed_metadata, strong_metadata],
        )
    ]
    _, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0},
        FAILURE_PRIORITY,
    )
    assert "signal_strength_guard_preserved_legacy_to_avoid_false_incomplete" in trace["warnings"]


# 15. false_clean_remains_zero_after_false_incomplete_fix
def test_false_clean_remains_zero_after_false_incomplete_fix() -> None:
    unsupported_metadata = EvidenceSignalMetadata(
        signal_name="candidate_backed_unsupported_claim",
        source_analyzer="ClaimGroundingAnalyzer",
        method="claim_grounding_support_contract",
        method_status="structured_deterministic",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )
    weak_metadata = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.RETRIEVAL,
            signal_metadata=[weak_metadata],
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[unsupported_metadata],
        ),
    ]
    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0, "ClaimGroundingAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.UNSUPPORTED_CLAIM
    assert "signal_strength_guard_selected_stronger_candidate" in trace["selection_reason"]


# 16. false_security_remains_zero_after_false_incomplete_fix
def test_false_security_remains_zero_after_false_incomplete_fix() -> None:
    weak_anomaly = EvidenceSignalMetadata(
        signal_name="noisy_chunk_ids",
        source_analyzer="RetrievalDiagnosisAnalyzerV0",
        method="lexical_overlap",
        method_status="heuristic_baseline",
        calibration_status="uncalibrated",
        evidence_strength="weak",
        evidence_tier="heuristic",
    )
    unsupported_metadata = EvidenceSignalMetadata(
        signal_name="candidate_backed_unsupported_claim",
        source_analyzer="ClaimGroundingAnalyzer",
        method="claim_grounding_support_contract",
        method_status="practical_approximation",
        calibration_status="uncalibrated",
        evidence_strength="strong",
        evidence_tier="structured",
    )
    results = [
        _result(
            analyzer_name="RetrievalDiagnosisAnalyzerV0",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,
            signal_metadata=[weak_anomaly],
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
            signal_metadata=[unsupported_metadata],
        ),
    ]
    selected, _, _ = select_primary_failure_with_policy(
        results,
        {"RetrievalDiagnosisAnalyzerV0": 1.0, "ClaimGroundingAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )
    assert selected == FailureType.UNSUPPORTED_CLAIM

