from __future__ import annotations

from raggov.decision_policy import (
    EvidenceTier,
    classify_evidence_tier,
    select_primary_failure_with_policy,
)
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType, SufficiencyResult
from raggov.taxonomy import FAILURE_PRIORITY


def _result(
    *,
    analyzer_name: str,
    status: str,
    failure_type: FailureType,
    stage: FailureStage,
    evidence: list[str] | None = None,
) -> AnalyzerResult:
    return AnalyzerResult(
        analyzer_name=analyzer_name,
        status=status,
        failure_type=failure_type,
        stage=stage,
        evidence=evidence or [f"{analyzer_name}:{failure_type.value}"],
    )


def test_retrieval_anomaly_cannot_override_citation_mismatch() -> None:
    results = [
        _result(
            analyzer_name="RetrievalAnomalyAnalyzer",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,
        ),
        _result(
            analyzer_name="CitationMismatchAnalyzer",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.RETRIEVAL,
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {
            "RetrievalAnomalyAnalyzer": 0.99,
            "CitationMismatchAnalyzer": 0.10,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CITATION_MISMATCH
    assert trace["selected_tier"] == EvidenceTier.STRUCTURED_DIAGNOSTIC.value
    assert any(
        candidate["failure_type"] == FailureType.RETRIEVAL_ANOMALY.value
        for candidate in trace["suppressed_candidates"]
    )


def test_retrieval_anomaly_cannot_override_unsupported_claim() -> None:
    results = [
        _result(
            analyzer_name="RetrievalAnomalyAnalyzer",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, _, _ = select_primary_failure_with_policy(
        results,
        {
            "RetrievalAnomalyAnalyzer": 0.99,
            "ClaimGroundingAnalyzer": 0.10,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.UNSUPPORTED_CLAIM


def test_retrieval_anomaly_cannot_override_contradicted_claim() -> None:
    results = [
        _result(
            analyzer_name="RetrievalAnomalyAnalyzer",
            status="fail",
            failure_type=FailureType.RETRIEVAL_ANOMALY,
            stage=FailureStage.SECURITY,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, _, _ = select_primary_failure_with_policy(
        results,
        {
            "RetrievalAnomalyAnalyzer": 0.99,
            "ClaimGroundingAnalyzer": 0.10,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CONTRADICTED_CLAIM


def test_external_advisory_alone_cannot_create_primary_failure() -> None:
    results = [
        _result(
            analyzer_name="RAGASContextPrecisionAdapter",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.RETRIEVAL,
            evidence=["external advisory only"],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"RAGASContextPrecisionAdapter": 0.95},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.CLEAN
    assert trace["selected_primary_failure"] == FailureType.CLEAN.value
    assert trace["warnings"]


def test_deepeval_contextual_precision_is_external_advisory() -> None:
    result = _result(
        analyzer_name="DeepEvalContextualPrecisionAdapter",
        status="fail",
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        stage=FailureStage.RETRIEVAL,
    )

    assert classify_evidence_tier(result) == EvidenceTier.EXTERNAL_ADVISORY


def test_prompt_injection_remains_blocking() -> None:
    results = [
        _result(
            analyzer_name="PromptInjectionAnalyzer",
            status="fail",
            failure_type=FailureType.PROMPT_INJECTION,
            stage=FailureStage.SECURITY,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, _, _ = select_primary_failure_with_policy(
        results,
        {
            "PromptInjectionAnalyzer": 0.1,
            "ClaimGroundingAnalyzer": 1.0,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.PROMPT_INJECTION


def test_privacy_violation_remains_blocking() -> None:
    results = [
        _result(
            analyzer_name="PrivacyAnalyzer",
            status="fail",
            failure_type=FailureType.PRIVACY_VIOLATION,
            stage=FailureStage.SECURITY,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, _, _ = select_primary_failure_with_policy(
        results,
        {
            "PrivacyAnalyzer": 0.1,
            "ClaimGroundingAnalyzer": 1.0,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.PRIVACY_VIOLATION


def test_incomplete_diagnosis_remains_blocking() -> None:
    results = [
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.INCOMPLETE_DIAGNOSIS,
            stage=FailureStage.UNKNOWN,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzerBackup",
            status="fail",
            failure_type=FailureType.UNSUPPORTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, _, _ = select_primary_failure_with_policy(
        results,
        {
            "ClaimGroundingAnalyzer": 0.1,
            "ClaimGroundingAnalyzerBackup": 1.0,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INCOMPLETE_DIAGNOSIS


def test_parser_blocking_failure_remains_earliest_root_cause() -> None:
    results = [
        _result(
            analyzer_name="ParserValidationAnalyzer",
            status="fail",
            failure_type=FailureType.PARSER_STRUCTURE_LOSS,
            stage=FailureStage.PARSING,
        ),
        _result(
            analyzer_name="ClaimGroundingAnalyzer",
            status="fail",
            failure_type=FailureType.CONTRADICTED_CLAIM,
            stage=FailureStage.GROUNDING,
        ),
    ]

    selected, primary_result, trace = select_primary_failure_with_policy(
        results,
        {
            "ParserValidationAnalyzer": 0.05,
            "ClaimGroundingAnalyzer": 1.0,
        },
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.PARSER_STRUCTURE_LOSS


def test_structured_sufficiency_outranks_weak_citation_mismatch() -> None:
    results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["[sufficiency:missing_exception] missing exception coverage"],
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                sufficiency_label="insufficient",
                structured_failure_reason="missing_exception",
                recommended_fix_category="COVERAGE_EXPANSION",
                evidence_markers=["[sufficiency:missing_exception]"],
                method="structured_heuristic_v1",
            ),
        ),
        _result(
            analyzer_name="CitationMismatchAnalyzer",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.GROUNDING,
            evidence=["claim lacks source citations"],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"SufficiencyAnalyzer": 0.9, "CitationMismatchAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT
    assert any(
        candidate["failure_type"] == FailureType.CITATION_MISMATCH.value
        for candidate in trace["suppressed_candidates"]
    )


def test_strong_citation_mismatch_is_not_suppressed_by_sufficiency() -> None:
    results = [
        AnalyzerResult(
            analyzer_name="SufficiencyAnalyzer",
            status="fail",
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            stage=FailureStage.SUFFICIENCY,
            evidence=["[sufficiency:missing_scope_condition] missing scope coverage"],
            sufficiency_result=SufficiencyResult(
                sufficient=False,
                sufficiency_label="insufficient",
                structured_failure_reason="missing_scope_condition",
                recommended_fix_category="SCOPE_DISAMBIGUATION",
                evidence_markers=["[sufficiency:missing_scope_condition]"],
                method="structured_heuristic_v1",
            ),
        ),
        _result(
            analyzer_name="CitationMismatchAnalyzer",
            status="fail",
            failure_type=FailureType.CITATION_MISMATCH,
            stage=FailureStage.GROUNDING,
            evidence=["phantom citation: doc-404"],
        ),
    ]

    selected, _, trace = select_primary_failure_with_policy(
        results,
        {"SufficiencyAnalyzer": 0.9, "CitationMismatchAnalyzer": 0.8},
        FAILURE_PRIORITY,
    )

    assert selected == FailureType.INSUFFICIENT_CONTEXT
    assert not any(
        candidate["failure_type"] == FailureType.CITATION_MISMATCH.value
        for candidate in trace["suppressed_candidates"]
    )
