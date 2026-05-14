"""Final diagnosis decision policy with explicit evidence tiers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from raggov.calibration_status import CalibrationStatus, get_calibration_status
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType


class EvidenceTier(str, Enum):
    BLOCKING_DETERMINISTIC = "BLOCKING_DETERMINISTIC"
    STRUCTURED_DIAGNOSTIC = "STRUCTURED_DIAGNOSTIC"
    HEURISTIC_SUPPORTING = "HEURISTIC_SUPPORTING"
    EXTERNAL_ADVISORY = "EXTERNAL_ADVISORY"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class DecisionCandidate:
    failure_type: FailureType
    analyzer_name: str
    status: str
    stage: FailureStage | None
    evidence_tier: EvidenceTier
    weight: float
    original_index: int
    calibration_status: str
    recommended_for_gating: bool
    evidence_summary: str
    reason: str
    sufficiency_reason: str | None = None
    sufficiency_markers: tuple[str, ...] = ()
    version_severity: str | None = None
    version_doc_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "analyzer_name": self.analyzer_name,
            "status": self.status,
            "stage": self.stage.value if self.stage is not None else None,
            "evidence_tier": self.evidence_tier.value,
            "weight": self.weight,
            "original_index": self.original_index,
            "calibration_status": self.calibration_status,
            "recommended_for_gating": self.recommended_for_gating,
            "evidence_summary": self.evidence_summary,
            "reason": self.reason,
            "sufficiency_reason": self.sufficiency_reason,
            "sufficiency_markers": list(self.sufficiency_markers),
            "version_severity": self.version_severity,
            "version_doc_ids": list(self.version_doc_ids),
        }


@dataclass(slots=True)
class DiagnosisDecisionTrace:
    selected_primary_failure: str
    selected_analyzer: str | None
    selected_tier: str | None
    selection_reason: str
    alternatives_considered: list[dict[str, Any]]
    suppressed_candidates: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_primary_failure": self.selected_primary_failure,
            "selected_analyzer": self.selected_analyzer,
            "selected_tier": self.selected_tier,
            "selection_reason": self.selection_reason,
            "alternatives_considered": self.alternatives_considered,
            "suppressed_candidates": self.suppressed_candidates,
            "warnings": self.warnings,
        }


_BLOCKING_FAILURES = {
    FailureType.PROMPT_INJECTION,
    FailureType.PRIVACY_VIOLATION,
    FailureType.INCOMPLETE_DIAGNOSIS,
}
_PARSER_BLOCKING_FAILURES = {
    FailureType.TABLE_STRUCTURE_LOSS,
    FailureType.HIERARCHY_FLATTENING,
    FailureType.METADATA_LOSS,
    FailureType.PARSER_STRUCTURE_LOSS,
    FailureType.CHUNKING_BOUNDARY_ERROR,
}
_STRUCTURED_FAILURES = {
    FailureType.CONTRADICTED_CLAIM,
    FailureType.UNSUPPORTED_CLAIM,
    FailureType.CITATION_MISMATCH,
}
_HEURISTIC_FAILURES = {
    FailureType.RETRIEVAL_ANOMALY,
    FailureType.SCOPE_VIOLATION,
    FailureType.LOW_CONFIDENCE,
}
_EXTERNAL_PROVIDER_MARKERS = (
    "ragas",
    "deepeval",
    "ragchecker",
    "refchecker",
    "cross_encoder",
    "cross-encoder",
)
_META_HEURISTIC_ANALYZERS = {"Layer6TaxonomyClassifier", "A2PAttributionAnalyzer"}

_SUFFICIENCY_STRUCTURED_MARKERS = frozenset({
    "[sufficiency:missing_critical_requirement]",
    "[sufficiency:partial_requirement_coverage]",
    "[sufficiency:missing_exception]",
    "[sufficiency:missing_scope_condition]",
    "[sufficiency:missing_temporal_or_freshness_requirement]",
    "[sufficiency:stale_context_mistaken_as_sufficient]",
})


def _any_structured_sufficiency_marker(evidence_text: str) -> bool:
    return any(marker in evidence_text for marker in _SUFFICIENCY_STRUCTURED_MARKERS)


def classify_evidence_tier(result: AnalyzerResult) -> EvidenceTier:
    if result.failure_type in _BLOCKING_FAILURES:
        return EvidenceTier.BLOCKING_DETERMINISTIC
    if _is_parser_blocking_failure(result):
        return EvidenceTier.BLOCKING_DETERMINISTIC
    if (
        result.failure_type == FailureType.SUSPICIOUS_CHUNK
        and result.analyzer_name
        in {"PromptInjectionAnalyzer", "PoisoningHeuristicAnalyzer", "RetrievalAnomalyAnalyzer"}
    ):
        return EvidenceTier.BLOCKING_DETERMINISTIC
    if _is_external_advisory_result(result):
        return EvidenceTier.EXTERNAL_ADVISORY
    if result.failure_type in _STRUCTURED_FAILURES:
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if _is_structured_insufficient_context(result):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if _is_structured_stale_retrieval(result):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if _is_structured_ncv_failure(result):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if _is_structured_retrieval_diagnosis(result):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if (
        result.failure_type == FailureType.SCOPE_VIOLATION
        and any("quoted query entity missing" in evidence for evidence in result.evidence)
    ):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if (
        result.analyzer_name == "SemanticEntropyAnalyzer"
        and result.failure_type == FailureType.LOW_CONFIDENCE
        and any("query_understanding ambiguity" in evidence for evidence in result.evidence)
    ):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if result.failure_type in _HEURISTIC_FAILURES:
        return EvidenceTier.HEURISTIC_SUPPORTING
    if _is_heuristic_supporting_result(result):
        return EvidenceTier.HEURISTIC_SUPPORTING
    return EvidenceTier.UNKNOWN


def select_primary_failure_with_policy(
    results: list[AnalyzerResult],
    result_weights: dict[str, float],
    failure_priority: list[FailureType],
) -> tuple[FailureType, AnalyzerResult | None, dict[str, Any]]:
    candidates = _build_candidates(results, result_weights)
    if not candidates:
        return (
            FailureType.CLEAN,
            None,
            DiagnosisDecisionTrace(
                selected_primary_failure=FailureType.CLEAN.value,
                selected_analyzer=None,
                selected_tier=None,
                selection_reason="No fail/warn diagnosis candidates were available.",
                alternatives_considered=[],
                suppressed_candidates=[],
                warnings=[],
            ).to_dict(),
        )

    eligible, suppressed, warnings = _split_candidates(candidates)
    if not eligible:
        warnings.append("No eligible non-advisory candidate remained after policy suppression.")
        return (
            FailureType.CLEAN,
            None,
            DiagnosisDecisionTrace(
                selected_primary_failure=FailureType.CLEAN.value,
                selected_analyzer=None,
                selected_tier=None,
                selection_reason="All fail/warn candidates were external advisory only.",
                alternatives_considered=[],
                suppressed_candidates=[candidate.to_dict() for candidate in suppressed],
                warnings=warnings,
            ).to_dict(),
        )

    status_pool = [c for c in eligible if c.status == "fail"]
    if not status_pool:
        return (
            FailureType.CLEAN,
            None,
            DiagnosisDecisionTrace(
                selected_primary_failure=FailureType.CLEAN.value,
                selected_analyzer=None,
                selected_tier=None,
                selection_reason="No fail-level diagnosis candidate survived policy selection.",
                alternatives_considered=[candidate.to_dict() for candidate in sorted(
                    eligible,
                    key=lambda candidate: _candidate_sort_key(candidate, failure_priority),
                )],
                suppressed_candidates=[candidate.to_dict() for candidate in suppressed],
                warnings=warnings,
            ).to_dict(),
        )
    ranked_status_pool = sorted(
        status_pool,
        key=lambda candidate: _candidate_sort_key(candidate, failure_priority),
    )
    winner = ranked_status_pool[0]
    primary_result = results[winner.original_index]

    alternatives: list[DecisionCandidate] = []
    final_suppressed = list(suppressed)
    all_other_candidates = sorted(
        [candidate for candidate in eligible if candidate is not winner],
        key=lambda candidate: (
            0 if candidate.status == winner.status else 1,
            *_candidate_sort_key(candidate, failure_priority),
        ),
    )
    for candidate in all_other_candidates:
        if _should_suppress_candidate(candidate, winner):
            final_suppressed.append(candidate)
        else:
            alternatives.append(candidate)

    return (
        winner.failure_type,
        primary_result,
        DiagnosisDecisionTrace(
            selected_primary_failure=winner.failure_type.value,
            selected_analyzer=winner.analyzer_name,
            selected_tier=winner.evidence_tier.value,
            selection_reason=_selection_reason(winner, alternatives, final_suppressed),
            alternatives_considered=[candidate.to_dict() for candidate in alternatives],
            suppressed_candidates=[candidate.to_dict() for candidate in final_suppressed],
            warnings=warnings,
        ).to_dict(),
    )


def _build_candidates(
    results: list[AnalyzerResult],
    result_weights: dict[str, float],
) -> list[DecisionCandidate]:
    candidates: list[DecisionCandidate] = []
    for index, result in enumerate(results):
        if result.status not in {"fail", "warn"} or result.failure_type is None:
            continue
        tier = classify_evidence_tier(result)
        calibration_status = _resolve_calibration_status(result)
        candidates.append(
            DecisionCandidate(
                failure_type=result.failure_type,
                analyzer_name=result.analyzer_name,
                status=result.status,
                stage=result.stage,
                evidence_tier=tier,
                weight=result_weights.get(result.analyzer_name, 1.0),
                original_index=index,
                calibration_status=calibration_status,
                recommended_for_gating=_resolve_recommended_for_gating(result),
                evidence_summary=result.evidence[0] if result.evidence else "",
                reason=_candidate_reason(result, tier, calibration_status),
                sufficiency_reason=(
                    result.sufficiency_result.structured_failure_reason
                    if result.sufficiency_result is not None
                    else None
                ),
                sufficiency_markers=tuple(
                    result.sufficiency_result.evidence_markers
                    if result.sufficiency_result is not None
                    else ()
                ),
                version_severity=_version_severity(result),
                version_doc_ids=tuple(_version_doc_ids(result)),
            )
        )
    return candidates


def _split_candidates(
    candidates: list[DecisionCandidate],
) -> tuple[list[DecisionCandidate], list[DecisionCandidate], list[str]]:
    eligible: list[DecisionCandidate] = []
    suppressed: list[DecisionCandidate] = []
    warnings: list[str] = []
    has_non_meta_candidate = any(
        candidate.analyzer_name not in _META_HEURISTIC_ANALYZERS
        for candidate in candidates
    )
    for candidate in candidates:
        if candidate.evidence_tier == EvidenceTier.EXTERNAL_ADVISORY:
            suppressed.append(candidate)
            continue
        if has_non_meta_candidate and candidate.analyzer_name in _META_HEURISTIC_ANALYZERS:
            suppressed.append(candidate)
            continue
        eligible.append(candidate)
    if suppressed:
        warnings.append("External advisory candidates were suppressed from primary failure selection.")
    if has_non_meta_candidate and any(
        candidate.analyzer_name in _META_HEURISTIC_ANALYZERS for candidate in suppressed
    ):
        warnings.append(
            "Meta heuristic candidates were retained for traceability but suppressed from primary selection because direct analyzer evidence exists."
        )
    return eligible, suppressed, warnings


def _candidate_sort_key(
    candidate: DecisionCandidate,
    failure_priority: list[FailureType],
) -> tuple[int, float, int, int, float, int, int]:
    priority_index = {failure_type: index for index, failure_type in enumerate(failure_priority)}
    return (
        _tier_rank(candidate),
        -_specificity_rank(candidate),
        -candidate.weight,
        -_calibration_rank(candidate.calibration_status),
        -float(candidate.recommended_for_gating),
        priority_index.get(candidate.failure_type, len(failure_priority)),
        candidate.original_index,
    )


def _tier_rank(candidate: DecisionCandidate) -> int:
    return {
        EvidenceTier.BLOCKING_DETERMINISTIC: 0,
        EvidenceTier.STRUCTURED_DIAGNOSTIC: 1,
        EvidenceTier.HEURISTIC_SUPPORTING: 2,
        EvidenceTier.UNKNOWN: 3,
        EvidenceTier.EXTERNAL_ADVISORY: 4,
    }[candidate.evidence_tier]


def _specificity_rank(candidate: DecisionCandidate) -> int:
    if _is_parser_failure_type(candidate.failure_type):
        return 100
    if candidate.failure_type in {FailureType.PROMPT_INJECTION, FailureType.PRIVACY_VIOLATION}:
        return 95
    if candidate.failure_type == FailureType.INCOMPLETE_DIAGNOSIS:
        return 90
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name == "RetrievalDiagnosisAnalyzerV0"
    ):
        return 90
    if (
        candidate.failure_type == FailureType.SCOPE_VIOLATION
        and "quoted query entity missing" in candidate.evidence_summary
    ):
        return 91
    if (
        candidate.failure_type == FailureType.LOW_CONFIDENCE
        and candidate.analyzer_name == "SemanticEntropyAnalyzer"
        and "query_understanding ambiguity" in candidate.evidence_summary
    ):
        return 93
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name == "SufficiencyAnalyzer"
        and _candidate_has_explicit_sufficiency_root_cause(candidate)
    ):
        # Explicit structured sufficiency evidence outranks symptom-level failures
        # (CITATION_MISMATCH at 91, RetrievalDiagnosisAnalyzerV0 at 90) so that
        # the root-cause stage is SUFFICIENCY, not a downstream GROUNDING/RETRIEVAL.
        return 92
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name == "SufficiencyAnalyzer"
    ):
        if (
            candidate.sufficiency_reason == "stale_context_mistaken_as_sufficient"
            or "[sufficiency:stale_context_mistaken_as_sufficient]" in candidate.sufficiency_markers
            or "[sufficiency:stale_context_mistaken_as_sufficient]" in candidate.evidence_summary
        ):
            # Stale-context-mistaken-as-sufficient should outrank INSUFFICIENT_CONTEXT/RETRIEVAL
            # (RetrievalDiagnosisAnalyzerV0 at 90) to surface the freshness root cause.
            return 91
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1", "RetrievalDiagnosisAnalyzerV0"}
        and _candidate_has_strong_version_root_cause(candidate)
    ):
        return 89
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1", "RetrievalDiagnosisAnalyzerV0"}
        and _candidate_has_retrieval_quality_stale_root_cause(candidate)
    ):
        return 81
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name in {"SufficiencyAnalyzer", "ClaimAwareSufficiencyAnalyzer"}
    ):
        return 88
    if candidate.failure_type in _STRUCTURED_FAILURES:
        return 80
    if candidate.failure_type == FailureType.STALE_RETRIEVAL:
        return 75
    if candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT:
        return 70
    if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY:
        return 30
    if candidate.failure_type in {FailureType.SCOPE_VIOLATION, FailureType.LOW_CONFIDENCE}:
        return 20
    return 10


def _calibration_rank(calibration_status: str) -> int:
    return {
        CalibrationStatus.DETERMINISTIC.value: 5,
        CalibrationStatus.CALIBRATED.value: 4,
        CalibrationStatus.PROVISIONAL.value: 3,
        "preliminary_calibrated_v1": 2,
        CalibrationStatus.NOT_CALIBRATED.value: 1,
        "uncalibrated": 1,
        "uncalibrated_locally": 0,
    }.get(calibration_status, 0)


def _resolve_calibration_status(result: AnalyzerResult) -> str:
    if result.ncv_report is not None:
        status = result.ncv_report.get("calibration_status")
        if status:
            return str(status)
    if result.retrieval_diagnosis_report is not None:
        return result.retrieval_diagnosis_report.calibration_status.value
    if result.version_validity_report is not None:
        return result.version_validity_report.calibration_status.value
    if result.sufficiency_result is not None:
        return result.sufficiency_result.calibration_status
    return get_calibration_status(result.analyzer_name).value


def _resolve_recommended_for_gating(result: AnalyzerResult) -> bool:
    if result.ncv_report is not None:
        return bool(result.ncv_report.get("recommended_for_gating", False))
    if result.retrieval_diagnosis_report is not None:
        return result.retrieval_diagnosis_report.recommended_for_gating
    if result.version_validity_report is not None:
        return result.version_validity_report.recommended_for_gating
    return False


def _candidate_reason(result: AnalyzerResult, tier: EvidenceTier, calibration_status: str) -> str:
    return (
        f"{result.analyzer_name} emitted {result.failure_type.value} at tier "
        f"{tier.value} with calibration_status={calibration_status}."
    )


def _selection_reason(
    candidate: DecisionCandidate,
    alternatives: list[DecisionCandidate],
    suppressed: list[DecisionCandidate],
) -> str:
    if _is_parser_failure_type(candidate.failure_type):
        return "Parser blocking failure selected as earliest deterministic root cause."
    others = alternatives + suppressed
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and _candidate_has_strong_version_root_cause(candidate)
    ):
        return (
            "Claim support failure is downstream of invalid source lifecycle evidence. "
            f"Selected STALE_RETRIEVAL from {candidate.analyzer_name} because cited or answer-bearing invalid sources "
            "make downstream unsupported-claim signals secondary."
        )
    return (
        f"Selected {candidate.failure_type.value} from {candidate.analyzer_name} because "
        "fail before warn, then evidence tier, specificity, calibration status, analyzer weight, "
        "and failure priority are applied in order."
    )


def _should_suppress_candidate(candidate: DecisionCandidate, winner: DecisionCandidate) -> bool:
    if candidate.evidence_tier == EvidenceTier.EXTERNAL_ADVISORY:
        return True
    if (
        candidate.evidence_tier == EvidenceTier.HEURISTIC_SUPPORTING
        and winner.evidence_tier in {EvidenceTier.BLOCKING_DETERMINISTIC, EvidenceTier.STRUCTURED_DIAGNOSTIC}
    ):
        return True
    if (
        candidate.failure_type == FailureType.RETRIEVAL_ANOMALY
        and winner.failure_type
        in {
            FailureType.CONTRADICTED_CLAIM,
            FailureType.UNSUPPORTED_CLAIM,
            FailureType.CITATION_MISMATCH,
            FailureType.INSUFFICIENT_CONTEXT,
            FailureType.STALE_RETRIEVAL,
        }
    ):
        return True
    # Suppress CITATION_MISMATCH when explicit structured sufficiency evidence is the winner.
    # Citation problems are downstream symptoms of missing context; surfacing the root cause
    # (INSUFFICIENT_CONTEXT/SUFFICIENCY) is more actionable than the symptom.
    # Exception: do NOT suppress when citation evidence is strong (phantom docs, contradictions).
    if (
        candidate.failure_type == FailureType.CITATION_MISMATCH
        and winner.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and winner.analyzer_name == "SufficiencyAnalyzer"
        and _candidate_has_explicit_sufficiency_root_cause(winner)
        and not _citation_mismatch_is_strong(candidate)
    ):
        return True
    return False


def _candidate_has_explicit_sufficiency_root_cause(candidate: DecisionCandidate) -> bool:
    return bool(
        candidate.sufficiency_reason
        or candidate.sufficiency_markers
        or _any_structured_sufficiency_marker(candidate.evidence_summary)
    )


def _citation_mismatch_is_strong(candidate: DecisionCandidate) -> bool:
    evidence = candidate.evidence_summary.lower()
    strong_markers = (
        "phantom citation",
        "cited document not present",
        "absent from retrieved evidence",
        "contradict",
        "wrong source",
    )
    return any(marker in evidence for marker in strong_markers)


def _candidate_has_strong_version_root_cause(candidate: DecisionCandidate) -> bool:
    return candidate.version_severity in {"cited_invalid_source", "answer_bearing_invalid_source"}


def _candidate_has_retrieval_quality_stale_root_cause(candidate: DecisionCandidate) -> bool:
    return candidate.version_severity == "retrieved_only_stale_source"


def _version_severity(result: AnalyzerResult) -> str | None:
    report = result.version_validity_report
    if report is not None:
        if getattr(report, "cited_invalid_doc_ids", []):
            return "cited_invalid_source"
        if getattr(report, "answer_bearing_invalid_doc_ids", []):
            return "answer_bearing_invalid_source"
        if getattr(report, "retrieval_quality_affected_doc_ids", []):
            return "retrieved_only_stale_source"
        if getattr(report, "stale_but_irrelevant_doc_ids", []):
            return "stale_but_irrelevant_source"
    retrieval = result.retrieval_diagnosis_report
    if retrieval is not None:
        signal_names = {signal.signal_name for signal in retrieval.evidence_signals}
        if "invalid_cited_source_docs" in signal_names:
            return "cited_invalid_source"
        if "answer_bearing_invalid_source_docs" in signal_names:
            return "answer_bearing_invalid_source"
        if "retrieval_quality_affected_by_stale_source_docs" in signal_names:
            return "retrieved_only_stale_source"
    return None


def _version_doc_ids(result: AnalyzerResult) -> list[str]:
    report = result.version_validity_report
    if report is not None:
        return list(
            dict.fromkeys(
                list(getattr(report, "cited_invalid_doc_ids", []))
                + list(getattr(report, "answer_bearing_invalid_doc_ids", []))
                + list(getattr(report, "retrieval_quality_affected_doc_ids", []))
                + list(getattr(report, "stale_but_irrelevant_doc_ids", []))
            )
        )
    retrieval = result.retrieval_diagnosis_report
    if retrieval is not None:
        return list(dict.fromkeys(retrieval.invalid_cited_doc_ids + retrieval.invalid_retrieved_doc_ids))
    return []


def _is_parser_blocking_failure(result: AnalyzerResult) -> bool:
    return (
        result.status == "fail"
        and result.failure_type in _PARSER_BLOCKING_FAILURES
        or (
            result.analyzer_name == "ParserValidationAnalyzer"
            and result.status == "fail"
            and result.stage == FailureStage.PARSING
        )
    )


def _is_parser_failure_type(failure_type: FailureType) -> bool:
    return failure_type in _PARSER_BLOCKING_FAILURES


def _is_structured_insufficient_context(result: AnalyzerResult) -> bool:
    if result.failure_type != FailureType.INSUFFICIENT_CONTEXT:
        return False
    if result.analyzer_name in {"ClaimAwareSufficiencyAnalyzer", "RetrievalDiagnosisAnalyzerV0"}:
        return True
    if result.sufficiency_result is None:
        return False
    # Term-coverage WITH a structured marker is still structured evidence
    # (e.g. [sufficiency:missing_critical_requirement] prepended to a coverage string)
    evidence_text = " ".join(result.evidence)
    if result.sufficiency_result.evidence_markers:
        return True
    if result.sufficiency_result.structured_failure_reason:
        return True
    if _any_structured_sufficiency_marker(evidence_text):
        return True
    method = result.sufficiency_result.method.lower()
    # Pure term-coverage heuristic without a structured marker is heuristic, not structured
    return "lexical" not in method and "term_coverage" not in method


def _is_structured_stale_retrieval(result: AnalyzerResult) -> bool:
    if result.failure_type != FailureType.STALE_RETRIEVAL:
        return False
    if result.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1"}:
        severity = _version_severity(result)
        return severity in {
            "cited_invalid_source",
            "answer_bearing_invalid_source",
            "retrieved_only_stale_source",
        }
    if result.analyzer_name == "RetrievalDiagnosisAnalyzerV0":
        return _version_severity(result) in {
            "cited_invalid_source",
            "answer_bearing_invalid_source",
            "retrieved_only_stale_source",
        }
    if result.analyzer_name == "SufficiencyAnalyzer":
        if result.sufficiency_result is not None:
            if result.sufficiency_result.structured_failure_reason == "stale_context_mistaken_as_sufficient":
                return True
            if "[sufficiency:stale_context_mistaken_as_sufficient]" in result.sufficiency_result.evidence_markers:
                return True
        evidence_text = result.evidence[0] if result.evidence else ""
        return "[sufficiency:stale_context_mistaken_as_sufficient]" in evidence_text
    if result.version_validity_report is None:
        return False
    return bool(
        getattr(result.version_validity_report, "cited_invalid_doc_ids", [])
        or getattr(result.version_validity_report, "answer_bearing_invalid_doc_ids", [])
        or getattr(result.version_validity_report, "retrieval_quality_affected_doc_ids", [])
    )


def _is_structured_ncv_failure(result: AnalyzerResult) -> bool:
    if result.analyzer_name != "NCVPipelineVerifier" or result.ncv_report is None:
        return False
    method = str(result.ncv_report.get("method_type", "")).lower()
    return "evidence_aggregation" in method and result.status in {"fail", "warn"}


def _is_structured_retrieval_diagnosis(result: AnalyzerResult) -> bool:
    if result.analyzer_name != "RetrievalDiagnosisAnalyzerV0":
        return False
    report = result.retrieval_diagnosis_report
    if report is None:
        return False
    return report.primary_failure_type.value in {"retrieval_miss", "retrieval_noise"}


def _is_heuristic_supporting_result(result: AnalyzerResult) -> bool:
    if result.analyzer_name in {"SemanticEntropyAnalyzer", "Layer6TaxonomyClassifier"}:
        return True
    if result.analyzer_name == "A2PAttributionAnalyzer":
        return not _has_nonlegacy_a2p_evidence(result)
    if result.failure_type == FailureType.STALE_RETRIEVAL and result.analyzer_name == "StaleRetrievalAnalyzer":
        return result.version_validity_report is None
    if result.failure_type == FailureType.INSUFFICIENT_CONTEXT and result.sufficiency_result is not None:
        method = result.sufficiency_result.method.lower()
        # Demote to heuristic only when method is term-coverage AND no structured markers present
        evidence_text = " ".join(result.evidence)
        if _any_structured_sufficiency_marker(evidence_text):
            return False
        return "lexical" in method or "term_coverage" in method
    return False


def _has_nonlegacy_a2p_evidence(result: AnalyzerResult) -> bool:
    for attribution in result.claim_attributions or []:
        if attribution.fallback_used:
            continue
        if attribution.attribution_method == "legacy_failure_level_heuristic":
            continue
        if attribution.evidence:
            return True
    for attribution in result.claim_attributions_v2 or []:
        if attribution.fallback_used:
            continue
        if attribution.attribution_method == "legacy_failure_level_heuristic":
            continue
        if attribution.evidence_summary:
            return True
        if any(candidate.evidence_for for candidate in attribution.candidate_causes):
            return True
    return False


def _is_external_advisory_result(result: AnalyzerResult) -> bool:
    analyzer_name = result.analyzer_name.lower()
    if any(marker in analyzer_name for marker in _EXTERNAL_PROVIDER_MARKERS):
        return True
    if result.retrieval_diagnosis_report is not None:
        signals = result.retrieval_diagnosis_report.evidence_signals
        if signals and all(_signal_is_external(signal.source_report) for signal in signals):
            return True
    if result.ncv_report is not None:
        node_results = result.ncv_report.get("node_results", [])
        evidence_signals: list[dict[str, Any]] = []
        for node_result in node_results:
            evidence_signals.extend(node_result.get("evidence_signals", []))
        if evidence_signals and all(_signal_is_external(signal.get("source_report")) for signal in evidence_signals):
            return True
    return False


def _signal_is_external(source_report: str | None) -> bool:
    if source_report is None:
        return False
    lowered = source_report.lower()
    return "external" in lowered or any(marker in lowered for marker in _EXTERNAL_PROVIDER_MARKERS)
