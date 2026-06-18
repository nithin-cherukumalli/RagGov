"""Internal helpers for GovRAG's primary failure selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any

from raggov.calibration_status import CalibrationStatus, get_calibration_status
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.signals import EvidenceSignalMetadata


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
    citation_reason: str | None = None
    citation_markers: tuple[str, ...] = ()
    signal_metadata: tuple[EvidenceSignalMetadata, ...] = ()
    suppressed_reason: str | None = None
    method_status: str | None = None
    evidence_strength: str | None = None
    evidence_tier_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
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
            "citation_reason": self.citation_reason,
            "citation_markers": list(self.citation_markers),
            "signal_metadata": [sig.model_dump() for sig in self.signal_metadata],
        }
        if self.suppressed_reason is not None:
            d["suppressed_reason"] = self.suppressed_reason
        if self.method_status is not None:
            d["method_status"] = self.method_status
        if self.evidence_strength is not None:
            d["evidence_strength"] = self.evidence_strength
        if self.evidence_tier_name is not None:
            d["evidence_tier_name"] = self.evidence_tier_name
        return d


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
    FailureType.POST_RATIONALIZED_CITATION,
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


def classify_evidence_tier(result: AnalyzerResult) -> EvidenceTier:
    if result.failure_type in _BLOCKING_FAILURES or _is_parser_blocking_failure(result):
        return EvidenceTier.BLOCKING_DETERMINISTIC
    if (
        result.failure_type == FailureType.SUSPICIOUS_CHUNK
        and result.analyzer_name in {"PromptInjectionAnalyzer", "PoisoningHeuristicAnalyzer", "RetrievalAnomalyAnalyzer"}
    ):
        return EvidenceTier.BLOCKING_DETERMINISTIC
    if _is_external_advisory_result(result):
        return EvidenceTier.EXTERNAL_ADVISORY
    if result.analyzer_name == "RetrievalQualityAnalyzer" and result.stage == FailureStage.RETRIEVAL:
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if result.failure_type in _STRUCTURED_FAILURES:
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if any((
        _is_structured_insufficient_context(result),
        _is_structured_stale_retrieval(result),
        _is_structured_ncv_failure(result),
        _is_structured_retrieval_diagnosis(result),
    )):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if (
        result.failure_type == FailureType.SCOPE_VIOLATION
        and any("quoted query entity missing" in evidence for evidence in result.evidence)
    ):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if (
        result.analyzer_name == "SemanticEntropyAnalyzer"
        and result.failure_type == FailureType.LOW_CONFIDENCE
        and any(
            "query_understanding ambiguity" in evidence or "text-only uncertainty" in evidence
            for evidence in result.evidence
        )
    ):
        return EvidenceTier.STRUCTURED_DIAGNOSTIC
    if result.failure_type in _HEURISTIC_FAILURES or _is_heuristic_supporting_result(result):
        return EvidenceTier.HEURISTIC_SUPPORTING
    return EvidenceTier.UNKNOWN


def build_candidates(
    results: list[AnalyzerResult],
    result_weights: dict[str, float],
) -> list[DecisionCandidate]:
    candidates: list[DecisionCandidate] = []
    for index, result in enumerate(results):
        if result.status not in {"fail", "warn"} or result.failure_type is None:
            continue
        tier = classify_evidence_tier(result)
        calibration_status = _resolve_calibration_status(result)
        
        # Collect signal metadata for this result
        signal_metadata_list: list[EvidenceSignalMetadata] = []
        if result.signal_metadata:
            signal_metadata_list.extend(result.signal_metadata)
        if result.analyzer_report is not None:
            if hasattr(result.analyzer_report, "findings") and result.analyzer_report.findings:
                for finding in result.analyzer_report.findings:
                    if getattr(finding, "signal_metadata", None) is not None:
                        signal_metadata_list.append(finding.signal_metadata)
            if hasattr(result.analyzer_report, "selected_finding") and getattr(result.analyzer_report, "selected_finding") is not None:
                if getattr(result.analyzer_report.selected_finding, "signal_metadata", None) is not None:
                    signal_metadata_list.append(result.analyzer_report.selected_finding.signal_metadata)

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
                reason=(
                    f"{result.analyzer_name} emitted {result.failure_type.value} at tier "
                    f"{tier.value} with calibration_status={calibration_status}."
                ),
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
                citation_reason=_citation_reason(result),
                citation_markers=tuple(_citation_markers(result)),
                signal_metadata=tuple(signal_metadata_list),
            )
        )
    return candidates


def _signal_strength_rank(metadata: EvidenceSignalMetadata | None) -> int:
    if metadata is None:
        return 0
    return {
        "hard": 5,
        "strong": 4,
        "medium": 3,
        "weak": 2,
        "advisory": 1,
    }.get(metadata.evidence_strength, 0)


def _signal_tier_rank(metadata: EvidenceSignalMetadata | None) -> int:
    if metadata is None:
        return 0
    return {
        "structured": 3,
        "calibrated": 3,
        "proxy": 2,
        "external": 1,
        "heuristic": 1,
    }.get(metadata.evidence_tier, 0)


def _calibration_status_rank(metadata: EvidenceSignalMetadata | None) -> int:
    if metadata is None:
        return 0
    return {
        "calibrated_heldout": 4,
        "calibrated_dev": 3,
        "provisional_dataset": 2,
        "uncalibrated": 1,
        "unknown": 1,
    }.get(metadata.calibration_status, 0)


def _is_weak_uncalibrated_signal(candidate: DecisionCandidate) -> bool:
    if not candidate.signal_metadata:
        return False
    for sig in candidate.signal_metadata:
        is_weak = (
            sig.calibration_status in {"uncalibrated", "unknown"}
            and sig.evidence_strength in {"weak", "advisory"}
            and sig.evidence_tier in {"heuristic", "proxy", "external"}
            and sig.method_status in {"heuristic_baseline", "external_advisory", "practical_approximation"}
        )
        if not is_weak:
            return False
    return True


def _is_strong_structured_signal(candidate: DecisionCandidate) -> bool:
    if not candidate.signal_metadata:
        return False
    for sig in candidate.signal_metadata:
        if (
            sig.evidence_strength in {"hard", "strong"}
            and sig.evidence_tier in {"structured", "calibrated"}
            and sig.method_status in {"structured_deterministic", "calibrated_statistical", "practical_approximation"}
        ):
            return True
    return False


def _signal_can_override(candidate: DecisionCandidate, incumbent: DecisionCandidate) -> bool:
    import logging
    logger = logging.getLogger(__name__)

    if not candidate.signal_metadata and not incumbent.signal_metadata:
        return False
    if not candidate.signal_metadata:
        logger.warning(
            f"Trace Warning: Missing signal metadata for candidate {candidate.analyzer_name} "
            f"({candidate.failure_type.value}). Preserving legacy behavior."
        )
        return False
    if not incumbent.signal_metadata:
        logger.warning(
            f"Trace Warning: Missing signal metadata for incumbent {incumbent.analyzer_name} "
            f"({incumbent.failure_type.value}). Preserving legacy behavior."
        )
        return _is_strong_structured_signal(candidate)

    cand_strength = max(_signal_strength_rank(sig) for sig in candidate.signal_metadata)
    inc_strength = max(_signal_strength_rank(sig) for sig in incumbent.signal_metadata)
    if cand_strength != inc_strength:
        return cand_strength > inc_strength

    cand_tier = max(_signal_tier_rank(sig) for sig in candidate.signal_metadata)
    inc_tier = max(_signal_tier_rank(sig) for sig in incumbent.signal_metadata)
    if cand_tier != inc_tier:
        return cand_tier > inc_tier

    cand_calib = max(_calibration_status_rank(sig) for sig in candidate.signal_metadata)
    inc_calib = max(_calibration_status_rank(sig) for sig in incumbent.signal_metadata)
    if cand_calib != inc_calib:
        return cand_calib > inc_calib

    return False



def _candidate_has_structured_stronger_evidence(candidate: DecisionCandidate) -> bool:
    if not candidate.signal_metadata:
        return False
    for sig in candidate.signal_metadata:
        if (
            sig.evidence_strength in {"hard", "strong"}
            and sig.evidence_tier == "structured"
            and sig.method_status == "structured_deterministic"
        ):
            return True
    return False


def _is_lexical_overlap_scope_relevance(candidate: DecisionCandidate) -> bool:
    if not candidate.signal_metadata:
        return False
    for sig in candidate.signal_metadata:
        if (
            sig.signal_name in {"lexical_overlap_relevance", "scope_lexical_irrelevant", "noisy_chunk_ids"}
            or sig.method == "lexical_overlap"
        ):
            return True
    return False


def _is_external_advisory_retrieval_signal(candidate: DecisionCandidate) -> bool:
    if not candidate.signal_metadata:
        return False
    for sig in candidate.signal_metadata:
        if (
            sig.method_status == "external_advisory"
            or sig.evidence_tier == "external"
            or sig.evidence_strength == "advisory"
        ):
            if candidate.analyzer_name in {"RetrievalEvidenceProfilerV0", "ScopeViolationAnalyzer", "RetrievalDiagnosisAnalyzerV0"}:
                return True
    return False


def _is_actually_strong_structured(candidate: DecisionCandidate) -> bool:
    if candidate.signal_metadata:
        return _is_strong_structured_signal(candidate)
    return candidate.evidence_tier in {EvidenceTier.BLOCKING_DETERMINISTIC, EvidenceTier.STRUCTURED_DIAGNOSTIC}


def split_candidates(
    candidates: list[DecisionCandidate],
) -> tuple[list[DecisionCandidate], list[DecisionCandidate], list[str]]:
    eligible: list[DecisionCandidate] = []
    suppressed: list[DecisionCandidate] = []
    warnings: list[str] = []

    # 1. Identify if any candidate has stronger structured evidence:
    has_stronger_structured = False
    for candidate in candidates:
        if _is_actually_strong_structured(candidate):
            has_stronger_structured = True
            break
        if _candidate_has_structured_stronger_evidence(candidate):
            has_stronger_structured = True
            break

    # 2. Check for meta heuristic candidates
    has_non_meta_candidate = any(
        candidate.analyzer_name not in _META_HEURISTIC_ANALYZERS
        for candidate in candidates
    )

    for candidate in candidates:
        # Check if candidate is from retrieval/scope components:
        is_retrieval_scope = (
            candidate.analyzer_name in {"ScopeViolationAnalyzer", "RetrievalEvidenceProfilerV0", "RetrievalDiagnosisAnalyzerV0", "RetrievalQualityAnalyzer"}
            or candidate.stage == FailureStage.RETRIEVAL
            or candidate.failure_type == FailureType.SCOPE_VIOLATION
            or any(
                sig.source_analyzer in {"ScopeViolationAnalyzer", "RetrievalEvidenceProfilerV0", "RetrievalDiagnosisAnalyzerV0", "RetrievalQualityAnalyzer"}
                for sig in candidate.signal_metadata
            )
        )

        # Rule 5: If signal metadata is missing for a retrieval/scope candidate, warn and preserve
        if is_retrieval_scope and not candidate.signal_metadata:
            warnings.append(
                f"Signal metadata was missing for candidate {candidate.analyzer_name} ({candidate.failure_type.value}). Preserving current behavior."
            )

        core_analyzers = {
            "ClaimGroundingAnalyzer",
            "CitationFaithfulnessAnalyzerV0",
            "CitationFaithfulnessProbe",
            "SufficiencyAnalyzer",
            "ClaimAwareSufficiencyAnalyzer",
            "TemporalSourceValidityAnalyzerV1",
            "RetrievalDiagnosisAnalyzerV0",
            "RetrievalQualityAnalyzer",
            "ScopeViolationAnalyzer",
        }
        is_core = (
            candidate.analyzer_name in core_analyzers
            or any(sig.source_analyzer in core_analyzers for sig in candidate.signal_metadata)
        )
        if is_core and not candidate.signal_metadata:
            warnings.append(
                f"Signal metadata was missing for core candidate {candidate.analyzer_name} ({candidate.failure_type.value}). Preserving legacy behavior."
            )

        # Check for weak uncalibrated signal
        is_weak_uncalibrated = _is_weak_uncalibrated_signal(candidate)

        # Rule 1 General: Weak/advisory uncalibrated heuristic signals cannot override hard/strong structured signals
        if is_weak_uncalibrated and has_stronger_structured:
            candidate.suppressed_reason = "weak_uncalibrated_signal_cannot_override_structured_evidence"
            if candidate.signal_metadata:
                sig = candidate.signal_metadata[0]
                candidate.method_status = sig.method_status
                candidate.calibration_status = sig.calibration_status
                candidate.evidence_strength = sig.evidence_strength
                candidate.evidence_tier_name = sig.evidence_tier.value if hasattr(sig.evidence_tier, 'value') else str(sig.evidence_tier)
            suppressed.append(candidate)
            warnings.append(
                f"Suppressed weak uncalibrated signal {candidate.analyzer_name} ({candidate.failure_type.value}) because stronger structured evidence exists."
            )
            continue

        # Rule 1: lexical_overlap-based scope/relevance signals cannot override structured INSUFFICIENT_CONTEXT or structured grounding/citation evidence
        is_lexical_overlap = _is_lexical_overlap_scope_relevance(candidate)
        if is_lexical_overlap and has_stronger_structured:
            candidate.suppressed_reason = "weak_uncalibrated_signal_cannot_override_structured_evidence"
            if candidate.signal_metadata:
                sig = candidate.signal_metadata[0]
                candidate.method_status = sig.method_status
                candidate.calibration_status = sig.calibration_status
                candidate.evidence_strength = sig.evidence_strength
                candidate.evidence_tier_name = sig.evidence_tier.value if hasattr(sig.evidence_tier, 'value') else str(sig.evidence_tier)
            suppressed.append(candidate)
            warnings.append(
                f"Suppressed weak uncalibrated lexical overlap signal {candidate.analyzer_name} ({candidate.failure_type.value}) because stronger structured evidence exists."
            )
            continue

        # Rule 2: external_advisory retrieval signals cannot select primary failure by themselves
        is_external_adv = _is_external_advisory_retrieval_signal(candidate)
        if is_external_adv:
            candidate.suppressed_reason = "external_advisory_retrieval_signal_cannot_select_primary_by_default"
            if candidate.signal_metadata:
                sig = candidate.signal_metadata[0]
                candidate.method_status = sig.method_status
                candidate.calibration_status = sig.calibration_status
                candidate.evidence_strength = sig.evidence_strength
                candidate.evidence_tier_name = sig.evidence_tier.value if hasattr(sig.evidence_tier, 'value') else str(sig.evidence_tier)
            suppressed.append(candidate)
            warnings.append(
                f"Suppressed external advisory retrieval signal {candidate.analyzer_name} ({candidate.failure_type.value}) from selecting primary failure."
            )
            continue

        # Rule 3: retrieval anomaly based only on heuristic/noise proxy must not become SECURITY
        if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY and is_weak_uncalibrated:
            if candidate.stage == FailureStage.SECURITY:
                candidate.stage = FailureStage.RETRIEVAL
            if has_stronger_structured:
                candidate.suppressed_reason = "weak_uncalibrated_signal_cannot_override_structured_evidence"
                if candidate.signal_metadata:
                    sig = candidate.signal_metadata[0]
                    candidate.method_status = sig.method_status
                    candidate.calibration_status = sig.calibration_status
                    candidate.evidence_strength = sig.evidence_strength
                    candidate.evidence_tier_name = sig.evidence_tier.value if hasattr(sig.evidence_tier, 'value') else str(sig.evidence_tier)
                suppressed.append(candidate)
                warnings.append(
                    f"Suppressed weak uncalibrated retrieval anomaly {candidate.analyzer_name} because stronger structured evidence exists."
                )
                continue

        # Rule 3 cont: if it would produce SECURITY or SCOPE_VIOLATION, suppress/demote when stronger evidence exists
        if is_weak_uncalibrated and has_stronger_structured and (candidate.failure_type == FailureType.SCOPE_VIOLATION or candidate.stage == FailureStage.SECURITY):
            candidate.suppressed_reason = "weak_uncalibrated_signal_cannot_override_structured_evidence"
            if candidate.signal_metadata:
                sig = candidate.signal_metadata[0]
                candidate.method_status = sig.method_status
                candidate.calibration_status = sig.calibration_status
                candidate.evidence_strength = sig.evidence_strength
                candidate.evidence_tier_name = sig.evidence_tier.value if hasattr(sig.evidence_tier, 'value') else str(sig.evidence_tier)
            suppressed.append(candidate)
            warnings.append(
                f"Suppressed weak uncalibrated signal {candidate.analyzer_name} producing {candidate.failure_type.value} because stronger structured evidence exists."
            )
            continue

        # Standard filters:
        if candidate.evidence_tier == EvidenceTier.EXTERNAL_ADVISORY:
            suppressed.append(candidate)
            continue
        if has_non_meta_candidate and candidate.analyzer_name in _META_HEURISTIC_ANALYZERS:
            suppressed.append(candidate)
            continue

        eligible.append(candidate)

    if not eligible and any(
        c.suppressed_reason == "weak_uncalibrated_signal_cannot_override_structured_evidence"
        for c in suppressed
    ):
        restored = [
            c for c in suppressed
            if c.suppressed_reason == "weak_uncalibrated_signal_cannot_override_structured_evidence"
        ]
        for c in restored:
            c.suppressed_reason = None
            eligible.append(c)
            suppressed.remove(c)
        warnings.append("signal_strength_guard_preserved_legacy_to_avoid_false_incomplete")

    # Standard warning processing
    standard_suppressed_advisory = any(
        c.evidence_tier == EvidenceTier.EXTERNAL_ADVISORY
        for c in suppressed
        if not c.suppressed_reason
    )
    if standard_suppressed_advisory:
        warnings.append("External advisory candidates were suppressed from primary failure selection.")

    standard_suppressed_meta = has_non_meta_candidate and any(
        c.analyzer_name in _META_HEURISTIC_ANALYZERS
        for c in suppressed
        if not c.suppressed_reason
    )
    if standard_suppressed_meta:
        warnings.append(
            "Meta heuristic candidates were retained for traceability but suppressed from primary selection because direct analyzer evidence exists."
        )

    return eligible, suppressed, warnings


def candidate_sort_key(
    candidate: DecisionCandidate,
    failure_priority: list[FailureType],
) -> tuple[int, float, float, int, int, int]:
    priority_index = {failure_type: index for index, failure_type in enumerate(failure_priority)}
    return (
        _tier_rank(candidate),
        -_specificity_rank(candidate),
        -candidate.weight,
        -_calibration_rank(candidate.calibration_status),
        priority_index.get(candidate.failure_type, len(failure_priority)),
        candidate.original_index,
    )


def apply_named_exception_rules(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> tuple[DecisionCandidate, str | None]:
    rules = (
        _prefer_explicit_phantom_citation,
        _prefer_explicit_post_rationalized_over_incidental_phantom,
        _prefer_grounding_citation_authority,
        _suppress_citation_when_downstream_symptom,
        _prefer_grounding_over_related_non_supporting_probe,
        _prefer_low_confidence_over_generic_symptoms,
        _prefer_warn_retrieval_anomaly_over_missing_citation,
        _prefer_unsupported_over_generic_insufficient_context,
        _prefer_scope_violation_over_generic_insufficient_context,
        _prefer_unsupported_when_partial_support_exists,
        _require_candidate_backed_unsupported,
        _require_explicit_contradiction,
        _require_explicit_citation_root_for_post_rationalized,
        _prefer_stale_root_cause,
        _prefer_explicit_retrieval_anomaly_over_weak_scope,
        _apply_signal_strength_guard_v2,
    )
    current = winner
    applied: list[str] = []
    for rule in rules:
        updated = rule(current, ranked_status_pool, results)
        if updated is not current:
            current = updated
            applied.append(rule.__name__)
    return current, ", ".join(applied) if applied else None


def suppress_alternatives(
    *,
    winner: DecisionCandidate,
    eligible: list[DecisionCandidate],
    initially_suppressed: list[DecisionCandidate],
    failure_priority: list[FailureType],
) -> tuple[list[DecisionCandidate], list[DecisionCandidate]]:
    alternatives: list[DecisionCandidate] = []
    final_suppressed = list(initially_suppressed)
    all_other_candidates = sorted(
        [candidate for candidate in eligible if candidate is not winner],
        key=lambda candidate: (
            0 if candidate.status == winner.status else 1,
            *candidate_sort_key(candidate, failure_priority),
        ),
    )
    for candidate in all_other_candidates:
        if _should_suppress_candidate(candidate, winner):
            final_suppressed.append(candidate)
        else:
            alternatives.append(candidate)
    return alternatives, final_suppressed


def build_selection_reason(
    *,
    winner: DecisionCandidate,
    alternatives: list[DecisionCandidate],
    suppressed: list[DecisionCandidate],
    applied_rule: str | None,
    warnings: list[str] | None = None,
) -> str:
    if _is_parser_failure_type(winner.failure_type):
        return "Parser blocking failure selected as earliest deterministic root cause."
    if warnings and "signal_strength_guard_preserved_legacy_to_avoid_false_incomplete" in warnings:
        return (
            f"signal_strength_guard_preserved_legacy_to_avoid_false_incomplete: Selected {winner.failure_type.value} from "
            f"{winner.analyzer_name} because no other structured candidate was available."
        )
    if any(getattr(c, "suppressed_reason", None) == "weak_uncalibrated_signal_cannot_override_structured_evidence" for c in suppressed):
        return (
            f"signal_strength_guard_selected_stronger_candidate: Selected {winner.failure_type.value} from "
            f"{winner.analyzer_name} because weaker signals were suppressed."
        )
    if applied_rule:
        return (
            f"Selected {winner.failure_type.value} from {winner.analyzer_name} after applying named policy rules: "
            f"{applied_rule}."
        )
    if (
        winner.failure_type == FailureType.STALE_RETRIEVAL
        and _candidate_has_strong_version_root_cause(winner)
    ):
        return (
            "Selected STALE_RETRIEVAL because invalid or stale cited/answer-bearing sources make downstream "
            "claim-level symptoms secondary."
        )
    if (
        winner.failure_type in {FailureType.CITATION_MISMATCH, FailureType.POST_RATIONALIZED_CITATION}
        and _candidate_has_explicit_citation_root_cause(winner)
    ):
        return (
            "Selected citation failure because explicit citation-root evidence outranked downstream unsupported-claim symptoms."
        )
    if winner.signal_metadata:
        sig = winner.signal_metadata[0]
        return (
            f"Selected {winner.failure_type.value} from {winner.analyzer_name} using fail-before-warn, "
            f"then evidence tier, specificity, analyzer weight, calibration rank, and taxonomy priority. "
            f"[selected strength={sig.evidence_strength}, tier={sig.evidence_tier}, calibration={sig.calibration_status}]"
        )
    return (
        f"Selected {winner.failure_type.value} from {winner.analyzer_name} using fail-before-warn, "
        "then evidence tier, specificity, analyzer weight, calibration rank, and taxonomy priority."
    )


def _prefer_explicit_phantom_citation(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type not in {
        FailureType.POST_RATIONALIZED_CITATION,
        FailureType.CONTRADICTED_CLAIM,
    }:
        return winner
    phantom = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.CITATION_MISMATCH
            and candidate.citation_reason == "phantom_citation"
        ),
        None,
    )
    if phantom is not None:
        phantom.stage = FailureStage.GROUNDING
        return phantom
    return winner


def _prefer_grounding_citation_authority(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not (
        winner.failure_type == FailureType.CITATION_MISMATCH
        and winner.analyzer_name == "CitationMismatchAnalyzer"
    ):
        return winner
    explicit_citation = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type in {FailureType.CITATION_MISMATCH, FailureType.POST_RATIONALIZED_CITATION}
            and candidate.analyzer_name == "CitationFaithfulnessAnalyzerV0"
        ),
        None,
    )
    if explicit_citation is not None:
        return explicit_citation
    if winner.citation_reason == "phantom_citation":
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.POST_RATIONALIZED_CITATION
            and candidate.analyzer_name == "CitationFaithfulnessProbe"
        ),
        winner,
    )


def _prefer_explicit_post_rationalized_over_incidental_phantom(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type != FailureType.CITATION_MISMATCH:
        return winner
    post_rationalized = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.POST_RATIONALIZED_CITATION
            and candidate.analyzer_name == "CitationFaithfulnessProbe"
            and _candidate_has_explicit_citation_root_cause(candidate)
        ),
        None,
    )
    if post_rationalized is None or winner.citation_reason != "phantom_citation":
        return winner
    if not _probe_has_mixed_citation_attachment(post_rationalized, results):
        return winner
    return post_rationalized


def _prefer_grounding_over_related_non_supporting_probe(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not (
        winner.failure_type == FailureType.POST_RATIONALIZED_CITATION
        and winner.analyzer_name == "CitationFaithfulnessProbe"
        and _has_related_non_supporting_citation_evidence(results)
    ):
        return winner
    if any(
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and _candidate_has_strong_version_root_cause(candidate)
        for candidate in ranked_status_pool
    ):
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.UNSUPPORTED_CLAIM
            and candidate.analyzer_name == "ClaimGroundingAnalyzer"
            and candidate.status == "fail"
        ),
        winner,
    )


def _prefer_unsupported_when_partial_support_exists(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not (
        winner.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and winner.analyzer_name in {"SufficiencyAnalyzer", "RetrievalDiagnosisAnalyzerV0"}
    ):
        return winner
    grounding_candidate = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.UNSUPPORTED_CLAIM
            and candidate.analyzer_name == "ClaimGroundingAnalyzer"
        ),
        None,
    )
    if grounding_candidate is None:
        return winner
    grounding_result = results[grounding_candidate.original_index]
    if any(
        getattr(claim, "label_reason", None) == "partial_support"
        for claim in (grounding_result.claim_results or [])
    ):
        return grounding_candidate
    return winner


def _prefer_unsupported_over_generic_insufficient_context(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type != FailureType.INSUFFICIENT_CONTEXT:
        return winner
    if _candidate_has_explicit_sufficiency_root_cause(winner):
        return winner
    if not _insufficient_context_is_generic_or_grounding_derived(winner):
        return winner
    if winner.failure_type == FailureType.INSUFFICIENT_CONTEXT and "explicit_missing_context" in winner.evidence_summary.lower():
        return winner
    grounding_candidate = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type in {FailureType.UNSUPPORTED_CLAIM, FailureType.CONTRADICTED_CLAIM}
            and candidate.analyzer_name == "ClaimGroundingAnalyzer"
            and _claim_grounding_has_candidate_evidence(candidate, results)
        ),
        None,
    )
    return grounding_candidate or winner


def _prefer_scope_violation_over_generic_insufficient_context(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type != FailureType.INSUFFICIENT_CONTEXT:
        return winner
    if not _insufficient_context_is_generic_or_retrieval_miss(winner):
        return winner
    scope_candidate = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.SCOPE_VIOLATION
            and candidate.status == "fail"
            and _has_explicit_scope_violation(candidate)
        ),
        None,
    )
    return scope_candidate or winner


def _prefer_low_confidence_over_generic_symptoms(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type not in {
        FailureType.CITATION_MISMATCH,
        FailureType.INSUFFICIENT_CONTEXT,
        FailureType.SCOPE_VIOLATION,
    }:
        return winner
    if _is_stronger_concrete_root_cause(winner):
        return winner
    low_confidence = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.LOW_CONFIDENCE
            and candidate.analyzer_name == "SemanticEntropyAnalyzer"
            and _has_explicit_low_confidence_signal(candidate)
        ),
        None,
    )
    return low_confidence or winner


def _prefer_warn_retrieval_anomaly_over_missing_citation(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not (
        winner.failure_type == FailureType.CITATION_MISMATCH
        and winner.citation_reason == "citation_missing"
    ):
        return winner
    retrieval_anomaly = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY
            and candidate.status == "warn"
            and _has_explicit_retrieval_anomaly(candidate)
        ),
        None,
    )
    return retrieval_anomaly or winner


def _require_explicit_contradiction(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type != FailureType.CONTRADICTED_CLAIM or _has_explicit_contradiction(winner, results):
        return winner
    grounding_contradiction = next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.analyzer_name == "ClaimGroundingAnalyzer"
            and candidate.failure_type == FailureType.CONTRADICTED_CLAIM
            and candidate.status == "fail"
        ),
        None,
    )
    if grounding_contradiction is not None and grounding_contradiction is not winner:
        if _has_explicit_contradiction(grounding_contradiction, results):
            return grounding_contradiction
        if _claim_grounding_has_candidate_evidence(grounding_contradiction, results):
            grounding_contradiction.failure_type = FailureType.UNSUPPORTED_CLAIM
            grounding_contradiction.reason = (
                "ClaimGroundingAnalyzer emitted non-explicit contradiction evidence; "
                "policy classified it as unsupported candidate-backed grounding evidence."
            )
            return grounding_contradiction
    if (
        winner.analyzer_name == "ClaimGroundingAnalyzer"
        and _claim_grounding_has_candidate_evidence(winner, results)
    ):
        winner.failure_type = FailureType.UNSUPPORTED_CLAIM
        winner.reason = (
            "ClaimGroundingAnalyzer emitted non-explicit contradiction evidence; "
            "policy classified it as unsupported candidate-backed grounding evidence."
        )
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate is not winner
            and candidate.failure_type in {FailureType.UNSUPPORTED_CLAIM, FailureType.INSUFFICIENT_CONTEXT}
        ),
        winner,
    )


def _require_candidate_backed_unsupported(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not (
        winner.failure_type == FailureType.UNSUPPORTED_CLAIM
        and winner.analyzer_name == "ClaimGroundingAnalyzer"
        and not _claim_grounding_has_candidate_evidence(winner, results)
    ):
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate is not winner
            and candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
            and candidate.status == "fail"
        ),
        winner,
    )


def _require_explicit_citation_root_for_post_rationalized(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if (
        winner.failure_type != FailureType.POST_RATIONALIZED_CITATION
        or _candidate_has_explicit_citation_root_cause(winner)
    ):
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type in {FailureType.UNSUPPORTED_CLAIM, FailureType.INSUFFICIENT_CONTEXT}
            and candidate.analyzer_name in {"ClaimGroundingAnalyzer", "SufficiencyAnalyzer", "RetrievalDiagnosisAnalyzerV0"}
        ),
        winner,
    )


def _citation_is_downstream_symptom(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
) -> bool:
    if winner.failure_type not in {FailureType.CITATION_MISMATCH, FailureType.POST_RATIONALIZED_CITATION}:
        return False
    if (
        _candidate_has_explicit_citation_root_cause(winner)
        and winner.citation_reason not in {
            "citation_missing",
            "post_rationalized_citation",
            "related_non_supporting_citation",
        }
    ):
        return False
    return any(
        candidate for candidate in ranked_status_pool
        if candidate is not winner and candidate.status == "fail" and (
            _is_downstream_symptom_target(candidate)
        )
    )


def _is_downstream_symptom_target(candidate: DecisionCandidate) -> bool:
    if candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT:
        return candidate.analyzer_name in {
            "SufficiencyAnalyzer",
            "ClaimAwareSufficiencyAnalyzer",
            "RetrievalDiagnosisAnalyzerV0",
        }
    if candidate.failure_type == FailureType.UNSUPPORTED_CLAIM:
        return candidate.analyzer_name == "ClaimGroundingAnalyzer"
    if candidate.failure_type == FailureType.STALE_RETRIEVAL:
        return _candidate_has_strong_version_root_cause(candidate)
    if candidate.failure_type == FailureType.SCOPE_VIOLATION:
        return _has_explicit_scope_violation(candidate)
    if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY:
        return _has_explicit_retrieval_anomaly(candidate)
    if candidate.failure_type == FailureType.RETRIEVAL_DEPTH_LIMIT:
        return True
    return False


def _suppress_citation_when_downstream_symptom(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if not _citation_is_downstream_symptom(winner, ranked_status_pool):
        return winner
    root_cause_preference: list[FailureType] = [
        FailureType.STALE_RETRIEVAL,
        FailureType.UNSUPPORTED_CLAIM,
        FailureType.SCOPE_VIOLATION,
        FailureType.RETRIEVAL_ANOMALY,
        FailureType.RETRIEVAL_DEPTH_LIMIT,
        FailureType.INSUFFICIENT_CONTEXT,
    ]
    if winner.citation_reason == "post_rationalized_citation":
        return winner
    for preferred_type in root_cause_preference:
        for candidate in ranked_status_pool:
            if candidate is winner:
                continue
            if candidate.status != "fail":
                continue
            if (
                candidate.failure_type == FailureType.UNSUPPORTED_CLAIM
                and not _claim_grounding_has_candidate_evidence(candidate, results)
            ):
                continue
            if candidate.failure_type == preferred_type and _is_downstream_symptom_target(candidate):
                return candidate
    return winner


def _prefer_stale_root_cause(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type not in {
        FailureType.CONTRADICTED_CLAIM,
        FailureType.POST_RATIONALIZED_CITATION,
        FailureType.UNSUPPORTED_CLAIM,
    }:
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.STALE_RETRIEVAL
            and (
                _candidate_has_strong_version_root_cause(candidate)
                or candidate.analyzer_name in {"StaleRetrievalAnalyzer", "VersionValidityAnalyzerV1"}
            )
        ),
        winner,
    )


def _prefer_explicit_retrieval_anomaly_over_weak_scope(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    if winner.failure_type != FailureType.SCOPE_VIOLATION:
        return winner
    return next(
        (
            candidate
            for candidate in ranked_status_pool
            if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY
            and _has_explicit_retrieval_anomaly(candidate)
            and not _has_explicit_scope_violation(winner)
        ),
        winner,
    )


def _apply_signal_strength_guard_v2(
    winner: DecisionCandidate,
    ranked_status_pool: list[DecisionCandidate],
    results: list[AnalyzerResult],
) -> DecisionCandidate:
    # Rule 1: Weak/advisory uncalibrated heuristic signals cannot override hard/strong structured signals.
    if _is_weak_uncalibrated_signal(winner):
        for candidate in ranked_status_pool:
            if _is_strong_structured_signal(candidate):
                winner.suppressed_reason = "weak_uncalibrated_signal_cannot_override_structured_evidence"
                candidate.reason = (
                    f"Override Rule 1: {candidate.analyzer_name} ({candidate.failure_type.value}) [strong structured] "
                    f"overrode {winner.analyzer_name} ({winner.failure_type.value}) [weak uncalibrated]."
                )
                return candidate

    # Rule 2: Term-coverage-only sufficiency fallback cannot override candidate-backed unsupported grounding.
    is_term_coverage_winner = (
        winner.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and any(sig.signal_name == "term_coverage_fallback" for sig in winner.signal_metadata)
    )
    if is_term_coverage_winner:
        for candidate in ranked_status_pool:
            if (
                candidate.failure_type == FailureType.UNSUPPORTED_CLAIM
                and any(sig.signal_name == "candidate_backed_unsupported_claim" for sig in candidate.signal_metadata)
            ):
                winner.suppressed_reason = "term_coverage_sufficiency_cannot_override_unsupported_grounding"
                candidate.reason = (
                    f"Override Rule 2: Grounding unsupported claim ({candidate.analyzer_name}) "
                    f"overrode term-coverage sufficiency fallback ({winner.analyzer_name})."
                )
                return candidate

    # Rule 3: Missing citation without explicit requirement cannot override structured grounding/citation/source-validity failure.
    is_missing_citation_winner = (
        winner.failure_type == FailureType.CITATION_MISMATCH
        and winner.citation_reason == "citation_missing"
    )
    if is_missing_citation_winner:
        for candidate in ranked_status_pool:
            if candidate is winner:
                continue
            is_structured = _is_strong_structured_signal(candidate) or candidate.evidence_tier in {EvidenceTier.BLOCKING_DETERMINISTIC, EvidenceTier.STRUCTURED_DIAGNOSTIC}
            if is_structured and candidate.failure_type in {
                FailureType.UNSUPPORTED_CLAIM,
                FailureType.CONTRADICTED_CLAIM,
                FailureType.STALE_RETRIEVAL,
                FailureType.CITATION_MISMATCH
            }:
                winner.suppressed_reason = "missing_citation_cannot_override_structured_failure"
                candidate.reason = (
                    f"Override Rule 3: Structured failure ({candidate.analyzer_name}, {candidate.failure_type.value}) "
                    f"overrode missing citation symptom ({winner.analyzer_name})."
                )
                return candidate

    # Rule 4: Semantic/proxy/advisory signal cannot override explicit contradiction.
    is_proxy_advisory_winner = (
        winner.analyzer_name == "SemanticEntropyAnalyzer"
        or winner.evidence_tier in {EvidenceTier.EXTERNAL_ADVISORY, EvidenceTier.HEURISTIC_SUPPORTING}
        or _is_weak_uncalibrated_signal(winner)
    )
    if is_proxy_advisory_winner:
        for candidate in ranked_status_pool:
            if (
                candidate.failure_type == FailureType.CONTRADICTED_CLAIM
                and any(sig.signal_name == "grounding_contradiction" for sig in candidate.signal_metadata)
            ):
                winner.suppressed_reason = "proxy_advisory_cannot_override_explicit_contradiction"
                candidate.reason = (
                    f"Override Rule 4: Grounding contradiction ({candidate.analyzer_name}) "
                    f"overrode proxy/advisory signal ({winner.analyzer_name})."
                )
                return candidate

    # Rule 5: External advisory signals cannot select primary failure by default.
    if any(sig.method_status == "external_advisory" for sig in winner.signal_metadata):
        for candidate in ranked_status_pool:
            if not any(sig.method_status == "external_advisory" for sig in candidate.signal_metadata):
                winner.suppressed_reason = "external_advisory_cannot_select_primary_by_default"
                candidate.reason = (
                    f"Override Rule 5: Native candidate ({candidate.analyzer_name}) "
                    f"overrode external advisory signal ({winner.analyzer_name})."
                )
                return candidate

    # Rule 6: Hard/strong structured source-validity invalidation can override downstream unsupported/citation symptoms when source is answer-bearing.
    is_downstream_symptom = winner.failure_type in {
        FailureType.UNSUPPORTED_CLAIM,
        FailureType.CITATION_MISMATCH,
        FailureType.POST_RATIONALIZED_CITATION
    }
    if is_downstream_symptom:
        for candidate in ranked_status_pool:
            if (
                candidate.failure_type == FailureType.STALE_RETRIEVAL
                and candidate.analyzer_name == "TemporalSourceValidityAnalyzerV1"
                and any(
                    sig.evidence_strength in {"hard", "strong"}
                    and sig.evidence_tier == "structured"
                    for sig in candidate.signal_metadata
                )
            ):
                winner.suppressed_reason = "downstream_symptom_overridden_by_source_validity_invalidation"
                candidate.reason = (
                    f"Override Rule 6: Hard/strong structured source-validity invalidation ({candidate.analyzer_name}) "
                    f"overrode downstream claim symptom ({winner.analyzer_name})."
                )
                return candidate

    # Rule 7: Hard/strong structured grounding contradiction can override generic sufficiency/retrieval symptoms.
    is_generic_symptom = winner.failure_type in {
        FailureType.INSUFFICIENT_CONTEXT,
        FailureType.RETRIEVAL_ANOMALY,
        FailureType.SCOPE_VIOLATION
    }
    if is_generic_symptom:
        for candidate in ranked_status_pool:
            if (
                candidate.failure_type == FailureType.CONTRADICTED_CLAIM
                and any(
                    sig.signal_name == "grounding_contradiction"
                    and sig.evidence_strength in {"hard", "strong"}
                    for sig in candidate.signal_metadata
                )
            ):
                winner.suppressed_reason = "generic_symptom_overridden_by_grounding_contradiction"
                candidate.reason = (
                    f"Override Rule 7: Hard/strong structured grounding contradiction ({candidate.analyzer_name}) "
                    f"overrode generic symptom ({winner.analyzer_name})."
                )
                return candidate

    return winner



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
    if candidate.failure_type == FailureType.CONTRADICTED_CLAIM:
        return 95
    if (
        candidate.failure_type == FailureType.CITATION_MISMATCH
        and candidate.analyzer_name == "CitationFaithfulnessAnalyzerV0"
        and candidate.citation_reason == "phantom_citation"
    ):
        return 94
    if (
        candidate.failure_type == FailureType.CITATION_MISMATCH
        and candidate.analyzer_name == "CitationFaithfulnessAnalyzerV0"
        and _candidate_has_explicit_citation_root_cause(candidate)
    ):
        return 93
    if candidate.failure_type == FailureType.POST_RATIONALIZED_CITATION:
        return 92 if _candidate_has_explicit_citation_root_cause(candidate) else 82
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name == "SufficiencyAnalyzer"
        and _candidate_has_explicit_sufficiency_root_cause(candidate)
    ):
        return 92
    if (
        candidate.failure_type == FailureType.LOW_CONFIDENCE
        and candidate.analyzer_name == "SemanticEntropyAnalyzer"
        and "query_understanding ambiguity" in candidate.evidence_summary
    ):
        return 93
    if candidate.analyzer_name == "RetrievalQualityAnalyzer":
        if candidate.failure_type == FailureType.RETRIEVAL_DEPTH_LIMIT:
            return 93
        if candidate.failure_type == FailureType.SCOPE_VIOLATION:
            return 91
        if candidate.failure_type == FailureType.RETRIEVAL_ANOMALY:
            return 89
    if (
        candidate.failure_type == FailureType.SCOPE_VIOLATION
        and "quoted query entity missing" in candidate.evidence_summary
    ):
        return 91
    if candidate.failure_type == FailureType.INCOMPLETE_DIAGNOSIS:
        return 90
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name == "RetrievalDiagnosisAnalyzerV0"
    ):
        return 90
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name == "SufficiencyAnalyzer"
        and (
            candidate.sufficiency_reason == "stale_context_mistaken_as_sufficient"
            or "[sufficiency:stale_context_mistaken_as_sufficient]" in candidate.sufficiency_markers
            or "[sufficiency:stale_context_mistaken_as_sufficient]" in candidate.evidence_summary
        )
    ):
        # Staleness is a more specific retrieval root cause than generic
        # insufficiency: it identifies the wrong-version context that made
        # the otherwise present evidence unusable.
        return 91
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1", "RetrievalDiagnosisAnalyzerV0"}
        and _candidate_has_strong_version_root_cause(candidate)
    ):
        return 89
    if (
        candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and candidate.analyzer_name in {"SufficiencyAnalyzer", "ClaimAwareSufficiencyAnalyzer"}
    ):
        return 88
    if (
        candidate.failure_type == FailureType.STALE_RETRIEVAL
        and candidate.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1", "RetrievalDiagnosisAnalyzerV0"}
        and _candidate_has_retrieval_quality_stale_root_cause(candidate)
    ):
        return 81
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
        and winner.failure_type in {
            FailureType.CONTRADICTED_CLAIM,
            FailureType.UNSUPPORTED_CLAIM,
            FailureType.CITATION_MISMATCH,
            FailureType.INSUFFICIENT_CONTEXT,
            FailureType.STALE_RETRIEVAL,
        }
    ):
        return True
    if (
        candidate.failure_type == FailureType.CITATION_MISMATCH
        and winner.failure_type == FailureType.INSUFFICIENT_CONTEXT
        and winner.analyzer_name in {"SufficiencyAnalyzer", "ClaimAwareSufficiencyAnalyzer", "RetrievalDiagnosisAnalyzerV0"}
        and _candidate_has_explicit_sufficiency_root_cause(winner)
        and not _citation_mismatch_is_strong(candidate)
    ):
        return True
    if (
        candidate.failure_type == FailureType.UNSUPPORTED_CLAIM
        and winner.failure_type in {FailureType.CITATION_MISMATCH, FailureType.POST_RATIONALIZED_CITATION}
        and _candidate_has_explicit_citation_root_cause(winner)
    ):
        return True
    if (
        candidate.failure_type == FailureType.CITATION_MISMATCH
        and candidate.stage == FailureStage.RETRIEVAL
        and winner.stage == FailureStage.GROUNDING
        and winner.failure_type in {FailureType.CITATION_MISMATCH, FailureType.POST_RATIONALIZED_CITATION}
        and _candidate_has_explicit_citation_root_cause(winner)
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
    if candidate.citation_reason in {
        "phantom_citation",
        "related_non_supporting_citation",
        "post_rationalized_citation",
        "cited_source_contradicts_claim",
    }:
        return True
    if any(marker != "[citation:citation_missing]" for marker in candidate.citation_markers):
        return True
    evidence = candidate.evidence_summary.lower()
    return any(
        marker in evidence
        for marker in (
            "phantom citation",
            "cited document not present",
            "absent from retrieved evidence",
            "contradict",
            "wrong source",
        )
    )


def _candidate_has_explicit_citation_root_cause(candidate: DecisionCandidate) -> bool:
    if candidate.failure_type == FailureType.POST_RATIONALIZED_CITATION and candidate.analyzer_name == "CitationFaithfulnessProbe":
        return True
    if candidate.citation_reason in {
        "phantom_citation",
        "cited_source_contradicts_claim",
        "post_rationalized_citation",
        "related_non_supporting_citation",
        "citation_missing",
    }:
        return True
    return any(
        marker in {
            "[citation:phantom_citation]",
            "[citation:cited_source_contradicts_claim]",
            "[citation:post_rationalized_citation]",
            "[citation:related_non_supporting_citation]",
            "[citation:citation_missing]",
        }
        for marker in candidate.citation_markers
    )


def _candidate_has_strong_version_root_cause(candidate: DecisionCandidate) -> bool:
    return candidate.version_severity in {"cited_invalid_source", "answer_bearing_invalid_source"}


def _candidate_has_retrieval_quality_stale_root_cause(candidate: DecisionCandidate) -> bool:
    return candidate.version_severity == "retrieved_only_stale_source"


def _citation_reason(result: AnalyzerResult) -> str | None:
    evidence_text = " ".join(result.evidence)
    lowered = evidence_text.lower()
    if "phantom citation" in lowered or "[citation:phantom_citation]" in evidence_text:
        return "phantom_citation"
    if "post-rationalized" in lowered or "post rationalized" in lowered or "[citation:post_rationalized_citation]" in evidence_text:
        return "post_rationalized_citation"
    if "[citation:related_non_supporting_citation]" in evidence_text:
        return "related_non_supporting_citation"
    if "[citation:citation_missing]" in evidence_text:
        return "citation_missing"
    if "[citation:cited_source_contradicts_claim]" in evidence_text:
        return "cited_source_contradicts_claim"
    report = result.citation_faithfulness_report
    if report is not None:
        if report.phantom_citation_doc_ids:
            return "phantom_citation"
        if report.contradicted_claim_ids:
            return "cited_source_contradicts_claim"
        if report.missing_citation_claim_ids:
            return "citation_missing"
        if report.unsupported_claim_ids:
            return "related_non_supporting_citation"
    return None


def _has_related_non_supporting_citation_evidence(results: list[AnalyzerResult]) -> bool:
    return any(
        result.analyzer_name == "CitationFaithfulnessAnalyzerV0"
        and result.status == "warn"
        and _citation_reason(result) == "related_non_supporting_citation"
        for result in results
    )


def _probe_has_mixed_citation_attachment(
    candidate: DecisionCandidate,
    results: list[AnalyzerResult],
) -> bool:
    probe_results = results[candidate.original_index].citation_probe_results or []
    statuses = {
        str(probe_result.get("status"))
        for probe_result in probe_results
        if isinstance(probe_result, dict)
    }
    return {"unsupported_cited_claim", "citation_supported"} <= statuses


def _citation_markers(result: AnalyzerResult) -> list[str]:
    report = result.citation_faithfulness_report
    if report is not None and getattr(report, "evidence_markers", None):
        return list(report.evidence_markers)
    markers: list[str] = []
    evidence_text = " ".join(result.evidence)
    for marker in (
        "[citation:phantom_citation]",
        "[citation:related_non_supporting_citation]",
        "[citation:citation_missing]",
        "[citation:cited_source_contradicts_claim]",
        "[citation:post_rationalized_citation]",
    ):
        if marker in evidence_text:
            markers.append(marker)
    return list(dict.fromkeys(markers))


def _has_explicit_contradiction(candidate: DecisionCandidate, results: list[AnalyzerResult]) -> bool:
    # "explicit_contradiction" is the default label_reason for ANY contradicted claim
    # (see _default_label_reason in verifiers.py).  Counting it here would make
    # _require_explicit_contradiction a no-op — every CONTRADICTED_CLAIM would appear
    # "explicit" even when the contradiction is purely heuristic.  Only treat a
    # contradiction as explicit when there is substantive additional evidence:
    # a real value conflict, a "conflicting value" phrase, or a specific typed reason.
    result = results[candidate.original_index]
    for claim in result.claim_results or []:
        if claim.value_conflicts:
            if claim.value_matches and (claim.atomicity_status == "compound" or len(claim.value_conflicts) > 1):
                continue
            return True
        reason = (claim.evidence_reason or "").lower()
        if "conflicting value" in reason:
            return True
        if claim.label_reason in {"explicit_conflict", "value_conflict", "date_conflict", "unit_mismatch", "contradictory_evidence"}:
            return True
        if claim.label_reason == "explicit_contradiction" and _claim_has_textual_contradiction(claim.claim_text, result):
            return True
    evidence_text = " ".join(result.evidence).lower()
    return "conflicting value" in evidence_text


def _claim_has_textual_contradiction(claim_text: str, result: AnalyzerResult) -> bool:
    claim_lower = claim_text.lower()
    evidence_texts: list[str] = []
    if result.grounding_evidence_bundle is not None:
        for record in result.grounding_evidence_bundle.claim_evidence_records:
            if record.claim_text != claim_text:
                continue
            for chunk in getattr(record, "candidate_evidence_chunks", []) or []:
                chunk_text = getattr(chunk, "chunk_text", None)
                if chunk_text:
                    evidence_texts.append(chunk_text.lower())
    evidence_lower = " ".join(evidence_texts)
    if not evidence_lower:
        return False

    positive_permission = any(term in claim_lower for term in ("allowed", "permitted", "authorized", "eligible"))
    negative_permission = any(
        term in evidence_lower
        for term in ("not allowed", "not permitted", "prohibited", "ineligible", "not eligible")
    )
    negative_claim = any(
        term in claim_lower
        for term in ("not allowed", "not permitted", "prohibited", "ineligible", "not eligible")
    )
    positive_evidence = any(term in evidence_lower for term in ("allowed", "permitted", "authorized", "eligible"))
    # Task 16: a claim asserting BROAD/unrestricted permission contradicts evidence stating an
    # explicit restriction (e.g. "you can wear anything you want" vs "Policy: No blue shirts").
    broad_permission_claim = bool(
        re.search(
            r"\b(?:wear|do|use|say|bring|choose|eat|take|have)\s+(?:anything|whatever|any)\b"
            r"|\banything\s+(?:you|they)\s+(?:want|like|wish)\b"
            r"|\bno\s+(?:restrictions?|rules?|limits?|dress\s*code)\b",
            claim_lower,
        )
    )
    restriction_evidence = bool(
        re.search(
            r"\bno\s+[a-z]+(?:\s+[a-z]+)?\b"
            r"|\b(?:not\s+allowed|prohibit(?:ed)?|forbidden|banned|must\s+not|may\s+not"
            r"|only\s+[a-z]+\s+(?:allowed|permitted)|required\s+to)\b",
            evidence_lower,
        )
    )
    return (
        (positive_permission and negative_permission)
        or (negative_claim and positive_evidence)
        or (broad_permission_claim and restriction_evidence)
    )


def _has_explicit_scope_violation(candidate: DecisionCandidate) -> bool:
    text = candidate.evidence_summary.lower()
    has_irrelevant = "label=irrelevant" in text
    if has_irrelevant and "method=lexical_overlap" in text:
        has_irrelevant = False

    return has_irrelevant or any(
        marker in text
        for marker in (
            "quoted query entity missing",
            "query entity missing",
            "out-of-scope",
            "wrong entity",
            "wrong domain",
            "wrong task",
        )
    )


def _is_abstention(claim_text: str) -> bool:
    lowered = claim_text.lower().strip()
    lowered = lowered.rstrip(".?!\"'")
    
    abstention_phrases = {
        "i don't know",
        "i do not know",
        "i don't have",
        "i do not have",
        "not mentioned",
        "no information",
        "not provided",
        "unable to find",
        "unable to determine",
        "cannot determine",
        "cannot find",
        "cannot provide",
        "no mention",
        "not specified",
        "does not mention",
        "does not specify",
        "does not state",
        "does not contain",
        "is not mentioned",
        "is not provided",
        "is not available",
        "not available in",
        "not discussed",
        "no details",
        "no data",
    }
    
    if any(phrase in lowered for phrase in abstention_phrases):
        return True
        
    if lowered.startswith("i apologize") or lowered.startswith("i'm sorry"):
        return True
        
    return False


def _claim_grounding_has_candidate_evidence(
    candidate: DecisionCandidate,
    results: list[AnalyzerResult],
) -> bool:
    """Return True only when there is substantive candidate evidence."""
    result = results[candidate.original_index]
    
    if result.claim_results:
        unsupported_claims = [
            claim
            for claim in result.claim_results
            if claim.label in {"unsupported", "contradicted"}
        ]
        if not unsupported_claims:
            return False
        if all(_is_abstention(claim.claim_text) for claim in unsupported_claims):
            return False
        # Fast-exit: if every unsupported/contradicted claim explicitly states no
        # evidence was found, the populated chunk IDs are noise from corpus
        # enumeration, not actual candidate snippets. Return False unconditionally
        # so that purely off-topic retrievals do not trigger the unsupported-over-
        # insufficient-context preference rule.
        if all(
            getattr(claim, "label_reason", None) in {"unsupported_no_evidence", "contradicted_no_evidence"}
            for claim in unsupported_claims
        ):
            if not any(
                ("related but insufficient" in (getattr(claim, "evidence_reason", "") or "").lower())
                or bool(getattr(claim, "neutral_chunk_ids", None))
                for claim in unsupported_claims
            ):
                return False
        if result.grounding_evidence_bundle is not None:
            for record in result.grounding_evidence_bundle.claim_evidence_records:
                candidate_ids = getattr(record, "candidate_evidence_chunk_ids", [])
                candidate_chunks = getattr(record, "candidate_evidence_chunks", [])
                neutral_ids = getattr(record, "neutral_candidate_ids", [])
                contradicting_ids = getattr(record, "contradicting_candidate_ids", [])
                if candidate_ids or candidate_chunks or neutral_ids or contradicting_ids:
                    return True
        # Only count candidate/neutral chunk IDs as real evidence when the
        # label_reason confirms that some candidate evidence was actually found.
        # "unsupported_no_evidence" means the IDs are corpus-wide, not candidates.
        _no_evidence_reasons = {"unsupported_no_evidence", "contradicted_no_evidence"}
        for claim in result.claim_results:
            if getattr(claim, "label_reason", None) in _no_evidence_reasons:
                # These candidate IDs represent the searched corpus, not actual
                # candidate evidence snippets — skip the ID check for this claim.
                pass
            elif claim.candidate_chunk_ids or claim.neutral_chunk_ids or getattr(claim, "contradicting_chunk_ids", []):
                return True
            if claim.label_reason in {
                "unsupported_with_candidate_evidence",
                "partial_support",
                "related_but_non_supporting",
                "contradictory_evidence",
                "explicit_contradiction",
            }:
                return True
        return False

    evidence_text = " ".join(result.evidence).lower()
    if "claim grounding summary:" in evidence_text or "total=" in evidence_text:
        return True
    if "candidate_chunk_ids" in evidence_text and "candidate_chunk_ids\":[]" not in evidence_text:
        return True
    return False



def _insufficient_context_is_generic_or_retrieval_miss(candidate: DecisionCandidate) -> bool:
    if candidate.failure_type != FailureType.INSUFFICIENT_CONTEXT:
        return False
    if not _candidate_has_explicit_sufficiency_root_cause(candidate):
        return True
    text = candidate.evidence_summary.lower()
    return any(
        marker in text
        for marker in (
            '"primary_failure_type":"retrieval_miss"',
            "query term coverage",
            "insufficient_context_with_unsupported_claims",
            "insufficient_context_without_verifiable_claims",
        )
    )


def _insufficient_context_is_generic_or_grounding_derived(candidate: DecisionCandidate) -> bool:
    if candidate.failure_type != FailureType.INSUFFICIENT_CONTEXT:
        return False
    text = candidate.evidence_summary.lower()
    if candidate.analyzer_name == "RetrievalDiagnosisAnalyzerV0":
        if '"primary_failure_type":"retrieval_miss"' in text:
            if '"candidate_chunk_ids":[]' in text or '"candidate_chunk_ids"' not in text:
                return False
            return True
        return any(
            marker in text
            for marker in (
                '"primary_failure_type":"retrieval_miss"',
                "insufficient_context_with_unsupported_claims",
                "unsupported_claim",
                "candidate_chunk_ids",
            )
        )
    if candidate.analyzer_name == "SufficiencyAnalyzer":
        if not _candidate_has_explicit_sufficiency_root_cause(candidate):
            return True
        if candidate.sufficiency_reason == "missing_temporal_or_freshness_requirement":
            return True
        return False
    return not _candidate_has_explicit_sufficiency_root_cause(candidate)


def _has_explicit_low_confidence_signal(candidate: DecisionCandidate) -> bool:
    text = candidate.evidence_summary.lower()
    return any(
        marker in text
        for marker in (
            "text-only ambiguity",
            "text-only uncertainty",
            "query_understanding ambiguity",
            "explicit uncertainty alternatives",
            "multiple distinct retrieved contexts",
        )
    )


def _is_stronger_concrete_root_cause(candidate: DecisionCandidate) -> bool:
    if candidate.failure_type in {
        FailureType.CONTRADICTED_CLAIM,
        FailureType.PROMPT_INJECTION,
        FailureType.PRIVACY_VIOLATION,
        FailureType.SUSPICIOUS_CHUNK,
        FailureType.STALE_RETRIEVAL,
    }:
        return True
    if candidate.failure_type in {
        FailureType.CITATION_MISMATCH,
        FailureType.POST_RATIONALIZED_CITATION,
    }:
        return _citation_mismatch_is_strong(candidate)
    if candidate.failure_type == FailureType.INSUFFICIENT_CONTEXT:
        return (
            _candidate_has_explicit_sufficiency_root_cause(candidate)
            and not _insufficient_context_is_generic_or_retrieval_miss(candidate)
        )
    if candidate.failure_type == FailureType.SCOPE_VIOLATION:
        return _has_explicit_scope_violation(candidate)
    return False


def _has_explicit_retrieval_anomaly(candidate: DecisionCandidate) -> bool:
    text = candidate.evidence_summary
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "retrieval_noise",
            "score cliff",
            "duplicate chunk",
            "duplicate chunks",
            "noisy_chunk_ids",
            "score distribution",
            "reranking artifact",
        )
    ):
        return True
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception:
            return False
        return str(payload.get("primary_failure_type")) in {"retrieval_noise", "retrieval_anomaly"}
    return False


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
    ) or (
        result.analyzer_name == "ParserValidationAnalyzer"
        and result.status == "fail"
        and result.stage == FailureStage.PARSING
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
    evidence_text = " ".join(result.evidence)
    if "no retrieved chunks available" in evidence_text.lower() or "no retrieved chunks" in evidence_text.lower():
        return True
    if result.sufficiency_result.evidence_markers or result.sufficiency_result.structured_failure_reason:
        return True
    if _any_structured_sufficiency_marker(evidence_text):
        return True
    # Zero-coverage is a deterministic signal: the retrieved context has
    # absolutely no topical overlap with the query.  This is as certain as an
    # explicit structured marker and should be treated as STRUCTURED_DIAGNOSTIC
    # so it is not displaced by weaker heuristic signals.
    if result.analyzer_name == "SufficiencyAnalyzer" and result.sufficiency_result is not None:
        for cov in result.sufficiency_result.coverage:
            if cov.requirement_id == "term_coverage_v0" and cov.confidence == 0.0 and cov.status == "missing":
                return True
    method = result.sufficiency_result.method.lower()
    return "lexical" not in method and "term_coverage" not in method


def _is_structured_stale_retrieval(result: AnalyzerResult) -> bool:
    if result.failure_type != FailureType.STALE_RETRIEVAL:
        return False
    if result.analyzer_name in {"TemporalSourceValidityAnalyzerV1", "VersionValidityAnalyzerV1", "RetrievalDiagnosisAnalyzerV0"}:
        return _version_severity(result) in {
            "cited_invalid_source",
            "answer_bearing_invalid_source",
            "retrieved_only_stale_source",
        }
    if result.analyzer_name == "SufficiencyAnalyzer" and result.sufficiency_result is not None:
        if result.sufficiency_result.structured_failure_reason == "stale_context_mistaken_as_sufficient":
            return True
        if "[sufficiency:stale_context_mistaken_as_sufficient]" in result.sufficiency_result.evidence_markers:
            return True
    evidence_text = result.evidence[0] if result.evidence else ""
    return "[sufficiency:stale_context_mistaken_as_sufficient]" in evidence_text


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
        evidence_text = " ".join(result.evidence)
        if "no retrieved chunks" in evidence_text.lower() or "no retrieved chunks available" in evidence_text.lower():
            return False
        if _any_structured_sufficiency_marker(evidence_text):
            return False
        method = result.sufficiency_result.method.lower()
        return "lexical" in method or "term_coverage" in method
    return False


def _has_nonlegacy_a2p_evidence(result: AnalyzerResult) -> bool:
    for attribution in result.claim_attributions or []:
        if not attribution.fallback_used and attribution.attribution_method != "legacy_failure_level_heuristic" and attribution.evidence:
            return True
    for attribution in result.claim_attributions_v2 or []:
        if attribution.fallback_used or attribution.attribution_method == "legacy_failure_level_heuristic":
            continue
        if attribution.evidence_summary or any(candidate.evidence_for for candidate in attribution.candidate_causes):
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


def _any_structured_sufficiency_marker(evidence_text: str) -> bool:
    return any(marker in evidence_text for marker in _SUFFICIENCY_STRUCTURED_MARKERS)
