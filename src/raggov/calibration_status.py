"""Calibration status classification for RagGov analyzers.

CRITICAL: This module exists to prevent pseudo-rigor.

With only 10-50 labeled samples, we CANNOT claim serious threshold calibration.
ARESCalibrator recommends 150-300 samples for stable operating points.

This module explicitly classifies each analyzer's calibration status and
threshold provenance to maintain honesty about what is validated vs provisional.
"""

from __future__ import annotations

from enum import Enum


class CalibrationStatus(str, Enum):
    """Calibration status for an analyzer's operating thresholds."""

    CALIBRATED = "CALIBRATED"  # 150+ labels, thresholds validated with confidence intervals
    PROVISIONAL = "PROVISIONAL"  # 30-150 labels, thresholds tentative, may not generalize
    NOT_CALIBRATED = "NOT_CALIBRATED"  # <30 labels, thresholds are defaults only
    DETERMINISTIC = "DETERMINISTIC"  # No calibration needed (rule-based, no thresholds)


class AnalyzerType(str, Enum):
    """Classification of analyzer implementation approach."""

    DETERMINISTIC_CHECK = "DETERMINISTIC_CHECK"  # Rule-based, no ML or thresholds
    HEURISTIC = "HEURISTIC"  # Pattern matching with thresholds
    PROXY_METRIC = "PROXY_METRIC"  # Approximates a target metric
    PARTIALLY_RESEARCH_ALIGNED = "PARTIALLY_RESEARCH_ALIGNED"  # Based on paper, simplified
    VALIDATED_METRIC = "VALIDATED_METRIC"  # Empirically validated against ground truth


class ThresholdProvenance(str, Enum):
    """Origin and justification for analyzer thresholds."""

    PAPER_DERIVED = "PAPER_DERIVED"  # From cited research paper
    EMPIRICALLY_TUNED = "EMPIRICALLY_TUNED"  # Tuned on validation set
    HEURISTIC_DEFAULT = "HEURISTIC_DEFAULT"  # Reasonable default, not validated
    ARBITRARY = "ARBITRARY"  # No documented justification
    NOT_APPLICABLE = "NOT_APPLICABLE"  # Deterministic analyzer, no thresholds


# Analyzer Calibration Status Classification
# Based on baseline_validation_v1.json results (10 cases)
ANALYZER_CALIBRATION_STATUS: dict[str, CalibrationStatus] = {
    # Deterministic analyzers (no thresholds, no calibration needed)
    "PromptInjectionAnalyzer": CalibrationStatus.DETERMINISTIC,
    "PrivacyAnalyzer": CalibrationStatus.DETERMINISTIC,
    "NCVPipelineVerifier": CalibrationStatus.DETERMINISTIC,

    # Provisional (10-50 samples available, but insufficient for stable thresholds)
    "ClaimGroundingAnalyzer": CalibrationStatus.PROVISIONAL,  # 10 cases, 50% accuracy
    "CitationFaithfulnessProbe": CalibrationStatus.PROVISIONAL,  # 10 cases, 50% accuracy
    "SufficiencyAnalyzer": CalibrationStatus.PROVISIONAL,  # 10 cases, 20% accuracy - UNSTABLE
    "CitationMismatchAnalyzer": CalibrationStatus.PROVISIONAL,  # 10 cases, 20% accuracy - UNSTABLE

    # Not calibrated (no samples or unstable performance)
    "ParserValidationAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 30% accuracy - UNSTABLE
    "StaleRetrievalAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 20% accuracy - UNSTABLE
    "ScopeViolationAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 20% accuracy - UNSTABLE
    "PoisoningHeuristicAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 20% accuracy - UNSTABLE
    "RetrievalAnomalyAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 10% accuracy - UNSTABLE
    "SemanticEntropyAnalyzer": CalibrationStatus.NOT_CALIBRATED,  # 10% accuracy - UNSTABLE (misnomed)

    # Meta analyzers (aggregate others' results, not directly calibratable)
    "Layer6TaxonomyClassifier": CalibrationStatus.DETERMINISTIC,
    "A2PAttributionAnalyzer": CalibrationStatus.DETERMINISTIC,
}


# Analyzer Type Classification
ANALYZER_TYPE_CLASSIFICATION: dict[str, AnalyzerType] = {
    "ClaimGroundingAnalyzer": AnalyzerType.PARTIALLY_RESEARCH_ALIGNED,  # NLI-based grounding
    "CitationFaithfulnessProbe": AnalyzerType.HEURISTIC,  # Anchor/predicate overlap (NOT true faithfulness)
    "SufficiencyAnalyzer": AnalyzerType.HEURISTIC,  # Token count + keyword matching
    "CitationMismatchAnalyzer": AnalyzerType.DETERMINISTIC_CHECK,  # Set membership check
    "ParserValidationAnalyzer": AnalyzerType.HEURISTIC,  # Pattern detection for structure loss
    "StaleRetrievalAnalyzer": AnalyzerType.DETERMINISTIC_CHECK,  # Date comparison
    "ScopeViolationAnalyzer": AnalyzerType.HEURISTIC,  # Keyword overlap
    "PromptInjectionAnalyzer": AnalyzerType.PARTIALLY_RESEARCH_ALIGNED,  # Based on Greshake et al.
    "PoisoningHeuristicAnalyzer": AnalyzerType.HEURISTIC,  # Pattern matching (NOT research-aligned)
    "RetrievalAnomalyAnalyzer": AnalyzerType.HEURISTIC,  # Statistical outlier detection
    "PrivacyAnalyzer": AnalyzerType.HEURISTIC,  # Pattern matching for PII
    "SemanticEntropyAnalyzer": AnalyzerType.PROXY_METRIC,  # Claim-label entropy, NOT semantic entropy
    "Layer6TaxonomyClassifier": AnalyzerType.HEURISTIC,  # Threshold-based mapping
    "A2PAttributionAnalyzer": AnalyzerType.HEURISTIC,  # Rule cascade (NOT counterfactual reasoning)
    "NCVPipelineVerifier": AnalyzerType.DETERMINISTIC_CHECK,  # Node coverage verification
}


# Threshold Provenance Classification
THRESHOLD_PROVENANCE: dict[str, ThresholdProvenance] = {
    "ClaimGroundingAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,  # NLI thresholds not tuned
    "CitationFaithfulnessProbe": ThresholdProvenance.HEURISTIC_DEFAULT,  # Overlap thresholds arbitrary
    "SufficiencyAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,  # Token count thresholds arbitrary
    "CitationMismatchAnalyzer": ThresholdProvenance.NOT_APPLICABLE,  # Deterministic
    "ParserValidationAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,
    "StaleRetrievalAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,  # max_age_days=180 arbitrary
    "ScopeViolationAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,
    "PromptInjectionAnalyzer": ThresholdProvenance.PAPER_DERIVED,  # Based on Greshake taxonomy
    "PoisoningHeuristicAnalyzer": ThresholdProvenance.ARBITRARY,  # No research alignment
    "RetrievalAnomalyAnalyzer": ThresholdProvenance.ARBITRARY,
    "PrivacyAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,
    "SemanticEntropyAnalyzer": ThresholdProvenance.HEURISTIC_DEFAULT,  # entropy_threshold=1.2 arbitrary
    "Layer6TaxonomyClassifier": ThresholdProvenance.HEURISTIC_DEFAULT,  # Score bands arbitrary
    "A2PAttributionAnalyzer": ThresholdProvenance.ARBITRARY,
    "NCVPipelineVerifier": ThresholdProvenance.NOT_APPLICABLE,
}


# Implementation Gap Documentation (what differs from ideal/research version)
IMPLEMENTATION_GAPS: dict[str, str] = {
    "SemanticEntropyAnalyzer": (
        "Uses claim-label distribution entropy (entailed/unsupported/contradicted), "
        "NOT semantic entropy per Farquhar et al. which requires sampling N completions "
        "and clustering by bidirectional NLI entailment. The proxy is weaker but available "
        "without additional LLM calls."
    ),
    "CitationFaithfulnessProbe": (
        "Checks citation CORRECTNESS (does cited doc contain claim) via anchor/predicate overlap, "
        "NOT citation FAITHFULNESS (did model rely on evidence vs parametric memory). "
        "True faithfulness requires perturbation-based testing per Wallat et al."
    ),
    "A2PAttributionAnalyzer": (
        "Deterministic mode uses prioritized rule cascade based on failure co-occurrence, "
        "NOT true counterfactual reasoning. The 'Abduct-Act-Predict' framing is aspirational. "
        "LLM mode prompts for structured reasoning but depends on model quality."
    ),
    "Layer6TaxonomyClassifier": (
        "Rule-based threshold mapping to Layer6-style labels, NOT a trained ML classifier. "
        "The 'classification' is deterministic score-band mapping."
    ),
    "PoisoningHeuristicAnalyzer": (
        "Basic pattern matching for suspicious content. No alignment to corpus poisoning research. "
        "Heuristic-only detection with high false positive risk."
    ),
    "ParserValidationAnalyzer": (
        "Detects structure loss via heuristic patterns (table flattening, hierarchy loss). "
        "Not exhaustive; may miss novel parser failures."
    ),
}


# Research Alignment Documentation
RESEARCH_ALIGNMENT: dict[str, str] = {
    "SemanticEntropyAnalyzer": "Farquhar et al. 2024 (CLAIMED, implementation differs significantly)",
    "CitationFaithfulnessProbe": "Wallat et al. 2024 (PARTIAL, correctness only, not faithfulness)",
    "PromptInjectionAnalyzer": "Greshake et al. 2023, StruQ 2025 (WELL-ALIGNED)",
    "ClaimGroundingAnalyzer": "NLI-based grounding patterns (PARTIALLY ALIGNED)",
    "A2PAttributionAnalyzer": "A2P framework (CLAIMED, deterministic mode is heuristic)",
    "Layer6TaxonomyClassifier": "Layer6 AI taxonomy (PARTIAL, mapping not classification)",
    "PoisoningHeuristicAnalyzer": "None (NO RESEARCH ALIGNMENT)",
    "RetrievalAnomalyAnalyzer": "None (HEURISTIC ONLY)",
    "SufficiencyAnalyzer": "None (HEURISTIC ONLY)",
    "ScopeViolationAnalyzer": "None (HEURISTIC ONLY)",
    "StaleRetrievalAnalyzer": "None (STRAIGHTFORWARD DATE CHECK)",
    "ParserValidationAnalyzer": "None (HEURISTIC PATTERNS)",
}


def get_calibration_status(analyzer_name: str) -> CalibrationStatus:
    """Return calibration status for an analyzer."""
    return ANALYZER_CALIBRATION_STATUS.get(analyzer_name, CalibrationStatus.NOT_CALIBRATED)


def get_analyzer_type(analyzer_name: str) -> AnalyzerType:
    """Return analyzer type classification."""
    return ANALYZER_TYPE_CLASSIFICATION.get(analyzer_name, AnalyzerType.HEURISTIC)


def get_threshold_provenance(analyzer_name: str) -> ThresholdProvenance:
    """Return threshold provenance for an analyzer."""
    return THRESHOLD_PROVENANCE.get(analyzer_name, ThresholdProvenance.ARBITRARY)


def get_implementation_gap(analyzer_name: str) -> str | None:
    """Return implementation gap description if documented."""
    return IMPLEMENTATION_GAPS.get(analyzer_name)


def get_research_alignment(analyzer_name: str) -> str | None:
    """Return research alignment description if documented."""
    return RESEARCH_ALIGNMENT.get(analyzer_name)


def requires_calibration(analyzer_name: str) -> bool:
    """Return True if analyzer has thresholds that could benefit from calibration."""
    status = get_calibration_status(analyzer_name)
    return status not in {CalibrationStatus.DETERMINISTIC, CalibrationStatus.CALIBRATED}


def is_unstable(analyzer_name: str, accuracy: float) -> bool:
    """Return True if analyzer accuracy is below stability threshold (60%)."""
    return accuracy < 0.6
