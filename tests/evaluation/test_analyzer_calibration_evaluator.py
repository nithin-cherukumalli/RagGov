from __future__ import annotations

from raggov.evaluation.analyzer_calibration import (
    AnalyzerCalibrationCase,
    compute_analyzer_calibration_metrics,
    render_analyzer_calibration_markdown,
)
from raggov.models.diagnosis import AnalyzerResult, FailureStage, FailureType
from raggov.models.signals import EvidenceSignalMetadata


def _signal(
    *,
    strength: str = "medium",
    method_status: str = "heuristic_baseline",
    calibration_status: str = "uncalibrated",
) -> EvidenceSignalMetadata:
    return EvidenceSignalMetadata(
        signal_name="test_signal",
        source_analyzer="TestAnalyzer",
        method="test_method",
        method_status=method_status,  # type: ignore[arg-type]
        calibration_status=calibration_status,  # type: ignore[arg-type]
        evidence_strength=strength,  # type: ignore[arg-type]
        evidence_tier="structured",
    )


def _result(
    analyzer_name: str,
    failure_type: FailureType,
    stage: FailureStage,
    *,
    status: str = "fail",
    signal: EvidenceSignalMetadata | None = None,
) -> AnalyzerResult:
    return AnalyzerResult(
        analyzer_name=analyzer_name,
        status=status,  # type: ignore[arg-type]
        failure_type=failure_type,
        stage=stage,
        evidence=[f"{analyzer_name} evidence"],
        signal_metadata=[signal or _signal()],
    )


def test_metric_computation_counts_selected_true_and_false_positive() -> None:
    cases = [
        AnalyzerCalibrationCase(
            case_id="case_ok",
            category="retrieval",
            mode="native",
            expected_primary="SCOPE_VIOLATION",
            expected_stage="RETRIEVAL",
            actual_primary="SCOPE_VIOLATION",
            actual_stage="RETRIEVAL",
            selected_analyzer="ScopeAnalyzer",
            analyzer_results=[
                _result("ScopeAnalyzer", FailureType.SCOPE_VIOLATION, FailureStage.RETRIEVAL)
            ],
        ),
        AnalyzerCalibrationCase(
            case_id="case_bad",
            category="grounding",
            mode="native",
            expected_primary="UNSUPPORTED_CLAIM",
            expected_stage="GROUNDING",
            actual_primary="CITATION_MISMATCH",
            actual_stage="GROUNDING",
            selected_analyzer="CitationAnalyzer",
            analyzer_results=[
                _result("CitationAnalyzer", FailureType.CITATION_MISMATCH, FailureStage.GROUNDING)
            ],
        ),
    ]

    payload = compute_analyzer_calibration_metrics(cases)
    rows = {row["analyzer_name"]: row for row in payload["analyzers"]}

    assert rows["ScopeAnalyzer"]["candidate_count"] == 1
    assert rows["ScopeAnalyzer"]["selected_count"] == 1
    assert rows["ScopeAnalyzer"]["true_positive_count"] == 1
    assert rows["ScopeAnalyzer"]["precision"] == 1.0
    assert rows["CitationAnalyzer"]["false_positive_count"] == 1
    assert rows["CitationAnalyzer"]["precision"] == 0.0


def test_false_negative_when_expected_candidate_emitted_but_not_selected() -> None:
    cases = [
        AnalyzerCalibrationCase(
            case_id="case_suppressed",
            category="retrieval",
            mode="native",
            expected_primary="RETRIEVAL_ANOMALY",
            expected_stage="RETRIEVAL",
            actual_primary="UNSUPPORTED_CLAIM",
            actual_stage="GROUNDING",
            selected_analyzer="GroundingAnalyzer",
            analyzer_results=[
                _result("RetrievalAnalyzer", FailureType.RETRIEVAL_ANOMALY, FailureStage.RETRIEVAL),
                _result("GroundingAnalyzer", FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING),
            ],
        )
    ]

    payload = compute_analyzer_calibration_metrics(cases)
    rows = {row["analyzer_name"]: row for row in payload["analyzers"]}

    assert rows["RetrievalAnalyzer"]["false_negative_count"] == 1
    assert rows["GroundingAnalyzer"]["false_positive_count"] == 1
    assert payload["totals"]["unclaimed_expected_count"] == 0


def test_unclaimed_expected_count_when_no_analyzer_emits_expected_candidate() -> None:
    cases = [
        AnalyzerCalibrationCase(
            case_id="case_unclaimed",
            category="retrieval",
            mode="native",
            expected_primary="RETRIEVAL_DEPTH_LIMIT",
            expected_stage="RETRIEVAL",
            actual_primary="UNSUPPORTED_CLAIM",
            actual_stage="GROUNDING",
            selected_analyzer="GroundingAnalyzer",
            analyzer_results=[
                _result("GroundingAnalyzer", FailureType.UNSUPPORTED_CLAIM, FailureStage.GROUNDING)
            ],
        )
    ]

    payload = compute_analyzer_calibration_metrics(cases)

    assert payload["totals"]["unclaimed_expected_count"] == 1


def test_metadata_distributions_and_report_schema() -> None:
    signal = _signal(
        strength="strong",
        method_status="practical_approximation",
        calibration_status="uncalibrated",
    )
    cases = [
        AnalyzerCalibrationCase(
            case_id="case_meta",
            category="answer_quality",
            mode="external-enhanced",
            expected_primary="UNSUPPORTED_CLAIM",
            expected_stage="GENERATION",
            actual_primary="UNSUPPORTED_CLAIM",
            actual_stage="GENERATION",
            selected_analyzer="AnswerQualityAnalyzer",
            analyzer_results=[
                _result(
                    "AnswerQualityAnalyzer",
                    FailureType.UNSUPPORTED_CLAIM,
                    FailureStage.GENERATION,
                    signal=signal,
                )
            ],
        )
    ]

    payload = compute_analyzer_calibration_metrics(cases)
    row = payload["analyzers"][0]
    markdown = render_analyzer_calibration_markdown(payload)

    assert payload["audit_only"] is True
    assert payload["calibration_claim_made"] is False
    assert payload["production_gating_enabled"] is False
    assert payload["readiness_gaps"]
    assert row["evidence_strength_distribution"] == {"strong": 1}
    assert row["method_status_distribution"] == {"practical_approximation": 1}
    assert row["calibration_status_distribution"] == {"uncalibrated": 1}
    assert "Analyzer-wise Calibration Audit" in markdown
