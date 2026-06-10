import pytest
from pydantic import BaseModel

from raggov.evaluation.subtle_failure_triage import compute_triage_audit
from raggov.models.diagnosis import FailureStage, FailureType, AnalyzerResult
from raggov.evaluation.analyzer_calibration import AnalyzerCalibrationCase
from raggov.models.signals import EvidenceSignalMetadata

def test_triage_audit_basic_counts():
    cases = [
        AnalyzerCalibrationCase(
            case_id="1",
            category="test",
            mode="native",
            expected_primary=FailureType.CLEAN.value,
            expected_stage=FailureStage.UNKNOWN.value,
            actual_primary=FailureType.INSUFFICIENT_CONTEXT.value,
            actual_stage=FailureStage.SUFFICIENCY.value,
            selected_analyzer="Missing",
            analyzer_results=[]
        ),
        AnalyzerCalibrationCase(
            case_id="2",
            category="test",
            mode="native",
            expected_primary=FailureType.UNSUPPORTED_CLAIM.value,
            expected_stage=FailureStage.GROUNDING.value,
            actual_primary=FailureType.CLEAN.value,
            actual_stage=FailureStage.UNKNOWN.value,
            selected_analyzer=None,
            analyzer_results=[]
        )
    ]
    
    payload = compute_triage_audit(cases)
    
    assert payload["total_cases"] == 2
    assert payload["total_failures"] == 2
    assert payload["false_clean_count"] == 1
    assert payload["wrong_type_count"] == 1
    
    false_clean_case = next(f for f in payload["failures"] if f["case_id"] == "2")
    assert false_clean_case["false_clean"] is True
    assert false_clean_case["likely_root_cause"] == "missing_analyzer_evidence"

def test_triage_audit_decision_policy():
    cases = [
        AnalyzerCalibrationCase(
            case_id="3",
            category="test",
            mode="native",
            expected_primary=FailureType.SCOPE_VIOLATION.value,
            expected_stage=FailureStage.RETRIEVAL.value,
            actual_primary=FailureType.STALE_RETRIEVAL.value,
            actual_stage=FailureStage.RETRIEVAL.value,
            selected_analyzer="WrongAnalyzer",
            analyzer_results=[
                AnalyzerResult(
                    analyzer_name="CorrectAnalyzer",
                    status="fail",
                    score=0.9,
                    failure_type=FailureType.SCOPE_VIOLATION,
                    stage=FailureStage.RETRIEVAL,
                    signal_metadata=[
                        EvidenceSignalMetadata(
                            signal_name="test",
                            source_analyzer="CorrectAnalyzer",
                            method="test",
                            evidence_strength="medium",
                            evidence_tier="structured",
                            method_status="practical_approximation",
                            calibration_status="uncalibrated"
                        )
                    ]
                )
            ]
        )
    ]
    
    payload = compute_triage_audit(cases)
    
    assert payload["total_cases"] == 1
    assert payload["total_failures"] == 1
    
    failure = payload["failures"][0]
    assert failure["likely_root_cause"] == "decision_policy_selection"
    assert failure["likely_owner"] == "CorrectAnalyzer"
    assert len(failure["candidate_details"]) == 1
    assert failure["candidate_details"][0]["metadata_present"] is True

def test_triage_audit_weak_metadata():
    cases = [
        AnalyzerCalibrationCase(
            case_id="4",
            category="test",
            mode="native",
            expected_primary=FailureType.UNSUPPORTED_CLAIM.value,
            expected_stage=FailureStage.GROUNDING.value,
            actual_primary=FailureType.CLEAN.value,
            actual_stage=FailureStage.UNKNOWN.value,
            selected_analyzer=None,
            analyzer_results=[
                AnalyzerResult(
                    analyzer_name="ClaimGroundingAnalyzer",
                    status="fail",
                    score=0.5,
                    failure_type=FailureType.UNSUPPORTED_CLAIM,
                    stage=FailureStage.GROUNDING,
                    signal_metadata=[]
                )
            ]
        )
    ]
    
    payload = compute_triage_audit(cases)
    failure = payload["failures"][0]
    assert failure["likely_root_cause"] == "weak_signal_metadata"
    assert failure["candidate_details"][0]["metadata_present"] is False
