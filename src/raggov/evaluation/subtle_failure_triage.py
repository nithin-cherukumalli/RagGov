"""Subtle Suite Failure Triage Audit."""

from __future__ import annotations

from collections import Counter
from typing import Any

from raggov.models.diagnosis import FailureStage, FailureType


def compute_triage_audit(cases: list[Any]) -> dict[str, Any]:
    failures = []
    
    total_cases = len(cases)
    false_clean_count = 0
    false_security_count = 0
    false_incomplete_count = 0
    wrong_stage_count = 0
    wrong_type_count = 0

    root_cause_distribution: Counter[str] = Counter()
    likely_owner_distribution: Counter[str] = Counter()
    
    for case in cases:
        passed = (case.expected_primary == case.actual_primary and case.expected_stage == case.actual_stage)
        if passed:
            continue
            
        is_false_clean = case.actual_primary == FailureType.CLEAN.value and case.expected_primary != FailureType.CLEAN.value
        is_false_security = case.actual_stage == FailureStage.SECURITY.value and case.expected_stage != FailureStage.SECURITY.value
        is_false_incomplete = case.actual_primary == FailureType.INCOMPLETE_DIAGNOSIS.value and case.expected_primary != FailureType.INCOMPLETE_DIAGNOSIS.value
        is_wrong_stage = case.actual_stage != case.expected_stage and not is_false_clean and not is_false_security
        is_wrong_type = case.actual_primary != case.expected_primary and not is_false_clean and not is_false_incomplete
        
        if is_false_clean: false_clean_count += 1
        if is_false_security: false_security_count += 1
        if is_false_incomplete: false_incomplete_count += 1
        if is_wrong_stage: wrong_stage_count += 1
        if is_wrong_type: wrong_type_count += 1

        selected_result = None
        if case.selected_analyzer:
            for r in case.analyzer_results:
                if r.analyzer_name == case.selected_analyzer:
                    selected_result = r
                    break
        if not selected_result:
            for r in case.analyzer_results:
                if r.failure_type and r.failure_type.value == case.actual_primary and r.stage and r.stage.value == case.actual_stage:
                    selected_result = r
                    break

        candidate_details = []
        expected_analyzers = []
        has_expected_missing_metadata = False
        
        for r in case.analyzer_results:
            if r.status in {"fail", "warn"} and r.failure_type is not None:
                metadata_present = bool(r.signal_metadata)
                
                evidence_strength = "unknown"
                method_status = "unknown"
                calibration_status = "unknown"
                
                if metadata_present and r.signal_metadata:
                    sig = r.signal_metadata[0]
                    evidence_strength = getattr(sig, "evidence_strength", "unknown")
                    method_status = getattr(sig, "method_status", "unknown")
                    calibration_status = getattr(sig, "calibration_status", "unknown")
                    
                    if method_status == "unknown" or calibration_status == "unknown" or evidence_strength == "unknown":
                        pass
                
                candidate_details.append({
                    "analyzer_name": r.analyzer_name,
                    "failure_type": r.failure_type.value if r.failure_type else None,
                    "stage": r.stage.value if r.stage else None,
                    "status": r.status,
                    "confidence": r.score if hasattr(r, "score") else None,
                    "evidence_strength": str(evidence_strength),
                    "method_status": str(method_status),
                    "calibration_status": str(calibration_status),
                    "metadata_present": metadata_present,
                })
                
                if r.failure_type and r.failure_type.value == case.expected_primary and r.stage and r.stage.value == case.expected_stage:
                    expected_analyzers.append(r.analyzer_name)
                    if not metadata_present or method_status == "unknown" or calibration_status == "unknown":
                        has_expected_missing_metadata = True
        
        likely_owner = "unknown"
        root_cause = "unknown"
        
        has_retrieval_candidate = any("Retrieval" in c["analyzer_name"] for c in candidate_details)
        has_security_candidate = any("Security" in c["analyzer_name"] or "Injection" in c["analyzer_name"] for c in candidate_details)
        has_claim_candidate = any("Grounding" in c["analyzer_name"] or "Claim" in c["analyzer_name"] for c in candidate_details)
        
        if expected_analyzers and has_expected_missing_metadata:
            root_cause = "weak_signal_metadata"
            likely_owner = expected_analyzers[0]
        elif expected_analyzers and case.selected_analyzer not in expected_analyzers:
            root_cause = "decision_policy_selection"
            likely_owner = expected_analyzers[0]
        elif is_false_clean:
            root_cause = "missing_analyzer_evidence"
            likely_owner = "TBD"
        elif case.expected_stage == "RETRIEVAL" and not has_retrieval_candidate:
            root_cause = "retrieval_evidence_gap"
            likely_owner = "RetrievalAnomalyAnalyzer"
        elif case.expected_stage == "SECURITY" and not has_security_candidate:
            root_cause = "security_evidence_gap"
            likely_owner = "PromptInjectionAnalyzer"
        elif case.expected_stage == "GROUNDING" and not has_claim_candidate:
            root_cause = "claim_extraction_gap"
            likely_owner = "ClaimGroundingAnalyzer"
        elif not case.analyzer_results:
            root_cause = "benchmark_fixture_gap"
            likely_owner = "TBD"
        elif case.mode == "native":
            root_cause = "unknown"
        else:
            root_cause = "external_provider_degraded"
            
        root_cause_distribution[root_cause] += 1
        likely_owner_distribution[likely_owner] += 1

        failures.append({
            "case_id": case.case_id,
            "category": case.category,
            "mode": case.mode,
            "expected_failure_type": case.expected_primary,
            "expected_stage": case.expected_stage,
            "actual_failure_type": case.actual_primary,
            "actual_stage": case.actual_stage,
            "selected_analyzer": case.selected_analyzer or (selected_result.analyzer_name if selected_result else None),
            "candidate_details": candidate_details,
            "false_clean": is_false_clean,
            "false_security": is_false_security,
            "false_incomplete": is_false_incomplete,
            "wrong_stage": is_wrong_stage,
            "wrong_type": is_wrong_type,
            "likely_owner": likely_owner,
            "likely_root_cause": root_cause,
        })
        
    return {
        "audit_title": "Subtle Suite Failure Triage and Ownership Audit",
        "total_cases": total_cases,
        "total_failures": len(failures),
        "false_clean_count": false_clean_count,
        "false_security_count": false_security_count,
        "false_incomplete_count": false_incomplete_count,
        "wrong_stage_count": wrong_stage_count,
        "wrong_type_count": wrong_type_count,
        "root_cause_distribution": dict(root_cause_distribution),
        "likely_owner_distribution": dict(likely_owner_distribution),
        "failures": failures,
    }


def render_triage_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Subtle Suite Failure Triage Audit",
        "",
        f"- Total Cases: {payload['total_cases']}",
        f"- Total Failures: {payload['total_failures']}",
        f"- False CLEAN: {payload['false_clean_count']}",
        f"- False SECURITY: {payload['false_security_count']}",
        f"- False INCOMPLETE: {payload['false_incomplete_count']}",
        f"- Wrong Stage: {payload['wrong_stage_count']}",
        f"- Wrong Type: {payload['wrong_type_count']}",
        "",
        "## Root Cause Distribution",
        *[f"- {k}: {v}" for k, v in payload["root_cause_distribution"].items()],
        "",
        "## Likely Owner Distribution",
        *[f"- {k}: {v}" for k, v in payload["likely_owner_distribution"].items()],
        "",
        "## Failures",
        "",
        "| Case ID | Mode | Expected | Actual | Selected Analyzer | Root Cause | Likely Owner |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for f in payload["failures"]:
        expected = f"{f['expected_failure_type']} ({f['expected_stage']})"
        actual = f"{f['actual_failure_type']} ({f['actual_stage']})"
        lines.append(f"| {f['case_id']} | {f['mode']} | {expected} | {actual} | {f['selected_analyzer']} | {f['likely_root_cause']} | {f['likely_owner']} |")
        
    lines.append("")
    return "\n".join(lines)
