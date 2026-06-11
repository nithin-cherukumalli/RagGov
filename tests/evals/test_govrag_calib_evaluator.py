from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for path in [PROJECT_ROOT / "src", PROJECT_ROOT / "evals", PROJECT_ROOT / "scripts"]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import evaluate_govrag_calib
from raggov.engine import DiagnosisEngine
from raggov.models.diagnosis import FailureType


DATASET = PROJECT_ROOT / "evals" / "govrag_calib" / "calib_150_seed.jsonl"


class _FakeAnalyzerResult:
    def __init__(self, evidence):
        self.evidence = evidence


class _FakeDiagnosis:
    def __init__(self, evidence=None, analyzer_evidence=None):
        self.evidence = evidence or []
        self.analyzer_results = [
            _FakeAnalyzerResult(values)
            for values in (analyzer_evidence or [])
        ]


def test_evaluator_runs_on_seed_dataset_limit_two():
    report = evaluate_govrag_calib.evaluate_calib_dataset(DATASET, limit=2)

    assert report["dataset_summary"]["total_cases"] == 2
    assert "primary_failure_accuracy" in report["prediction_metrics"]
    assert "false_clean_count" in report["safety_metrics"]
    assert report["calibration_status"] == {
        "calibration_status": "not_calibrated",
        "production_gating_eligible": False,
        "confidence_intervals_available": False,
        "heldout_split_locked": False,
    }


def test_evaluator_reports_expected_metric_sections():
    report = evaluate_govrag_calib.evaluate_calib_dataset(DATASET, limit=3)

    assert "per_family_metrics" in report
    assert "confusion_matrices" in report
    assert "decision_policy_metrics" in report
    assert "expected_primary_failure_vs_actual_primary_failure" in report["confusion_matrices"]
    assert "expected_stage_vs_actual_stage" in report["confusion_matrices"]
    assert "acceptable_alternative_match_count" in report["decision_policy_metrics"]


def test_evaluator_missing_optional_node_label_is_unavailable(tmp_path):
    report = evaluate_govrag_calib.evaluate_calib_dataset(DATASET, limit=1)

    assert report["prediction_metrics"]["first_failing_node_available_count"] == 0
    assert report["prediction_metrics"]["first_failing_node_accuracy"] == "unavailable"


def test_evaluator_writes_json_and_markdown_reports(tmp_path):
    report = evaluate_govrag_calib.evaluate_calib_dataset(DATASET, limit=1)
    json_path = tmp_path / "seed_eval_native.json"
    md_path = json_path.with_suffix(".md")

    evaluate_govrag_calib.write_eval_json_report(report, json_path)
    evaluate_govrag_calib.write_eval_markdown_report(report, md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["calibration_status"]["production_gating_eligible"] is False
    assert "not_calibrated" in md_path.read_text(encoding="utf-8")


def test_evaluator_supports_external_enhanced_mode_without_gating():
    report = evaluate_govrag_calib.evaluate_calib_dataset(
        DATASET,
        mode="external-enhanced",
        limit=1,
    )

    assert report["mode"] == "external-enhanced"
    assert report["calibration_status"]["production_gating_eligible"] is False


def test_evaluator_emits_evidence_diagnostics_for_missing_ids():
    record = {
        "expected_primary_failure": "UNSUPPORTED_CLAIM",
        "failure_family": "grounding",
        "claims": [{"claim_id": "claim-1", "text": "A claim."}],
        "citations": [],
        "retrieved_chunks": [{"chunk_id": "chunk-1", "doc_id": "doc-1", "text": "Context."}],
        "expected_affected_claim_ids": ["missing-claim"],
        "expected_affected_doc_ids": ["missing-doc"],
        "expected_first_failing_node": None,
        "expected_human_review_required": True,
    }
    diagnosis = _FakeDiagnosis(analyzer_evidence=[["no claims extracted from final answer"]])

    diagnostics = evaluate_govrag_calib._record_evidence_diagnostics(
        record,
        diagnosis,
        expected_candidate_generated=False,
    )

    assert diagnostics["expected_claim_count"] == 1
    assert diagnostics["extracted_claim_count"] == 0
    assert diagnostics["skipped_claim_count"] == 1
    assert diagnostics["missing_expected_claim_ids"] == ["missing-claim"]
    assert diagnostics["missing_expected_doc_ids"] == ["missing-doc"]
    assert "missing_claim_ids" in diagnostics["evidence_gap_flags"]
    assert "missing_doc_ids" in diagnostics["evidence_gap_flags"]
    assert "diagnosis_extracted_no_claims" in diagnostics["reason_not_scored"]


def test_evaluator_emits_citation_evidence_gap_diagnostics():
    record = {
        "expected_primary_failure": "CITATION_MISMATCH",
        "failure_family": "citation",
        "claims": [{"claim_id": "claim-1", "text": "A claim."}],
        "citations": [],
        "retrieved_chunks": [{"chunk_id": "chunk-1", "doc_id": "doc-1", "text": "Context."}],
        "expected_affected_claim_ids": ["claim-1"],
        "expected_affected_doc_ids": ["doc-1"],
        "expected_first_failing_node": None,
        "expected_human_review_required": True,
    }
    diagnosis = _FakeDiagnosis(
        analyzer_evidence=[["no claim evidence available for citation faithfulness analysis"]]
    )

    diagnostics = evaluate_govrag_calib._record_evidence_diagnostics(
        record,
        diagnosis,
        expected_candidate_generated=False,
    )

    assert diagnostics["citation_count"] == 0
    assert diagnostics["diagnosis_has_citation_evidence"] is False
    assert "no_citations" in diagnostics["evidence_gap_flags"]
    assert "citation_family_without_citations" in diagnostics["reason_not_scored"]


def test_evaluator_diagnostics_do_not_change_optional_metric_scoring():
    report = evaluate_govrag_calib.evaluate_calib_dataset(DATASET, limit=1)

    assert report["prediction_metrics"]["first_failing_node_accuracy"] == "unavailable"
    assert "evidence_diagnostics" in report["per_case"][0]
    assert "unsupported_optional_metric" in report["per_case"][0]["evidence_diagnostics"]["evidence_gap_flags"]
    assert report["calibration_status"]["production_gating_eligible"] is False


def test_security_relevant_clean_counts_as_dangerous_clean_miss():
    record = {
        "expected_primary_failure": "INSUFFICIENT_CONTEXT",
        "expected_stage": "SUFFICIENCY",
        "expected_human_review_required": True,
        "security_relevant": True,
        "adversarial": False,
    }

    classification = evaluate_govrag_calib._safety_classification(
        record,
        actual_primary="CLEAN",
        actual_stage="UNKNOWN",
        human_review_required=False,
    )

    assert classification["false_clean"] is True
    assert classification["dangerous_clean_miss"] is True
    assert classification["safety_outcome"] == "dangerous_clean_miss"


def test_security_relevant_nonclean_review_wrong_stage_is_acceptable_not_dangerous_clean():
    record = {
        "expected_primary_failure": "INSUFFICIENT_CONTEXT",
        "expected_stage": "SUFFICIENCY",
        "expected_human_review_required": True,
        "security_relevant": True,
        "adversarial": False,
    }

    classification = evaluate_govrag_calib._safety_classification(
        record,
        actual_primary="UNSUPPORTED_CLAIM",
        actual_stage="GROUNDING",
        human_review_required=True,
    )

    assert classification["dangerous_clean_miss"] is False
    assert classification["security_stage_miss"] is False
    assert classification["acceptable_nonclean_human_review"] is True
    assert classification["safety_outcome"] == "acceptable_nonclean_human_review"


def test_expected_security_wrong_stage_remains_security_stage_miss():
    record = {
        "expected_primary_failure": "PRIVACY_VIOLATION",
        "expected_stage": "SECURITY",
        "expected_human_review_required": True,
        "security_relevant": True,
        "adversarial": False,
    }

    classification = evaluate_govrag_calib._safety_classification(
        record,
        actual_primary="CITATION_MISMATCH",
        actual_stage="GROUNDING",
        human_review_required=True,
    )

    assert classification["dangerous_clean_miss"] is False
    assert classification["security_stage_miss"] is True
    assert classification["acceptable_nonclean_human_review"] is False
    assert classification["safety_outcome"] == "security_stage_miss"


def test_human_review_miss_remains_separate_for_non_security_case():
    record = {
        "expected_primary_failure": "STALE_RETRIEVAL",
        "expected_stage": "RETRIEVAL",
        "expected_human_review_required": True,
        "security_relevant": False,
        "adversarial": False,
    }

    classification = evaluate_govrag_calib._safety_classification(
        record,
        actual_primary="STALE_RETRIEVAL",
        actual_stage="RETRIEVAL",
        human_review_required=False,
    )

    assert classification["human_review_miss"] is True
    assert classification["dangerous_clean_miss"] is False
    assert classification["safety_outcome"] == "human_review_miss"


def _load_record(case_id: str) -> dict:
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["case_id"] == case_id:
            return record
    raise AssertionError(f"missing case {case_id}")


def test_calib_false_clean_safety_cases_are_non_clean_after_repair():
    engine = DiagnosisEngine(config={"mode": "native"})
    repaired_case_ids = [
        "govrag-calib-seed-023",
        "govrag-calib-seed-030",
        "govrag-calib-seed-033",
        "govrag-calib-seed-034",
        "govrag-calib-seed-036",
        "govrag-calib-seed-037",
        "govrag-calib-seed-049",
    ]

    for case_id in repaired_case_ids:
        diagnosis = engine.diagnose(evaluate_govrag_calib._record_to_run(_load_record(case_id)))
        assert diagnosis.primary_failure != FailureType.CLEAN, case_id
        assert diagnosis.human_review_required(), case_id


def test_calib_privacy_dangerous_miss_is_security_failure_after_repair():
    engine = DiagnosisEngine(config={"mode": "native"})
    diagnosis = engine.diagnose(
        evaluate_govrag_calib._record_to_run(_load_record("govrag-calib-seed-046"))
    )

    assert diagnosis.primary_failure == FailureType.PRIVACY_VIOLATION
    assert diagnosis.human_review_required()
    assert diagnosis.calibration_status == "uncalibrated"
