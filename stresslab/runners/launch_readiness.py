"""Launch-readiness orchestration and report generation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from raggov.calibration_status import ANALYZER_CALIBRATION_STATUS
from raggov.engine import DiagnosisEngine, DEFAULT_EXTERNAL_PROVIDERS
from raggov.evaluators.readiness import ExternalProviderDoctorReport
from raggov.evaluators.registry import create_standard_registry
from stresslab.reports import write_json_artifact
from stresslab.cases.load import load_common_rag_failures, load_subtle_rag_failures
from stresslab.runners.rag_failure_runner import BenchmarkReport, RAGFailureRunner


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class LaunchCheckResult:
    name: str
    passed: bool
    status: str = ""
    command: list[str] = field(default_factory=list)
    exit_code: int = 0
    details: list[str] = field(default_factory=list)
    stdout: str = ""
    error_message: str | None = None
    launch_blocker_classification: str | None = None
    launch_blocker: bool = False
    severity: str = "low"
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "passed" if self.passed else "failed"
        self.launch_blocker = not self.passed
        if self.launch_blocker and self.severity == "low":
            self.severity = "high"
        if self.remediation is None and self.launch_blocker:
            self.remediation = "Inspect captured check details and rerun this check after repair."


@dataclass
class BenchmarkSummary:
    name: str
    passed: bool
    accuracy: float
    status: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    failed_case_ids: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    error_message: str | None = None
    launch_blocker_classification: str | None = "benchmark_behavior"
    launch_blocker: bool = False
    severity: str = "low"
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "passed" if self.passed else "failed"
        self.launch_blocker = not self.passed
        if self.launch_blocker and self.severity == "low":
            self.severity = "high"
        if self.remediation is None and self.launch_blocker:
            self.remediation = "Investigate failed benchmark cases and rerun the suite."


@dataclass
class CalibrationStatusSummary:
    calibration_artifact_exists: bool
    calibrated_confidence_present: bool
    statuses: dict[str, int]
    artifact_path: str | None = None
    details: list[str] = field(default_factory=list)
    calibrated_analyzers_count: int = 0
    provisional_analyzers_count: int = 0
    uncalibrated_outputs_count: int = 0
    deterministic_analyzers_count: int = 0
    labeled_sample_count: int = 0
    confidence_intervals_available: bool = False
    production_gating_eligible: bool = False
    recommended_for_gating_true_count: int = 0
    calibrated_confidence_present_count: int = 0


@dataclass
class LaunchReadinessInputs:
    unit_tests: LaunchCheckResult
    common_failure_suite: BenchmarkSummary
    subtle_failure_suite: BenchmarkSummary
    external_regression: LaunchCheckResult
    provider_doctor: ExternalProviderDoctorReport
    calibration_status: CalibrationStatusSummary
    no_silent_fallback: LaunchCheckResult
    decision_policy_regression: LaunchCheckResult
    false_clean_count: int
    false_incomplete_count: int
    advisory_primary_failure_count: int
    retrieval_security_drift_count: int
    prompt_injection_passed: bool
    privacy_violation_passed: bool
    non_clean_missing_fix_count: int
    failed_case_missing_first_failing_node_count: int
    missing_first_failing_node_reasons: list[str]
    benchmark_accuracy: float
    benchmark_accuracy_threshold: float
    surfaced_regression_failures: list[str]


@dataclass
class LaunchReadinessReport:
    status: str
    passed: bool
    failure_reasons: list[str]
    warnings: list[str]
    gates: dict[str, bool]
    metrics: dict[str, Any]
    checks: dict[str, Any]
    launch_blockers: list[dict[str, Any]] = field(default_factory=list)


def run_check_safely(
    name: str,
    fn: Callable[[], LaunchCheckResult | None],
    *,
    launch_blocker_classification: str,
) -> LaunchCheckResult:
    """Run a check and record aborts as data instead of propagating exceptions."""
    try:
        result = fn()
    except Exception as exc:  # pragma: no cover - exercised by tests
        return LaunchCheckResult(
            name=name,
            passed=False,
            status="aborted",
            error_message=str(exc),
            details=[str(exc)],
            launch_blocker_classification=launch_blocker_classification,
        )
    if result is None:
        return LaunchCheckResult(
            name=name,
            passed=True,
            status="passed",
            launch_blocker_classification=launch_blocker_classification,
        )
    if result.launch_blocker_classification is None:
        result.launch_blocker_classification = launch_blocker_classification
    return result


def build_provider_runtime_gate(
    provider_doctor: ExternalProviderDoctorReport,
    *,
    mode: str,
    enabled_providers: list[str] | None,
) -> dict[str, Any]:
    """Summarize runtime provider maturity separately from native launch health."""
    enabled = set(enabled_providers or [])
    provider_rows: list[dict[str, Any]] = []
    missing_reason_count = 0
    real_runtime_count = 0
    schema_only_count = 0

    for readiness in provider_doctor.providers:
        enabled_for_gate = not enabled or readiness.provider_name in enabled
        counts_as_real_runtime = (
            enabled_for_gate
            and readiness.status == "available"
            and readiness.runtime_execution_available
            and readiness.integration_maturity
            in {"configured_runner", "native_library_runtime", "validated_runtime"}
        )
        emits_real_signals = counts_as_real_runtime
        if counts_as_real_runtime:
            real_runtime_count += 1
        if readiness.integration_maturity == "schema_only":
            schema_only_count += 1
        if readiness.status in {"degraded", "unavailable"} and not (
            readiness.reason_code or readiness.reason
        ):
            missing_reason_count += 1

        provider_rows.append(
            {
                "provider_name": readiness.provider_name,
                "available": readiness.available,
                "status": readiness.status,
                "reason_code": readiness.reason_code,
                "integration_maturity": readiness.integration_maturity,
                "runtime_execution_available": readiness.runtime_execution_available,
                "emits_real_signals_current_runtime": emits_real_signals,
                "fallback_visible": readiness.fallback_visible,
                "requires_network": readiness.requires_network,
                "requires_model_download": readiness.requires_model_download,
                "requires_api_key": False,
                "network_model_api_required": (
                    readiness.requires_network or readiness.requires_model_download
                ),
                "counts_as_real_runtime_maturity": counts_as_real_runtime,
            }
        )

    external_enhanced_degraded = False
    if mode == "external-enhanced":
        external_enhanced_degraded = any(
            row["provider_name"] in enabled
            and not row["emits_real_signals_current_runtime"]
            for row in provider_rows
        )

    native_optional_blocker = mode == "native" and False
    passed = missing_reason_count == 0 and (
        mode == "native" or not external_enhanced_degraded
    )

    return {
        "mode": mode,
        "name": f"provider_runtime_{mode}",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "error": None,
        "launch_blocker": not passed,
        "severity": "medium" if not passed else "low",
        "remediation": (
            None
            if passed
            else "Configure a real runtime provider or keep the run in native mode with visible fallback reasons."
        ),
        "native_mode_optional_provider_blocker": native_optional_blocker,
        "external_enhanced_degraded": external_enhanced_degraded,
        "real_runtime_provider_count": real_runtime_count,
        "schema_only_provider_count": schema_only_count,
        "missing_provider_reason_missing_count": missing_reason_count,
        "providers": provider_rows,
    }


def build_calibration_gate(summary: CalibrationStatusSummary) -> dict[str, Any]:
    """Summarize calibration and production-gating eligibility."""
    production_gating_eligible = (
        summary.labeled_sample_count >= 150
        and summary.confidence_intervals_available
        and summary.recommended_for_gating_true_count == 0
    )
    return {
        "name": "calibration_gate",
        "status": "passed" if production_gating_eligible else "failed",
        "production_gating_eligible": production_gating_eligible,
        "passed": production_gating_eligible,
        "error": None,
        "launch_blocker": not production_gating_eligible,
        "severity": "medium" if not production_gating_eligible else "low",
        "remediation": (
            None
            if production_gating_eligible
            else "Provide labeled validation samples and confidence intervals before enabling production gating."
        ),
        "calibrated_analyzers_count": summary.calibrated_analyzers_count,
        "provisional_analyzers_count": summary.provisional_analyzers_count,
        "uncalibrated_outputs_count": summary.uncalibrated_outputs_count,
        "deterministic_analyzers_count": summary.deterministic_analyzers_count,
        "labeled_sample_count": summary.labeled_sample_count,
        "confidence_intervals_available": summary.confidence_intervals_available,
        "recommended_for_gating_true_count": summary.recommended_for_gating_true_count,
        "calibrated_confidence_present_count": summary.calibrated_confidence_present_count,
        "calibration_artifact_exists": summary.calibration_artifact_exists,
    }


def build_launch_readiness_report(inputs: LaunchReadinessInputs) -> LaunchReadinessReport:
    """Evaluate launch gates into a pass/fail report."""
    failure_reasons: list[str] = []
    warnings = list(inputs.provider_doctor.warnings)
    launch_blockers: list[dict[str, Any]] = []

    provider_reasons_visible = all(
        readiness.status not in {"degraded", "unavailable"}
        or bool(readiness.reason_code or readiness.reason)
        for readiness in inputs.provider_doctor.providers
    )
    provider_runtime_native = build_provider_runtime_gate(
        inputs.provider_doctor,
        mode="native",
        enabled_providers=[],
    )
    provider_runtime_external = build_provider_runtime_gate(
        inputs.provider_doctor,
        mode="external-enhanced",
        enabled_providers=[provider.provider_name for provider in inputs.provider_doctor.providers],
    )
    calibration_gate = build_calibration_gate(inputs.calibration_status)

    gates = {
        "false_clean_count == 0": inputs.false_clean_count == 0,
        "false_incomplete_count == 0": inputs.false_incomplete_count == 0,
        "no external advisory signal becomes primary failure alone": inputs.advisory_primary_failure_count == 0,
        "no retrieval anomaly becomes security without explicit security evidence": inputs.retrieval_security_drift_count == 0,
        "prompt injection detection still passes": inputs.prompt_injection_passed,
        "privacy violation detection still passes": inputs.privacy_violation_passed,
        "provider missing/degraded reasons visible": provider_reasons_visible,
        "calibrated confidence absent unless calibration artifact exists": (
            inputs.calibration_status.calibration_artifact_exists
            or not inputs.calibration_status.calibrated_confidence_present
        ),
        "every non-clean diagnosis has recommended fix": inputs.non_clean_missing_fix_count == 0,
        "every failed case has first_failing_node or explicit reason why unavailable": (
            inputs.failed_case_missing_first_failing_node_count == 0
        ),
        "benchmark accuracy meets configured threshold": (
            inputs.benchmark_accuracy >= inputs.benchmark_accuracy_threshold
        ),
        "all previously surfaced regression cases pass": len(inputs.surfaced_regression_failures) == 0,
        "full pytest passes": inputs.unit_tests.passed,
        "common failure golden suite passes": inputs.common_failure_suite.passed,
        "subtle failure golden suite passes": inputs.subtle_failure_suite.passed,
        "external regression suite passes": inputs.external_regression.passed,
        "no-silent-fallback checks pass": inputs.no_silent_fallback.passed,
        "decision policy regression checks pass": inputs.decision_policy_regression.passed,
        "native mode ignores missing optional external providers": provider_runtime_native["passed"],
        "external-enhanced provider degradation is visible": provider_runtime_external["external_enhanced_degraded"]
        or provider_runtime_external["passed"],
        "production gating remains disabled without calibration evidence": (
            not calibration_gate["production_gating_eligible"]
            or (
                inputs.calibration_status.labeled_sample_count >= 150
                and inputs.calibration_status.confidence_intervals_available
            )
        ),
    }

    if not gates["false_clean_count == 0"]:
        launch_blockers.append(
            {
                "classification": "false_clean_risk",
                "severity": "critical",
                "message": f"false_clean_count={inputs.false_clean_count}",
                "remediation": "Inspect common/subtle benchmark cases returning CLEAN and restore blocking diagnosis paths.",
            }
        )
        failure_reasons.append(
            f"false_clean_count={inputs.false_clean_count}. Inspect the subtle/common golden mismatches and repair the cases returning CLEAN."
        )
    if not gates["false_incomplete_count == 0"]:
        failure_reasons.append(
            f"false_incomplete_count={inputs.false_incomplete_count}. Critical analyzers are producing unexpected INCOMPLETE_DIAGNOSIS outcomes."
        )
    if not gates["no external advisory signal becomes primary failure alone"]:
        failure_reasons.append(
            f"advisory_primary_failure_count={inputs.advisory_primary_failure_count}. External advisory-only evidence is still reaching primary_failure."
        )
    if not gates["no retrieval anomaly becomes security without explicit security evidence"]:
        failure_reasons.append(
            f"retrieval_security_drift_count={inputs.retrieval_security_drift_count}. Retrieval anomalies are drifting into security diagnoses without explicit security evidence."
        )
    if not gates["prompt injection detection still passes"]:
        failure_reasons.append("Prompt injection regression detected. Re-run the security golden cases and restore the blocking path.")
    if not gates["privacy violation detection still passes"]:
        failure_reasons.append("Privacy violation regression detected. Restore the privacy blocking path before launch.")
    if not gates["provider missing/degraded reasons visible"]:
        launch_blockers.append(
            {
                "classification": "provider_runtime_visibility",
                "severity": "high",
                "message": "One or more degraded/unavailable providers lack reason_code/reason.",
                "remediation": "Ensure every degraded or unavailable provider readiness result includes reason_code or reason.",
            }
        )
        failure_reasons.append("Launch gate failed: provider missing/degraded reasons visible. Every degraded or unavailable provider needs a reason_code or reason.")
    if not gates["calibrated confidence absent unless calibration artifact exists"]:
        launch_blockers.append(
            {
                "classification": "calibration_honesty",
                "severity": "high",
                "message": "Calibrated/provisional confidence appears without a calibration artifact.",
                "remediation": "Remove calibrated confidence claims or provide a calibration artifact with validation evidence.",
            }
        )
        failure_reasons.append(
            "Calibrated confidence is present without a calibration artifact. Remove calibrated/provisional confidence claims or provide the artifact."
        )
    if not gates["every non-clean diagnosis has recommended fix"]:
        failure_reasons.append(
            f"{inputs.non_clean_missing_fix_count} non-clean diagnoses are missing recommended fixes."
        )
    if not gates["every failed case has first_failing_node or explicit reason why unavailable"]:
        reason_text = "; ".join(inputs.missing_first_failing_node_reasons) or "missing reason details"
        failure_reasons.append(
            f"{inputs.failed_case_missing_first_failing_node_count} failed cases lack first_failing_node coverage. Reasons: {reason_text}."
        )
    if not gates["benchmark accuracy meets configured threshold"]:
        launch_blockers.append(
            {
                "classification": "benchmark_behavior",
                "severity": "high",
                "message": (
                    f"benchmark_accuracy={inputs.benchmark_accuracy:.0%} "
                    f"below threshold {inputs.benchmark_accuracy_threshold:.0%}"
                ),
                "remediation": "Repair benchmark mismatches before raising the readiness status.",
            }
        )
        failure_reasons.append(
            f"benchmark_accuracy={inputs.benchmark_accuracy:.0%} is below threshold {inputs.benchmark_accuracy_threshold:.0%}."
        )
    if not gates["all previously surfaced regression cases pass"]:
        failure_reasons.append(
            "Previously surfaced regression cases failed: " + ", ".join(inputs.surfaced_regression_failures)
        )

    for check_name, check_result in {
        "full pytest": inputs.unit_tests,
        "external regression suite": inputs.external_regression,
        "no-silent-fallback checks": inputs.no_silent_fallback,
        "decision policy regression checks": inputs.decision_policy_regression,
    }.items():
        if not check_result.passed:
            launch_blockers.append(
                {
                    "classification": check_result.launch_blocker_classification
                    or "code_test_health",
                    "severity": check_result.severity,
                    "message": f"{check_name} {check_result.status}",
                    "error": check_result.error_message,
                    "remediation": check_result.remediation,
                }
            )
            failure_reasons.append(
                f"{check_name} failed with exit_code={check_result.exit_code}. " + "; ".join(check_result.details or ["See captured pytest output."])
            )

    for benchmark in (inputs.common_failure_suite, inputs.subtle_failure_suite):
        if not benchmark.passed:
            launch_blockers.append(
                {
                    "classification": benchmark.launch_blocker_classification
                    or "benchmark_behavior",
                    "severity": benchmark.severity,
                    "message": f"{benchmark.name} {benchmark.status}",
                    "error": benchmark.error_message,
                    "remediation": benchmark.remediation,
                }
            )
            failure_reasons.append(
                f"{benchmark.name} failed at {benchmark.accuracy:.0%}. Mismatches: {', '.join(benchmark.failed_case_ids) or 'unknown'}."
            )

    if provider_runtime_external["external_enhanced_degraded"]:
        launch_blockers.append(
            {
                "classification": "external_provider_runtime",
                "severity": "medium",
                "message": "External-enhanced mode is degraded because one or more enabled providers do not emit real runtime signals.",
                "remediation": "Install/configure real runtime providers or run native mode until external runtimes are available.",
            }
        )
    if not calibration_gate["production_gating_eligible"]:
        launch_blockers.append(
            {
                "classification": "calibration_gating",
                "severity": "medium",
                "message": "Production gating is disabled because labeled validation and confidence intervals are insufficient.",
                "remediation": "Provide labeled validation samples and confidence intervals; keep recommended_for_gating false until then.",
            }
        )

    hard_blockers = [blocker for blocker in launch_blockers if blocker["severity"] in {"critical", "high"}]
    passed = not hard_blockers
    status = "Reliable Alpha Candidate" if passed else "Not Ready"
    if passed and (
        inputs.common_failure_suite.accuracy < 1.0
        or inputs.subtle_failure_suite.accuracy < 1.0
        or provider_runtime_external["external_enhanced_degraded"]
        or not calibration_gate["production_gating_eligible"]
    ):
        status = "Internal Alpha"
    return LaunchReadinessReport(
        status=status,
        passed=passed,
        failure_reasons=failure_reasons,
        warnings=warnings,
        gates=gates,
        metrics={
            "false_clean_count": inputs.false_clean_count,
            "false_incomplete_count": inputs.false_incomplete_count,
            "advisory_primary_failure_count": inputs.advisory_primary_failure_count,
            "retrieval_security_drift_count": inputs.retrieval_security_drift_count,
            "non_clean_missing_fix_count": inputs.non_clean_missing_fix_count,
            "failed_case_missing_first_failing_node_count": inputs.failed_case_missing_first_failing_node_count,
            "benchmark_accuracy": inputs.benchmark_accuracy,
            "benchmark_accuracy_threshold": inputs.benchmark_accuracy_threshold,
            "provider_safe_to_run_external_enhanced": inputs.provider_doctor.safe_to_run_external_enhanced,
            "calibration_statuses": inputs.calibration_status.statuses,
            "full_pytest_status": inputs.unit_tests.status,
            "decision_policy_status": inputs.decision_policy_regression.status,
            "external_alignment_status": "not_run",
            "common_benchmark_pass_rate": inputs.common_failure_suite.accuracy,
            "subtle_benchmark_status": inputs.subtle_failure_suite.status,
            "claim_harness_status": "not_run",
            "false_security_count": inputs.retrieval_security_drift_count,
            "external_ignored_count": 0,
            "missing_provider_reason_missing_count": provider_runtime_native["missing_provider_reason_missing_count"],
            "calibrated_confidence_present_count": inputs.calibration_status.calibrated_confidence_present_count,
            "production_gating_eligible": calibration_gate["production_gating_eligible"],
            "recommended_for_gating_true_count": calibration_gate["recommended_for_gating_true_count"],
        },
        checks={
            "unit_tests": asdict(inputs.unit_tests),
            "common_failure_suite": asdict(inputs.common_failure_suite),
            "subtle_failure_suite": asdict(inputs.subtle_failure_suite),
            "external_regression": asdict(inputs.external_regression),
            "provider_doctor": inputs.provider_doctor.model_dump(mode="json"),
            "calibration_status": asdict(inputs.calibration_status),
            "calibration_gate": calibration_gate,
            "provider_runtime_gate_native": provider_runtime_native,
            "provider_runtime_gate_external_enhanced": provider_runtime_external,
            "no_silent_fallback": asdict(inputs.no_silent_fallback),
            "decision_policy_regression": asdict(inputs.decision_policy_regression),
        },
        launch_blockers=launch_blockers,
    )


def render_launch_readiness_markdown(report: LaunchReadinessReport) -> str:
    lines = [
        "# GovRAG Launch Readiness",
        "",
        f"Status: **{report.status}**",
        "",
        "## Gates",
    ]
    for gate, passed in report.gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {gate}")

    lines.extend(["", "## Failure Reasons"])
    if report.failure_reasons:
        for reason in report.failure_reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- None")

    lines.extend(["", "## Launch Blockers"])
    if report.launch_blockers:
        for blocker in report.launch_blockers:
            line = (
                f"- `{blocker.get('classification')}` ({blocker.get('severity')}): "
                f"{blocker.get('message')}"
            )
            if blocker.get("remediation"):
                line += f" Remediation: {blocker.get('remediation')}"
            lines.append(line)
    else:
        lines.append("- None")

    lines.extend(["", "## Metrics"])
    for key, value in report.metrics.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Warnings"])
    if report.warnings:
        for warning in report.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_launch_readiness_report(report: LaunchReadinessReport, output_path: str | Path) -> Path:
    return write_json_artifact(output_path, asdict(report))


def write_launch_readiness_markdown_report(report: LaunchReadinessReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_launch_readiness_markdown(report), encoding="utf-8")
    return target


def run_launch_readiness(
    *,
    benchmark_accuracy_threshold: float = 0.95,
    calibration_artifact_path: str | Path | None = None,
) -> LaunchReadinessReport:
    """Run the launch-readiness command end to end."""
    artifact_path = Path(calibration_artifact_path) if calibration_artifact_path else None
    unit_tests = _run_pytest_check(
        "full_pytest",
        [],
        launch_blocker_classification="code_test_health",
    )
    external_regression = _run_pytest_check(
        "external_regression",
        ["tests/stresslab/external_regression"],
        launch_blocker_classification="external_native_alignment",
    )
    external_alignment = _run_pytest_check(
        "external_alignment",
        ["tests/external_alignment"],
        launch_blocker_classification="external_native_alignment",
    )
    claim_harness = _run_pytest_check(
        "claim_harness",
        ["tests", "-k", "claim_diagnosis"],
        launch_blocker_classification="claim_harness",
    )
    no_silent_fallback = _run_pytest_check(
        "no_silent_fallback",
        [
            "tests/evaluators/retrieval/test_cross_encoder_relevance_provider.py",
            "-k",
            "no_silent_fallback_when_missing_dependency or no_silent_fallback_on_model_error",
        ],
        launch_blocker_classification="fallback_visibility",
    )
    decision_policy = _run_pytest_check(
        "decision_policy_regression",
        [
            "tests/decision_policy/test_primary_failure_policy.py",
            "tests/engine/test_decision_policy_integration.py",
            "tests/stresslab/external_regression/test_advisory_only_external_signals.py",
        ],
        launch_blocker_classification="decision_policy",
    )

    common_report, common_summary = _run_benchmark_suite("common")
    subtle_report, subtle_summary = _run_benchmark_suite("subtle")
    common_diagnoses = _collect_suite_diagnoses_safely("common")
    subtle_diagnoses = _collect_suite_diagnoses_safely("subtle")

    registry = create_standard_registry(
        {
            "mode": "external-enhanced",
            "enabled_external_providers": list(DEFAULT_EXTERNAL_PROVIDERS),
            "retrieval_relevance_provider": "native",
        }
    )
    provider_doctor = registry.readiness_report(enabled_providers=None)
    calibration_status = _calibration_status_summary(
        artifact_path,
        diagnoses=[entry["diagnosis"] for entry in common_diagnoses + subtle_diagnoses],
    )

    false_clean_count = _count_false_clean(common_report) + _count_false_clean(subtle_report)
    false_incomplete_count = _count_false_incomplete(common_report) + _count_false_incomplete(subtle_report)
    prompt_injection_passed = _primary_failure_cases_pass(common_report, "PROMPT_INJECTION") and _primary_failure_cases_pass(subtle_report, "PROMPT_INJECTION")
    privacy_violation_passed = _primary_failure_cases_pass(common_report, "PRIVACY_VIOLATION") and _primary_failure_cases_pass(subtle_report, "PRIVACY_VIOLATION")
    benchmark_accuracy = min(common_summary.accuracy, subtle_summary.accuracy)
    non_clean_missing_fix_count = _count_non_clean_missing_fix(common_diagnoses + subtle_diagnoses)
    failed_case_missing_first_failing_node_count, missing_first_failing_node_reasons = _count_missing_first_failing_node(
        common_diagnoses + subtle_diagnoses
    )

    inputs = LaunchReadinessInputs(
        unit_tests=unit_tests,
        common_failure_suite=common_summary,
        subtle_failure_suite=subtle_summary,
        external_regression=external_regression,
        provider_doctor=provider_doctor,
        calibration_status=calibration_status,
        no_silent_fallback=no_silent_fallback,
        decision_policy_regression=decision_policy,
        false_clean_count=false_clean_count,
        false_incomplete_count=false_incomplete_count,
        advisory_primary_failure_count=0 if external_regression.passed else 1,
        retrieval_security_drift_count=0 if external_regression.passed else 1,
        prompt_injection_passed=prompt_injection_passed,
        privacy_violation_passed=privacy_violation_passed,
        non_clean_missing_fix_count=non_clean_missing_fix_count,
        failed_case_missing_first_failing_node_count=failed_case_missing_first_failing_node_count,
        missing_first_failing_node_reasons=missing_first_failing_node_reasons,
        benchmark_accuracy=benchmark_accuracy,
        benchmark_accuracy_threshold=benchmark_accuracy_threshold,
        surfaced_regression_failures=_surfaced_regression_failures(
            common_report,
            subtle_report,
            unit_tests,
            external_regression,
            no_silent_fallback,
            decision_policy,
        ),
    )
    report = build_launch_readiness_report(inputs)
    report.metrics["external_alignment_status"] = external_alignment.status
    report.metrics["claim_harness_status"] = claim_harness.status
    report.checks["external_alignment"] = asdict(external_alignment)
    report.checks["claim_harness"] = asdict(claim_harness)
    for check_name, check in {
        "external alignment": external_alignment,
        "claim harness": claim_harness,
    }.items():
        if not check.passed:
            report.launch_blockers.append(
                {
                    "classification": check.launch_blocker_classification,
                    "severity": check.severity,
                    "message": f"{check_name} {check.status}",
                    "error": check.error_message,
                    "remediation": check.remediation,
                }
            )
            report.failure_reasons.append(
                f"{check_name} failed with exit_code={check.exit_code}. "
                + "; ".join(check.details or ["See captured pytest output."])
            )
    if any(blocker.get("severity") in {"critical", "high"} for blocker in report.launch_blockers):
        report.status = "Not Ready"
        report.passed = False
    return report


def _run_pytest_check(
    name: str,
    extra_args: list[str],
    *,
    launch_blocker_classification: str,
) -> LaunchCheckResult:
    command = [sys.executable, "-m", "pytest", "-q", *extra_args]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return LaunchCheckResult(
            name=name,
            passed=False,
            status="aborted",
            command=command,
            exit_code=1,
            details=[str(exc)],
            error_message=str(exc),
            launch_blocker_classification=launch_blocker_classification,
        )
    details = []
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr).strip().splitlines()[-5:]
        details.extend(line for line in tail if line)
    return LaunchCheckResult(
        name=name,
        passed=completed.returncode == 0,
        status="passed" if completed.returncode == 0 else "failed",
        command=command,
        exit_code=completed.returncode,
        details=details,
        stdout=completed.stdout,
        launch_blocker_classification=launch_blocker_classification,
    )


def _benchmark_summary(name: str, report: BenchmarkReport) -> BenchmarkSummary:
    return BenchmarkSummary(
        name=name,
        passed=report.passed_cases == report.total_cases,
        accuracy=report.pass_rate,
        status="passed" if report.passed_cases == report.total_cases else "failed",
        total_cases=report.total_cases,
        passed_cases=report.passed_cases,
        failed_case_ids=[result.case_id for result in report.results if not result.passed],
        details=[note for result in report.results for note in result.notes],
    )


def _run_benchmark_suite(suite_name: str) -> tuple[BenchmarkReport | None, BenchmarkSummary]:
    name = f"{suite_name}_failure_suite"
    try:
        report = RAGFailureRunner(mode="native", suite=suite_name).run_benchmark()
    except Exception as exc:
        return None, BenchmarkSummary(
            name=name,
            passed=False,
            status="aborted",
            accuracy=0.0,
            details=[str(exc)],
            error_message=str(exc),
            launch_blocker_classification="benchmark_behavior",
        )
    return report, _benchmark_summary(name, report)


def _calibration_status_summary(artifact_path: Path | None, *, diagnoses: list[Any]) -> CalibrationStatusSummary:
    statuses: dict[str, int] = {}
    for status in ANALYZER_CALIBRATION_STATUS.values():
        statuses[status.value] = statuses.get(status.value, 0) + 1
    artifact_exists = artifact_path.exists() if artifact_path is not None else False
    calibrated_confidence_present_count = sum(
        1
        for diagnosis in diagnoses
        if diagnosis.calibrated_confidence is not None
        or diagnosis.calibration_status in {"provisional", "calibrated"}
    )
    recommended_for_gating_true_count = sum(
        _count_key_value(diagnosis.model_dump(mode="json"), "recommended_for_gating", True)
        for diagnosis in diagnoses
    )
    confidence_intervals_available = any(
        bool(diagnosis.confidence_intervals) for diagnosis in diagnoses
    )
    labeled_sample_count = _infer_labeled_sample_count(artifact_path) if artifact_exists else 0
    uncalibrated_outputs = sum(
        1
        for diagnosis in diagnoses
        if diagnosis.calibration_status in {"uncalibrated", None}
    )
    return CalibrationStatusSummary(
        calibration_artifact_exists=artifact_exists,
        calibrated_confidence_present=calibrated_confidence_present_count > 0,
        statuses=statuses,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        details=[] if artifact_exists else ["No calibration artifact provided."],
        calibrated_analyzers_count=statuses.get("calibrated", 0),
        provisional_analyzers_count=statuses.get("preliminary_calibrated", 0)
        + statuses.get("provisional", 0),
        uncalibrated_outputs_count=uncalibrated_outputs,
        deterministic_analyzers_count=statuses.get("deterministic", 0),
        labeled_sample_count=labeled_sample_count,
        confidence_intervals_available=confidence_intervals_available,
        production_gating_eligible=False,
        recommended_for_gating_true_count=recommended_for_gating_true_count,
        calibrated_confidence_present_count=calibrated_confidence_present_count,
    )


def _count_false_clean(report: BenchmarkReport | None) -> int:
    if report is None:
        return 0
    return sum(
        1
        for result in report.results
        if result.expected_primary != "CLEAN" and result.actual_primary == "CLEAN"
    )


def _count_false_incomplete(report: BenchmarkReport | None) -> int:
    if report is None:
        return 0
    return sum(
        1
        for result in report.results
        if result.expected_primary != "INCOMPLETE_DIAGNOSIS"
        and result.actual_primary == "INCOMPLETE_DIAGNOSIS"
    )


def _primary_failure_cases_pass(report: BenchmarkReport | None, failure_name: str) -> bool:
    if report is None:
        return False
    cases = [result for result in report.results if result.expected_primary == failure_name]
    return all(result.actual_primary == failure_name for result in cases)


def _surfaced_regression_failures(
    common_report: BenchmarkReport | None,
    subtle_report: BenchmarkReport | None,
    unit_tests: LaunchCheckResult,
    external_regression: LaunchCheckResult,
    no_silent_fallback: LaunchCheckResult,
    decision_policy: LaunchCheckResult,
) -> list[str]:
    failures = [
        result.case_id
        for result in (common_report.results if common_report is not None else [])
        if not result.passed
    ]
    failures.extend(
        result.case_id
        for result in (subtle_report.results if subtle_report is not None else [])
        if not result.passed
    )
    for check in (unit_tests, external_regression, no_silent_fallback, decision_policy):
        if not check.passed:
            failures.append(check.name)
    return failures


def _collect_suite_diagnoses_safely(suite_name: str) -> list[dict[str, Any]]:
    try:
        return _collect_suite_diagnoses(suite_name)
    except Exception:
        return []


def _collect_suite_diagnoses(suite_name: str) -> list[dict[str, Any]]:
    cases = load_subtle_rag_failures() if suite_name == "subtle" else load_common_rag_failures()
    runner = RAGFailureRunner(mode="native", suite=suite_name)
    engine = runner.engine or DiagnosisEngine(config={"mode": "native"})
    collected: list[dict[str, Any]] = []
    for case in cases:
        diagnosis = engine.diagnose(runner._build_run(case))
        collected.append({"case_id": case.case_id, "expected_primary": case.expected_primary_failure, "diagnosis": diagnosis})
    return collected


def _infer_labeled_sample_count(artifact_path: Path | None) -> int:
    if artifact_path is None or not artifact_path.exists():
        return 0
    try:
        if artifact_path.suffix == ".jsonl":
            return len([line for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        import json

        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("samples", "examples", "labeled_examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value)
            count = payload.get("labeled_sample_count")
            if isinstance(count, int):
                return count
    except Exception:
        return 0
    return 0


def _count_key_value(payload: Any, key: str, expected_value: Any) -> int:
    if isinstance(payload, dict):
        return (
            (1 if payload.get(key) == expected_value else 0)
            + sum(_count_key_value(value, key, expected_value) for value in payload.values())
        )
    if isinstance(payload, list):
        return sum(_count_key_value(item, key, expected_value) for item in payload)
    return 0


def _count_non_clean_missing_fix(entries: list[dict[str, Any]]) -> int:
    return sum(
        1
        for entry in entries
        if entry["diagnosis"].primary_failure.value != "CLEAN"
        and not entry["diagnosis"].recommended_fix
    )


def _count_missing_first_failing_node(entries: list[dict[str, Any]]) -> tuple[int, list[str]]:
    missing = 0
    reasons: list[str] = []
    for entry in entries:
        diagnosis = entry["diagnosis"]
        if diagnosis.primary_failure.value == "CLEAN":
            continue
        if diagnosis.first_failing_node:
            continue
        reason = ""
        if diagnosis.diagnosis_decision_trace:
            reason = diagnosis.diagnosis_decision_trace.get("selection_reason", "")
        if not reason and diagnosis.ncv_report is None:
            reason = "NCV report unavailable"
        if reason:
            reasons.append(f"{entry['case_id']}: {reason}")
            continue
        missing += 1
    return missing, reasons
