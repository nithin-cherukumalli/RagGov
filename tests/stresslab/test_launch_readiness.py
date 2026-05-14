from __future__ import annotations

from raggov.evaluators.readiness import ExternalProviderDoctorReport, ProviderReadiness
from stresslab.runners.launch_readiness import (
    BenchmarkSummary,
    CalibrationStatusSummary,
    LaunchCheckResult,
    LaunchReadinessInputs,
    build_calibration_gate,
    build_launch_readiness_report,
    build_provider_runtime_gate,
    run_check_safely,
)


def _passing_inputs() -> LaunchReadinessInputs:
    return LaunchReadinessInputs(
        unit_tests=LaunchCheckResult(name="unit_tests", passed=True),
        common_failure_suite=BenchmarkSummary(name="common_failure_suite", passed=True, accuracy=1.0),
        subtle_failure_suite=BenchmarkSummary(name="subtle_failure_suite", passed=True, accuracy=1.0),
        external_regression=LaunchCheckResult(name="external_regression", passed=True),
        provider_doctor=ExternalProviderDoctorReport(
            providers=[
                ProviderReadiness(
                    provider_name="ragas",
                    available=False,
                    status="degraded",
                    reason_code="runtime_execution_not_configured",
                    reason="Runtime execution is not configured for this provider.",
                    fallback_provider="native_retrieval_signals_only",
                )
            ],
            available_providers=[],
            unavailable_providers=[],
            degraded_providers=["ragas"],
            safe_to_run_external_enhanced=False,
            warnings=["ragas degraded: runtime_execution_not_configured"],
        ),
        calibration_status=CalibrationStatusSummary(
            calibration_artifact_exists=False,
            calibrated_confidence_present=False,
            statuses={"uncalibrated": 3},
        ),
        no_silent_fallback=LaunchCheckResult(name="no_silent_fallback", passed=True),
        decision_policy_regression=LaunchCheckResult(name="decision_policy_regression", passed=True),
        false_clean_count=0,
        false_incomplete_count=0,
        advisory_primary_failure_count=0,
        retrieval_security_drift_count=0,
        prompt_injection_passed=True,
        privacy_violation_passed=True,
        non_clean_missing_fix_count=0,
        failed_case_missing_first_failing_node_count=0,
        missing_first_failing_node_reasons=[],
        benchmark_accuracy=1.0,
        benchmark_accuracy_threshold=0.95,
        surfaced_regression_failures=[],
    )


def test_launch_readiness_fails_on_false_clean() -> None:
    inputs = _passing_inputs()
    inputs.false_clean_count = 1

    report = build_launch_readiness_report(inputs)

    assert report.passed is False
    assert report.status == "Not Ready"
    assert any("false_clean_count" in reason for reason in report.failure_reasons)


def test_launch_readiness_fails_on_missing_provider_reason() -> None:
    inputs = _passing_inputs()
    inputs.provider_doctor = ExternalProviderDoctorReport(
        providers=[
            ProviderReadiness(
                provider_name="ragas",
                available=False,
                status="degraded",
            )
        ],
        available_providers=[],
        unavailable_providers=[],
        degraded_providers=["ragas"],
        safe_to_run_external_enhanced=False,
        warnings=[],
    )

    report = build_launch_readiness_report(inputs)

    assert report.passed is False
    assert any("provider missing/degraded reasons visible" in reason for reason in report.failure_reasons)


def test_launch_readiness_fails_on_uncalibrated_confidence() -> None:
    inputs = _passing_inputs()
    inputs.calibration_status = CalibrationStatusSummary(
        calibration_artifact_exists=False,
        calibrated_confidence_present=True,
        statuses={"calibrated": 1},
    )

    report = build_launch_readiness_report(inputs)

    assert report.passed is False
    assert any("calibrated confidence" in reason.lower() for reason in report.failure_reasons)


def test_launch_readiness_passes_on_clean_expected_report() -> None:
    report = build_launch_readiness_report(_passing_inputs())

    assert report.passed is True
    assert report.status == "Internal Alpha"
    assert report.failure_reasons == []


def test_launch_readiness_records_aborted_check_without_crashing() -> None:
    check = run_check_safely(
        "subtle_benchmark",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        launch_blocker_classification="benchmark_behavior",
    )

    assert check.passed is False
    assert check.status == "aborted"
    assert check.error_message == "boom"
    assert check.launch_blocker_classification == "benchmark_behavior"


def test_native_mode_does_not_fail_for_missing_optional_providers() -> None:
    doctor = ExternalProviderDoctorReport(
        providers=[
            ProviderReadiness(
                provider_name="ragas",
                available=False,
                status="unavailable",
                reason_code="package_missing",
                integration_maturity="schema_only",
                runtime_execution_available=False,
            )
        ],
        unavailable_providers=["ragas"],
        safe_to_run_external_enhanced=False,
    )

    gate = build_provider_runtime_gate(doctor, mode="native", enabled_providers=[])

    assert gate["native_mode_optional_provider_blocker"] is False
    assert gate["passed"] is True


def test_external_enhanced_degraded_when_schema_only_providers_enabled() -> None:
    doctor = ExternalProviderDoctorReport(
        providers=[
            ProviderReadiness(
                provider_name="ragas",
                available=False,
                status="degraded",
                reason_code="runtime_execution_not_configured",
                integration_maturity="schema_only",
                runtime_execution_available=False,
            )
        ],
        degraded_providers=["ragas"],
        safe_to_run_external_enhanced=False,
    )

    gate = build_provider_runtime_gate(doctor, mode="external-enhanced", enabled_providers=["ragas"])

    assert gate["external_enhanced_degraded"] is True
    assert gate["passed"] is False
    assert gate["providers"][0]["emits_real_signals_current_runtime"] is False


def test_calibration_gate_rejects_gating_without_labeled_samples() -> None:
    gate = build_calibration_gate(
        CalibrationStatusSummary(
            calibration_artifact_exists=False,
            calibrated_confidence_present=False,
            statuses={"uncalibrated": 3},
            labeled_sample_count=0,
            confidence_intervals_available=False,
        )
    )

    assert gate["production_gating_eligible"] is False
    assert gate["recommended_for_gating_true_count"] == 0


def test_provider_runtime_gate_distinguishes_schema_only_from_native_runtime() -> None:
    doctor = ExternalProviderDoctorReport(
        providers=[
            ProviderReadiness(
                provider_name="ragas",
                available=True,
                status="available",
                integration_maturity="schema_only",
                runtime_execution_available=False,
            ),
            ProviderReadiness(
                provider_name="cross_encoder_relevance",
                available=True,
                status="available",
                integration_maturity="native_library_runtime",
                runtime_execution_available=True,
            ),
        ],
        available_providers=["ragas", "cross_encoder_relevance"],
        safe_to_run_external_enhanced=True,
    )

    gate = build_provider_runtime_gate(
        doctor,
        mode="external-enhanced",
        enabled_providers=["ragas", "cross_encoder_relevance"],
    )

    providers = {provider["provider_name"]: provider for provider in gate["providers"]}
    assert providers["ragas"]["counts_as_real_runtime_maturity"] is False
    assert providers["cross_encoder_relevance"]["counts_as_real_runtime_maturity"] is True
    assert gate["real_runtime_provider_count"] == 1


def test_report_contains_launch_blockers() -> None:
    inputs = _passing_inputs()
    inputs.false_clean_count = 1

    report = build_launch_readiness_report(inputs)

    assert report.status == "Not Ready"
    assert report.launch_blockers
    assert any(blocker["classification"] == "false_clean_risk" for blocker in report.launch_blockers)
