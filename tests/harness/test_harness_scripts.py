from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import harness_common
import harness_post_edit_validation
import harness_preflight
import workspace_audit


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_harness_config(tmp_path: Path) -> None:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    for name in ("protected_paths.json", "protected_baseline.json"):
        shutil.copy(REPO_ROOT / "harness" / name, harness_dir / name)


def _write_common_report(
    tmp_path: Path,
    native: dict | None = None,
    external: dict | None = None,
    external_key: str = "external-enhanced",
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    modes = {}
    if native is not None:
        modes["native"] = native
    if external is not None:
        modes[external_key] = external
    (reports_dir / "common_failure_triage.json").write_text(
        json.dumps({"modes": modes}),
        encoding="utf-8",
    )


def _clean_git_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness_post_edit_validation,
        "git_status",
        lambda cwd: {"dirty_files": [], "deleted_tracked_files": [], "untracked_files": []},
    )


def test_preflight_detects_protected_fixture_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_harness_config(tmp_path)
    monkeypatch.setattr(
        harness_preflight,
        "git_status",
        lambda cwd: {
            "dirty_files": ["tests/fixtures/retrieval_diagnosis_v0.jsonl"],
            "deleted_tracked_files": [],
            "untracked_files": [],
        },
    )
    monkeypatch.setattr(harness_preflight, "branch", lambda cwd: "test")
    monkeypatch.setattr(harness_preflight, "last_commit", lambda cwd: "abc123")

    report = harness_preflight.build_report(cwd=tmp_path)

    assert report["status"] == "warn"
    assert report["protected_changes"] == ["tests/fixtures/retrieval_diagnosis_v0.jsonl"]


def test_git_porcelain_parser_preserves_first_path_character() -> None:
    parsed = harness_common.parse_porcelain(" M src/raggov/models/diagnosis.py\n?? harness/README.md\n")

    assert parsed["dirty_files"][0] == "src/raggov/models/diagnosis.py"
    assert parsed["untracked_files"] == ["harness/README.md"]


def test_preflight_detects_deleted_tracked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_harness_config(tmp_path)
    monkeypatch.setattr(
        harness_preflight,
        "git_status",
        lambda cwd: {
            "dirty_files": ["src/raggov/models/signals.py"],
            "deleted_tracked_files": ["src/raggov/models/signals.py"],
            "untracked_files": [],
        },
    )
    monkeypatch.setattr(harness_preflight, "branch", lambda cwd: "test")
    monkeypatch.setattr(harness_preflight, "last_commit", lambda cwd: "abc123")

    report = harness_preflight.build_report(cwd=tmp_path)

    assert report["status"] == "fail"
    assert "Tracked files are deleted." in report["errors"]


def test_post_edit_detects_false_clean_regression_from_mock_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    _write_common_report(
        tmp_path,
        native={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 1,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
        external={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "fail"
    assert any(
        regression["metric"] == "native_false_clean_count"
        for regression in report["protected_baseline_regressions"]
    )


def test_post_edit_detects_production_gating_enabled_from_mock_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "launch_readiness_report.json").write_text(
        json.dumps({"production_gating_eligible": True}),
        encoding="utf-8",
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "fail"
    assert report["production_gating_eligible"] is True
    assert any(
        regression["metric"] == "production_gating_eligible"
        for regression in report["protected_baseline_regressions"]
    )


def test_post_edit_detects_native_pass_count_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    _write_common_report(
        tmp_path,
        native={
            "passed_cases": 39,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
        external={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "fail"
    assert report["mode_results"]["native"]["passed"] == 39
    assert any(
        regression["metric"] == "common_native_passed"
        for regression in report["protected_baseline_regressions"]
    )


def test_post_edit_detects_external_pass_count_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    _write_common_report(
        tmp_path,
        native={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
        external={
            "passed_cases": 39,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "fail"
    assert report["mode_results"]["external_enhanced"]["passed"] == 39
    assert any(
        regression["metric"] == "common_external_passed"
        for regression in report["protected_baseline_regressions"]
    )


def test_post_edit_keeps_modes_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_harness_config(tmp_path)
    _write_common_report(
        tmp_path,
        native={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
        external={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 2,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["mode_results"]["native"]["false_clean_count"] == 0
    assert report["mode_results"]["external_enhanced"]["false_clean_count"] == 2
    assert any(
        regression["metric"] == "external_enhanced_false_clean_count"
        for regression in report["protected_baseline_regressions"]
    )


def test_post_edit_passes_when_both_modes_meet_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    mode = {
        "passed_cases": 41,
        "total_cases": 46,
        "false_clean_count": 0,
        "false_security_count": 0,
        "false_incomplete_count": 0,
    }
    _write_common_report(tmp_path, native=mode, external=mode, external_key="external_enhanced")
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "pass"
    assert report["protected_baseline_regressions"] == []


def test_missing_external_mode_warns_without_using_native_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_harness_config(tmp_path)
    _write_common_report(
        tmp_path,
        native={
            "passed_cases": 41,
            "total_cases": 46,
            "false_clean_count": 0,
            "false_security_count": 0,
            "false_incomplete_count": 0,
        },
        external=None,
    )
    _clean_git_status(monkeypatch)

    report = harness_post_edit_validation.build_report(cwd=tmp_path)

    assert report["status"] == "warn"
    assert report["mode_results"]["native"]["passed"] == 41
    assert report["mode_results"]["external_enhanced"] is None
    assert "Common benchmark mode missing: external_enhanced." in report["warnings"]


def test_workspace_audit_outputs_json_and_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_harness_config(tmp_path)
    monkeypatch.setattr(
        workspace_audit,
        "git_status",
        lambda cwd: {"dirty_files": [], "deleted_tracked_files": [], "untracked_files": []},
    )
    monkeypatch.setattr(workspace_audit, "branch", lambda cwd: "test")
    monkeypatch.setattr(workspace_audit, "last_commit", lambda cwd: "abc123")

    report = workspace_audit.build_report(tmp_path)
    harness_common.write_json(tmp_path / "reports" / "workspace_audit.json", report)
    harness_common.write_markdown_report(tmp_path / "reports" / "workspace_audit.md", "Workspace Audit", report)

    assert (tmp_path / "reports" / "workspace_audit.json").exists()
    assert (tmp_path / "reports" / "workspace_audit.md").exists()
    assert json.loads((tmp_path / "reports" / "workspace_audit.json").read_text())["status"] == "pass"


def test_failure_mode_registry_schema_valid() -> None:
    data = json.loads((REPO_ROOT / "harness" / "failure_mode_registry.json").read_text(encoding="utf-8"))
    categories = {mode["category"] for mode in data["failure_modes"]}
    required = {
        "retrieval",
        "grounding",
        "citation",
        "sufficiency",
        "version_validity",
        "parser_chunking",
        "answer_quality",
        "confidence",
        "security",
        "decision_policy",
        "external_provider_runtime",
        "calibration",
        "benchmark_integrity",
        "workspace_integrity",
        "report_integrity",
    }
    assert required <= categories
    for mode in data["failure_modes"]:
        assert {
            "id",
            "category",
            "description",
            "symptoms",
            "protected_files",
            "likely_owner",
            "recommended_first_diagnostic_command",
            "safe_action",
            "unsafe_action",
        } <= set(mode)


def test_protected_baseline_schema_valid() -> None:
    data = json.loads((REPO_ROOT / "harness" / "protected_baseline.json").read_text(encoding="utf-8"))
    assert data["common_native_passed"] == 41
    assert data["common_native_total"] == 46
    assert data["common_external_passed"] == 41
    assert data["common_external_total"] == 46
    assert data["false_clean_count"] == 0
    assert data["false_security_count"] == 0
    assert data["false_incomplete_count"] == 0
    assert data["production_gating_eligible"] is False
    assert data["calibration_status"] == "not_production_calibrated"
