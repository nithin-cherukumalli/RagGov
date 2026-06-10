"""Audit current GovRAG workspace state for coding-agent safety."""

from __future__ import annotations

from pathlib import Path

from harness_common import (
    ROOT,
    branch,
    generated_report_changes,
    git_status,
    last_commit,
    protected_changes,
    status_from_messages,
    write_json,
    write_markdown_report,
)


def build_report(cwd: Path = ROOT) -> dict:
    status = git_status(cwd)
    dirty_files = status["dirty_files"]
    protected = protected_changes(dirty_files, cwd)
    generated_reports = generated_report_changes(dirty_files, cwd)
    warnings: list[str] = []
    errors: list[str] = []

    if status["deleted_tracked_files"]:
        errors.append("Tracked files are deleted.")
    if protected:
        warnings.append("Protected files are changed.")
    if dirty_files:
        warnings.append("Workspace has dirty files.")

    safe_to_continue = not errors and not protected
    report = {
        "status": status_from_messages(errors, warnings),
        "branch": branch(cwd),
        "last_commit": last_commit(cwd),
        "git_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "deleted_tracked_files": status["deleted_tracked_files"],
        "untracked_files": status["untracked_files"],
        "protected_files_changed": protected,
        "generated_reports_changed": generated_reports,
        "safe_to_continue": safe_to_continue,
        "warnings": warnings,
        "errors": errors,
        "recommended_action": (
            "Stop and review deleted/protected files before editing."
            if not safe_to_continue
            else "Proceed with a narrow patch and run preflight before edits."
        ),
    }
    return report


def main() -> int:
    report = build_report(ROOT)
    write_json(ROOT / "reports" / "workspace_audit.json", report)
    write_markdown_report(ROOT / "reports" / "workspace_audit.md", "Workspace Audit", report)
    print(f"workspace audit: {report['status']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
