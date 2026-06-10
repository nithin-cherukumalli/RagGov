"""Shared helpers for lightweight GovRAG workspace harness scripts."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = "reports"


def run_command(args: list[str], cwd: Path = ROOT, timeout: int = 120) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"command": args, "returncode": 127, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": args,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "command timed out",
        }
    return {
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_output(args: list[str], cwd: Path = ROOT) -> str:
    result = run_command(["git", *args], cwd=cwd)
    if result["returncode"] != 0:
        return ""
    return result["stdout"].strip()


def branch(cwd: Path = ROOT) -> str:
    return git_output(["branch", "--show-current"], cwd=cwd) or "unknown"


def last_commit(cwd: Path = ROOT) -> str:
    return git_output(["rev-parse", "--short", "HEAD"], cwd=cwd) or "unknown"


def parse_porcelain(output: str) -> dict[str, list[str]]:
    dirty: list[str] = []
    deleted: list[str] = []
    untracked: list[str] = []
    for raw_line in output.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path = raw_line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty.append(path)
        if status == "??":
            untracked.append(path)
        if "D" in status:
            deleted.append(path)
    return {"dirty_files": dirty, "deleted_tracked_files": deleted, "untracked_files": untracked}


def git_status(cwd: Path = ROOT) -> dict[str, list[str]]:
    result = run_command(["git", "status", "--porcelain"], cwd=cwd)
    if result["returncode"] != 0:
        return {"dirty_files": [], "deleted_tracked_files": [], "untracked_files": []}
    return parse_porcelain(result["stdout"])


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def protected_config(cwd: Path = ROOT) -> dict[str, Any]:
    return load_json(cwd / "harness" / "protected_paths.json", {})


def baseline_config(cwd: Path = ROOT) -> dict[str, Any]:
    return load_json(cwd / "harness" / "protected_baseline.json", {})


def _normalize_pattern(pattern: str) -> str:
    return pattern.replace("\\", "/")


def matches_any(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        pat = _normalize_pattern(pattern)
        if fnmatch.fnmatch(normalized, pat):
            return True
        if pat.endswith("/**") and normalized.startswith(pat[:-3]):
            return True
    return False


def contains_protected_token(path: str, cwd: Path, tokens: list[str]) -> bool:
    file_path = cwd / path
    if not file_path.is_file():
        return False
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def protected_changes(paths: list[str], cwd: Path = ROOT) -> list[str]:
    config = protected_config(cwd)
    patterns = config.get("protected_patterns", [])
    tokens = config.get("protected_content_tokens", [])
    return sorted(
        {
            path
            for path in paths
            if matches_any(path, patterns) or contains_protected_token(path, cwd, tokens)
        }
    )


def threshold_or_gate_changes(paths: list[str], cwd: Path = ROOT) -> list[str]:
    config = protected_config(cwd)
    patterns = config.get("threshold_or_gate_patterns", [])
    tokens = ["threshold", "production_gating_eligible", "launch readiness", "launch_readiness"]
    return sorted(
        {
            path
            for path in paths
            if matches_any(path, patterns) or contains_protected_token(path, cwd, tokens)
        }
    )


def generated_report_changes(paths: list[str], cwd: Path = ROOT) -> list[str]:
    patterns = protected_config(cwd).get("generated_report_patterns", ["reports/*.json", "reports/*.md"])
    return sorted({path for path in paths if matches_any(path, patterns)})


def classify_risk(paths: list[str], cwd: Path = ROOT) -> dict[str, list[str]]:
    config = protected_config(cwd)
    risk_patterns = config.get("risk_classes", {})
    result = {"low": [], "medium": [], "high": [], "critical": []}
    for path in paths:
        assigned = False
        for risk in ("critical", "high", "medium", "low"):
            if matches_any(path, risk_patterns.get(risk, [])):
                result[risk].append(path)
                assigned = True
                break
        if not assigned:
            result["medium"].append(path)
    return {risk: sorted(values) for risk, values in result.items()}


def find_key(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _mode_metric(mode_data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _as_int(mode_data.get(key))
        if value is not None:
            return value
    return None


def parse_common_benchmark_modes(report: dict[str, Any]) -> dict[str, dict[str, int | None] | None]:
    """Return common benchmark metrics by mode without cross-mode fallback."""
    modes = report.get("modes")
    if not isinstance(modes, dict):
        return {"native": None, "external_enhanced": None}

    external_data = modes.get("external-enhanced")
    if external_data is None:
        external_data = modes.get("external_enhanced")

    mode_sources = {
        "native": modes.get("native"),
        "external_enhanced": external_data,
    }
    parsed: dict[str, dict[str, int | None] | None] = {}
    for mode_name, mode_data in mode_sources.items():
        if not isinstance(mode_data, dict):
            parsed[mode_name] = None
            continue
        parsed[mode_name] = {
            "passed": _mode_metric(mode_data, "passed", "passed_cases"),
            "total": _mode_metric(mode_data, "total", "total_cases"),
            "false_clean_count": _mode_metric(mode_data, "false_clean_count"),
            "false_security_count": _mode_metric(mode_data, "false_security_count"),
            "false_incomplete_count": _mode_metric(mode_data, "false_incomplete_count"),
        }
    return parsed


def common_benchmark_mode_warnings(mode_results: dict[str, dict[str, int | None] | None]) -> list[str]:
    warnings: list[str] = []
    for mode_name in ("native", "external_enhanced"):
        if mode_results.get(mode_name) is None:
            warnings.append(f"Common benchmark mode missing: {mode_name}.")
    return warnings


def recent_report_status(cwd: Path = ROOT) -> dict[str, bool]:
    return {
        "common_failure_triage_json": (cwd / "reports" / "common_failure_triage.json").exists(),
        "common_failure_triage_md": (cwd / "reports" / "common_failure_triage.md").exists(),
        "launch_readiness_json": (cwd / "reports" / "launch_readiness_report.json").exists(),
        "launch_readiness_md": (cwd / "reports" / "launch_readiness_report.md").exists(),
    }


def status_from_messages(errors: list[str], warnings: list[str], strict: bool = False) -> str:
    if errors or (strict and warnings):
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_list(values: list[Any]) -> str:
    if not values:
        return "- None\n"
    return "".join(f"- `{value}`\n" for value in values)


def write_markdown_report(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"- Status: `{payload.get('status', 'unknown')}`"]
    if "branch" in payload:
        lines.append(f"- Branch: `{payload.get('branch')}`")
    if "last_commit" in payload:
        lines.append(f"- Last commit: `{payload.get('last_commit')}`")
    if "recommended_action" in payload:
        lines.append(f"- Recommended action: {payload.get('recommended_action')}")
    lines.append("")
    for key, value in payload.items():
        if key in {"status", "branch", "last_commit", "recommended_action"}:
            continue
        heading = key.replace("_", " ").title()
        lines.append(f"## {heading}")
        if isinstance(value, list):
            lines.append(markdown_list(value).rstrip())
        elif isinstance(value, dict):
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, sort_keys=True))
            lines.append("```")
        else:
            lines.append(f"`{value}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_common_report(cwd: Path = ROOT) -> dict[str, Any]:
    for name in ("common_failure_triage.json", "harness_common_benchmark.json"):
        path = cwd / "reports" / name
        if path.exists():
            data = load_json(path, {})
            if isinstance(data, dict):
                return data
    return {}


def load_launch_report(cwd: Path = ROOT) -> dict[str, Any]:
    path = cwd / "reports" / "launch_readiness_report.json"
    data = load_json(path, {})
    return data if isinstance(data, dict) else {}


def summarize_command_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": " ".join(result.get("command", [])),
        "returncode": result.get("returncode"),
        "stdout_tail": result.get("stdout", "")[-2000:],
        "stderr_tail": result.get("stderr", "")[-2000:],
    }
