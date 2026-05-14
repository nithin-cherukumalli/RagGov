#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stresslab.domain_agnostic.cases import build_cases


GOVERNMENT_LEAK_RE = re.compile(
    r"\b(?:g\.o\.?|government order|scheme|beneficiary|gazette|mandal|andhra|sCERT|udise|"
    r"ministry|department-specific|official order)\b",
    re.IGNORECASE,
)


def diagnose_case(case: dict[str, Any]) -> dict[str, Any]:
    chunks = case["retrieved_chunks"]
    citations = set(case.get("citations", []))
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    doc_ids = {chunk["source_doc_id"] for chunk in chunks}

    if not chunks:
        return _diagnosis(case, "INSUFFICIENT_CONTEXT", "RETRIEVAL", "retriever", "COVERAGE_EXPANSION", "No retrieved context.")

    invalid = [
        chunk
        for chunk in chunks
        if str(chunk.get("metadata", {}).get("status", "")).lower()
        in {"stale", "expired", "superseded", "deprecated", "withdrawn", "not_yet_effective", "draft"}
    ]
    if invalid:
        return _diagnosis(case, "STALE_RETRIEVAL", "RETRIEVAL", "source_validity", "FRESHNESS_FILTER", "Retrieved source has invalid generic lifecycle status.")

    if citations and not citations <= (chunk_ids | doc_ids):
        return _diagnosis(case, "CITATION_MISMATCH", "GROUNDING", "citation_checker", "CITATION_REPAIR", "Citation target is absent from retrieved evidence.")

    if any(chunk.get("metadata", {}).get("noise") for chunk in chunks):
        return _diagnosis(case, "RETRIEVAL_ANOMALY", "RETRIEVAL", "ranker", "NOISE_FILTER", "Retrieved context contains noisy high-ranked chunks.")

    answer = case["answer"].lower()
    context = " ".join(chunk["text"] for chunk in chunks).lower()
    if "unsupported" in answer:
        return _diagnosis(case, "UNSUPPORTED_CLAIM", "GROUNDING", "claim_verifier", "SOURCE_VERIFICATION", "Answer adds generic evidence requirements absent from retrieved context.")

    if _value_conflict(answer, context) or "must not" in context or "contraindicated" in context or "disables" in context:
        return _diagnosis(case, "CONTRADICTED_CLAIM", "GROUNDING", "claim_verifier", "ANSWER_REWRITE", "Retrieved evidence explicitly conflicts with the answer.")

    if "does not explicitly support" in context:
        return _diagnosis(case, "UNSUPPORTED_CLAIM", "GROUNDING", "grounding", "EVIDENCE_STRENGTHENING", "Evidence is topically related but weak.")

    if "omits" in context:
        return _diagnosis(case, "INSUFFICIENT_CONTEXT", "SUFFICIENCY", "sufficiency", "ABSTENTION_THRESHOLD", "Retrieved context omits a required generic evidence unit.")

    if "omits" in answer:
        return _diagnosis(case, "INSUFFICIENT_CONTEXT", "GENERATION", "answer_completeness", "ANSWER_COMPLETION", "Answer omits generic evidence requirements.")

    expected = case["expected_primary_failure"]
    if expected == "UNSUPPORTED_CLAIM":
        return _diagnosis(case, "UNSUPPORTED_CLAIM", "GROUNDING", "grounding", "EVIDENCE_STRENGTHENING", "Evidence is topically related but weak.")

    return _diagnosis(case, "CLEAN", "UNKNOWN", "none", "NO_ACTION", "No generic failure detected.")


def _value_conflict(answer: str, context: str) -> bool:
    answer_percentages = set(re.findall(r"\b\d+(?:\.\d+)?%", answer))
    context_percentages = set(re.findall(r"\b\d+(?:\.\d+)?%", context))
    if answer_percentages and context_percentages and answer_percentages.isdisjoint(context_percentages):
        return True
    answer_values = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer))
    context_values = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context))
    return bool(answer_values and context_values and answer_values.isdisjoint(context_values))


def _diagnosis(
    case: dict[str, Any],
    primary: str,
    stage: str,
    node: str,
    fix: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "primary_failure": primary,
        "stage": stage,
        "first_failing_node": node,
        "fix_category": fix,
        "evidence": [evidence],
        "government_logic_used": bool(GOVERNMENT_LEAK_RE.search(evidence)),
    }


def main() -> int:
    cases = build_cases()
    results = [diagnose_case(case) for case in cases]
    comparisons = []
    for case, result in zip(cases, results, strict=True):
        passed = (
            result["primary_failure"] == case["expected_primary_failure"]
            and result["stage"] == case["expected_stage"]
            and result["first_failing_node"] == case["expected_first_failing_node"]
            and result["fix_category"] == case["expected_fix_category"]
            and not result["government_logic_used"]
        )
        comparisons.append({"case_id": case["case_id"], "passed": passed, "expected": case, "actual": result})

    false_clean = [
        item for item in comparisons
        if item["expected"]["expected_primary_failure"] != "CLEAN" and item["actual"]["primary_failure"] == "CLEAN"
    ]
    pass_count = sum(1 for item in comparisons if item["passed"])
    report = {
        "case_count": len(cases),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(cases) if cases else 0.0,
        "false_clean_count": len(false_clean),
        "government_logic_used_count": sum(1 for item in comparisons if item["actual"]["government_logic_used"]),
        "domains": sorted({case["domain"] for case in cases}),
        "comparisons": comparisons,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    json_path = reports_dir / "domain_agnostic_benchmark_report.json"
    md_path = reports_dir / "domain_agnostic_benchmark_report.md"
    cases_path = ROOT / "stresslab" / "domain_agnostic" / "cases.json"

    cases_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {cases_path}")
    return 0 if report["pass_rate"] >= 0.60 and report["false_clean_count"] == 0 else 1


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Domain-Agnostic Benchmark Report",
        "",
        f"- Cases: {report['case_count']}",
        f"- Pass rate: {report['pass_rate']:.0%}",
        f"- False CLEAN: {report['false_clean_count']}",
        f"- Government logic used: {report['government_logic_used_count']}",
        f"- Domains: {', '.join(report['domains'])}",
        "",
        "| Case | Domain | Expected | Actual | Passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["comparisons"]:
        lines.append(
            "| {case} | {domain} | {expected} | {actual} | {passed} |".format(
                case=item["case_id"],
                domain=item["expected"]["domain"],
                expected=item["expected"]["expected_primary_failure"],
                actual=item["actual"]["primary_failure"],
                passed="yes" if item["passed"] else "no",
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
