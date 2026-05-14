"""Runner for the common RAG pipeline failure golden suite."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raggov.engine import DiagnosisEngine
from raggov.evaluators.base import (
    ExternalEvaluationResult,
    ExternalEvaluatorProvider,
)
from raggov.models.chunk import RetrievedChunk
from raggov.models.diagnosis import AnalyzerResult, Diagnosis, FailureStage, FailureType
from raggov.models.run import RAGRun

from stresslab.cases.load import load_common_rag_failures, load_subtle_rag_failures
from stresslab.cases.models import RAGFailureGoldenCase

logger = logging.getLogger(__name__)

@dataclass
class CaseResult:
    case_id: str
    category: str
    expected_primary: str
    actual_primary: str
    expected_stage: str
    actual_stage: str
    passed: bool
    notes: list[str] = field(default_factory=list)

@dataclass
class BenchmarkReport:
    total_cases: int
    passed_cases: int
    pass_rate: float
    results: list[CaseResult]
    category_stats: dict[str, dict[str, int]] = field(default_factory=dict)


EXPECTED_ANALYZER_BY_CATEGORY = {
    "parser_chunking": "ParserValidationAnalyzer",
    "retrieval": "RetrievalDiagnosisAnalyzerV0",
    "sufficiency": "SufficiencyAnalyzer",
    "grounding": "ClaimGroundingAnalyzer",
    "citation": "CitationFaithfulnessAnalyzerV0",
    "version_validity": "TemporalSourceValidityAnalyzerV1",
    "security": "PromptInjectionAnalyzer/PoisoningHeuristicAnalyzer/PrivacyAnalyzer",
    "answer_quality": "ClaimGroundingAnalyzer/SemanticEntropyAnalyzer/CitationFaithfulnessAnalyzerV0",
}

class RAGFailureRunner:
    def __init__(self, mode: str = "native", mock_native: bool = False, suite: str = "common"):
        self.mode = mode
        self.mock_native = mock_native
        self.suite_name = suite
        # We'll create the engine per case if mocking, to inject specific analyzers
        if not mock_native:
            self.engine = DiagnosisEngine(config={"mode": mode})
        else:
            self.engine = None

    def run_benchmark(self) -> BenchmarkReport:
        results = []
        category_stats = {}

        if self.suite_name == "subtle":
            cases = load_subtle_rag_failures()
        else:
            cases = load_common_rag_failures()

        for case in cases:
            result = self._run_case(case)
            results.append(result)

            cat = case.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if result.passed:
                category_stats[cat]["passed"] += 1

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        return BenchmarkReport(
            total_cases=total,
            passed_cases=passed,
            pass_rate=(passed / total) if total > 0 else 0.0,
            results=results,
            category_stats=category_stats
        )

    def _run_case(self, case: RAGFailureGoldenCase) -> CaseResult:
        run = self._build_run(case)
        
        if self.mock_native:
            # Create a set of mock analyzers that exactly match expectations
            from raggov.analyzers.base import BaseAnalyzer
            
            class _MockAnalyzer(BaseAnalyzer):
                def __init__(self, res: AnalyzerResult):
                    super().__init__()
                    self._res = res
                def analyze(self, r: RAGRun) -> AnalyzerResult:
                    return self._res

            mock_analyzers = []
            # Primary failure mock
            mock_analyzers.append(_MockAnalyzer(
                AnalyzerResult(
                    analyzer_name="GoldenPrimaryMock",
                    status="pass" if case.expected_primary_failure == FailureType.CLEAN.value else "fail",
                    failure_type=FailureType(case.expected_primary_failure),
                    stage=FailureStage(case.expected_root_cause_stage),
                    evidence=["Golden primary mock evidence"]
                )
            ))
            
            if case.expected_primary_failure == FailureType.CLEAN.value:
                # Mock critical analyzers to avoid INCOMPLETE_DIAGNOSIS
                for critical_name in ["RetrievalDiagnosisAnalyzerV0", "ClaimGroundingAnalyzer", "ClaimAwareSufficiencyAnalyzer"]:
                    mock_analyzers.append(_MockAnalyzer(
                        AnalyzerResult(
                            analyzer_name=critical_name,
                            status="pass",
                            failure_type=FailureType.CLEAN,
                            stage=FailureStage.UNKNOWN,
                            evidence=[f"Mock {critical_name} pass"]
                        )
                    ))
            # Secondary failures mocks
            for sec in case.expected_secondary_failures:
                mock_analyzers.append(_MockAnalyzer(
                    AnalyzerResult(
                        analyzer_name=f"GoldenSecondaryMock_{sec}",
                        status="warn",
                        failure_type=FailureType(sec),
                        stage=FailureStage.UNKNOWN,
                        evidence=["Golden secondary mock evidence"]
                    )
                ))
            
            # Configure external providers properly via config
            provider_specific_results = {}
            if case.expected_external_signals:
                for s in case.expected_external_signals:
                    p = s["provider"]
                    m_name = s["metric_name"]
                    # Strip provider prefix if present
                    if m_name.startswith(f"{p}_"):
                        m_name = m_name[len(p)+1:]
                    
                    if p not in provider_specific_results:
                        provider_specific_results[p] = {}
                    
                    # Special handling for refchecker/ragchecker which might have nested lists
                    # (but for now our golden cases are simple enough)
                    provider_specific_results[p][m_name] = s["value"]

            # Create engine with all external enabled and results injected
            config = {
                "mode": self.mode,
                "enabled_external_providers": list(provider_specific_results.keys()) if self.mode == "external-enhanced" else []
            }
            # Inject provider-specific results so adapters can pick them up
            for p, results in provider_specific_results.items():
                config[f"{p}_metric_results"] = results

            engine = DiagnosisEngine(config=config, analyzers=mock_analyzers)
        else:
            engine = self.engine

        diagnosis = engine.diagnose(run)

        actual_primary = diagnosis.primary_failure.value
        actual_stage = diagnosis.root_cause_stage.value

        passed = True
        notes = []

        if actual_primary != case.expected_primary_failure:
            passed = False
            notes.append(f"Primary mismatch: expected {case.expected_primary_failure}, got {actual_primary}")

        if actual_stage != case.expected_root_cause_stage:
            passed = False
            notes.append(f"Stage mismatch: expected {case.expected_root_cause_stage}, got {actual_stage}")

        if actual_primary == FailureType.CLEAN.value and self.suite_name == "subtle":
            if case.case_id != "subtle_external_disagreement_09": # Exception for controlled test
                passed = False
                notes.append("Subtle failure returned CLEAN, which is forbidden.")

        if case.expected_human_review_required and not diagnosis.human_review_required:
            passed = False
            notes.append("Expected human review required but not found.")

        return CaseResult(
            case_id=case.case_id,
            category=case.category,
            expected_primary=case.expected_primary_failure,
            actual_primary=actual_primary,
            expected_stage=case.expected_root_cause_stage,
            actual_stage=actual_stage,
            passed=passed,
            notes=notes
        )

    def _build_run(self, case: RAGFailureGoldenCase) -> RAGRun:
        chunks = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id"),
                text=c.get("text"),
                source_doc_id=c.get("source_doc_id"),
                score=c.get("score", 0.9),
                metadata=c.get("metadata", {})
            )
            for c in case.retrieved_chunks
        ]
        
        run = RAGRun(
            run_id=case.case_id,
            query=case.query,
            retrieved_chunks=chunks,
            final_answer=case.final_answer,
            cited_doc_ids=case.cited_doc_ids,
            metadata=case.metadata.copy() if case.metadata else {}
        )

        # Inject external signals if present
        if case.expected_external_signals:
            from raggov.evaluators.base import ExternalSignalRecord
            
            signals_by_provider = {}
            for s_dict in case.expected_external_signals:
                s = ExternalSignalRecord.model_validate(s_dict)
                signals_by_provider.setdefault(s.provider, []).append(s)
            
            ext_results = []
            for provider, sigs in signals_by_provider.items():
                ext_results.append(
                    ExternalEvaluationResult(
                        provider=provider,
                        succeeded=True,
                        signals=sigs
                    ).model_dump()
                )
            run.metadata["external_evaluation_results"] = ext_results

        return run

def generate_markdown_report(report: BenchmarkReport) -> str:
    lines = [
        "# RAG Failure Benchmark Coverage Matrix",
        "",
        f"**Date:** 2026-05-07",
        f"**Total Cases:** {report.total_cases}",
        f"**Passed:** {report.passed_cases}",
        f"**Pass Rate:** {report.pass_rate:.1%}",
        "",
        "## Performance by Category",
        "",
        "| Category | Total | Passed | Pass Rate |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    for cat, stats in sorted(report.category_stats.items()):
        rate = (stats["passed"] / stats["total"]) if stats["total"] > 0 else 0.0
        lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {rate:.1%} |")
    
    lines.extend([
        "",
        "## Detailed Case Results",
        "",
        "| Case ID | Category | Expected Primary | Actual Primary | Result |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])
    
    for res in report.results:
        status = "✅ PASS" if res.passed else "❌ FAIL"
        lines.append(f"| `{res.case_id}` | {res.category} | `{res.expected_primary}` | `{res.actual_primary}` | {status} |")
    
    return "\n".join(lines)


def generate_common_failure_triage(
    *,
    output_json: Path = Path("reports/common_failure_triage.json"),
    output_md: Path = Path("reports/common_failure_triage.md"),
) -> dict[str, Any]:
    """Write detailed common-suite triage artifacts for native and external modes."""
    return generate_failure_triage(suite="common", output_json=output_json, output_md=output_md)


def generate_failure_triage(
    *,
    suite: str,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    """Write detailed suite triage artifacts for native and external modes."""
    if output_json is None:
        output_json = Path(f"reports/{suite}_failure_triage.json")
    if output_md is None:
        output_md = Path(f"reports/{suite}_failure_triage.md")

    triage = {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "modes": {},
    }
    for mode in ("native", "external-enhanced"):
        runner = RAGFailureRunner(mode=mode, suite=suite)
        mode_payload = _triage_mode(runner, mode, suite)
        triage["modes"][mode] = mode_payload

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(triage, indent=2, sort_keys=True))
    output_md.write_text(_failure_triage_markdown(triage))
    return triage


def _triage_mode(runner: RAGFailureRunner, mode: str, suite: str) -> dict[str, Any]:
    cases = load_subtle_rag_failures() if suite == "subtle" else load_common_rag_failures()
    failures: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, int]] = {}
    false_clean_count = 0
    false_security_count = 0
    false_incomplete_count = 0
    passed_count = 0

    for case in cases:
        run = runner._build_run(case)
        diagnosis = runner.engine.diagnose(run)
        actual_primary = diagnosis.primary_failure.value
        actual_stage = diagnosis.root_cause_stage.value
        passed = (
            actual_primary == case.expected_primary_failure
            and actual_stage == case.expected_root_cause_stage
        )
        category_stats.setdefault(case.category, {"total": 0, "passed": 0})
        category_stats[case.category]["total"] += 1
        if passed:
            category_stats[case.category]["passed"] += 1
            passed_count += 1
            continue

        false_clean = actual_primary == FailureType.CLEAN.value and case.expected_primary_failure != FailureType.CLEAN.value
        false_security = (
            actual_stage == FailureStage.SECURITY.value
            and case.expected_root_cause_stage != FailureStage.SECURITY.value
        )
        false_incomplete = (
            actual_primary == FailureType.INCOMPLETE_DIAGNOSIS.value
            and case.expected_primary_failure != FailureType.INCOMPLETE_DIAGNOSIS.value
        )
        false_clean_count += int(false_clean)
        false_security_count += int(false_security)
        false_incomplete_count += int(false_incomplete)

        failures.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "expected_primary_failure": case.expected_primary_failure,
                "actual_primary_failure": actual_primary,
                "expected_stage": case.expected_root_cause_stage,
                "actual_stage": actual_stage,
                "expected_first_failing_node": case.expected_first_failing_node,
                "actual_first_failing_node": diagnosis.first_failing_node,
                "expected_fix": case.expected_recommended_fix_category,
                "actual_fix": diagnosis.proposed_fix or diagnosis.recommended_fix,
                "false_clean": false_clean,
                "false_security": false_security,
                "false_incomplete": false_incomplete,
                "external_probes": [
                    probe.model_dump(mode="json")
                    for probe in diagnosis.external_diagnosis_probes
                ],
                "likely_failing_analyzer": _likely_failing_analyzer(case, diagnosis),
                "likely_code_cause": _likely_code_cause(case, diagnosis),
            }
        )

    return {
        "mode": mode,
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "pass_rate": passed_count / len(cases) if cases else 0.0,
        "false_clean_count": false_clean_count,
        "false_security_count": false_security_count,
        "false_incomplete_count": false_incomplete_count,
        "category_stats": {
            category: {
                **stats,
                "pass_rate": stats["passed"] / stats["total"] if stats["total"] else 0.0,
            }
            for category, stats in sorted(category_stats.items())
        },
        "failures": failures,
    }


def _likely_failing_analyzer(case: RAGFailureGoldenCase, diagnosis: Diagnosis) -> str:
    expected_primary = case.expected_primary_failure
    expected_stage = case.expected_root_cause_stage
    for result in diagnosis.analyzer_results:
        if (
            result.failure_type is not None
            and result.failure_type.value == expected_primary
            and result.stage is not None
            and result.stage.value == expected_stage
        ):
            if result.status == "warn":
                return f"{result.analyzer_name} (warn-level evidence not final)"
            return f"{result.analyzer_name} (emitted expected evidence)"
    return EXPECTED_ANALYZER_BY_CATEGORY.get(case.category, "DiagnosisEngine")


def _likely_code_cause(case: RAGFailureGoldenCase, diagnosis: Diagnosis) -> str:
    expected_primary = case.expected_primary_failure
    actual_primary = diagnosis.primary_failure.value
    if actual_primary == FailureType.CLEAN.value:
        return "Expected failure evidence was absent or only advisory/warn-level, so clean was not blocked."
    for result in diagnosis.analyzer_results:
        if result.failure_type is not None and result.failure_type.value == expected_primary:
            return "Expected evidence exists but decision policy selected a different higher-ranked failure."
    if case.category == "answer_quality":
        return "Answer-quality fixture needs stronger claim/citation/confidence evidence generation."
    if case.category == "retrieval":
        return "Retrieval evidence is generated but subtype mapping remains too coarse for this fixture."
    return "Expected analyzer did not emit the gold failure type for the available fixture evidence."


def _failure_triage_markdown(triage: dict[str, Any]) -> str:
    title = str(triage["suite"]).replace("_", " ").title()
    lines = [
        f"# {title} Failure Triage",
        "",
        f"Generated: `{triage['generated_at']}`",
        "",
    ]
    for mode, payload in triage["modes"].items():
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Pass rate: {payload['passed_cases']}/{payload['total_cases']} ({payload['pass_rate']:.1%})",
                f"False CLEAN: {payload['false_clean_count']}",
                f"False SECURITY: {payload['false_security_count']}",
                f"False INCOMPLETE: {payload['false_incomplete_count']}",
                "",
                "### Category Pass Rates",
                "",
                "| Category | Passed | Total | Pass Rate |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for category, stats in payload["category_stats"].items():
            lines.append(
                f"| {category} | {stats['passed']} | {stats['total']} | {stats['pass_rate']:.1%} |"
            )
        lines.extend(
            [
                "",
                "### Failed Cases",
                "",
                "| Case ID | Category | Expected | Actual | Flags | Likely Analyzer | Likely Code Cause |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for failure in payload["failures"]:
            flags = ", ".join(
                flag
                for flag, enabled in (
                    ("false_clean", failure["false_clean"]),
                    ("false_security", failure["false_security"]),
                    ("false_incomplete", failure["false_incomplete"]),
                )
                if enabled
            ) or "none"
            lines.append(
                "| `{case_id}` | {category} | `{expected_primary_failure}` / `{expected_stage}` "
                "| `{actual_primary_failure}` / `{actual_stage}` | {flags} | {likely_failing_analyzer} | {likely_code_cause} |".format(
                    **failure,
                    flags=flags,
                )
            )
        lines.append("")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Run RAG failure benchmark")
    parser.add_argument("--mode", choices=["native", "external-enhanced"], default="native")
    parser.add_argument("--suite", choices=["common", "subtle"], default="common")
    parser.add_argument("--mock-native", action="store_true", help="Use mock native signals from golden cases")
    parser.add_argument("--output", type=str, help="Path to save markdown report")
    args = parser.parse_args()

    runner = RAGFailureRunner(mode=args.mode, mock_native=args.mock_native, suite=args.suite)
    report = runner.run_benchmark()

    print(f"Benchmark completed: {report.passed_cases}/{report.total_cases} passed ({report.pass_rate:.1%})")

    if args.output:
        md = generate_markdown_report(report)
        Path(args.output).write_text(md)
        print(f"Report saved to {args.output}")

    if args.suite in {"common", "subtle"}:
        generate_failure_triage(suite=args.suite)
        print(f"Triage saved to reports/{args.suite}_failure_triage.json and reports/{args.suite}_failure_triage.md")

if __name__ == "__main__":
    main()
