"""
Evaluation harness for retrieval analyzer v0 regression fixtures.

Loads retrieval diagnosis JSONL, builds RAGRun objects, runs
RetrievalEvidenceProfilerV0 and all four retrieval analyzers, then
compares outputs against expected labels from the fixture.

IMPORTANT:
- This harness validates v0 heuristic regression behavior only.
- It is NOT a calibration tool.
- It is NOT a gold-set evaluator.
- Exact-match accuracy here means the heuristic behaves as it did when
  the fixture was authored — not that the heuristic is correct.
- Timestamp-based staleness fixtures may drift as documents age past
  the default 180-day threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raggov.analyzers.retrieval.citation import CitationMismatchAnalyzer
from raggov.analyzers.retrieval.evidence_profile import RetrievalEvidenceProfilerV0
from raggov.analyzers.retrieval.inconsistency import InconsistentChunksAnalyzer
from raggov.analyzers.retrieval.scope import ScopeViolationAnalyzer
from raggov.analyzers.retrieval.stale import StaleRetrievalAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile
from raggov.models.run import RAGRun

_ANALYZERS: dict[str, Any] = {
    "CitationMismatchAnalyzer": CitationMismatchAnalyzer(),
    "ScopeViolationAnalyzer": ScopeViolationAnalyzer(),
    "InconsistentChunksAnalyzer": InconsistentChunksAnalyzer(),
    "StaleRetrievalAnalyzer": StaleRetrievalAnalyzer(),
}

_FIXTURE_DEFAULT = ROOT / "tests" / "fixtures" / "retrieval_diagnosis_v0.jsonl"

_DISCLAIMER = (
    "v0 heuristic regression only — not calibrated, not gold-labeled, "
    "not proof of semantic correctness"
)

_PASS_SKIP = {"pass", "skip"}
_WARN_FAIL = {"warn", "fail"}


# ---------------------------------------------------------------------------
# Data classes for metrics
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerMetrics:
    analyzer_name: str
    total: int = 0
    exact_match: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    skip_count: int = 0
    profile_used_count: int = 0
    fallback_count: int = 0

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_match / self.total if self.total else 0.0


@dataclass
class EvalReport:
    total_cases: int
    per_analyzer: dict[str, AnalyzerMetrics]
    profile_match_count: int
    profile_mismatch_count: int
    case_mismatches: list[dict[str, Any]]
    run_timestamp: str
    disclaimer: str = _DISCLAIMER


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load and parse a retrieval diagnosis JSONL fixture file."""
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {lineno}: {exc}") from exc
    return cases


# ---------------------------------------------------------------------------
# RAGRun construction
# ---------------------------------------------------------------------------

def build_run(case: dict[str, Any]) -> RAGRun:
    """Deserialize a fixture case into a RAGRun (no profile attached yet)."""
    chunks = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source_doc_id=c["source_doc_id"],
            score=c.get("score"),
        )
        for c in case.get("retrieved_chunks", [])
    ]

    corpus_entries = []
    for entry in case.get("corpus_entries", []):
        ts_raw = entry.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        corpus_entries.append(
            CorpusEntry(doc_id=entry["doc_id"], text=entry.get("text", ""), timestamp=ts)
        )

    return RAGRun(
        query=case["query"],
        retrieved_chunks=chunks,
        final_answer="",
        cited_doc_ids=case.get("cited_doc_ids", []),
        corpus_entries=corpus_entries,
    )


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def _compare_profile(
    actual: RetrievalEvidenceProfile,
    expected: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare key outcome fields of the built profile against expected dict.

    Returns (matches: bool, mismatches: list[str]).
    """
    mismatches: list[str] = []

    def _check_set(field_name: str) -> None:
        actual_val = set(getattr(actual, field_name, []))
        expected_val = set(expected.get(field_name, []))
        if actual_val != expected_val:
            mismatches.append(
                f"{field_name}: actual={sorted(actual_val)} expected={sorted(expected_val)}"
            )

    def _check_pairs(field_name: str) -> None:
        actual_pairs = set(tuple(p) for p in getattr(actual, field_name, []))
        expected_pairs = set(tuple(p) for p in expected.get(field_name, []))
        if actual_pairs != expected_pairs:
            mismatches.append(
                f"{field_name}: actual={sorted(actual_pairs)} expected={sorted(expected_pairs)}"
            )

    def _check_scalar(field_name: str) -> None:
        actual_val = getattr(actual, field_name, None)
        expected_val = expected.get(field_name)
        if actual_val != expected_val:
            mismatches.append(f"{field_name}: actual={actual_val!r} expected={expected_val!r}")

    _check_scalar("overall_retrieval_status")
    _check_set("phantom_citation_doc_ids")
    _check_set("stale_doc_ids")
    _check_set("noisy_chunk_ids")
    _check_pairs("contradictory_pairs")
    _check_scalar("recommended_for_gating")

    return (len(mismatches) == 0), mismatches


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run the full profiler + analyzer pipeline on one fixture case."""
    run = build_run(case)

    profile = RetrievalEvidenceProfilerV0().build(run)
    run.retrieval_evidence_profile = profile

    profile_matches, profile_diffs = _compare_profile(profile, case.get("expected_profile", {}))

    analyzer_results: dict[str, dict[str, Any]] = {}
    for name, analyzer in _ANALYZERS.items():
        result = analyzer.analyze(run)
        expected = case.get("expected_analyzer_results", {}).get(name, {})
        expected_status = expected.get("status")
        actual_status = result.status

        exact_match = actual_status == expected_status

        if not exact_match:
            if expected_status in _PASS_SKIP and actual_status in _WARN_FAIL:
                verdict = "false_positive"
            elif expected_status in _WARN_FAIL and actual_status in _PASS_SKIP:
                verdict = "false_negative"
            else:
                verdict = "status_mismatch"
        else:
            verdict = "correct"

        analyzer_results[name] = {
            "status": actual_status,
            "analysis_source": result.analysis_source,
            "expected_status": expected_status,
            "exact_match": exact_match,
            "verdict": verdict,
        }

    return {
        "case_id": case["case_id"],
        "description": case.get("description", ""),
        "profile_matches": profile_matches,
        "profile_diffs": profile_diffs,
        "analyzer_results": analyzer_results,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_metrics(case_results: list[dict[str, Any]]) -> EvalReport:
    """Aggregate per-case results into per-analyzer metrics."""
    metrics: dict[str, AnalyzerMetrics] = {
        name: AnalyzerMetrics(analyzer_name=name) for name in _ANALYZERS
    }
    profile_match_count = 0
    profile_mismatch_count = 0
    case_mismatches: list[dict[str, Any]] = []

    for cr in case_results:
        if cr["profile_matches"]:
            profile_match_count += 1
        else:
            profile_mismatch_count += 1

        case_has_mismatch = not cr["profile_matches"]

        for name, ar in cr["analyzer_results"].items():
            m = metrics[name]
            m.total += 1

            if ar["exact_match"]:
                m.exact_match += 1
            else:
                case_has_mismatch = True

            verdict = ar["verdict"]
            if verdict == "false_positive":
                m.false_positives += 1
            elif verdict == "false_negative":
                m.false_negatives += 1

            if ar["status"] == "skip":
                m.skip_count += 1
            if ar["analysis_source"] == "retrieval_evidence_profile":
                m.profile_used_count += 1
            elif ar["analysis_source"] == "legacy_heuristic_fallback":
                m.fallback_count += 1

        if case_has_mismatch:
            case_mismatches.append({
                "case_id": cr["case_id"],
                "profile_diffs": cr["profile_diffs"],
                "analyzer_mismatches": {
                    name: {
                        "actual": ar["status"],
                        "expected": ar["expected_status"],
                        "verdict": ar["verdict"],
                    }
                    for name, ar in cr["analyzer_results"].items()
                    if not ar["exact_match"]
                },
            })

    return EvalReport(
        total_cases=len(case_results),
        per_analyzer=metrics,
        profile_match_count=profile_match_count,
        profile_mismatch_count=profile_mismatch_count,
        case_mismatches=case_mismatches,
        run_timestamp=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_summary(report: EvalReport) -> str:
    lines = [
        "Retrieval Analyzer v0 Evaluation",
        "=" * 60,
        f"DISCLAIMER: {report.disclaimer}",
        "",
        f"Total cases : {report.total_cases}",
        f"Profile match : {report.profile_match_count}/{report.total_cases}",
        "",
        "Per-analyzer results:",
        f"  {'Analyzer':<32} {'Acc':>6}  {'FP':>4}  {'FN':>4}  {'Skip':>5}  {'Profile':>8}  {'Fallback':>9}",
        "  " + "-" * 78,
    ]
    for name, m in report.per_analyzer.items():
        short = name.replace("Analyzer", "")
        lines.append(
            f"  {short:<32} {m.exact_match_accuracy:>6.1%}"
            f"  {m.false_positives:>4}"
            f"  {m.false_negatives:>4}"
            f"  {m.skip_count:>5}"
            f"  {m.profile_used_count:>8}"
            f"  {m.fallback_count:>9}"
        )

    if report.case_mismatches:
        lines += ["", f"Mismatched cases ({len(report.case_mismatches)}):"]
        for cm in report.case_mismatches:
            lines.append(f"  {cm['case_id']}")
            for f_name, diff in cm.get("analyzer_mismatches", {}).items():
                lines.append(f"    {f_name}: {diff['actual']} (expected {diff['expected']}) [{diff['verdict']}]")
            for diff_str in cm.get("profile_diffs", []):
                lines.append(f"    [profile] {diff_str}")
    else:
        lines += ["", "No mismatches."]

    lines += ["", "=" * 60]
    return "\n".join(lines)


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Serialise EvalReport to a JSON-compatible dict."""
    return {
        "disclaimer": report.disclaimer,
        "total_cases": report.total_cases,
        "profile_match_count": report.profile_match_count,
        "profile_mismatch_count": report.profile_mismatch_count,
        "per_analyzer": {
            name: {
                "total": m.total,
                "exact_match": m.exact_match,
                "exact_match_accuracy": m.exact_match_accuracy,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "skip_count": m.skip_count,
                "profile_used_count": m.profile_used_count,
                "fallback_count": m.fallback_count,
            }
            for name, m in report.per_analyzer.items()
        },
        "case_mismatches": report.case_mismatches,
        "run_timestamp": report.run_timestamp,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    fixture_path: Path = _FIXTURE_DEFAULT,
    output_path: Path | None = None,
) -> EvalReport:
    """Run the evaluation harness and return the report."""
    cases = load_fixture(fixture_path)
    case_results = [evaluate_case(c) for c in cases]
    report = compute_metrics(case_results)

    print(render_summary(report))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report_to_dict(report), indent=2), encoding="utf-8"
        )
        print(f"\nReport written to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval analyzer v0 against regression fixtures."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=_FIXTURE_DEFAULT,
        help="Path to the JSONL fixture file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    args = parser.parse_args()
    main(fixture_path=args.fixture, output_path=args.output)
