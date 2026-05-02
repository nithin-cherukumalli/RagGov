"""
Evaluation harness for citation faithfulness v0 regression fixtures.

IMPORTANT:
- These fixtures validate v0 behavior only.
- They are not human-labeled calibration data.
- They do not prove citation faithfulness or model reliance.
- They do not prove post-rationalized citation detection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raggov.analyzers.citation_faithfulness import CitationFaithfulnessAnalyzerV0  # noqa: E402
from raggov.models.chunk import RetrievedChunk  # noqa: E402
from raggov.models.grounding import ClaimEvidenceRecord  # noqa: E402
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile  # noqa: E402
from raggov.models.run import RAGRun  # noqa: E402

_FIXTURE_DEFAULT = ROOT / "tests" / "fixtures" / "citation_faithfulness_v0.jsonl"
_DISCLAIMER = (
    "citation faithfulness v0 regression only — not calibration data, "
    "not proof of citation faithfulness, not proof of model reliance"
)

_ISSUE_FIELDS = {
    "unsupported_claim_ids",
    "phantom_citation_doc_ids",
    "missing_citation_claim_ids",
    "contradicted_claim_ids",
}


@dataclass
class EvalReport:
    total_cases: int
    exact_match_count: int
    unsupported_tp: int
    unsupported_fp: int
    unsupported_fn: int
    phantom_tp: int
    phantom_fp: int
    phantom_fn: int
    missing_tp: int
    missing_fp: int
    missing_fn: int
    contradicted_citation_count: int
    false_positive_count: int
    false_negative_count: int
    skip_count: int
    case_mismatches: list[dict[str, Any]]
    run_timestamp: str
    disclaimer: str = _DISCLAIMER

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_match_count / self.total_cases if self.total_cases else 0.0

    @property
    def unsupported_citation_precision(self) -> float:
        return _precision(self.unsupported_tp, self.unsupported_fp)

    @property
    def unsupported_citation_recall(self) -> float:
        return _recall(self.unsupported_tp, self.unsupported_fn)

    @property
    def phantom_citation_precision(self) -> float:
        return _precision(self.phantom_tp, self.phantom_fp)

    @property
    def phantom_citation_recall(self) -> float:
        return _recall(self.phantom_tp, self.phantom_fn)

    @property
    def missing_citation_precision(self) -> float:
        return _precision(self.missing_tp, self.missing_fp)

    @property
    def missing_citation_recall(self) -> float:
        return _recall(self.missing_tp, self.missing_fn)


def _precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load and parse a citation faithfulness JSONL fixture file."""
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


def build_run(case: dict[str, Any]) -> RAGRun:
    """Deserialize a fixture case into a RAGRun."""
    chunks = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source_doc_id=c["source_doc_id"],
            score=c.get("score"),
        )
        for c in case.get("retrieved_chunks", [])
    ]
    claim_records = [
        ClaimEvidenceRecord.model_validate(record)
        for record in case.get("claim_evidence_records", [])
    ]
    profile_raw = case.get("retrieval_evidence_profile")
    retrieval_profile = (
        RetrievalEvidenceProfile.model_validate(profile_raw)
        if profile_raw is not None
        else None
    )
    return RAGRun(
        run_id=case.get("case_id", "citation-faithfulness-fixture"),
        query=case["query"],
        retrieved_chunks=chunks,
        final_answer=case.get("generated_answer", ""),
        cited_doc_ids=case.get("cited_doc_ids", []),
        retrieval_evidence_profile=retrieval_profile,
        metadata={"claim_evidence_records": claim_records},
    )


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run CitationFaithfulnessAnalyzerV0 on one fixture case."""
    run = build_run(case)
    result = CitationFaithfulnessAnalyzerV0().analyze(run)
    actual_report = (
        result.citation_faithfulness_report.model_dump(mode="json")
        if result.citation_faithfulness_report is not None
        else None
    )
    expected_report = case.get("expected_citation_faithfulness_report", {})
    expected_result = case.get("expected_analyzer_result", {})

    report_matches, report_diffs = _compare_report(actual_report, expected_report)
    status_matches = result.status == expected_result.get("status")
    exact_match = report_matches and status_matches

    return {
        "case_id": case["case_id"],
        "description": case.get("description", ""),
        "actual_status": result.status,
        "expected_status": expected_result.get("status"),
        "actual_report": actual_report,
        "expected_report": expected_report,
        "exact_match": exact_match,
        "report_diffs": report_diffs,
    }


def _compare_report(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    if actual is None:
        if expected is None:
            return True, []
        return False, ["actual report is None"]

    for field_name in _ISSUE_FIELDS:
        actual_set = set(actual.get(field_name, []))
        expected_set = set(expected.get(field_name, []))
        if actual_set != expected_set:
            diffs.append(
                f"{field_name}: actual={sorted(actual_set)} expected={sorted(expected_set)}"
            )

    actual_records = {
        record["claim_id"]: record
        for record in actual.get("records", [])
    }
    expected_records = {
        record["claim_id"]: record
        for record in expected.get("records", [])
    }
    if set(actual_records) != set(expected_records):
        diffs.append(
            f"record claim_ids: actual={sorted(actual_records)} expected={sorted(expected_records)}"
        )

    for claim_id, expected_record in expected_records.items():
        actual_record = actual_records.get(claim_id)
        if actual_record is None:
            continue
        for field_name in (
            "citation_support_label",
            "faithfulness_risk",
            "evidence_source",
        ):
            if actual_record.get(field_name) != expected_record.get(field_name):
                diffs.append(
                    f"{claim_id}.{field_name}: actual={actual_record.get(field_name)!r} "
                    f"expected={expected_record.get(field_name)!r}"
                )

    for field_name in (
        "method_type",
        "calibration_status",
        "recommended_for_gating",
        "claim_grounding_used",
        "retrieval_evidence_profile_used",
        "legacy_citation_fallback_used",
    ):
        if field_name in expected and actual.get(field_name) != expected.get(field_name):
            diffs.append(
                f"{field_name}: actual={actual.get(field_name)!r} expected={expected.get(field_name)!r}"
            )

    return len(diffs) == 0, diffs


def compute_metrics(case_results: list[dict[str, Any]]) -> EvalReport:
    """Aggregate case results into citation faithfulness v0 metrics."""
    exact_match_count = 0
    skip_count = 0
    false_positive_count = 0
    false_negative_count = 0
    case_mismatches: list[dict[str, Any]] = []
    unsupported_tp = unsupported_fp = unsupported_fn = 0
    phantom_tp = phantom_fp = phantom_fn = 0
    missing_tp = missing_fp = missing_fn = 0
    contradicted_count = 0

    for result in case_results:
        if result["exact_match"]:
            exact_match_count += 1
        else:
            case_mismatches.append(
                {
                    "case_id": result["case_id"],
                    "actual_status": result["actual_status"],
                    "expected_status": result["expected_status"],
                    "report_diffs": result["report_diffs"],
                }
            )
        if result["actual_status"] == "skip":
            skip_count += 1

        actual = result["actual_report"] or {}
        expected = result["expected_report"] or {}

        unsupported_tp, unsupported_fp, unsupported_fn = _accumulate_set_counts(
            actual.get("unsupported_claim_ids", []),
            expected.get("unsupported_claim_ids", []),
            unsupported_tp,
            unsupported_fp,
            unsupported_fn,
        )
        phantom_tp, phantom_fp, phantom_fn = _accumulate_set_counts(
            actual.get("phantom_citation_doc_ids", []),
            expected.get("phantom_citation_doc_ids", []),
            phantom_tp,
            phantom_fp,
            phantom_fn,
        )
        missing_tp, missing_fp, missing_fn = _accumulate_set_counts(
            actual.get("missing_citation_claim_ids", []),
            expected.get("missing_citation_claim_ids", []),
            missing_tp,
            missing_fp,
            missing_fn,
        )

        contradicted_count += len(actual.get("contradicted_claim_ids", []))
        false_positive_count += _issue_fp_count(actual, expected)
        false_negative_count += _issue_fn_count(actual, expected)

    return EvalReport(
        total_cases=len(case_results),
        exact_match_count=exact_match_count,
        unsupported_tp=unsupported_tp,
        unsupported_fp=unsupported_fp,
        unsupported_fn=unsupported_fn,
        phantom_tp=phantom_tp,
        phantom_fp=phantom_fp,
        phantom_fn=phantom_fn,
        missing_tp=missing_tp,
        missing_fp=missing_fp,
        missing_fn=missing_fn,
        contradicted_citation_count=contradicted_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        skip_count=skip_count,
        case_mismatches=case_mismatches,
        run_timestamp=datetime.now(UTC).isoformat(),
    )


def _accumulate_set_counts(
    actual_values: list[str],
    expected_values: list[str],
    tp: int,
    fp: int,
    fn: int,
) -> tuple[int, int, int]:
    actual = set(actual_values)
    expected = set(expected_values)
    return (
        tp + len(actual & expected),
        fp + len(actual - expected),
        fn + len(expected - actual),
    )


def _issue_fp_count(actual: dict[str, Any], expected: dict[str, Any]) -> int:
    return sum(
        len(set(actual.get(field_name, [])) - set(expected.get(field_name, [])))
        for field_name in _ISSUE_FIELDS
    )


def _issue_fn_count(actual: dict[str, Any], expected: dict[str, Any]) -> int:
    return sum(
        len(set(expected.get(field_name, [])) - set(actual.get(field_name, [])))
        for field_name in _ISSUE_FIELDS
    )


def render_summary(report: EvalReport) -> str:
    lines = [
        "Citation Faithfulness v0 Evaluation",
        "=" * 60,
        f"DISCLAIMER: {report.disclaimer}",
        "",
        f"Total cases: {report.total_cases}",
        f"Exact match accuracy: {report.exact_match_accuracy:.1%}",
        f"Unsupported precision/recall: {report.unsupported_citation_precision:.1%} / {report.unsupported_citation_recall:.1%}",
        f"Phantom precision/recall: {report.phantom_citation_precision:.1%} / {report.phantom_citation_recall:.1%}",
        f"Missing precision/recall: {report.missing_citation_precision:.1%} / {report.missing_citation_recall:.1%}",
        f"Contradicted citation count: {report.contradicted_citation_count}",
        f"False positives: {report.false_positive_count}",
        f"False negatives: {report.false_negative_count}",
        f"Skips: {report.skip_count}",
    ]
    if report.case_mismatches:
        lines += ["", f"Mismatched cases ({len(report.case_mismatches)}):"]
        for mismatch in report.case_mismatches:
            lines.append(f"  {mismatch['case_id']}")
            for diff in mismatch["report_diffs"]:
                lines.append(f"    {diff}")
    else:
        lines += ["", "No mismatches."]
    lines += ["", "=" * 60]
    return "\n".join(lines)


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    payload = asdict(report)
    payload.update(
        {
            "exact_match_accuracy": report.exact_match_accuracy,
            "unsupported_citation_precision": report.unsupported_citation_precision,
            "unsupported_citation_recall": report.unsupported_citation_recall,
            "phantom_citation_precision": report.phantom_citation_precision,
            "phantom_citation_recall": report.phantom_citation_recall,
            "missing_citation_precision": report.missing_citation_precision,
            "missing_citation_recall": report.missing_citation_recall,
        }
    )
    return payload


def main(
    fixture_path: Path = _FIXTURE_DEFAULT,
    output_path: Path | None = None,
) -> EvalReport:
    cases = load_fixture(fixture_path)
    case_results = [evaluate_case(case) for case in cases]
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
        description="Evaluate citation faithfulness v0 regression fixtures."
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
