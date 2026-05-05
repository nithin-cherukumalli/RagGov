"""
Evaluation harness for temporal source validity v1 regression fixtures.

IMPORTANT:
- These fixtures validate rule behavior only.
- They are not calibration data.
- They are not domain-specific temporal proof.
- They are not production gating evidence.
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
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from raggov.analyzers.version_validity import TemporalSourceValidityAnalyzerV1  # noqa: E402
from raggov.models.chunk import RetrievedChunk  # noqa: E402
from raggov.models.citation_faithfulness import CitationFaithfulnessReport  # noqa: E402
from raggov.models.corpus import CorpusEntry  # noqa: E402
from raggov.models.retrieval_evidence import RetrievalEvidenceProfile  # noqa: E402
from raggov.models.run import RAGRun  # noqa: E402

_FIXTURE_DEFAULT = ROOT / "tests" / "fixtures" / "version_validity_v1.jsonl"
_DISCLAIMER = (
    "version validity v1 rule regression only — not calibration data, "
    "not domain-specific temporal proof, not production gating evidence"
)

_ISSUE_FIELDS = {
    "superseded_doc_ids",
    "withdrawn_doc_ids",
    "expired_doc_ids",
    "not_yet_effective_doc_ids",
    "metadata_missing_doc_ids",
    "high_risk_claim_ids",
}


@dataclass
class EvalReport:
    total_cases: int
    exact_match_count: int
    superseded_tp: int
    superseded_fp: int
    superseded_fn: int
    withdrawn_tp: int
    withdrawn_fp: int
    expired_tp: int
    expired_fp: int
    invalid_cited_tp: int
    invalid_cited_fn: int
    false_positive_count: int
    false_negative_count: int
    skip_count: int
    metadata_missing_count: int
    age_based_fallback_count: int
    case_mismatches: list[dict[str, Any]]
    run_timestamp: str
    disclaimer: str = _DISCLAIMER

    @property
    def exact_match_accuracy(self) -> float:
        return self.exact_match_count / self.total_cases if self.total_cases else 0.0

    @property
    def superseded_detection_precision(self) -> float:
        return _precision(self.superseded_tp, self.superseded_fp)

    @property
    def superseded_detection_recall(self) -> float:
        return _recall(self.superseded_tp, self.superseded_fn)

    @property
    def withdrawn_detection_precision(self) -> float:
        return _precision(self.withdrawn_tp, self.withdrawn_fp)

    @property
    def expired_detection_precision(self) -> float:
        return _precision(self.expired_tp, self.expired_fp)

    @property
    def invalid_cited_source_recall(self) -> float:
        return _recall(self.invalid_cited_tp, self.invalid_cited_fn)


def _precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def _recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load and parse a version validity JSONL fixture file."""
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
            metadata=c.get("metadata", {}),
        )
        for c in case.get("retrieved_chunks", [])
    ]
    corpus_entries = []
    for entry in case.get("corpus_entries", []):
        timestamp = _parse_datetime(entry.get("timestamp"))
        corpus_entries.append(
            CorpusEntry(
                doc_id=entry["doc_id"],
                text=entry.get("text", ""),
                timestamp=timestamp,
                metadata=entry.get("metadata", {}),
            )
        )
    profile_raw = case.get("retrieval_evidence_profile")
    citation_raw = case.get("citation_faithfulness_report")
    return RAGRun(
        run_id=case.get("case_id", "version-validity-fixture"),
        query=case["query"],
        retrieved_chunks=chunks,
        final_answer=case.get("generated_answer", "Answer."),
        cited_doc_ids=case.get("cited_doc_ids", []),
        corpus_entries=corpus_entries,
        retrieval_evidence_profile=(
            RetrievalEvidenceProfile.model_validate(profile_raw)
            if profile_raw is not None
            else None
        ),
        citation_faithfulness_report=(
            CitationFaithfulnessReport.model_validate(citation_raw)
            if citation_raw is not None
            else None
        ),
        metadata={"query_date": case["query_date"]},
    )


def _parse_datetime(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    value = str(raw)
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return datetime.fromisoformat(value)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run TemporalSourceValidityAnalyzerV1 on one fixture case."""
    result = TemporalSourceValidityAnalyzerV1({"max_age_days": 180}).analyze(build_run(case))
    actual_report = (
        result.version_validity_report.model_dump(mode="json")
        if result.version_validity_report is not None
        else None
    )
    expected_report = case.get("expected_version_validity_report")
    expected_result = case.get("expected_analyzer_result", {})
    report_matches, report_diffs = _compare_report(actual_report, expected_report)
    status_matches = result.status == expected_result.get("status")
    return {
        "case_id": case["case_id"],
        "description": case.get("description", ""),
        "actual_status": result.status,
        "expected_status": expected_result.get("status"),
        "actual_report": actual_report,
        "expected_report": expected_report,
        "exact_match": report_matches and status_matches,
        "report_diffs": report_diffs,
    }


def _compare_report(
    actual: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    if actual is None:
        if expected is None:
            return True, []
        return False, ["actual report is None"]
    if expected is None:
        return False, ["expected report is None"]

    diffs: list[str] = []
    for field_name in (
        "active_doc_ids",
        "stale_doc_ids",
        "superseded_doc_ids",
        "amended_doc_ids",
        "withdrawn_doc_ids",
        "expired_doc_ids",
        "not_yet_effective_doc_ids",
        "metadata_missing_doc_ids",
        "high_risk_claim_ids",
    ):
        actual_set = set(actual.get(field_name, []))
        expected_set = set(expected.get(field_name, []))
        if actual_set != expected_set:
            diffs.append(
                f"{field_name}: actual={sorted(actual_set)} expected={sorted(expected_set)}"
            )

    actual_records = {r["doc_id"]: r for r in actual.get("document_records", [])}
    expected_records = {r["doc_id"]: r for r in expected.get("document_records", [])}
    if set(actual_records) != set(expected_records):
        diffs.append(
            f"document doc_ids: actual={sorted(actual_records)} expected={sorted(expected_records)}"
        )
    for doc_id, expected_record in expected_records.items():
        actual_record = actual_records.get(doc_id)
        if actual_record is None:
            continue
        for field_name in ("validity_status", "validity_risk", "evidence_source"):
            if actual_record.get(field_name) != expected_record.get(field_name):
                diffs.append(
                    f"{doc_id}.{field_name}: actual={actual_record.get(field_name)!r} "
                    f"expected={expected_record.get(field_name)!r}"
                )

    actual_claims = {r["claim_id"]: r for r in actual.get("claim_source_records", [])}
    expected_claims = {r["claim_id"]: r for r in expected.get("claim_source_records", [])}
    if set(actual_claims) != set(expected_claims):
        diffs.append(
            f"claim ids: actual={sorted(actual_claims)} expected={sorted(expected_claims)}"
        )
    for claim_id, expected_record in expected_claims.items():
        actual_record = actual_claims.get(claim_id)
        if actual_record is None:
            continue
        for field_name in (
            "invalid_cited_doc_ids",
            "claim_validity_status",
            "claim_validity_risk",
        ):
            if field_name == "invalid_cited_doc_ids":
                if set(actual_record.get(field_name, [])) != set(expected_record.get(field_name, [])):
                    diffs.append(
                        f"{claim_id}.{field_name}: actual={actual_record.get(field_name)} "
                        f"expected={expected_record.get(field_name)}"
                    )
            elif actual_record.get(field_name) != expected_record.get(field_name):
                diffs.append(
                    f"{claim_id}.{field_name}: actual={actual_record.get(field_name)!r} "
                    f"expected={expected_record.get(field_name)!r}"
                )

    for field_name in (
        "method_type",
        "calibration_status",
        "recommended_for_gating",
        "retrieval_evidence_profile_used",
        "citation_faithfulness_report_used",
        "lineage_metadata_used",
        "age_based_fallback_used",
    ):
        if field_name in expected and actual.get(field_name) != expected.get(field_name):
            diffs.append(
                f"{field_name}: actual={actual.get(field_name)!r} expected={expected.get(field_name)!r}"
            )
    return len(diffs) == 0, diffs


def compute_metrics(case_results: list[dict[str, Any]]) -> EvalReport:
    """Aggregate version validity fixture results."""
    exact_match_count = 0
    skip_count = 0
    false_positive_count = 0
    false_negative_count = 0
    metadata_missing_count = 0
    age_based_fallback_count = 0
    superseded_tp = superseded_fp = superseded_fn = 0
    withdrawn_tp = withdrawn_fp = 0
    expired_tp = expired_fp = 0
    invalid_cited_tp = invalid_cited_fn = 0
    case_mismatches: list[dict[str, Any]] = []

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
        superseded_tp, superseded_fp, superseded_fn = _accumulate_counts(
            actual.get("superseded_doc_ids", []),
            expected.get("superseded_doc_ids", []),
            superseded_tp,
            superseded_fp,
            superseded_fn,
        )
        withdrawn_tp, withdrawn_fp, _ = _accumulate_counts(
            actual.get("withdrawn_doc_ids", []),
            expected.get("withdrawn_doc_ids", []),
            withdrawn_tp,
            withdrawn_fp,
            0,
        )
        expired_tp, expired_fp, _ = _accumulate_counts(
            actual.get("expired_doc_ids", []),
            expected.get("expired_doc_ids", []),
            expired_tp,
            expired_fp,
            0,
        )
        invalid_cited_tp, _, invalid_cited_fn = _accumulate_counts(
            actual.get("high_risk_claim_ids", []),
            expected.get("high_risk_claim_ids", []),
            invalid_cited_tp,
            0,
            invalid_cited_fn,
        )
        metadata_missing_count += len(actual.get("metadata_missing_doc_ids", []))
        age_based_fallback_count += int(bool(actual.get("age_based_fallback_used")))
        false_positive_count += _issue_fp_count(actual, expected)
        false_negative_count += _issue_fn_count(actual, expected)

    return EvalReport(
        total_cases=len(case_results),
        exact_match_count=exact_match_count,
        superseded_tp=superseded_tp,
        superseded_fp=superseded_fp,
        superseded_fn=superseded_fn,
        withdrawn_tp=withdrawn_tp,
        withdrawn_fp=withdrawn_fp,
        expired_tp=expired_tp,
        expired_fp=expired_fp,
        invalid_cited_tp=invalid_cited_tp,
        invalid_cited_fn=invalid_cited_fn,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        skip_count=skip_count,
        metadata_missing_count=metadata_missing_count,
        age_based_fallback_count=age_based_fallback_count,
        case_mismatches=case_mismatches,
        run_timestamp=datetime.now(UTC).isoformat(),
    )


def _accumulate_counts(
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
        "Version Validity v1 Evaluation",
        "=" * 60,
        f"DISCLAIMER: {report.disclaimer}",
        "",
        f"Total cases: {report.total_cases}",
        f"Exact match accuracy: {report.exact_match_accuracy:.1%}",
        f"Superseded precision/recall: {report.superseded_detection_precision:.1%} / {report.superseded_detection_recall:.1%}",
        f"Withdrawn precision: {report.withdrawn_detection_precision:.1%}",
        f"Expired precision: {report.expired_detection_precision:.1%}",
        f"Invalid cited source recall: {report.invalid_cited_source_recall:.1%}",
        f"False positives: {report.false_positive_count}",
        f"False negatives: {report.false_negative_count}",
        f"Skips: {report.skip_count}",
        f"Metadata missing count: {report.metadata_missing_count}",
        f"Age fallback count: {report.age_based_fallback_count}",
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
            "superseded_detection_precision": report.superseded_detection_precision,
            "superseded_detection_recall": report.superseded_detection_recall,
            "withdrawn_detection_precision": report.withdrawn_detection_precision,
            "expired_detection_precision": report.expired_detection_precision,
            "invalid_cited_source_recall": report.invalid_cited_source_recall,
        }
    )
    return payload


def main(
    fixture_path: Path = _FIXTURE_DEFAULT,
    output_path: Path | None = None,
) -> EvalReport:
    cases = load_fixture(fixture_path)
    report = compute_metrics([evaluate_case(case) for case in cases])
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
        description="Evaluate version validity v1 regression fixtures."
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
