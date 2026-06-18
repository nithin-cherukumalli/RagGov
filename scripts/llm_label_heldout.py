#!/usr/bin/env python3
"""LLM-assisted heldout labeling scaffold, with offline mock judges.

Governance guarantees:
  - This script never writes to the canonical calibration dataset, dataset lock,
    gates, thresholds, engine, analyzer, policy, or label files.
  - Every output row is tagged `label_source=llm_assisted_provisional`; this
    script never emits `gold`.
  - K independent judge callables are collected per row. The default judges are
    deterministic offline mocks so the harness runs in the sandbox without API
    keys or network access.
  - The voted label is a majority vote over judge verdicts. Per-row inter-judge
    agreement (`max_votes / K`) is recorded as provisional `label_confidence`.
  - Human spot-audit worklist entries are emitted for every disagreement, every
    voted `CONTRADICTED_CLAIM`, and every row below the agreement threshold.
  - `add_calib_case.py` validation is run in validate-only mode through its
    validation function. No append path is called.

Real LLM judges should plug in as callables with the same interface as
`JudgeCallable`: input row -> `JudgeVerdict`. Keep K independent prompts/models
where possible; agreement is evidence for triage, not a gold label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evals/govrag_calib/staging/raw/llm_labeled_provisional.jsonl"
DEFAULT_WORKLIST = ROOT / "evals/govrag_calib/staging/raw/llm_labeled_worklist.jsonl"
DEFAULT_CHANGELOG = ROOT / "reports/calibration/LABEL_CHANGELOG_STUB.md"
FORBIDDEN_OUTPUT_NAMES = {
    "govrag_calib_150.jsonl",
    "DATASET_LOCK.json",
    "DATASET_MANIFEST.json",
    "LABEL_CHANGELOG.md",
}

FAILURE_TO_STAGE = {
    "CLEAN": "UNKNOWN",
    "STALE_RETRIEVAL": "RETRIEVAL",
    "SCOPE_VIOLATION": "RETRIEVAL",
    "CITATION_MISMATCH": "CITATION",
    "INCONSISTENT_CHUNKS": "RETRIEVAL",
    "INSUFFICIENT_CONTEXT": "SUFFICIENCY",
    "UNSUPPORTED_CLAIM": "GROUNDING",
    "CONTRADICTED_CLAIM": "GROUNDING",
    "PROMPT_INJECTION": "SECURITY",
    "SUSPICIOUS_CHUNK": "SECURITY",
    "RETRIEVAL_ANOMALY": "RETRIEVAL",
    "PRIVACY_VIOLATION": "SECURITY",
    "LOW_CONFIDENCE": "CONFIDENCE",
    "TABLE_STRUCTURE_LOSS": "PARSING",
    "HIERARCHY_FLATTENING": "CHUNKING",
    "METADATA_LOSS": "RETRIEVAL",
    "POST_RATIONALIZED_CITATION": "CITATION",
    "PARSER_STRUCTURE_LOSS": "PARSING",
    "CHUNKING_BOUNDARY_ERROR": "CHUNKING",
    "EMBEDDING_DRIFT": "EMBEDDING",
    "RETRIEVAL_DEPTH_LIMIT": "RETRIEVAL",
    "RERANKER_FAILURE": "RERANKING",
    "GENERATION_IGNORE": "GENERATION",
    "INCOMPLETE_DIAGNOSIS": "GENERATION",
    "CLAIM_EXTRACTION_FAILED": "GROUNDING",
}


@dataclass(frozen=True)
class JudgeVerdict:
    judge_id: str
    expected_primary: str
    rationale: str


JudgeCallable = Callable[[dict[str, Any]], JudgeVerdict]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _refuse_forbidden_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _refuse_forbidden_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected file: {resolved}")
    if "evals/govrag_calib/govrag_calib_150.jsonl" in str(resolved):
        raise ValueError(f"refusing to write canonical dataset: {resolved}")


def _row_text(row: dict[str, Any]) -> str:
    chunks = row.get("retrieved_chunks") or row.get("passages") or []
    chunk_text = " ".join(str(chunk.get("text", "")) for chunk in chunks if isinstance(chunk, dict))
    return " ".join(
        str(value or "")
        for value in (
            row.get("query"),
            row.get("answer") or row.get("reference_answer"),
            row.get("source_label"),
            row.get("expected_primary_failure"),
            row.get("notes"),
            row.get("rationale"),
            chunk_text,
        )
    ).lower()


def _source_label_guess(row: dict[str, Any]) -> str | None:
    label = str(row.get("source_label") or row.get("original_source_label") or "").lower()
    if label in {"contradicted", "conflict", "contradiction"}:
        return "CONTRADICTED_CLAIM"
    if label in {"unsupported", "baseless", "baseless_info"}:
        return "UNSUPPORTED_CLAIM"
    if label in {"citation", "citation_mismatch"}:
        return "CITATION_MISMATCH"
    if label == "prompt_injection":
        return "PROMPT_INJECTION"
    if row.get("kind") == "clean":
        return "CLEAN"
    return None


def _heuristic_guess(row: dict[str, Any], variant: int) -> tuple[str, str]:
    text = _row_text(row)
    source_guess = _source_label_guess(row)
    existing = row.get("expected_primary_failure") or row.get("proposed_expected_primary_failure")

    if variant % 3 == 0 and source_guess:
        return source_guess, "Mock judge used source label mapping as primary evidence."
    if "prompt injection" in text or "ignore previous" in text:
        return "PROMPT_INJECTION", "Mock judge saw prompt-injection language."
    if "citation" in text and "mismatch" in text:
        return "CITATION_MISMATCH", "Mock judge saw citation-mismatch language."
    if "insufficient" in text or "not enough context" in text:
        return "INSUFFICIENT_CONTEXT", "Mock judge saw insufficiency language."
    if variant % 3 == 1 and source_guess == "CONTRADICTED_CLAIM":
        return "UNSUPPORTED_CLAIM", "Mock judge treated contradiction source label as ambiguous."
    if existing:
        return str(existing), "Mock judge retained existing expected label as prior."
    if source_guess:
        return source_guess, "Mock judge fell back to source label mapping."
    return "CLEAN", "Mock judge found no explicit failure signal."


def build_mock_judges(k: int) -> list[JudgeCallable]:
    judges: list[JudgeCallable] = []
    for index in range(k):
        judge_id = f"mock-{index + 1}"

        def judge(row: dict[str, Any], *, _index: int = index, _judge_id: str = judge_id) -> JudgeVerdict:
            label, rationale = _heuristic_guess(row, _index)
            return JudgeVerdict(
                judge_id=_judge_id,
                expected_primary=label,
                rationale=rationale,
            )

        judges.append(judge)
    return judges


def majority_vote(verdicts: list[JudgeVerdict]) -> tuple[str, float, bool]:
    counts = Counter(v.expected_primary for v in verdicts)
    max_votes = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == max_votes)
    voted = winners[0]
    agreement = max_votes / len(verdicts)
    disagreement = len(counts) > 1
    return voted, agreement, disagreement


def label_rows(
    rows: list[dict[str, Any]],
    judges: list[JudgeCallable],
    agreement_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    worklist: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        verdicts = [judge(row) for judge in judges]
        voted, agreement, disagreement = majority_vote(verdicts)
        audit_reasons = []
        if disagreement:
            audit_reasons.append("judge_disagreement")
        if voted == "CONTRADICTED_CLAIM":
            audit_reasons.append("contradicted_claim_requires_human_adjudication")
        if agreement < agreement_threshold:
            audit_reasons.append("agreement_below_threshold")

        labeled = dict(row)
        labeled["expected_primary_failure"] = voted
        labeled["expected_stage"] = FAILURE_TO_STAGE.get(voted, "UNKNOWN")
        labeled["label_source"] = "llm_assisted_provisional"
        labeled["label_confidence"] = round(agreement, 4)
        labeled["llm_label_verdicts"] = [asdict(verdict) for verdict in verdicts]
        labeled["llm_label_majority"] = {
            "voted_label": voted,
            "agreement": round(agreement, 4),
            "judge_count": len(judges),
            "agreement_threshold": agreement_threshold,
        }
        labeled["expected_human_review_required"] = bool(
            labeled.get("expected_human_review_required") or audit_reasons
        )
        labeled["notes"] = _append_note(
            labeled.get("notes"),
            "LLM-assisted provisional label; not gold; human audit required="
            + str(bool(audit_reasons)).lower(),
        )
        output_rows.append(labeled)

        if audit_reasons:
            worklist.append(
                {
                    "row_index": index,
                    "case_id": row.get("case_id"),
                    "source_id": row.get("source_id"),
                    "voted_label": voted,
                    "agreement": round(agreement, 4),
                    "audit_reasons": audit_reasons,
                    "judge_verdicts": [asdict(verdict) for verdict in verdicts],
                }
            )
    return output_rows, worklist


def _append_note(existing: Any, note: str) -> str:
    prefix = str(existing or "").strip()
    return f"{prefix} {note}".strip()


def validate_with_add_calib_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import add_calib_case  # noqa: PLC0415

    existing_ids = {row["case_id"] for row in add_calib_case._rows()}
    failure_types = add_calib_case._failure_types()
    rejects: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        errors = add_calib_case.validate(row, existing_ids, failure_types)
        hard_errors = [error for error in errors if not error.startswith("WARN")]
        if hard_errors:
            rejects.append(
                {
                    "row_index": index,
                    "case_id": row.get("case_id"),
                    "errors": hard_errors,
                }
            )
    return rejects


def write_changelog_stub(
    path: Path,
    input_path: Path,
    output_path: Path,
    worklist_path: Path,
    row_count: int,
    worklist_count: int,
) -> None:
    _refuse_forbidden_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    text = f"""# LABEL_CHANGELOG Stub

## {now} - LLM-assisted provisional heldout labels

- Input: `{input_path}`
- Output: `{output_path}`
- Human audit worklist: `{worklist_path}`
- Rows labeled: {row_count}
- Rows requiring human audit: {worklist_count}
- Label source applied: `llm_assisted_provisional`
- Gold/canonical status: NOT GOLD; not appended to canonical dataset; lock not regenerated.
- Acceptance owner: Opus/human adjudicator.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    parser.add_argument("--judges", type=int, default=3, help="number of mock judges")
    parser.add_argument("--agreement-threshold", type=float, default=2 / 3)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument(
        "--judge-provider",
        default="mock",
        choices=["mock"],
        help="Only offline mock judges are wired in this scaffold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.judges < 1:
        raise SystemExit("--judges must be >= 1")
    rows = load_jsonl(args.input_jsonl)
    judges = build_mock_judges(args.judges)
    labeled_rows, worklist = label_rows(rows, judges, args.agreement_threshold)
    rejects = [] if args.no_validate else validate_with_add_calib_case(labeled_rows)

    write_jsonl(args.output, labeled_rows)
    write_jsonl(args.worklist, worklist)
    write_changelog_stub(
        args.changelog,
        args.input_jsonl,
        args.output,
        args.worklist,
        len(labeled_rows),
        len(worklist),
    )

    print(
        f"llm_label_heldout: wrote {len(labeled_rows)} provisional rows -> {args.output}"
    )
    print(f"llm_label_heldout: wrote {len(worklist)} audit rows -> {args.worklist}")
    print(f"llm_label_heldout: wrote changelog stub -> {args.changelog}")
    if rejects:
        print(f"llm_label_heldout: validation rejects={len(rejects)}")
        for reject in rejects[:20]:
            print(json.dumps(reject, sort_keys=True))
        return 1
    print("llm_label_heldout: add_calib_case validate-only PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

