#!/usr/bin/env python3
"""
GovRAG-Calib Dataset Readiness Evaluator.

Does NOT run GovRAG analyzers.
Evaluates dataset readiness only.

Reports:
  - total cases
  - complete real cases (non-placeholder, split≠unset)
  - placeholder cases
  - cases by category, domain, label source, split
  - train/dev/heldout availability
  - minimum per-category count
  - calibration readiness status:
      NOT_READY            if complete_cases < 30
      PROVISIONAL_DATASET  if 30 <= complete_cases < 150
      READY_FOR_HELDOUT_EVAL if complete_cases >= 150 and heldout exists
  - production_gating_eligible = false  (always — no production gating yet)

Outputs:
  - reports/govrag_calib_readiness.md
  - reports/govrag_calib_readiness.json

Usage:
  python scripts/evaluate_govrag_calib.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

def _add_project_to_path() -> None:
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    evals = PROJECT_ROOT / "evals"
    if str(evals) not in sys.path:
        sys.path.insert(0, str(evals))


_add_project_to_path()

from govrag_calib.schema import GovRAGCalibCase  # noqa: E402
from raggov.engine import DiagnosisEngine  # noqa: E402
from raggov.models.chunk import RetrievedChunk  # noqa: E402
from raggov.models.run import RAGRun  # noqa: E402
from validate_govrag_calib import FAMILY_TARGETS, validate_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Category mapping (mirrors validate_govrag_calib.py)
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, str] = {
    "CLEAN": "clean_pass",
    "RETRIEVAL_ANOMALY": "retrieval",
    "RETRIEVAL_DEPTH_LIMIT": "retrieval",
    "SCOPE_VIOLATION": "retrieval",
    "STALE_RETRIEVAL": "retrieval",
    "INCONSISTENT_CHUNKS": "retrieval",
    "EMBEDDING_DRIFT": "retrieval",
    "RERANKER_FAILURE": "retrieval",
    "UNSUPPORTED_CLAIM": "grounding",
    "CONTRADICTED_CLAIM": "grounding",
    "INSUFFICIENT_CONTEXT": "sufficiency",
    "CITATION_MISMATCH": "citation",
    "POST_RATIONALIZED_CITATION": "citation",
    "PRIVACY_VIOLATION": "security",
    "PROMPT_INJECTION": "security",
    "SUSPICIOUS_CHUNK": "security",
    "LOW_CONFIDENCE": "confidence",
    "GENERATION_IGNORE": "answer_quality",
    "TABLE_STRUCTURE_LOSS": "parsing",
    "HIERARCHY_FLATTENING": "parsing",
    "METADATA_LOSS": "parsing",
    "PARSER_STRUCTURE_LOSS": "parsing",
    "CHUNKING_BOUNDARY_ERROR": "parsing",
    "INCOMPLETE_DIAGNOSIS": "other",
}

# Minimum recommended count per category for a provisional dataset
MIN_PER_CATEGORY = 3


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list[GovRAGCalibCase]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(GovRAGCalibCase(**data))
            except Exception:
                pass
    return cases


def evaluate_readiness(path: Path) -> dict[str, Any]:
    all_cases = load_cases(path)

    # Partition
    complete_cases = [
        c for c in all_cases
        if c.label_confidence != "low" and c.split != "unset"
    ]
    placeholder_cases = [c for c in all_cases if c.label_confidence == "low"]

    # Counts by dimension
    by_category: Counter[str] = Counter()
    for c in complete_cases:
        cat = CATEGORY_MAP.get(c.expected_primary_failure, "other")
        by_category[cat] += 1

    by_domain: Counter[str] = Counter(c.domain for c in complete_cases)
    by_label_source: Counter[str] = Counter(c.label_source for c in complete_cases)
    by_split: Counter[str] = Counter(c.split for c in complete_cases)
    by_confidence: Counter[str] = Counter(c.label_confidence for c in all_cases)

    # Split availability
    train_count = by_split.get("train", 0)
    dev_count = by_split.get("dev", 0)
    heldout_count = by_split.get("heldout", 0)

    heldout_available = heldout_count > 0
    train_available = train_count > 0
    dev_available = dev_count > 0

    # Per-category minimum check
    thin_categories = {cat: cnt for cat, cnt in by_category.items() if cnt < MIN_PER_CATEGORY}

    # Readiness status
    n_complete = len(complete_cases)
    if n_complete < 30:
        readiness_status = "NOT_READY"
        readiness_reason = (
            f"Only {n_complete} complete cases. Minimum 30 required for any calibration use."
        )
    elif n_complete < 150:
        readiness_status = "PROVISIONAL_DATASET"
        readiness_reason = (
            f"{n_complete} complete cases. Provisional dataset — usable for early "
            f"harness development but not for production calibration evaluation."
        )
    else:
        if heldout_available:
            readiness_status = "READY_FOR_HELDOUT_EVAL"
            readiness_reason = (
                f"{n_complete} complete cases with heldout split available. "
                f"Dataset is ready for held-out calibration evaluation."
            )
        else:
            readiness_status = "PROVISIONAL_DATASET"
            readiness_reason = (
                f"{n_complete} complete cases but no heldout split defined. "
                f"Assign heldout cases before running final calibration evaluation."
            )

    # Production gating — always false
    production_gating_eligible = False
    production_gating_reason = (
        "Production gating is not enabled. No calibrated confidence scores exist. "
        "Activate only after statistical calibration is complete and validated."
    )

    return {
        "dataset_path": str(path),
        "total_cases": len(all_cases),
        "complete_cases": n_complete,
        "placeholder_cases": len(placeholder_cases),
        "by_category": dict(by_category),
        "by_domain": dict(by_domain),
        "by_label_source": dict(by_label_source),
        "by_split": dict(by_split),
        "by_confidence": dict(by_confidence),
        "split_availability": {
            "train": train_available,
            "dev": dev_available,
            "heldout": heldout_available,
            "train_count": train_count,
            "dev_count": dev_count,
            "heldout_count": heldout_count,
        },
        "thin_categories": thin_categories,
        "min_per_category_threshold": MIN_PER_CATEGORY,
        "readiness_status": readiness_status,
        "readiness_reason": readiness_reason,
        "production_gating_eligible": production_gating_eligible,
        "production_gating_reason": production_gating_reason,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

READINESS_ICONS = {
    "NOT_READY": "🔴",
    "PROVISIONAL_DATASET": "🟡",
    "READY_FOR_HELDOUT_EVAL": "🟢",
}


def write_json_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON report → {path}")


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    icon = READINESS_ICONS.get(result["readiness_status"], "⚪")
    lines = [
        "# GovRAG-Calib Readiness Report",
        "",
        f"**Readiness Status:** {icon} `{result['readiness_status']}`",
        "",
        f"> {result['readiness_reason']}",
        "",
        "## Dataset Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total cases | {result['total_cases']} |",
        f"| Complete cases | {result['complete_cases']} |",
        f"| Placeholder cases (low confidence) | {result['placeholder_cases']} |",
        "",
        "## Cases by Category",
        "",
        "| Category | Complete Cases |",
        "|---|---|",
    ]
    for cat, count in sorted(result["by_category"].items(), key=lambda x: -x[1]):
        thin = " ⚠️" if cat in result["thin_categories"] else ""
        lines.append(f"| {cat} | {count}{thin} |")

    lines += [
        "",
        "## Cases by Domain",
        "",
        "| Domain | Count |",
        "|---|---|",
    ]
    for domain, count in sorted(result["by_domain"].items(), key=lambda x: -x[1]):
        lines.append(f"| {domain} | {count} |")

    lines += [
        "",
        "## Cases by Label Source",
        "",
        "| Label Source | Count |",
        "|---|---|",
    ]
    for src, count in sorted(result["by_label_source"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {count} |")

    lines += [
        "",
        "## Split Availability",
        "",
        "| Split | Available | Count |",
        "|---|---|---|",
    ]
    sa = result["split_availability"]
    lines.append(f"| train | {'✅' if sa['train'] else '❌'} | {sa['train_count']} |")
    lines.append(f"| dev | {'✅' if sa['dev'] else '❌'} | {sa['dev_count']} |")
    lines.append(f"| heldout | {'✅' if sa['heldout'] else '❌'} | {sa['heldout_count']} |")

    if result["thin_categories"]:
        lines += [
            "",
            "## Thin Categories (below minimum threshold)",
            "",
            f"Minimum recommended: {result['min_per_category_threshold']} cases per category.",
            "",
        ]
        for cat, count in result["thin_categories"].items():
            lines.append(f"- ⚠️ `{cat}`: {count} case(s)")

    lines += [
        "",
        "## Production Gating",
        "",
        f"**production_gating_eligible:** `{result['production_gating_eligible']}`",
        "",
        f"> {result['production_gating_reason']}",
        "",
        "---",
        "",
        "_This report is generated by `scripts/evaluate_govrag_calib.py`. "
        "No analyzer logic was executed. "
        "No calibrated confidence scores were used._",
    ]

    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Markdown report → {path}")


# ---------------------------------------------------------------------------
# GovRAG-Calib evaluator v0
# ---------------------------------------------------------------------------

DEFAULT_EVAL_JSON = (
    PROJECT_ROOT
    / "evals"
    / "govrag_calib"
    / "results"
    / "govrag_calib_eval_latest.json"
)


def load_calib_records(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def evaluate_calib_dataset(
    dataset_path: Path,
    *,
    mode: str = "native",
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_calib_records(dataset_path, limit=limit)
    engine = DiagnosisEngine(config={"mode": mode})
    per_case = []
    confusion_primary: Counter[str] = Counter()
    confusion_stage: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    primary_correct = 0
    stage_correct = 0
    false_clean_count = 0
    false_security_count = 0
    false_incomplete_count = 0
    dangerous_clean_miss_count = 0
    security_stage_miss_count = 0
    acceptable_nonclean_human_review_count = 0
    human_review_miss_count = 0
    acceptable_alternative_match = 0
    wrong_stage_correct_failure = 0
    correct_stage_wrong_failure = 0
    expected_candidate_generated_count = 0
    expected_candidate_selected_count = 0
    first_failing_available = 0
    first_failing_correct = 0
    fix_available = 0
    fix_correct = 0
    root_available = 0
    root_correct = 0
    evidence_gap_counts: Counter[str] = Counter()
    reason_not_scored_counts: Counter[str] = Counter()
    family_pr = {
        family: {"tp": 0, "fp": 0, "fn": 0}
        for family in FAMILY_TARGETS
    }

    for record in records:
        diagnosis = engine.diagnose(_record_to_run(record))
        expected_primary = record["expected_primary_failure"]
        expected_stage = record["expected_stage"]
        expected_family = record["failure_family"]
        actual_primary = diagnosis.primary_failure.value
        actual_stage = diagnosis.root_cause_stage.value
        actual_family = _failure_to_family(actual_primary)
        expected_alternatives = set(record.get("acceptable_alternative_failures", []))

        primary_match = actual_primary == expected_primary
        stage_match = actual_stage == expected_stage
        selected_alternative_acceptable = (
            actual_primary in expected_alternatives and actual_primary != expected_primary
        )
        expected_candidate_generated = _expected_candidate_generated(diagnosis, expected_primary)
        evidence_diagnostics = _record_evidence_diagnostics(
            record,
            diagnosis,
            expected_candidate_generated=expected_candidate_generated,
        )
        safety_classification = _safety_classification(
            record,
            actual_primary=actual_primary,
            actual_stage=actual_stage,
            human_review_required=diagnosis.human_review_required(),
        )
        evidence_gap_counts.update(evidence_diagnostics["evidence_gap_flags"])
        reason_not_scored_counts.update(evidence_diagnostics["reason_not_scored"])

        primary_correct += int(primary_match)
        stage_correct += int(stage_match)
        expected_candidate_generated_count += int(expected_candidate_generated)
        expected_candidate_selected_count += int(primary_match)
        acceptable_alternative_match += int(selected_alternative_acceptable)
        wrong_stage_correct_failure += int(primary_match and not stage_match)
        correct_stage_wrong_failure += int(stage_match and not primary_match)
        false_clean_count += int(actual_primary == "CLEAN" and expected_primary != "CLEAN")
        false_security_count += int(actual_stage == "SECURITY" and expected_stage != "SECURITY")
        false_incomplete_count += int(
            actual_primary == "INCOMPLETE_DIAGNOSIS"
            and expected_primary != "INCOMPLETE_DIAGNOSIS"
        )
        dangerous_clean_miss_count += int(safety_classification["dangerous_clean_miss"])
        security_stage_miss_count += int(safety_classification["security_stage_miss"])
        acceptable_nonclean_human_review_count += int(
            safety_classification["acceptable_nonclean_human_review"]
        )
        human_review_miss_count += int(
            safety_classification["human_review_miss"]
        )

        if record.get("expected_first_failing_node") is not None:
            first_failing_available += 1
            first_failing_correct += int(
                diagnosis.first_failing_node == record["expected_first_failing_node"]
            )
        if record.get("expected_fix_category") is not None:
            fix_available += 1
            fix_correct += int(
                _normalize(record["expected_fix_category"])
                in _normalize(diagnosis.proposed_fix or diagnosis.recommended_fix)
            )
        if record.get("expected_root_cause") is not None:
            root_available += 1
            root_text = " ".join(
                value or ""
                for value in [
                    diagnosis.root_cause_attribution,
                    diagnosis.ncv_bottleneck_description,
                    " ".join(diagnosis.evidence[:3]),
                ]
            )
            root_correct += int(_root_cause_overlap(record["expected_root_cause"], root_text))

        family_totals[expected_family] += 1
        family_correct[expected_family] += int(primary_match)
        split_counts[record["calibration_split"]] += 1
        status_counts[record["calibration_status"]] += 1
        confusion_primary[f"{expected_primary} -> {actual_primary}"] += 1
        confusion_stage[f"{expected_stage} -> {actual_stage}"] += 1

        if expected_family in family_pr:
            if actual_family == expected_family:
                family_pr[expected_family]["tp"] += 1
            else:
                family_pr[expected_family]["fn"] += 1
        if actual_family in family_pr and actual_family != expected_family:
            family_pr[actual_family]["fp"] += 1

        per_case.append(
            {
                "case_id": record["case_id"],
                "source_case_id": record["source_case_id"],
                "failure_family": expected_family,
                "expected_primary_failure": expected_primary,
                "actual_primary_failure": actual_primary,
                "expected_stage": expected_stage,
                "actual_stage": actual_stage,
                "expected_failure_candidate_generated": expected_candidate_generated,
                "expected_candidate_selected": primary_match,
                "selected_alternative_acceptable": selected_alternative_acceptable,
                "actual_first_failing_node": diagnosis.first_failing_node,
                "expected_first_failing_node": record.get("expected_first_failing_node"),
                "human_review_required": diagnosis.human_review_required(),
                "expected_human_review_required": record.get("expected_human_review_required"),
                "evidence_diagnostics": evidence_diagnostics,
                "safety_classification": safety_classification,
            }
        )

    total = len(records)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "mode": mode,
        "limit": limit,
        "dataset_summary": {
            "total_cases": total,
            "cases_by_failure_family": dict(sorted(family_totals.items())),
            "cases_by_calibration_split": dict(sorted(split_counts.items())),
            "cases_by_calibration_status": dict(sorted(status_counts.items())),
            "seed_count": status_counts.get("seed", 0),
            "reviewed_count": status_counts.get("reviewed", 0),
            "adjudicated_count": status_counts.get("adjudicated", 0),
            "heldout_locked_count": status_counts.get("heldout_locked", 0),
        },
        "prediction_metrics": {
            "primary_failure_accuracy": _ratio(primary_correct, total),
            "stage_accuracy": _ratio(stage_correct, total),
            "first_failing_node_accuracy": _metric_or_unavailable(first_failing_correct, first_failing_available),
            "first_failing_node_available_count": first_failing_available,
            "fix_category_accuracy": _metric_or_unavailable(fix_correct, fix_available),
            "fix_category_available_count": fix_available,
            "root_cause_accuracy": _metric_or_unavailable(root_correct, root_available),
            "root_cause_available_count": root_available,
        },
        "safety_metrics": {
            "false_clean_count": false_clean_count,
            "false_security_count": false_security_count,
            "false_incomplete_count": false_incomplete_count,
            "dangerous_clean_miss_count": dangerous_clean_miss_count,
            "security_stage_miss_count": security_stage_miss_count,
            "acceptable_nonclean_human_review_count": acceptable_nonclean_human_review_count,
            "dangerous_miss_count": dangerous_clean_miss_count,
            "dangerous_miss_count_definition": (
                "Deprecated alias for dangerous_clean_miss_count: a security/privacy/"
                "adversarial/high-risk case returned CLEAN or failed to require human review."
            ),
            "human_review_miss_count": human_review_miss_count,
        },
        "per_family_metrics": _family_metrics(family_totals, family_correct, family_pr),
        "confusion_matrices": {
            "expected_primary_failure_vs_actual_primary_failure": dict(sorted(confusion_primary.items())),
            "expected_stage_vs_actual_stage": dict(sorted(confusion_stage.items())),
        },
        "decision_policy_metrics": {
            "expected_failure_candidate_generated_count": expected_candidate_generated_count,
            "expected_candidate_selected_count": expected_candidate_selected_count,
            "acceptable_alternative_match_count": acceptable_alternative_match,
            "wrong_stage_correct_failure_count": wrong_stage_correct_failure,
            "correct_stage_wrong_failure_count": correct_stage_wrong_failure,
        },
        "evidence_diagnostics_summary": {
            "evidence_gap_flag_counts": dict(sorted(evidence_gap_counts.items())),
            "reason_not_scored_counts": dict(sorted(reason_not_scored_counts.items())),
        },
        "calibration_status": {
            "calibration_status": "not_calibrated",
            "production_gating_eligible": False,
            "confidence_intervals_available": False,
            "heldout_split_locked": False,
        },
        "per_case": per_case,
    }
    return report


def _record_evidence_diagnostics(
    record: dict[str, Any],
    diagnosis: Any,
    *,
    expected_candidate_generated: bool,
) -> dict[str, Any]:
    claim_ids = {
        claim.get("claim_id")
        for claim in record.get("claims", [])
        if isinstance(claim, dict) and claim.get("claim_id")
    }
    retrieved_doc_ids = {
        chunk.get("doc_id")
        for chunk in record.get("retrieved_chunks", [])
        if isinstance(chunk, dict) and chunk.get("doc_id")
    }
    citation_doc_ids = {
        citation.get("doc_id")
        for citation in record.get("citations", [])
        if isinstance(citation, dict) and citation.get("doc_id")
    }
    all_doc_ids = retrieved_doc_ids | citation_doc_ids
    expected_claim_ids = set(record.get("expected_affected_claim_ids", []))
    expected_doc_ids = set(record.get("expected_affected_doc_ids", []))

    extracted_claim_count = _extracted_claim_count(diagnosis)
    skipped_claim_count = None
    if isinstance(extracted_claim_count, int):
        skipped_claim_count = max(len(record.get("claims", [])) - extracted_claim_count, 0)

    evidence_gap_flags: list[str] = []
    reason_not_scored: list[str] = []
    missing_expected_claim_ids = sorted(expected_claim_ids - claim_ids)
    missing_expected_doc_ids = sorted(expected_doc_ids - all_doc_ids)

    if missing_expected_claim_ids:
        evidence_gap_flags.append("missing_claim_ids")
    if missing_expected_doc_ids:
        evidence_gap_flags.append("missing_doc_ids")
    if not record.get("citations"):
        evidence_gap_flags.append("no_citations")
    if not record.get("retrieved_chunks"):
        evidence_gap_flags.append("no_retrieved_chunks")
    if not expected_claim_ids and not expected_doc_ids and record.get("expected_primary_failure") != "CLEAN":
        evidence_gap_flags.append("no_expected_affected_ids")
    if not isinstance(record.get("expected_human_review_required"), bool):
        evidence_gap_flags.append("missing_human_review_label")
    if record.get("expected_first_failing_node") is None:
        evidence_gap_flags.append("unsupported_optional_metric")
        reason_not_scored.append("first_failing_node_label_unavailable")
    if extracted_claim_count == 0 and record.get("claims"):
        reason_not_scored.append("diagnosis_extracted_no_claims")
    if not record.get("citations") and record.get("failure_family") == "citation":
        reason_not_scored.append("citation_family_without_citations")

    return {
        "expected_claim_count": len(record.get("claims", [])),
        "extracted_claim_count": extracted_claim_count,
        "skipped_claim_count": skipped_claim_count,
        "citation_count": len(record.get("citations", [])),
        "cited_doc_ids": sorted(citation_doc_ids),
        "missing_expected_claim_ids": missing_expected_claim_ids,
        "missing_expected_doc_ids": missing_expected_doc_ids,
        "diagnosis_has_claim_evidence": _diagnosis_has_claim_evidence(diagnosis),
        "diagnosis_has_citation_evidence": _diagnosis_has_citation_evidence(diagnosis),
        "expected_candidate_generated": expected_candidate_generated,
        "reason_not_scored": reason_not_scored,
        "evidence_gap_flags": evidence_gap_flags,
    }


def _safety_classification(
    record: dict[str, Any],
    *,
    actual_primary: str,
    actual_stage: str,
    human_review_required: bool,
) -> dict[str, bool | str]:
    expected_primary = record.get("expected_primary_failure")
    expected_stage = record.get("expected_stage")
    expected_review = record.get("expected_human_review_required") is True
    security_relevant = record.get("security_relevant") is True
    adversarial = record.get("adversarial") is True
    high_risk = security_relevant or adversarial or expected_stage == "SECURITY"
    non_clean = actual_primary != "CLEAN"
    false_clean = actual_primary == "CLEAN" and expected_primary != "CLEAN"
    human_review_miss = expected_review and not human_review_required
    dangerous_clean_miss = high_risk and (actual_primary == "CLEAN" or not human_review_required)
    security_stage_miss = (
        expected_stage == "SECURITY"
        and non_clean
        and human_review_required
        and actual_stage != "SECURITY"
    )
    acceptable_nonclean_human_review = (
        high_risk
        and expected_stage != "SECURITY"
        and non_clean
        and human_review_required
        and actual_stage != "SECURITY"
    )
    if dangerous_clean_miss:
        safety_outcome = "dangerous_clean_miss"
    elif security_stage_miss:
        safety_outcome = "security_stage_miss"
    elif acceptable_nonclean_human_review:
        safety_outcome = "acceptable_nonclean_human_review"
    elif human_review_miss:
        safety_outcome = "human_review_miss"
    else:
        safety_outcome = "not_safety_miss"

    return {
        "false_clean": false_clean,
        "dangerous_clean_miss": dangerous_clean_miss,
        "security_stage_miss": security_stage_miss,
        "acceptable_nonclean_human_review": acceptable_nonclean_human_review,
        "human_review_miss": human_review_miss,
        "safety_outcome": safety_outcome,
    }


def _extracted_claim_count(diagnosis: Any) -> int | None:
    for text in _diagnosis_evidence_strings(diagnosis):
        lowered = text.lower()
        if "no claims extracted" in lowered:
            return 0
        marker = "claim grounding summary: total="
        if marker in lowered:
            tail = lowered.split(marker, 1)[1]
            digits = []
            for char in tail:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return int("".join(digits))
    return None


def _diagnosis_has_claim_evidence(diagnosis: Any) -> bool | str:
    saw_skip = False
    for text in _diagnosis_evidence_strings(diagnosis):
        lowered = text.lower()
        if "claim grounding summary" in lowered or "claim verification methods" in lowered:
            return True
        if "no claims extracted" in lowered or "no claim evidence" in lowered:
            saw_skip = True
    return False if saw_skip else "unknown"


def _diagnosis_has_citation_evidence(diagnosis: Any) -> bool | str:
    saw_skip = False
    for text in _diagnosis_evidence_strings(diagnosis):
        lowered = text.lower()
        if "citation faithfulness summary" in lowered or "citation_missing" in lowered:
            return True
        if "no citations to verify" in lowered or "no claim evidence available for citation" in lowered:
            saw_skip = True
    return False if saw_skip else "unknown"


def _diagnosis_evidence_strings(diagnosis: Any) -> list[str]:
    values: list[str] = []
    for value in getattr(diagnosis, "evidence", []) or []:
        values.append(str(value))
    for result in getattr(diagnosis, "analyzer_results", []) or []:
        for value in getattr(result, "evidence", []) or []:
            values.append(str(value))
    return values


def _record_to_run(record: dict[str, Any]) -> RAGRun:
    chunks = [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            source_doc_id=chunk["doc_id"],
            score=chunk.get("score"),
            metadata=chunk.get("metadata", {}),
        )
        for chunk in record["retrieved_chunks"]
    ]
    return RAGRun(
        run_id=record["case_id"],
        query=record["query"],
        retrieved_chunks=chunks,
        final_answer=record["answer"],
        cited_doc_ids=[citation["doc_id"] for citation in record.get("citations", [])],
        metadata={
            "govrag_calib_case_id": record["case_id"],
            "source_suite": record["source_suite"],
            "source_case_id": record["source_case_id"],
        },
    )


def _expected_candidate_generated(diagnosis: Any, expected_primary: str) -> bool:
    return any(
        result.failure_type is not None and result.failure_type.value == expected_primary
        for result in diagnosis.analyzer_results
    )


def _failure_to_family(failure: str) -> str:
    if failure == "CLEAN":
        return "clean_pass"
    if failure in {
        "RETRIEVAL_ANOMALY",
        "RETRIEVAL_DEPTH_LIMIT",
        "SCOPE_VIOLATION",
        "INCONSISTENT_CHUNKS",
        "EMBEDDING_DRIFT",
        "RERANKER_FAILURE",
    }:
        return "retrieval"
    if failure in {"UNSUPPORTED_CLAIM", "CONTRADICTED_CLAIM"}:
        return "grounding"
    if failure in {"CITATION_MISMATCH", "POST_RATIONALIZED_CITATION"}:
        return "citation"
    if failure == "INSUFFICIENT_CONTEXT":
        return "sufficiency"
    if failure == "STALE_RETRIEVAL":
        return "version_validity"
    if failure in {"PROMPT_INJECTION", "PRIVACY_VIOLATION", "SUSPICIOUS_CHUNK"}:
        return "security_privacy"
    return "answer_quality"


def _family_metrics(
    family_totals: Counter[str],
    family_correct: Counter[str],
    family_pr: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    metrics = {}
    for family in sorted(FAMILY_TARGETS):
        pr = family_pr[family]
        precision = _ratio(pr["tp"], pr["tp"] + pr["fp"])
        recall = _ratio(pr["tp"], pr["tp"] + pr["fn"])
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metrics[family] = {
            "total": family_totals.get(family, 0),
            "accuracy": _ratio(family_correct.get(family, 0), family_totals.get(family, 0)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_or_unavailable(correct: int, total: int) -> float | str:
    if total == 0:
        return "unavailable"
    return correct / total


def _normalize(value: str | None) -> str:
    return (value or "").lower().replace("_", " ").replace("-", " ")


def _root_cause_overlap(expected: str, actual: str) -> bool:
    expected_terms = {term for term in _normalize(expected).split() if len(term) >= 5}
    actual_terms = set(_normalize(actual).split())
    return bool(expected_terms & actual_terms)


def write_eval_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  JSON report -> {path}")


def write_eval_markdown_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = report["prediction_metrics"]
    safety = report["safety_metrics"]
    calibration = report["calibration_status"]
    diagnostics = report.get("evidence_diagnostics_summary", {})
    evidence_gap_counts = diagnostics.get("evidence_gap_flag_counts", {})
    reason_not_scored_counts = diagnostics.get("reason_not_scored_counts", {})
    lines = [
        "# GovRAG-Calib Evaluation Report",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Mode: `{report['mode']}`",
        f"- Total cases: `{report['dataset_summary']['total_cases']}`",
        "",
        "## Calibration Status",
        "",
        f"- `calibration_status`: `{calibration['calibration_status']}`",
        f"- `production_gating_eligible`: `{calibration['production_gating_eligible']}`",
        f"- `confidence_intervals_available`: `{calibration['confidence_intervals_available']}`",
        f"- `heldout_split_locked`: `{calibration['heldout_split_locked']}`",
        "",
        "## Prediction Metrics",
        "",
        f"- `primary_failure_accuracy`: `{metrics['primary_failure_accuracy']}`",
        f"- `stage_accuracy`: `{metrics['stage_accuracy']}`",
        f"- `first_failing_node_accuracy`: `{metrics['first_failing_node_accuracy']}`",
        f"- `fix_category_accuracy`: `{metrics['fix_category_accuracy']}`",
        f"- `root_cause_accuracy`: `{metrics['root_cause_accuracy']}`",
        "",
        "## Safety Metrics",
        "",
        f"- `false_clean_count`: `{safety['false_clean_count']}`",
        f"- `false_security_count`: `{safety['false_security_count']}`",
        f"- `false_incomplete_count`: `{safety['false_incomplete_count']}`",
        f"- `dangerous_clean_miss_count`: `{safety['dangerous_clean_miss_count']}`",
        f"- `security_stage_miss_count`: `{safety['security_stage_miss_count']}`",
        f"- `acceptable_nonclean_human_review_count`: `{safety['acceptable_nonclean_human_review_count']}`",
        f"- `dangerous_miss_count`: `{safety['dangerous_miss_count']}`",
        f"- `dangerous_miss_count_definition`: {safety['dangerous_miss_count_definition']}",
        f"- `human_review_miss_count`: `{safety['human_review_miss_count']}`",
        "",
        "## Family Metrics",
        "",
        "| Family | Total | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, family_metrics in report["per_family_metrics"].items():
        lines.append(
            "| {family} | {total} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |".format(
                family=family,
                total=family_metrics["total"],
                accuracy=family_metrics["accuracy"],
                precision=family_metrics["precision"],
                recall=family_metrics["recall"],
                f1=family_metrics["f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision Policy Metrics",
            "",
            f"- `expected_failure_candidate_generated_count`: `{report['decision_policy_metrics']['expected_failure_candidate_generated_count']}`",
            f"- `expected_candidate_selected_count`: `{report['decision_policy_metrics']['expected_candidate_selected_count']}`",
            f"- `acceptable_alternative_match_count`: `{report['decision_policy_metrics']['acceptable_alternative_match_count']}`",
            f"- `wrong_stage_correct_failure_count`: `{report['decision_policy_metrics']['wrong_stage_correct_failure_count']}`",
            f"- `correct_stage_wrong_failure_count`: `{report['decision_policy_metrics']['correct_stage_wrong_failure_count']}`",
            "",
            "## Evidence Diagnostics",
            "",
            "### Evidence Gap Flags",
            "",
            "| Flag | Count |",
            "| --- | ---: |",
        ]
    )
    if evidence_gap_counts:
        for flag, count in evidence_gap_counts.items():
            lines.append(f"| `{flag}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "### Reasons Not Scored",
            "",
            "| Reason | Count |",
            "| --- | ---: |",
        ]
    )
    if reason_not_scored_counts:
        for reason, count in reason_not_scored_counts.items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "_This report is evaluation-only. It does not claim calibrated confidence or production gating eligibility._",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Markdown report -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GovRAG-Calib dataset readiness or run GovRAG-Calib evaluator v0.")
    parser.add_argument("dataset_path", nargs="?", type=Path)
    parser.add_argument("--dataset", type=Path, default=None, help="Legacy readiness dataset path.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["native", "external-enhanced"], default="native")
    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help="Run the legacy dataset readiness evaluator instead of GovRAG diagnosis evaluation.",
    )
    args = parser.parse_args()

    path: Path = args.dataset_path or args.dataset or DATASET_PATH
    if not path.exists():
        print(f"ERROR: Dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    if args.dataset_path is not None and not args.readiness_only:
        validation = validate_dataset(path)
        if not validation.ok:
            for error in validation.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(1)

        output_json = args.output or DEFAULT_EVAL_JSON
        output_md = output_json.with_suffix(".md")
        print(f"Evaluating GovRAG-Calib: {path}")
        report = evaluate_calib_dataset(path, mode=args.mode, limit=args.limit)
        write_eval_json_report(report, output_json)
        write_eval_markdown_report(report, output_md)
        print("Results:")
        print(f"  Total cases: {report['dataset_summary']['total_cases']}")
        print(f"  Primary accuracy: {report['prediction_metrics']['primary_failure_accuracy']:.3f}")
        print(f"  Stage accuracy: {report['prediction_metrics']['stage_accuracy']:.3f}")
        print(f"  Calibration status: {report['calibration_status']['calibration_status']}")
        print(f"  Prod gating: {report['calibration_status']['production_gating_eligible']}")
        return

    print(f"Evaluating readiness: {path}")
    result = evaluate_readiness(path)

    print(f"\nResults:")
    print(f"  Total cases:      {result['total_cases']}")
    print(f"  Complete cases:   {result['complete_cases']}")
    print(f"  Placeholder:      {result['placeholder_cases']}")
    print(f"  Readiness:        {result['readiness_status']}")
    print(f"  Prod gating:      {result['production_gating_eligible']}")

    write_json_report(result, REPORTS_DIR / "govrag_calib_readiness.json")
    write_markdown_report(result, REPORTS_DIR / "govrag_calib_readiness.md")

    print(f"\n  {READINESS_ICONS.get(result['readiness_status'], '⚪')} {result['readiness_status']}")
    print(f"  {result['readiness_reason']}")


if __name__ == "__main__":
    main()
