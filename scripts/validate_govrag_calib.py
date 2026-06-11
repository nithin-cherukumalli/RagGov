"""Validate GovRAG-Calib-150 JSONL records.

This script validates dataset structure only. It does not run analyzers, alter
thresholds, or produce calibration claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "case_id",
    "domain",
    "source_suite",
    "source_case_id",
    "query",
    "retrieved_chunks",
    "answer",
    "claims",
    "citations",
    "expected_primary_failure",
    "expected_stage",
    "expected_first_failing_node",
    "expected_root_cause",
    "expected_fix_category",
    "expected_affected_claim_ids",
    "expected_affected_doc_ids",
    "expected_human_review_required",
    "acceptable_alternative_failures",
    "failure_family",
    "difficulty",
    "adversarial",
    "security_relevant",
    "metadata_requirements",
    "labeler",
    "label_status",
    "label_confidence",
    "notes",
    "calibration_split",
    "calibration_status",
    "provenance",
}

FAILURE_LABELS = {
    "CLEAN",
    "CITATION_MISMATCH",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTED_CLAIM",
    "STALE_RETRIEVAL",
    "SCOPE_VIOLATION",
    "PROMPT_INJECTION",
    "INSUFFICIENT_CONTEXT",
    "RETRIEVAL_ANOMALY",
    "RETRIEVAL_DEPTH_LIMIT",
    "INCONSISTENT_CHUNKS",
    "POST_RATIONALIZED_CITATION",
    "LOW_CONFIDENCE",
    "PRIVACY_VIOLATION",
    "SUSPICIOUS_CHUNK",
    "TABLE_STRUCTURE_LOSS",
    "HIERARCHY_FLATTENING",
    "METADATA_LOSS",
    "PARSER_STRUCTURE_LOSS",
    "CHUNKING_BOUNDARY_ERROR",
    "EMBEDDING_DRIFT",
    "RERANKER_FAILURE",
    "GENERATION_IGNORE",
    "INCOMPLETE_DIAGNOSIS",
}

STAGE_LABELS = {
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "RETRIEVAL",
    "RERANKING",
    "CONTEXT_ASSEMBLY",
    "GROUNDING",
    "SUFFICIENCY",
    "GENERATION",
    "CITATION",
    "SECURITY",
    "CONFIDENCE",
    "UNKNOWN",
}

FAMILY_TARGETS = {
    "clean_pass": 15,
    "retrieval": 25,
    "grounding": 25,
    "citation": 20,
    "sufficiency": 15,
    "version_validity": 15,
    "security_privacy": 20,
    "answer_quality": 15,
}

DIFFICULTIES = {"low", "medium", "high"}
CALIBRATION_SPLITS = {"train", "calibration", "heldout", "unset"}
CALIBRATION_STATUSES = {"seed", "reviewed", "adjudicated", "heldout_locked"}
LABEL_STATUSES = {"seed", "reviewed", "adjudicated"}
LABEL_CONFIDENCES = {"low", "medium", "high"}
PROVENANCE_FIELDS = {"source", "created_at", "derived_from", "transformation_notes"}


@dataclass
class ValidationResult:
    path: Path
    total_cases: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    family_counts: Counter[str] = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(path: Path) -> ValidationResult:
    result = ValidationResult(path=path)
    case_ids: list[str] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            case = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            result.errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(case, dict):
            result.errors.append(f"line {line_number}: record must be a JSON object")
            continue

        result.total_cases += 1
        case_id = str(case.get("case_id", f"line-{line_number}"))
        case_ids.append(case_id)
        _validate_case(case, line_number, result)

    duplicates = [case_id for case_id, count in Counter(case_ids).items() if count > 1]
    for case_id in sorted(duplicates):
        result.errors.append(f"case_id {case_id!r}: duplicate case_id")

    if result.total_cases < 150:
        result.warnings.append(
            f"dataset has {result.total_cases} cases; GovRAG-Calib-150 target is 150"
        )
    for family, target in FAMILY_TARGETS.items():
        count = result.family_counts.get(family, 0)
        if count < target:
            result.warnings.append(
                f"family {family!r} has {count} cases; target is {target}"
            )

    return result


def _validate_case(case: dict[str, Any], line_number: int, result: ValidationResult) -> None:
    prefix = f"line {line_number} case {case.get('case_id', '<missing>')!r}"

    if _contains_key(case, "production_gating_eligible"):
        result.errors.append(f"{prefix}: production_gating_eligible must not appear")

    missing = sorted(REQUIRED_FIELDS - set(case))
    if missing:
        result.errors.append(f"{prefix}: missing required fields: {', '.join(missing)}")
        return

    _check_enum(case["expected_primary_failure"], FAILURE_LABELS, "expected_primary_failure", prefix, result)
    _check_enum(case["expected_stage"], STAGE_LABELS, "expected_stage", prefix, result)
    _check_enum(case["failure_family"], set(FAMILY_TARGETS), "failure_family", prefix, result)
    _check_enum(case["difficulty"], DIFFICULTIES, "difficulty", prefix, result)
    _check_enum(case["calibration_split"], CALIBRATION_SPLITS, "calibration_split", prefix, result)
    _check_enum(case["calibration_status"], CALIBRATION_STATUSES, "calibration_status", prefix, result)
    _check_enum(case["label_status"], LABEL_STATUSES, "label_status", prefix, result)
    _check_enum(case["label_confidence"], LABEL_CONFIDENCES, "label_confidence", prefix, result)

    if case["calibration_split"] == "heldout" and case["calibration_status"] != "heldout_locked":
        result.errors.append(f"{prefix}: heldout cases require calibration_status='heldout_locked'")

    for field_name in (
        "retrieved_chunks",
        "claims",
        "citations",
        "expected_affected_claim_ids",
        "expected_affected_doc_ids",
        "acceptable_alternative_failures",
        "metadata_requirements",
    ):
        if not isinstance(case[field_name], list):
            result.errors.append(f"{prefix}: {field_name} must be a list")

    if not isinstance(case["expected_human_review_required"], bool):
        result.errors.append(f"{prefix}: expected_human_review_required must be boolean")
    if not isinstance(case["adversarial"], bool):
        result.errors.append(f"{prefix}: adversarial must be boolean")
    if not isinstance(case["security_relevant"], bool):
        result.errors.append(f"{prefix}: security_relevant must be boolean")

    if isinstance(case["acceptable_alternative_failures"], list):
        for failure in case["acceptable_alternative_failures"]:
            _check_enum(failure, FAILURE_LABELS, "acceptable_alternative_failures", prefix, result)

    chunk_ids, doc_ids = _collect_chunk_and_doc_ids(case, prefix, result)
    claim_ids = _collect_claim_ids(case, prefix, result)
    citation_doc_ids, citation_chunk_ids = _collect_citation_ids(case, prefix, result)
    doc_ids.update(citation_doc_ids)

    for claim_id in case.get("expected_affected_claim_ids", []):
        if claim_id not in claim_ids:
            result.errors.append(f"{prefix}: affected claim_id {claim_id!r} does not exist")
    for doc_id in case.get("expected_affected_doc_ids", []):
        if doc_id not in doc_ids:
            result.errors.append(f"{prefix}: affected doc_id {doc_id!r} does not exist")
    for chunk_id in citation_chunk_ids:
        if chunk_id is not None and chunk_id not in chunk_ids:
            result.errors.append(f"{prefix}: citation chunk_id {chunk_id!r} does not exist")

    _add_evidence_gap_warnings(case, prefix, result)

    provenance = case.get("provenance")
    if not isinstance(provenance, dict):
        result.errors.append(f"{prefix}: provenance must be an object")
    else:
        missing_provenance = sorted(PROVENANCE_FIELDS - set(provenance))
        if missing_provenance:
            result.errors.append(
                f"{prefix}: provenance missing fields: {', '.join(missing_provenance)}"
            )

    if case["failure_family"] in FAMILY_TARGETS:
        result.family_counts[case["failure_family"]] += 1


def _add_evidence_gap_warnings(
    case: dict[str, Any],
    prefix: str,
    result: ValidationResult,
) -> None:
    primary_failure = case.get("expected_primary_failure")
    family = case.get("failure_family")
    notes = str(case.get("notes") or "").lower()

    if primary_failure != "CLEAN" and not case.get("expected_affected_claim_ids"):
        result.warnings.append(
            f"{prefix}: non-clean case has empty expected_affected_claim_ids"
        )

    doc_required_families = {
        "citation",
        "retrieval",
        "grounding",
        "sufficiency",
        "version_validity",
    }
    if (
        family in doc_required_families
        and case.get("retrieved_chunks")
        and not case.get("expected_affected_doc_ids")
    ):
        result.warnings.append(
            f"{prefix}: {family} case has retrieved_chunks but empty expected_affected_doc_ids"
        )

    if family == "citation" and not case.get("citations"):
        result.warnings.append(f"{prefix}: citation-family case has no citations")

    if (
        family == "version_validity" or primary_failure == "STALE_RETRIEVAL"
    ) and not case.get("metadata_requirements"):
        result.warnings.append(
            f"{prefix}: version/staleness case has empty metadata_requirements"
        )

    if (
        (case.get("security_relevant") is True or case.get("adversarial") is True)
        and case.get("difficulty") == "high"
        and case.get("expected_human_review_required") is False
    ):
        result.warnings.append(
            f"{prefix}: high-risk security/adversarial case does not require human review"
        )

    ambiguous_families = {"retrieval", "sufficiency", "grounding", "answer_quality"}
    if (
        family in ambiguous_families
        and "ambig" in notes
        and not case.get("acceptable_alternative_failures")
    ):
        result.warnings.append(
            f"{prefix}: ambiguous {family} case has empty acceptable_alternative_failures"
        )


def _check_enum(
    value: Any,
    allowed: set[str],
    field_name: str,
    prefix: str,
    result: ValidationResult,
) -> None:
    if value not in allowed:
        result.errors.append(f"{prefix}: invalid {field_name}={value!r}")


def _collect_chunk_and_doc_ids(
    case: dict[str, Any],
    prefix: str,
    result: ValidationResult,
) -> tuple[set[str], set[str]]:
    chunk_ids: set[str] = set()
    doc_ids: set[str] = set()
    chunks = case.get("retrieved_chunks", [])
    if not isinstance(chunks, list):
        return chunk_ids, doc_ids
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            result.errors.append(f"{prefix}: retrieved_chunks[{index}] must be an object")
            continue
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        text = chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            result.errors.append(f"{prefix}: retrieved_chunks[{index}].chunk_id is required")
        elif chunk_id in chunk_ids:
            result.errors.append(f"{prefix}: duplicate chunk_id {chunk_id!r}")
        else:
            chunk_ids.add(chunk_id)
        if not isinstance(doc_id, str) or not doc_id:
            result.errors.append(f"{prefix}: retrieved_chunks[{index}].doc_id is required")
        else:
            doc_ids.add(doc_id)
        if not isinstance(text, str) or not text:
            result.errors.append(f"{prefix}: retrieved_chunks[{index}].text is required")
    return chunk_ids, doc_ids


def _collect_claim_ids(case: dict[str, Any], prefix: str, result: ValidationResult) -> set[str]:
    claim_ids: set[str] = set()
    claims = case.get("claims", [])
    if not isinstance(claims, list):
        return claim_ids
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            result.errors.append(f"{prefix}: claims[{index}] must be an object")
            continue
        claim_id = claim.get("claim_id")
        text = claim.get("text")
        if not isinstance(claim_id, str) or not claim_id:
            result.errors.append(f"{prefix}: claims[{index}].claim_id is required")
        elif claim_id in claim_ids:
            result.errors.append(f"{prefix}: duplicate claim_id {claim_id!r}")
        else:
            claim_ids.add(claim_id)
        if not isinstance(text, str) or not text:
            result.errors.append(f"{prefix}: claims[{index}].text is required")
    return claim_ids


def _collect_citation_ids(
    case: dict[str, Any],
    prefix: str,
    result: ValidationResult,
) -> tuple[set[str], set[str | None]]:
    doc_ids: set[str] = set()
    chunk_ids: set[str | None] = set()
    citation_ids: set[str] = set()
    citations = case.get("citations", [])
    if not isinstance(citations, list):
        return doc_ids, chunk_ids
    for index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            result.errors.append(f"{prefix}: citations[{index}] must be an object")
            continue
        citation_id = citation.get("citation_id")
        doc_id = citation.get("doc_id")
        chunk_id = citation.get("chunk_id")
        if not isinstance(citation_id, str) or not citation_id:
            result.errors.append(f"{prefix}: citations[{index}].citation_id is required")
        elif citation_id in citation_ids:
            result.errors.append(f"{prefix}: duplicate citation_id {citation_id!r}")
        else:
            citation_ids.add(citation_id)
        if not isinstance(doc_id, str) or not doc_id:
            result.errors.append(f"{prefix}: citations[{index}].doc_id is required")
        else:
            doc_ids.add(doc_id)
        if chunk_id is not None:
            if not isinstance(chunk_id, str) or not chunk_id:
                result.errors.append(f"{prefix}: citations[{index}].chunk_id must be string or null")
            else:
                chunk_ids.add(chunk_id)
    return doc_ids, chunk_ids


def _contains_key(value: Any, target_key: str) -> bool:
    if isinstance(value, dict):
        return any(key == target_key or _contains_key(child, target_key) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, target_key) for child in value)
    return False


def print_result(result: ValidationResult) -> None:
    print(f"GovRAG-Calib validation: {result.path}")
    print(f"cases: {result.total_cases}")
    print("family distribution:")
    for family, target in FAMILY_TARGETS.items():
        print(f"- {family}: {result.family_counts.get(family, 0)}/{target}")

    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"- {error}")
    print("status: passed" if result.ok else "status: failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)

    result = validate_dataset(args.dataset)
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
