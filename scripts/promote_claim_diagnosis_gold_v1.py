"""Promote legacy hard stress cases into claim-diagnosis gold v1."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "stresslab" / "cases" / "golden"
DIAGNOSIS_FIXTURE_DIR = ROOT / "stresslab" / "cases" / "diagnosis_fixtures"
ARCHIVE_DIR = ROOT / "stresslab" / "archive"
LEGACY_GOLD_SOURCE = GOLDEN_DIR / "golden_set_v1.json"
ARCHIVED_GOLD_SOURCE = ARCHIVE_DIR / "cases" / "golden_set_v1_source.json"


def main() -> None:
    v0 = json.loads((GOLDEN_DIR / "claim_diagnosis_gold_v0.json").read_text(encoding="utf-8"))
    hard_source = LEGACY_GOLD_SOURCE if LEGACY_GOLD_SOURCE.exists() else ARCHIVED_GOLD_SOURCE
    hard_set = json.loads(hard_source.read_text(encoding="utf-8"))

    examples = []
    seen_case_ids: set[str] = set()

    for example in v0["examples"]:
        enriched = dict(example)
        enriched.setdefault("category", _category_for_v0_case(example["case_id"]))
        if "expected_primary_stage" in enriched and "expected_stage" not in enriched:
            enriched["expected_stage"] = enriched.pop("expected_primary_stage")
        seen_case_ids.add(enriched["case_id"])
        examples.append(enriched)

    for item in hard_set["items"]:
        case_id = f"hard_{item['gold_id']}"
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        examples.append(_convert_hard_item(item))

    for fixture_name in ("prompt_injection", "poisoned_chunk"):
        case = _security_case(fixture_name)
        if case["case_id"] not in seen_case_ids:
            seen_case_ids.add(case["case_id"])
            examples.append(case)

    payload = {
        "evaluation_status": "diagnostic_gold_v0_small_unvalidated",
        "examples": examples,
    }
    output_path = GOLDEN_DIR / "claim_diagnosis_gold_v1.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    _archive_legacy_assets()


def _convert_hard_item(item: dict[str, object]) -> dict[str, object]:
    support = item.get("supporting_evidence", [])
    required_doc_ids = item.get("required_doc_ids") or []
    source_docs = item.get("source_docs") or []
    required_sections = item.get("required_sections") or []

    cited_doc_ids = list(required_doc_ids) or list(source_docs)
    primary_doc_id = cited_doc_ids[0] if cited_doc_ids else "unknown-doc"
    final_answer = item.get("gold_short_answer") or item.get("gold_answer") or "No answer provided."

    retrieved_chunks = []
    corpus_entries = []
    for index, evidence in enumerate(support):
        if not isinstance(evidence, dict):
            continue
        doc_id = evidence.get("doc_id") or primary_doc_id
        text = evidence.get("quoted_fact_summary") or str(final_answer)
        section_label = evidence.get("section_label")
        page = evidence.get("page")
        metadata = {}
        if section_label is not None:
            metadata["section_label"] = section_label
        if page is not None:
            metadata["page"] = page
        retrieved_chunks.append(
            {
                "chunk_id": f"{item['gold_id']}-chunk-{index + 1}",
                "text": text,
                "source_doc_id": doc_id,
                "score": round(max(0.5, 0.93 - (index * 0.03)), 2),
                "metadata": metadata,
            }
        )
        corpus_entries.append(
            {
                "doc_id": doc_id,
                "text": text,
                "timestamp": "2011-01-01T00:00:00Z",
            }
        )

    if not retrieved_chunks:
        retrieved_chunks.append(
            {
                "chunk_id": f"{item['gold_id']}-chunk-1",
                "text": str(final_answer),
                "source_doc_id": primary_doc_id,
                "score": 0.75,
                "metadata": {"fallback": True},
            }
        )
        corpus_entries.append(
            {
                "doc_id": primary_doc_id,
                "text": str(final_answer),
                "timestamp": "2011-01-01T00:00:00Z",
            }
        )

    return {
        "case_id": f"hard_{item['gold_id']}",
        "category": _category_for_hard_item(item),
        "query": item["question"],
        "retrieved_chunks": retrieved_chunks,
        "final_answer": final_answer,
        "expected_claims": [
            {
                "claim_text": final_answer,
                "expected_claim_label": "entailed",
                "expected_citation_validity": "valid" if cited_doc_ids else "not_applicable",
                "expected_freshness_validity": "unknown",
                "expected_a2p_primary_cause": "none",
            }
        ],
        "expected_sufficiency": True,
        "expected_stage": "UNKNOWN",
        "expected_fix_category": "other",
        "cited_doc_ids": cited_doc_ids,
        "corpus_entries": corpus_entries,
        "metadata": {
            "source": "legacy_golden_set_v1",
            "gold_id": item["gold_id"],
            "question_type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
            "required_sections": required_sections,
            "likely_primary_failure_if_wrong": item.get("likely_primary_failure_if_wrong"),
        },
    }


def _security_case(fixture_name: str) -> dict[str, object]:
    fixture_meta = json.loads((DIAGNOSIS_FIXTURE_DIR / f"{fixture_name}.json").read_text(encoding="utf-8"))
    run_payload = json.loads((ROOT / fixture_meta["run_fixture"]).read_text(encoding="utf-8"))
    return {
        "case_id": f"security_{fixture_name}",
        "category": "security",
        "query": run_payload["query"],
        "retrieved_chunks": run_payload.get("retrieved_chunks", []),
        "final_answer": run_payload["final_answer"],
        "expected_claims": [
            {
                "claim_text": run_payload["final_answer"],
                "expected_claim_label": "entailed" if fixture_name == "prompt_injection" else "unsupported",
                "expected_citation_validity": "valid",
                "expected_freshness_validity": "unknown",
                "expected_a2p_primary_cause": "none",
            }
        ],
        "expected_sufficiency": True,
        "expected_stage": "SECURITY",
        "expected_fix_category": "security",
        "cited_doc_ids": run_payload.get("cited_doc_ids", []),
        "corpus_entries": run_payload.get("corpus_entries", []),
        "metadata": {
            "source": "diagnosis_fixture",
            "fixture_name": fixture_name,
            "expected_primary_failure": fixture_meta.get("expected_primary_failure"),
        },
    }


def _category_for_v0_case(case_id: str) -> str:
    mapping = {
        "supported_1": "grounding",
        "supported_2": "grounding",
        "unsupported_missing_1": "sufficiency",
        "unsupported_missing_2": "robustness/missing-data",
        "contradicted_1": "A2P",
        "contradicted_2": "A2P",
        "stale_source_case": "freshness",
        "citation_mismatch_case": "citation",
        "weak_ambiguous_case": "grounding",
        "insufficient_context_abstain_case": "sufficiency",
    }
    return mapping.get(case_id, "grounding")


def _category_for_hard_item(item: dict[str, object]) -> str:
    question_type = str(item.get("question_type", ""))
    likely_failure = str(item.get("likely_primary_failure_if_wrong", ""))
    if question_type == "abstention" or likely_failure == "SUFFICIENCY":
        return "sufficiency"
    if likely_failure == "GROUNDING":
        return "grounding"
    if likely_failure in {"PARSING", "CHUNKING", "EMBEDDING", "RETRIEVAL"}:
        return "robustness/missing-data"
    return "grounding"


def _archive_legacy_assets() -> None:
    archive_cases = ARCHIVE_DIR / "cases"
    archive_reports = ARCHIVE_DIR / "reports"
    archive_cases.mkdir(parents=True, exist_ok=True)
    archive_reports.mkdir(parents=True, exist_ok=True)

    for src, dst in (
        (GOLDEN_DIR / "golden_set_v1.json", archive_cases / "golden_set_v1_source.json"),
        (GOLDEN_DIR / "ares_calibration_report_v1.json", archive_cases / "ares_calibration_report_v1.json"),
        (GOLDEN_DIR / "ares_calibration_samples_v1.jsonl", archive_cases / "ares_calibration_samples_v1.jsonl"),
        (ROOT / "stresslab" / "reports" / "baseline_validation_v1.json", archive_reports / "baseline_validation_v1.json"),
    ):
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))


if __name__ == "__main__":
    main()
