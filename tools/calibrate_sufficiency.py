#!/usr/bin/env python
"""Calibrate sufficiency analyzer modes against a JSONL gold set."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raggov.analyzers.sufficiency.sufficiency import SufficiencyAnalyzer
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun


LABELS = ("sufficient", "partial", "insufficient", "unknown")
LOGGER = logging.getLogger("calibrate_sufficiency")


@dataclass(frozen=True)
class GoldExample:
    payload: dict[str, Any]

    @property
    def example_id(self) -> str:
        return str(self.payload["example_id"])

    @property
    def gold_label(self) -> str:
        return str(self.payload["gold_sufficiency_label"])

    @property
    def gold_should_abstain(self) -> bool | None:
        value = self.payload.get("gold_should_abstain")
        return value if isinstance(value, bool) else None


class MockSufficiencyLLM:
    """Deterministic mock that returns reasonable requirements based on query keywords."""

    def chat(self, prompt: str) -> str:
        query = self._query_from_prompt(prompt)
        return json.dumps({"required_evidence": self.extract_requirements(query)})

    def complete(self, prompt: str) -> str:
        return self.chat(prompt)

    def extract_requirements(self, query: str) -> list[dict]:
        """Returns 2-4 evidence requirements based on query content."""
        lowered = query.lower()
        requirements: list[dict] = []

        if any(word in lowered for word in ["rule", "policy", "go", "circular"]):
            requirements.append({
                "description": "The specific government order or circular that establishes the rule",
                "type": "citation",
                "importance": "critical",
            })

        if any(word in lowered for word in ["school", "education", "student", "teacher"]):
            requirements.append({
                "description": "School Education Department applicability or academic calendar reference",
                "type": "scope",
                "importance": "critical",
            })

        if any(word in lowered for word in ["holiday", "holidays"]):
            requirements.append({
                "description": "The applicable holiday list, holiday dates, or holiday limit",
                "type": "date",
                "importance": "critical",
            })

        if any(word in lowered for word in ["date", "effective", "when", "deadline"]):
            requirements.append({
                "description": "The effective date or implementation timeline",
                "type": "date",
                "importance": "critical",
            })

        if any(word in lowered for word in ["who", "authority", "issued", "competent"]):
            requirements.append({
                "description": "The issuing authority or competent authority designation",
                "type": "authority",
                "importance": "critical",
            })

        if any(word in lowered for word in ["exception", "exempt", "special case", "not applicable"]):
            requirements.append({
                "description": "Any exceptions, exemptions, or special cases",
                "type": "exception",
                "importance": "supporting",
            })

        if any(word in lowered for word in ["supersed", "replace", "old", "new", "current"]):
            requirements.append({
                "description": "Whether this rule supersedes or is superseded by another order",
                "type": "supersession",
                "importance": "supporting",
            })

        if not requirements:
            requirements.append({
                "description": "The relevant rule, policy, or factual information needed to answer",
                "type": "rule",
                "importance": "critical",
            })

        return requirements[:4]

    def _query_from_prompt(self, prompt: str) -> str:
        match = re.search(r"^Query:\s*(.+)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--llm",
        choices=("mock", "openai", "anthropic", "vertexai"),
        default=None,
        help="Optional LLM client for requirement-aware sufficiency calibration.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    llm_client = build_llm_client(args.llm)
    examples, skipped = load_gold_set(args.gold_set)
    per_example = evaluate_examples(examples, llm_client=llm_client)
    report = build_report(args.gold_set, examples, skipped, per_example)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)


def build_llm_client(name: str | None) -> object | None:
    if name is None:
        return None
    if name == "mock":
        return MockSufficiencyLLM()

    try:
        from raggov.llm import build_llm_client as project_build_llm_client
    except ImportError as exc:
        raise RuntimeError(
            f"LLM provider '{name}' requested, but this GovRAG checkout does not expose "
            "raggov.llm.build_llm_client. Configure an existing project LLM client first."
        ) from exc

    return project_build_llm_client(name)


def load_gold_set(path: Path) -> tuple[list[GoldExample], list[dict[str, str]]]:
    examples: list[GoldExample] = []
    skipped: list[dict[str, str]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            skipped.append({"line": str(line_number), "reason": f"bad_json: {exc}"})
            LOGGER.warning("Skipping line %s: bad JSON", line_number)
            continue

        missing = [
            field
            for field in ("example_id", "query", "retrieved_chunks", "gold_sufficiency_label")
            if field not in payload
        ]
        if missing:
            skipped.append({"line": str(line_number), "reason": f"missing_required: {missing}"})
            LOGGER.warning("Skipping line %s: missing required fields %s", line_number, missing)
            continue

        if payload["gold_sufficiency_label"] not in LABELS:
            skipped.append({"line": str(line_number), "reason": "invalid_gold_sufficiency_label"})
            LOGGER.warning("Skipping line %s: invalid gold_sufficiency_label", line_number)
            continue

        if not payload["retrieved_chunks"]:
            skipped.append({"line": str(line_number), "reason": "empty_retrieved_chunks"})
            LOGGER.warning("Skipping line %s: empty retrieved_chunks", line_number)
            continue

        examples.append(GoldExample(payload=payload))

    return examples, skipped


def evaluate_examples(
    examples: list[GoldExample],
    llm_client: object | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for example in examples:
        rag_run = to_rag_run(example)
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "gold_label": example.gold_label,
        }

        for mode in ("term_coverage", "requirement_aware"):
            try:
                config: dict[str, Any] = {"sufficiency_mode": mode}
                if mode == "requirement_aware" and llm_client is not None:
                    config["llm_client"] = llm_client
                analyzer = SufficiencyAnalyzer(config)
                result = analyzer.analyze(rag_run)
                sufficiency = result.sufficiency_result
                prediction = sufficiency.sufficiency_label if sufficiency else "unknown"
                row[f"{mode}_prediction"] = prediction
                row[f"{mode}_correct"] = prediction == example.gold_label
                row[f"{mode}_details"] = (
                    sufficiency.model_dump(mode="json") if sufficiency else {}
                )
            except Exception as exc:
                LOGGER.error(
                    "Mode %s failed for %s: %s",
                    mode,
                    example.example_id,
                    exc,
                )
                row[f"{mode}_prediction"] = "unknown"
                row[f"{mode}_correct"] = False
                row[f"{mode}_details"] = {"error": str(exc)}

        results.append(row)

    return results


def to_rag_run(example: GoldExample) -> RAGRun:
    chunks = []
    for raw in example.payload["retrieved_chunks"]:
        metadata = raw.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(raw["chunk_id"]),
                text=str(raw["text"]),
                source_doc_id=str(metadata.get("source_doc_id", raw["chunk_id"])),
                score=raw.get("score"),
                metadata=metadata,
            )
        )

    return RAGRun(
        query=str(example.payload["query"]),
        retrieved_chunks=chunks,
        final_answer=str(example.payload.get("generated_answer") or ""),
    )


def build_report(
    gold_set_path: Path,
    examples: list[GoldExample],
    skipped: list[dict[str, str]],
    per_example: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "gold_set_size": len(examples),
        "gold_set_path": str(gold_set_path),
        "timestamp": datetime.now(UTC).isoformat(),
        "skipped_count": len(skipped),
        "skipped_examples": skipped,
        "modes": {
            "term_coverage": {
                **mode_metrics(examples, per_example, "term_coverage"),
                "threshold_sweep": threshold_sweep(examples),
            },
            "requirement_aware": {
                **mode_metrics(examples, per_example, "requirement_aware"),
                "note": "Lexical overlap thresholds (0.5, 0.25) not swept - add threshold sweep in v2",
            },
        },
        "per_example_results": per_example,
    }


def mode_metrics(
    examples: list[GoldExample],
    per_example: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    gold_by_id = {example.example_id: example for example in examples}
    gold_labels = [gold_by_id[row["example_id"]].gold_label for row in per_example]
    predictions = [row[f"{mode}_prediction"] for row in per_example]

    correct = sum(1 for gold, pred in zip(gold_labels, predictions) if gold == pred)
    accuracy_overall = safe_div(correct, len(gold_labels))
    accuracy_per_label = {}
    for label in LABELS:
        label_total = sum(1 for gold in gold_labels if gold == label)
        label_correct = sum(
            1 for gold, pred in zip(gold_labels, predictions)
            if gold == label and pred == label
        )
        accuracy_per_label[label] = safe_div(label_correct, label_total)

    insufficient_tp = sum(
        1 for gold, pred in zip(gold_labels, predictions)
        if gold == "insufficient" and pred == "insufficient"
    )
    insufficient_gold = sum(1 for gold in gold_labels if gold == "insufficient")
    insufficient_pred = sum(1 for pred in predictions if pred == "insufficient")
    recall = safe_div(insufficient_tp, insufficient_gold)
    precision = safe_div(insufficient_tp, insufficient_pred)
    f1 = safe_div(2 * precision * recall, precision + recall)

    risky_gold_count = sum(1 for gold in gold_labels if gold in {"insufficient", "partial"})
    risky_missed = sum(
        1 for gold, pred in zip(gold_labels, predictions)
        if gold in {"insufficient", "partial"} and pred != "insufficient"
    )
    false_pass_rate = safe_div(risky_missed, risky_gold_count)

    sufficient_count = sum(1 for gold in gold_labels if gold == "sufficient")
    false_fails = sum(
        1 for gold, pred in zip(gold_labels, predictions)
        if gold == "sufficient" and pred == "insufficient"
    )
    false_fail_rate = safe_div(false_fails, sufficient_count)

    abstention_pairs = [
        (
            gold_by_id[row["example_id"]].gold_should_abstain,
            bool((row[f"{mode}_details"] or {}).get("should_abstain")),
        )
        for row in per_example
        if gold_by_id[row["example_id"]].gold_should_abstain is not None
    ]
    abstention_accuracy = safe_div(
        sum(1 for gold, pred in abstention_pairs if gold == pred),
        len(abstention_pairs),
    )

    return {
        "accuracy_overall": accuracy_overall,
        "accuracy_per_label": accuracy_per_label,
        "insufficient_recall": recall,
        "insufficient_precision": precision,
        "insufficient_f1": f1,
        "false_pass_rate": false_pass_rate,
        "false_fail_rate": false_fail_rate,
        "abstention_accuracy": abstention_accuracy,
        "confusion_matrix": confusion_matrix(gold_labels, predictions),
    }


def threshold_sweep(examples: list[GoldExample]) -> dict[str, Any]:
    sweep_results = []
    best_threshold = 0.1
    best_f1 = -1.0

    for step in range(10, 91, 5):
        threshold = step / 100
        per_example = []
        for example in examples:
            result = SufficiencyAnalyzer(
                {
                    "sufficiency_mode": "term_coverage",
                    "min_coverage_ratio": threshold,
                }
            ).analyze(to_rag_run(example))
            sufficiency = result.sufficiency_result
            per_example.append(
                {
                    "example_id": example.example_id,
                    "term_coverage_prediction": (
                        sufficiency.sufficiency_label if sufficiency else "unknown"
                    ),
                    "term_coverage_details": (
                        sufficiency.model_dump(mode="json") if sufficiency else {}
                    ),
                }
            )

        metrics = mode_metrics(examples, per_example, "term_coverage")
        row = {
            "threshold": threshold,
            "insufficient_f1": metrics["insufficient_f1"],
            "false_pass_rate": metrics["false_pass_rate"],
        }
        sweep_results.append(row)
        if row["insufficient_f1"] > best_f1:
            best_f1 = row["insufficient_f1"]
            best_threshold = threshold

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "sweep_results": sweep_results,
    }


def confusion_matrix(gold_labels: list[str], predictions: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for gold, pred in zip(gold_labels, predictions):
        matrix[index.get(gold, index["unknown"])][index.get(pred, index["unknown"])] += 1
    return matrix


def print_summary(report: dict[str, Any]) -> None:
    print("=== SUFFICIENCY CALIBRATION REPORT ===")
    print(f"Gold set: {report['gold_set_size']} examples")
    for mode in ("term_coverage", "requirement_aware"):
        metrics = report["modes"][mode]
        if mode == "term_coverage":
            print("Mode: term_coverage (min_coverage_ratio=0.3)")
        else:
            print("Mode: requirement_aware")
        print(f"Overall Accuracy: {metrics['accuracy_overall']:.3f}")
        print("Insufficient Context Detection:")
        print(f"  Recall: {metrics['insufficient_recall']:.3f}")
        print(f"  Precision: {metrics['insufficient_precision']:.3f}")
        print(f"  F1: {metrics['insufficient_f1']:.3f}")
        print(f"FALSE PASS RATE: {metrics['false_pass_rate']:.3f}")
        print(f"False Fail Rate: {metrics['false_fail_rate']:.3f}")
        if mode == "term_coverage":
            sweep = metrics["threshold_sweep"]
            print(
                f"Best Threshold from Sweep: {sweep['best_threshold']:.2f} "
                f"(F1: {sweep['best_f1']:.3f})"
            )
        print()

    print("=== PER-EXAMPLE DETAILS ===")
    for row in report["per_example_results"]:
        print(
            f"{row['example_id']}: gold={row['gold_label']} "
            f"term={row['term_coverage_prediction']} "
            f"requirement={row['requirement_aware_prediction']}"
        )


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
