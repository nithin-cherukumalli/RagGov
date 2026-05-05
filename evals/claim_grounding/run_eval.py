"""
Claim-grounding evaluation harness.

Runs the GovRAG HeuristicValueOverlapVerifier (and optionally other verifiers)
against the claim-grounding gold dataset and computes calibration metrics.

Usage:
    # From the repo root:
    python evals/claim_grounding/run_eval.py

    # Point to a different dataset:
    python evals/claim_grounding/run_eval.py --dataset path/to/cases.jsonl

    # Use a different verifier mode:
    python evals/claim_grounding/run_eval.py --verifier heuristic

    # Save outputs:
    python evals/claim_grounding/run_eval.py --json-out report.json --md-out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve repo root so the harness can be run from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.schema import ClaimGroundingCase, ChunkRecord  # noqa: E402
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector  # noqa: E402
from raggov.analyzers.grounding.verifiers import (  # noqa: E402
    HeuristicValueOverlapVerifier,
    StructuredLLMClaimVerifier,
    LLMTripletVerifierV1,
    VerificationResult,
)
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder  # noqa: E402
from raggov.analyzers.grounding.triplets import build_triplet_extractor  # noqa: E402
from raggov.models.chunk import RetrievedChunk  # noqa: E402

_DEFAULT_DATASET = _EVAL_DIR / "seed_cases.jsonl"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[ClaimGroundingCase]:
    """Load JSONL dataset and validate each record against the schema."""
    cases: list[ClaimGroundingCase] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                cases.append(ClaimGroundingCase.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"Line {line_no}: invalid record — {exc}") from exc
    return cases


# ---------------------------------------------------------------------------
# Adapter: ClaimGroundingCase → verifier inputs
# ---------------------------------------------------------------------------

def _to_retrieved_chunks(chunk_records: list[ChunkRecord]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=cr.chunk_id,
            text=cr.text,
            source_doc_id=cr.source_doc_id,
            score=cr.score,
        )
        for cr in chunk_records
    ]


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------

def predict(
    case: ClaimGroundingCase,
    builder: ClaimEvidenceBuilder,
) -> VerificationResult:
    """Run verifier on a single case and return the VerificationResult."""
    chunks = _to_retrieved_chunks(case.retrieved_chunks)
    # Use private _build_single to get the raw VerificationResult via aggregate logic
    record = builder._build_single(case.claim_text, 0, case.query, chunks)
    
    # Map ClaimEvidenceRecord back to VerificationResult for metrics
    return VerificationResult(
        label=record.verification_label,
        raw_score=record.verifier_score,
        evidence_chunk_id=record.supporting_chunk_ids[0] if record.supporting_chunk_ids else None,
        evidence_span=None,
        rationale=record.evidence_reason,
        verifier_name=record.verifier_method,
        fallback_used=record.fallback_used,
        supporting_chunk_ids=record.supporting_chunk_ids,
        contradicting_chunk_ids=record.contradicting_chunk_ids,
        triplet_results=getattr(record, "triplet_results", [])
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_metrics(
    cases: list[ClaimGroundingCase],
    predictions: list[VerificationResult],
) -> dict[str, Any]:
    """
    Compute all required evaluation metrics.

    Metric definitions
    ------------------
    false_pass_rate:
        Among cases where gold_label ∈ {unsupported, contradicted},
        the fraction predicted as entailed.
        High false-pass rate is the most dangerous failure mode for a
        high-trust RAG system — it means the system silently passes bad answers.

    false_fail_rate:
        Among cases where gold_label = entailed,
        the fraction predicted as unsupported or contradicted.

    contradiction_detection_rate:
        Among cases where gold_label = contradicted,
        the fraction correctly predicted as contradicted.

    evidence_chunk_recall:
        For entailed cases, the fraction of gold_supporting_chunk_ids
        that appear in the verifier's supporting_chunk_ids list.

    fallback_rate:
        Fraction of all predictions that used a fallback heuristic.
    """
    assert len(cases) == len(predictions), "cases/predictions length mismatch"

    total = len(cases)
    if total == 0:
        return {}

    # ---- per-label accumulators -------------------------------------------
    stats: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for label in ("entailed", "unsupported", "contradicted")
    }

    correct = 0
    false_pass_denom = 0
    false_pass_num = 0
    false_fail_denom = 0
    false_fail_num = 0
    contradiction_denom = 0
    contradiction_num = 0
    chunk_recall_sum = 0.0
    chunk_recall_count = 0
    fallback_count = 0

    for case, pred in zip(cases, predictions):
        gold = case.gold_label
        predicted = pred.label

        # ---- accuracy -------------------------------------------------------
        if predicted == gold:
            correct += 1

        # ---- per-label TP/FP/FN -------------------------------------------
        for label in ("entailed", "unsupported", "contradicted"):
            if gold == label and predicted == label:
                stats[label]["tp"] += 1
            elif gold != label and predicted == label:
                stats[label]["fp"] += 1
            elif gold == label and predicted != label:
                stats[label]["fn"] += 1

        # ---- false_pass (most dangerous) -----------------------------------
        if gold in ("unsupported", "contradicted"):
            false_pass_denom += 1
            if predicted == "entailed":
                false_pass_num += 1

        # ---- false_fail (over-rejection) -----------------------------------
        if gold == "entailed":
            false_fail_denom += 1
            if predicted in ("unsupported", "contradicted"):
                false_fail_num += 1

        # ---- contradiction_detection_rate ----------------------------------
        if gold == "contradicted":
            contradiction_denom += 1
            if predicted == "contradicted":
                contradiction_num += 1

        # ---- evidence chunk recall -----------------------------------------
        if gold == "entailed" and case.gold_supporting_chunk_ids:
            retrieved_ids = set(pred.supporting_chunk_ids)
            gold_ids = set(case.gold_supporting_chunk_ids)
            chunk_recall_sum += len(retrieved_ids & gold_ids) / len(gold_ids)
            chunk_recall_count += 1

        # ---- fallback count ------------------------------------------------
        if pred.fallback_used:
            fallback_count += 1

    # ---- aggregate per-label metrics --------------------------------------
    label_metrics: dict[str, dict[str, float]] = {}
    for label in ("entailed", "unsupported", "contradicted"):
        s = stats[label]
        precision = _safe_div(s["tp"], s["tp"] + s["fp"])
        recall = _safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        label_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": s["tp"],
            "false_positives": s["fp"],
            "false_negatives": s["fn"],
        }

    return {
        "total_cases": total,
        "overall_accuracy": round(_safe_div(correct, total), 4),
        "label_metrics": label_metrics,
        "false_pass_rate": round(_safe_div(false_pass_num, false_pass_denom), 4),
        "false_fail_rate": round(_safe_div(false_fail_num, false_fail_denom), 4),
        "contradiction_detection_rate": round(
            _safe_div(contradiction_num, contradiction_denom), 4
        ),
        "evidence_chunk_recall": round(
            _safe_div(chunk_recall_sum, chunk_recall_count), 4
        ),
        "fallback_rate": round(_safe_div(fallback_count, total), 4),
        "raw_counts": {
            "correct": correct,
            "false_pass": false_pass_num,
            "false_fail": false_fail_num,
            "contradiction_detected": contradiction_num,
            "fallback_used": fallback_count,
        },
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown(
    metrics: dict[str, Any],
    verifier_name: str,
    dataset_path: Path,
) -> str:
    lm = metrics["label_metrics"]
    lines = [
        "# Claim-Grounding Evaluation Report",
        "",
        f"- **Dataset**: `{dataset_path.name}`",
        f"- **Total cases**: {metrics['total_cases']}",
        f"- **Verifier**: `{verifier_name}`",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall accuracy | `{metrics['overall_accuracy']:.1%}` |",
        f"| **False-pass rate** | `{metrics['false_pass_rate']:.1%}` |",
        f"| False-fail rate | `{metrics['false_fail_rate']:.1%}` |",
        f"| Contradiction detection rate | `{metrics['contradiction_detection_rate']:.1%}` |",
        f"| Evidence chunk recall | `{metrics['evidence_chunk_recall']:.1%}` |",
        f"| Fallback rate | `{metrics['fallback_rate']:.1%}` |",
        "",
        "## Per-Label Metrics",
        "",
        "| Label | Precision | Recall | F1 | TP | FP | FN |",
        "|-------|-----------|--------|----|----|----|----|",
    ]

    for label in ("entailed", "unsupported", "contradicted"):
        m = lm[label]
        lines.append(
            f"| {label} "
            f"| {m['precision']:.3f} "
            f"| {m['recall']:.3f} "
            f"| {m['f1']:.3f} "
            f"| {m['true_positives']} "
            f"| {m['false_positives']} "
            f"| {m['false_negatives']} |"
        )

    rc = metrics["raw_counts"]
    lines += [
        "",
        "## Raw Counts",
        "",
        f"- Correct predictions: **{rc['correct']}** / {metrics['total_cases']}",
        f"- False passes (bad answers let through): **{rc['false_pass']}**",
        f"- False fails (good answers rejected): **{rc['false_fail']}**",
        f"- Contradictions correctly caught: **{rc['contradiction_detected']}**",
        f"- Predictions using fallback: **{rc['fallback_used']}**",
        "",
        "> **Note**: False-pass rate is the primary risk metric for high-trust RAG.",
        "> A false pass means a fabricated or contradicted claim was silently accepted.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GovRAG claim-grounding evaluation harness."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to JSONL dataset file (default: seed_cases.jsonl).",
    )
    p.add_argument(
        "--verifier",
        choices=["heuristic", "structured_llm"],
        default="heuristic",
        help="Verifier mode to evaluate (default: heuristic).",
    )
    p.add_argument(
        "--triplet-mode",
        action="store_true",
        help="Enable triplet-level verification.",
    )
    p.add_argument(
        "--triplet-extractor",
        choices=["rule_v0", "llm_v1"],
        default="rule_v0",
        help="Triplet extractor to use (default: rule_v0).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON report to this path.",
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Write markdown report to this path.",
    )
    return p


def run_eval(
    dataset_path: Path = _DEFAULT_DATASET,
    verifier_mode: str = "heuristic",
    json_out: Path | None = None,
    md_out: Path | None = None,
) -> dict[str, Any]:
    """Run the evaluation and return the metrics dict."""
    # ---- Load ---------------------------------------------------------------
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases from {dataset_path.name}")

    # ---- Build verifier + selector + builder -------------------------------
    config: dict[str, Any] = {
        "claim_verifier_mode": verifier_mode,
        "enable_triplet_extraction": verifier_mode == "triplet" or True, # extractor needed if verifier is triplet
        "triplet_extractor_method": "rule_v0" # default
    }
    
    if verifier_mode == "structured_llm":
        verifier = StructuredLLMClaimVerifier(config)
    else:
        verifier = HeuristicValueOverlapVerifier(config)
        
    selector = EvidenceCandidateSelector(config)
    extractor = build_triplet_extractor(config)
    
    builder = ClaimEvidenceBuilder(verifier, selector, triplet_extractor=extractor)
    
    # Triplet verifier setup
    # In eval mode, we don't always have an LLM client, but we allow forcing it if needed
    # If verifier_mode is specifically 'triplet_llm', we use it
    if verifier_mode == "triplet_llm" or config.get("enable_triplet_verification"):
        triplet_verifier = LLMTripletVerifierV1(config)
        builder.set_triplet_verifier(triplet_verifier)
    
    # Simplified override for CLI
    # (Actual implementation would pass the full config)

    # ---- Run predictions ---------------------------------------------------
    predictions: list[VerificationResult] = []
    for case in cases:
        result = predict(case, builder)
        predictions.append(result)

    # ---- Compute metrics ---------------------------------------------------
    metrics = compute_metrics(cases, predictions)
    verifier_name = verifier_mode
    if config.get("enable_triplet_verification"):
        verifier_name += "+triplet"

    # ---- Render markdown ---------------------------------------------------
    md = render_markdown(metrics, verifier_name, dataset_path)
    print()
    print(md)

    # ---- Optionally write outputs ------------------------------------------
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report written to: {json_out}")

    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(md, encoding="utf-8")
        print(f"Markdown report written to: {md_out}")

    return metrics


def main() -> None:
    args = _build_parser().parse_args()
    run_eval(
        dataset_path=args.dataset,
        verifier_mode=args.verifier,
        json_out=args.json_out,
        md_out=args.md_out,
    )


if __name__ == "__main__":
    main()
