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
from raggov.analyzers.grounding.claims import ExtractedClaim  # noqa: E402
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector  # noqa: E402
from raggov.analyzers.grounding.verifiers import (  # noqa: E402
    HeuristicValueOverlapVerifier,
    LLMClaimEntailmentVerifierV1,
    StructuredLLMClaimVerifier,
    LLMTripletVerifierV1,
    VerificationResult,
    ConservativeEnsembleVerifier,
)
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder  # noqa: E402
from raggov.analyzers.grounding.triplets import build_triplet_extractor  # noqa: E402
from raggov.connectors.groq_client import build_groq_client_from_env  # noqa: E402
import os  # noqa: E402
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
    source_sentence = case.answer
    source_start = case.answer.find(case.claim_text)
    if source_start < 0:
        source_start = 0
    extracted_claim = ExtractedClaim(
        claim_id=case.case_id,
        claim_text=case.claim_text,
        source_sentence=source_sentence,
        source_start_char=source_start,
        source_end_char=source_start + len(case.claim_text),
        atomicity_status=case.atomicity_status,
        claim_type=_map_eval_claim_type(case.claim_type),
        entities=list(case.critical_entities),
        dates=list(case.critical_dates),
        numbers=list(case.critical_values),
        extraction_method="eval_fixture",
        extraction_reason="eval_case",
        extraction_confidence=None,
        extraction_warnings=[],
        should_verify=True,
        skip_reason=None,
    )
    # Use private _build_single to get the raw VerificationResult via aggregate logic
    record = builder._build_single(extracted_claim, 0, case.query, chunks)
    
    # Map ClaimEvidenceRecord back to VerificationResult for metrics
    return VerificationResult(
        label=record.verification_label,
        support_label=record.support_label or (
            "supported"
            if str(record.verification_label) == "entailed"
            else "contradicted"
            if str(record.verification_label) == "contradicted"
            else "unverifiable"
            if str(record.verification_label) in {"neutral", "unverified"}
            else "insufficient_evidence"
        ),
        raw_score=record.verifier_score,
        evidence_chunk_id=record.supporting_chunk_ids[0] if record.supporting_chunk_ids else None,
        evidence_span=None,
        rationale=record.evidence_reason,
        verifier_name=record.verifier_method,
        fallback_used=record.fallback_used,
        supporting_chunk_ids=record.supporting_chunk_ids,
        candidate_chunk_ids=record.candidate_evidence_chunk_ids,
        contradicting_chunk_ids=record.contradicting_chunk_ids,
        neutral_chunk_ids=record.neutral_candidate_ids,
        confidence_status=record.confidence_status or "unavailable",
        triplet_results=getattr(record, "triplet_results", []),
        verifier_policy=getattr(record, "verifier_policy", None),
        verifier_disagreement=getattr(record, "verifier_disagreement", False),
        safety_gate_triggered=getattr(record, "safety_gate_triggered", False),
        safety_gate_reason=getattr(record, "safety_gate_reason", None),
        safety_gate_category=getattr(record, "safety_gate_category", None),
        critical_fact_check_summary=getattr(record, "critical_fact_check_summary", {}),
        llm_label=getattr(record, "llm_label", None),
        heuristic_label=getattr(record, "heuristic_label", None),
        deterministic_gate_labels=getattr(record, "deterministic_gate_labels", []),
        normalized_values_checked=getattr(record, "normalized_values_checked", []),
        normalized_dates_checked=getattr(record, "normalized_dates_checked", []),
        normalized_units_checked=getattr(record, "normalized_units_checked", []),
        normalized_entities_checked=getattr(record, "normalized_entities_checked", []),
    )


def _map_eval_claim_type(claim_type: str) -> str:
    mapping = {
        "general_factual": "other",
        "date_or_deadline": "temporal",
        "go_number": "other",
    }
    return mapping.get(claim_type, claim_type)


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
    
    llm_supported_heuristic_unsupported_count = 0
    llm_supported_heuristic_contradicted_count = 0
    safety_gate_downgrade_count = 0
    safety_gate_true_positive_count = 0
    false_pass_cases: list[dict[str, Any]] = []
    false_fail_cases: list[dict[str, Any]] = []
    contradiction_missed_cases: list[dict[str, Any]] = []
    
    domain_stats: dict[str, dict[str, int]] = {}

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
            
        # ---- safety gate and disagreement ---------------------------------
        if getattr(pred, "safety_gate_triggered", False):
            safety_gate_downgrade_count += 1
            
        if getattr(pred, "verifier_disagreement", False):
            # We know heuristic contradicted because it triggered the gate
            llm_supported_heuristic_contradicted_count += 1

        # ---- false pass case tracking --------------------------------------
        if gold in ("unsupported", "contradicted") and predicted == "entailed":
            false_pass_cases.append({
                "case_id": case.case_id,
                "claim_text": case.claim_text,
                "predicted_label": predicted,
                "expected_label": gold,
                "verifier_reason": getattr(pred, "rationale", ""),
                "safety_gate_triggered": getattr(pred, "safety_gate_triggered", False)
            })
            
        if gold == "entailed" and predicted in ("unsupported", "contradicted"):
            false_fail_cases.append({
                "case_id": case.case_id,
                "claim_text": case.claim_text,
                "predicted_label": predicted,
                "expected_label": gold,
                "verifier_reason": getattr(pred, "rationale", "")
            })
            
        if gold == "contradicted" and predicted != "contradicted":
            contradiction_missed_cases.append({
                "case_id": case.case_id,
                "claim_text": case.claim_text,
                "predicted_label": predicted,
                "expected_label": gold,
                "verifier_reason": getattr(pred, "rationale", "")
            })

        # ---- domain breakdown -----------------------------------------------
        domain = getattr(case, "domain", None) or "unknown"
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "correct": 0, "false_pass": 0, "false_pass_denom": 0}
        domain_stats[domain]["total"] += 1
        if predicted == gold:
            domain_stats[domain]["correct"] += 1
        if gold in ("unsupported", "contradicted"):
            domain_stats[domain]["false_pass_denom"] += 1
            if predicted == "entailed":
                domain_stats[domain]["false_pass"] += 1
        
        # ---- safety gate precision ----------------------------------------
        if getattr(pred, "safety_gate_triggered", False) and gold in ("unsupported", "contradicted"):
            safety_gate_true_positive_count += 1

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

    domain_breakdown: dict[str, dict[str, float]] = {}
    for dom, ds in domain_stats.items():
        domain_breakdown[dom] = {
            "total": ds["total"],
            "accuracy": round(_safe_div(ds["correct"], ds["total"]), 4),
            "false_pass_rate": round(_safe_div(ds["false_pass"], ds["false_pass_denom"]), 4),
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
        "safety_gate_precision": round(_safe_div(safety_gate_true_positive_count, safety_gate_downgrade_count), 4),
        "raw_counts": {
            "correct": correct,
            "false_pass": false_pass_num,
            "false_fail": false_fail_num,
            "contradiction_detected": contradiction_num,
            "fallback_used": fallback_count,
            "llm_supported_heuristic_unsupported_count": llm_supported_heuristic_unsupported_count,
            "llm_supported_heuristic_contradicted_count": llm_supported_heuristic_contradicted_count,
            "safety_gate_downgrade_count": safety_gate_downgrade_count,
            "safety_gate_true_positive_count": safety_gate_true_positive_count,
        },
        "false_pass_cases": false_pass_cases,
        "false_fail_cases": false_fail_cases,
        "contradiction_missed_cases": contradiction_missed_cases,
        "domain_breakdown": domain_breakdown,
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
        f"| Safety gate precision | `{metrics.get('safety_gate_precision', 0.0):.1%}` |",
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
        "## Ensemble & Disagreement Metrics",
        "",
        f"- Safety gate downgrades: **{rc.get('safety_gate_downgrade_count', 0)}**",
        f"- LLM Supported vs Heuristic Contradicted: **{rc.get('llm_supported_heuristic_contradicted_count', 0)}**",
        "",
        "> **Note**: False-pass rate is the primary risk metric for high-trust RAG.",
        "> A false pass means a fabricated or contradicted claim was silently accepted.",
    ]
    
    fp_cases = metrics.get("false_pass_cases", [])
    if fp_cases:
        lines.extend([
            "",
            "## False Pass Cases",
            ""
        ])
        for c in fp_cases:
            lines.extend([
                f"- **Case ID**: `{c['case_id']}`",
                f"  - Claim: {c['claim_text']}",
                f"  - Expected: `{c['expected_label']}` | Predicted: `{c['predicted_label']}`",
                f"  - Rationale: {c['verifier_reason']}",
                f"  - Safety Gate Caught It: {c.get('safety_gate_triggered', False)}"
            ])

    domain_breakdown = metrics.get("domain_breakdown", {})
    if domain_breakdown:
        lines.extend([
            "",
            "## Domain-Wise Breakdown",
            "",
            "| Domain | Cases | Accuracy | False-Pass Rate |",
            "|--------|-------|----------|----------------|",
        ])
        for dom, dm in sorted(domain_breakdown.items()):
            lines.append(
                f"| {dom} | {dm['total']} | `{dm['accuracy']:.1%}` | `{dm['false_pass_rate']:.1%}` |"
            )

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
        choices=["heuristic", "structured_llm", "llm_entailment", "conservative_ensemble"],
        default="heuristic",
        help="Verifier mode to evaluate (default: heuristic).",
    )
    p.add_argument(
        "--provider",
        choices=["groq"],
        default=None,
        help="Optional live LLM provider for llm_entailment verification.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name to override the default for the provider.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit for live/provider-backed evals.",
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
    llm_client: object | None = None,
    provider: str | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the evaluation and return the metrics dict."""
    # ---- Load ---------------------------------------------------------------
    cases = load_dataset(dataset_path)
    provider_reason: str | None = None
    if provider == "groq" and llm_client is None:
        if model:
            os.environ["GROQ_MODEL"] = model
        llm_client, provider_reason = build_groq_client_from_env()
    if provider == "groq" and limit is None:
        limit = 3
    if limit is not None:
        cases = cases[:limit]
    print(f"Loaded {len(cases)} cases from {dataset_path.name}")

    # ---- Build verifier + selector + builder -------------------------------
    config: dict[str, Any] = {
        "claim_verifier_mode": verifier_mode,
        "enable_triplet_extraction": verifier_mode == "triplet" or True, # extractor needed if verifier is triplet
        "triplet_extractor_method": "rule_v0", # default
        "llm_client": llm_client,
    }
    
    if provider == "groq" and llm_client is None:
        raise RuntimeError(provider_reason or "groq provider unavailable")

    if verifier_mode == "structured_llm":
        verifier = StructuredLLMClaimVerifier(config)
    elif verifier_mode == "llm_entailment":
        verifier = LLMClaimEntailmentVerifierV1(config)
    elif verifier_mode == "conservative_ensemble":
        verifier = ConservativeEnsembleVerifier(config)
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
    metrics["provider"] = provider
    metrics["model"] = getattr(llm_client, "model_name", None)
    metrics["call_count"] = getattr(getattr(llm_client, "stats", None), "call_count", 0)
    metrics["rate_limited"] = bool(getattr(getattr(llm_client, "stats", None), "rate_limited", False))

    # ---- Render markdown ---------------------------------------------------
    md = render_markdown(metrics, verifier_name, dataset_path)
    print()
    print(md)
    if provider:
        print(f"provider={provider}")
        print(f"model={metrics['model'] or 'unknown'}")
        print(f"call_count={metrics['call_count']}")
        print(f"rate_limited={str(metrics['rate_limited']).lower()}")

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
    try:
        run_eval(
            dataset_path=args.dataset,
            verifier_mode=args.verifier,
            json_out=args.json_out,
            md_out=args.md_out,
            provider=args.provider,
            model=args.model,
            limit=args.limit,
        )
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
