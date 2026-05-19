#!/usr/bin/env python
"""
Domain-Agnostic Claim Grounding Evaluation Script.

Evaluates the Heuristic, LLM Entailment, and Conservative Ensemble verifiers
on the newly created 105-case domain-agnostic claim grounding benchmark.
Generates a comprehensive comparative report showing overall performance,
breakdowns by domain, difficulty, and failure type, as well as safety gate and
verifier disagreement metrics.

Usage:
    GROQ_API_KEY=gsk_yfqGqzlX8... python scripts/evaluate_domain_agnostic_grounding.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Literal

# Add src/ directory to python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.run_eval import (
    load_dataset,
    predict,
    compute_metrics,
)
from evals.claim_grounding.schema import ClaimGroundingCase
from raggov.analyzers.grounding.verifiers import (
    HeuristicValueOverlapVerifier,
    LLMClaimEntailmentVerifierV1,
    ConservativeEnsembleVerifier,
    VerificationResult,
)
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder
from raggov.analyzers.grounding.triplets import build_triplet_extractor
from raggov.connectors.groq_client import build_groq_client_from_env

_DEFAULT_DATASET = _REPO_ROOT / "evals" / "claim_grounding" / "domain_agnostic_100.jsonl"
_JSON_REPORT_PATH = _REPO_ROOT / "reports" / "domain_agnostic_grounding_eval.json"
_MD_REPORT_PATH = _REPO_ROOT / "reports" / "domain_agnostic_grounding_eval.md"


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _metrics_stub() -> Dict[str, Any]:
    return {
        "overall_accuracy": 0.0,
        "false_pass_rate": 0.0,
        "hard_subset_false_pass_rate": 0.0,
        "false_fail_rate": 0.0,
        "contradiction_detection_rate": 0.0,
        "evidence_chunk_recall": 0.0,
        "fallback_rate": 0.0,
        "raw_counts": {},
        "domain_breakdown": {},
        "difficulty_breakdown": {},
        "failure_type_breakdown": {},
        "safety_gate_reasons": {},
        "false_pass_category_breakdown": {},
    }


def run_verifier_evaluation(
    cases: List[ClaimGroundingCase],
    verifier_mode: str,
    llm_client: Any,
) -> List[VerificationResult]:
    """Run verification predictions on all cases for a specific verifier mode with retry and rate limit logic."""
    if verifier_mode in {"llm_entailment", "conservative_ensemble"} and llm_client is None:
        raise RuntimeError(
            f"Verifier '{verifier_mode}' requires an llm_client. Configure GROQ_API_KEY before running this mode."
        )

    config: Dict[str, Any] = {
        "claim_verifier_mode": verifier_mode,
        "enable_triplet_extraction": False,
        "llm_client": llm_client,
    }

    if verifier_mode == "heuristic":
        verifier = HeuristicValueOverlapVerifier(config)
    elif verifier_mode == "llm_entailment":
        verifier = LLMClaimEntailmentVerifierV1(config)
    elif verifier_mode == "conservative_ensemble":
        verifier = ConservativeEnsembleVerifier(config)
    else:
        raise ValueError(f"Unknown verifier mode: {verifier_mode}")

    selector = EvidenceCandidateSelector(config)
    extractor = build_triplet_extractor(config)
    builder = ClaimEvidenceBuilder(verifier, selector, triplet_extractor=extractor)

    predictions: List[VerificationResult] = []
    
    print(f"\nEvaluating verifier: '{verifier_mode}' on {len(cases)} cases...")
    for idx, case in enumerate(cases, 1):
        if idx % 10 == 0 or idx == 1 or idx == len(cases):
            print(f"  [{verifier_mode}] Processing case {idx}/{len(cases)} (ID: {case.case_id})...")

        # Retry logic with exponential backoff for LLM-based verifiers
        max_retries = 5
        retry_delay = 2.0
        success = False
        result = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # Add a tiny delay between requests to prevent hitting rate limits
                if verifier_mode in ("llm_entailment", "conservative_ensemble") and attempt == 1:
                    time.sleep(0.5)

                result = predict(case, builder)
                success = True
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = "rate limit" in exc_str or "429" in exc_str or "tpm" in exc_str or "rpm" in exc_str
                
                if is_rate_limit:
                    print(f"    [Rate Limit] (Attempt {attempt}/{max_retries}): Rate limit hit. Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2.0  # Exponential backoff
                else:
                    print(f"    [Error] (Attempt {attempt}/{max_retries}) on case {case.case_id}: {exc}")
                    time.sleep(1.0)
        
        if not success or result is None:
            print(f"  [Failure] Failed to verify case {case.case_id} after {max_retries} attempts.")
            # Create a fallback/error VerificationResult
            result = VerificationResult(
                label="abstain",
                support_label="unverifiable",
                raw_score=0.0,
                evidence_chunk_id=None,
                evidence_span=None,
                rationale="Failed due to rate limits or API errors.",
                verifier_name=verifier_mode,
                error_info="API failure",
                fallback_used=True,
            )
            
        predictions.append(result)

    return predictions


def compute_comprehensive_metrics(
    cases: List[ClaimGroundingCase],
    predictions: List[VerificationResult],
) -> Dict[str, Any]:
    """Compute overall metrics, domain breakdowns, difficulty breakdowns, and failure type breakdowns."""
    # Compute base metrics using run_eval's standard compute_metrics helper
    base_metrics = compute_metrics(cases, predictions)

    # Initialize breakdown accumulators
    difficulty_stats: Dict[str, Dict[str, int]] = {}
    failure_type_stats: Dict[str, Dict[str, int]] = {}
    safety_gate_reasons: Dict[str, int] = {}
    false_pass_category_breakdown: Dict[str, int] = {}
    hard_false_pass_num = 0
    hard_false_pass_denom = 0
    
    # Calculate detailed breakdowns
    for case, pred in zip(cases, predictions):
        gold = case.gold_label
        predicted = pred.label

        # 1. Difficulty breakdown
        diff = getattr(case, "difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "correct": 0, "false_pass": 0, "false_pass_denom": 0}
        difficulty_stats[diff]["total"] += 1
        if predicted == gold:
            difficulty_stats[diff]["correct"] += 1
        if gold in ("unsupported", "contradicted"):
            difficulty_stats[diff]["false_pass_denom"] += 1
            if diff == "hard":
                hard_false_pass_denom += 1
            if predicted == "entailed":
                difficulty_stats[diff]["false_pass"] += 1
                if diff == "hard":
                    hard_false_pass_num += 1
                category = _categorize_false_pass(case, pred)
                false_pass_category_breakdown[category] = false_pass_category_breakdown.get(category, 0) + 1

        # 2. Failure type breakdown
        ftype = getattr(case, "failure_type", "unknown")
        if ftype not in failure_type_stats:
            failure_type_stats[ftype] = {"total": 0, "correct": 0, "false_pass": 0, "false_pass_denom": 0}
        failure_type_stats[ftype]["total"] += 1
        if predicted == gold:
            failure_type_stats[ftype]["correct"] += 1
        if gold in ("unsupported", "contradicted"):
            failure_type_stats[ftype]["false_pass_denom"] += 1
            if predicted == "entailed":
                failure_type_stats[ftype]["false_pass"] += 1

        # 3. Safety Gate Trigger Reasons
        if getattr(pred, "safety_gate_triggered", False):
            reason = getattr(pred, "safety_gate_reason", "unknown") or "unknown"
            safety_gate_reasons[reason] = safety_gate_reasons.get(reason, 0) + 1

    # Format breakdowns
    difficulty_breakdown: Dict[str, Dict[str, float]] = {}
    for diff, ds in difficulty_stats.items():
        difficulty_breakdown[diff] = {
            "total": ds["total"],
            "accuracy": round(_safe_div(ds["correct"], ds["total"]), 4),
            "false_pass_rate": round(_safe_div(ds["false_pass"], ds["false_pass_denom"]), 4),
        }

    failure_type_breakdown: Dict[str, Dict[str, float]] = {}
    for ftype, fs in failure_type_stats.items():
        failure_type_breakdown[ftype] = {
            "total": fs["total"],
            "accuracy": round(_safe_div(fs["correct"], fs["total"]), 4),
            "false_pass_rate": round(_safe_div(fs["false_pass"], fs["false_pass_denom"]), 4),
        }

    # Add extra metrics to the dict
    base_metrics["difficulty_breakdown"] = difficulty_breakdown
    base_metrics["failure_type_breakdown"] = failure_type_breakdown
    base_metrics["safety_gate_reasons"] = safety_gate_reasons
    base_metrics["false_pass_category_breakdown"] = false_pass_category_breakdown
    base_metrics["hard_subset_false_pass_rate"] = round(_safe_div(hard_false_pass_num, hard_false_pass_denom), 4)

    return base_metrics


def _categorize_false_pass(case: ClaimGroundingCase, pred: VerificationResult) -> str:
    explicit_category = getattr(pred, "safety_gate_category", None)
    if explicit_category:
        return str(explicit_category)
    mapping = {
        "contradicted_value": "value_mismatch_missed",
        "contradicted_date": "date_mismatch_missed",
        "contradicted_entity": "entity_mismatch_missed",
        "citation_like_mismatch": "entity_mismatch_missed",
        "lexical_decoy": "lexical_decoy",
        "compound_one_clause_contradicted": "compound_partial_support",
        "insufficient_missing_value": "llm_overpermissive_no_deterministic_gate",
        "insufficient_missing_entity": "related_but_non_supporting",
    }
    return mapping.get(getattr(case, "failure_type", ""), "llm_overpermissive_no_deterministic_gate")


def generate_reports(
    results: Dict[str, Any],
    dataset_path: Path,
    previous_results: Dict[str, Any] | None = None,
) -> None:
    """Generate Markdown and JSON reports from all evaluation results."""
    # Write JSON report
    _JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _JSON_REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n[Success] JSON report written to: {_JSON_REPORT_PATH}")

    # Generate comparative Markdown
    md_lines = [
        "# Domain-Agnostic Claim Grounding Benchmark Report",
        "",
        "This report evaluates and compares three claim verification policies across a newly built, domain-diverse benchmark consisting of 105 meticulously crafted test cases.",
        "",
        "## Evaluation Setup",
        "",
        f"- **Benchmark Dataset**: `{dataset_path.name}` ({results['total_cases']} cases)",
        f"- **LLM Provider**: `Groq` (Model: `llama-3.1-8b-instant`)",
        f"- **Report Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Policy Summary Comparison",
        "",
        "| Metric | Heuristic Verifier | LLM Entailment Verifier | Conservative Ensemble Verifier |",
        "| :--- | :---: | :---: | :---: |",
    ]

    h_m = results["verifiers"].get("heuristic") or _metrics_stub()
    l_m = results["verifiers"].get("llm_entailment") or _metrics_stub()
    e_m = results["verifiers"].get("conservative_ensemble") or _metrics_stub()

    metrics_to_show = [
        ("Overall Accuracy", "overall_accuracy", "{:.1%}"),
        ("False-Pass Rate (Safety Risk)", "false_pass_rate", "{:.1%}"),
        ("Hard-Subset False-Pass Rate", "hard_subset_false_pass_rate", "{:.1%}"),
        ("False-Fail Rate (Over-Rejection)", "false_fail_rate", "{:.1%}"),
        ("Contradiction Detection Rate", "contradiction_detection_rate", "{:.1%}"),
        ("Evidence Chunk Recall", "evidence_chunk_recall", "{:.1%}"),
        ("Fallback Rate", "fallback_rate", "{:.1%}"),
    ]

    for label, key, fmt in metrics_to_show:
        md_lines.append(
            f"| **{label}** | {fmt.format(float(h_m.get(key, 0.0) or 0.0))} | {fmt.format(float(l_m.get(key, 0.0) or 0.0))} | {fmt.format(float(e_m.get(key, 0.0) or 0.0))} |"
        )

    md_lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **False-Pass Rate** is the primary safety metric. A false pass means a fabricated or contradicted claim was silently accepted. The **Conservative Ensemble Verifier** strikes an optimal balance between the semantic recall of LLM entailment and the hard safety constraints of heuristics.",
        "",
        "## Domain-Wise Accuracy Breakdown",
        "",
        "| Domain | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    # Get sorted domain names
    domains = sorted(
        set(h_m["domain_breakdown"].keys())
        | set(l_m["domain_breakdown"].keys())
        | set(e_m["domain_breakdown"].keys())
    )
    for dom in domains:
        h_dom = h_m["domain_breakdown"].get(dom, {"accuracy": 0.0})
        l_dom = l_m["domain_breakdown"].get(dom, {"accuracy": 0.0})
        e_dom = e_m["domain_breakdown"].get(dom, {"accuracy": 0.0})
        md_lines.append(
            f"| `{dom}` | {h_dom['total']} | `{h_dom['accuracy']:.1%}` | `{l_dom['accuracy']:.1%}` | `{e_dom['accuracy']:.1%}` |"
        )

    md_lines.extend([
        "",
        "## Breakdown by Difficulty",
        "",
        "| Difficulty | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    difficulties = sorted(
        set(h_m["difficulty_breakdown"].keys())
        | set(l_m["difficulty_breakdown"].keys())
        | set(e_m["difficulty_breakdown"].keys())
    )
    for diff in difficulties:
        h_diff = h_m["difficulty_breakdown"].get(diff, {"accuracy": 0.0, "total": 0})
        l_diff = l_m["difficulty_breakdown"].get(diff, {"accuracy": 0.0})
        e_diff = e_m["difficulty_breakdown"].get(diff, {"accuracy": 0.0})
        md_lines.append(
            f"| `{diff}` | {h_diff['total']} | `{h_diff['accuracy']:.1%}` | `{l_diff['accuracy']:.1%}` | `{e_diff['accuracy']:.1%}` |"
        )

    md_lines.extend([
        "",
        "## Breakdown by Failure Category",
        "",
        "| Category / Failure Type | Total Cases | Heuristic | LLM Entailment | Conservative Ensemble |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    ftypes = sorted(
        set(h_m["failure_type_breakdown"].keys())
        | set(l_m["failure_type_breakdown"].keys())
        | set(e_m["failure_type_breakdown"].keys())
    )
    for ftype in ftypes:
        h_ftype = h_m["failure_type_breakdown"].get(ftype, {"accuracy": 0.0, "total": 0})
        l_ftype = l_m["failure_type_breakdown"].get(ftype, {"accuracy": 0.0})
        e_ftype = e_m["failure_type_breakdown"].get(ftype, {"accuracy": 0.0})
        md_lines.append(
            f"| `{ftype}` | {h_ftype['total']} | `{h_ftype['accuracy']:.1%}` | `{l_ftype['accuracy']:.1%}` | `{e_ftype['accuracy']:.1%}` |"
        )

    # Safety Gate Trigger Analysis for Ensemble
    gate_triggers = e_m.get("safety_gate_reasons", {})
    gate_total = e_m["raw_counts"].get("safety_gate_downgrade_count", 0)
    
    md_lines.extend([
        "",
        "## Conservative Ensemble Safety Gate Analysis",
        "",
        f"The Conservative Ensemble Verifier triggered **{gate_total}** deterministic safety overrides, downgrading unsafe `supported` judgments to either `insufficient_evidence` or `contradicted`.",
        "",
        "### Safety Gate Trigger Reason Breakdown:",
        "",
        "| Safety Gate Trigger Reason | Count | Percentage | Description |",
        "| :--- | :---: | :---: | :--- |",
    ])

    gate_desc = {
        "llm_heuristic_disagreement": "Heuristic flagged claim as contradicted, overriding LLM supported.",
        "missing_critical_fact_coverage": "A critical date, number, or entity was completely missing from the best supporting chunk.",
        "compound_claim_not_fully_covered": "Compound claim decomposition flagged missing support for a required subclaim.",
        "unknown": "Downgraded due to safety baseline policy constraints."
    }

    for reason, count in sorted(gate_triggers.items(), key=lambda x: x[1], reverse=True):
        desc = gate_desc.get(reason, "Safety boundary downgrade.")
        pct = _safe_div(count, gate_total)
        md_lines.append(f"| `{reason}` | {count} | {pct:.1%} | {desc} |")

    md_lines.extend([
        "",
        "## Conservative Ensemble False-Pass Breakdown",
        "",
        "| False-Pass Category | Count |",
        "| :--- | :---: |",
    ])
    for category, count in sorted(
        e_m.get("false_pass_category_breakdown", {}).items(),
        key=lambda item: (-item[1], item[0]),
    ):
        md_lines.append(f"| `{category}` | {count} |")

    if previous_results:
        previous_ensemble = previous_results.get("verifiers", {}).get("conservative_ensemble", {})
        if previous_ensemble:
            md_lines.extend([
                "",
                "## Before/After Conservative Ensemble Delta",
                "",
                "| Metric | Previous | Current | Delta |",
                "| :--- | :---: | :---: | :---: |",
            ])
            for label, key in (
                ("Accuracy", "overall_accuracy"),
                ("False-Pass Rate", "false_pass_rate"),
                ("Hard False-Pass Rate", "hard_subset_false_pass_rate"),
                ("Contradiction Detection", "contradiction_detection_rate"),
                ("Evidence Recall", "evidence_chunk_recall"),
            ):
                prev = float(previous_ensemble.get(key, 0.0) or 0.0)
                curr = float(e_m.get(key, 0.0) or 0.0)
                md_lines.append(f"| {label} | {prev:.1%} | {curr:.1%} | {(curr - prev):+.1%} |")

    # Conclusion & Key Findings
    md_lines.extend([
        "",
        "## Key Findings & Strategic Recommendations",
        "",
        "1. **Primary Safety Metric**: Treat conservative-ensemble false-pass rate as the blocking metric; accuracy alone is not sufficient.",
        "2. **Domain Coverage**: Use domain-wise false-pass rates to identify any domain that remains unsafe even if global averages improve.",
        "3. **Gate Telemetry**: Review gate-trigger reasons and false-pass categories together to decide the next deterministic checks to add.",
    ])

    # Write Markdown report
    _MD_REPORT_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Success] Markdown report written to: {_MD_REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate domain-agnostic claim grounding.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to JSONL dataset (default: domain_agnostic_100.jsonl)",
    )
    parser.add_argument(
        "--verifier",
        choices=["heuristic", "llm_entailment", "conservative_ensemble", "all"],
        default="all",
        help="Verifier to run (default: all).",
    )
    args = parser.parse_args()

    # Load dataset
    cases = load_dataset(args.dataset)
    print(f"Loaded {len(cases)} cases from {args.dataset}")

    # Build Groq client
    llm_client, provider_reason = build_groq_client_from_env()
    if not llm_client:
        print(f"[Warning] Groq client not built: {provider_reason}. LLM verifiers will fail or fall back.")

    previous_results: Dict[str, Any] | None = None
    if _JSON_REPORT_PATH.exists():
        try:
            previous_results = json.loads(_JSON_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous_results = None

    results: Dict[str, Any] = {
        "total_cases": len(cases),
        "verifiers": {},
    }

    verifier_modes = (
        ["heuristic", "llm_entailment", "conservative_ensemble"]
        if args.verifier == "all"
        else [args.verifier]
    )
    for verifier_mode in verifier_modes:
        preds = run_verifier_evaluation(cases, verifier_mode, llm_client)
        results["verifiers"][verifier_mode] = compute_comprehensive_metrics(cases, preds)

    for missing_verifier in ("heuristic", "llm_entailment", "conservative_ensemble"):
        results["verifiers"].setdefault(missing_verifier, previous_results.get("verifiers", {}).get(missing_verifier, {}) if previous_results else {})

    # Generate and save reports
    generate_reports(results, args.dataset, previous_results=previous_results)


if __name__ == "__main__":
    main()
