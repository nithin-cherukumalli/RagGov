#!/usr/bin/env python
"""
Analyze and document false passes for the ConservativeEnsembleVerifier.
Generates claim_grounding_false_pass_analysis.json and claim_grounding_false_pass_analysis.md.
"""

from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List

# Add src/ to python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.run_eval import load_dataset, predict
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

def parse_units(text: str) -> List[str]:
    """Extract potential units from text (e.g. mg, %, $, Rs., days, months, GB, MB, ms, version-like values, etc.)."""
    units = []
    # Match percentage, currency symbols
    if "%" in text:
        units.append("%")
    if "$" in text:
        units.append("$")
    if "₹" in text or "rs." in text.lower() or "rupees" in text.lower() or "lakhs" in text.lower():
        units.append("Rs. / Lakhs")
    
    # Match common unit terms
    common_units = [
        "mg", "ml", "days", "months", "years", "hours", "seconds", "minutes",
        "gb", "mb", "kb", "ms", "km", "meters", "degrees", "celsius", "fahrenheit",
        "citizens", "people", "requests", "patients", "participants", "users", "version", "v"
    ]
    for unit in common_units:
        if re.search(r'\b' + re.escape(unit) + r'\b', text.lower()):
            units.append(unit)
            
    return sorted(list(set(units)))

def main():
    dataset_path = _REPO_ROOT / "evals" / "claim_grounding" / "domain_agnostic_100.jsonl"
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases.")

    llm_client, _ = build_groq_client_from_env()
    if not llm_client:
        print("Error: Groq client not built. GROQ_API_KEY required.")
        sys.exit(1)

    config = {
        "claim_verifier_mode": "conservative_ensemble",
        "enable_triplet_extraction": False,
        "llm_client": llm_client,
    }

    selector = EvidenceCandidateSelector(config)
    extractor = build_triplet_extractor(config)
    
    heur_verifier = HeuristicValueOverlapVerifier(config)
    llm_verifier = LLMClaimEntailmentVerifierV1(config)
    ensemble_verifier = ConservativeEnsembleVerifier(config)

    heur_builder = ClaimEvidenceBuilder(heur_verifier, selector, triplet_extractor=extractor)
    llm_builder = ClaimEvidenceBuilder(llm_verifier, selector, triplet_extractor=extractor)
    ensemble_builder = ClaimEvidenceBuilder(ensemble_verifier, selector, triplet_extractor=extractor)

    false_passes = []

    print("Running predictions...")
    for idx, case in enumerate(cases, 1):
        gold = case.gold_label
        
        # Get Ensemble prediction (1 LLM call)
        ensemble_res = predict(case, ensemble_builder)
        
        # Check if Ensemble is a false pass
        is_false_pass = (ensemble_res.label == "entailed" and gold in ("unsupported", "contradicted"))
        
        if is_false_pass:
            # Reconstruct Heuristic and LLM labels on false pass
            heur_res = predict(case, heur_builder)
            from raggov.analyzers.grounding.verifiers import _verification_label_for_support_label
            llm_label = _verification_label_for_support_label(ensemble_res.final_support_label_before_gate)
        
        if is_false_pass:
            print(f"False pass detected: ID: {case.case_id} | Gold: {gold} | Ensemble: {ensemble_res.label}")
            
            # Find best supporting chunk
            best_chunk_id = ensemble_res.best_candidate_id or (case.retrieved_chunks[0].chunk_id if case.retrieved_chunks else None)
            best_chunk_text = ""
            if best_chunk_id and case.retrieved_chunks:
                for chunk in case.retrieved_chunks:
                    if chunk.chunk_id == best_chunk_id:
                        best_chunk_text = chunk.text
                        break
            
            # Parse units
            critical_units = parse_units(case.claim_text)
            
            # Check matches / conflicts
            failure_type = getattr(case, "failure_type", "unknown")
            is_value_mismatch = "value" in failure_type or "contradicted_value" == failure_type or "insufficient_missing_value" == failure_type
            is_entity_mismatch = "entity" in failure_type or "insufficient_missing_entity" == failure_type
            is_date_mismatch = "date" in failure_type or "contradicted_date" == failure_type or "insufficient_missing_date" == failure_type
            is_related_non_supporting = "lexical_decoy" in failure_type or "citation_like_mismatch" == failure_type
            is_compound = getattr(case, "is_compound", False)
            
            # Assign likely failure cause category
            likely_cause = "llm_overpermissive_no_deterministic_gate"
            if is_value_mismatch:
                if any(u in ["mg", "%", "$", "Rs. / Lakhs", "GB", "MB", "ms", "days", "months", "version"] for u in critical_units):
                    likely_cause = "unit_mismatch_missed"
                else:
                    likely_cause = "value_mismatch_missed"
            elif is_date_mismatch:
                likely_cause = "date_mismatch_missed"
            elif is_entity_mismatch:
                likely_cause = "entity_mismatch_missed"
            elif is_related_non_supporting:
                likely_cause = "related_but_non_supporting"
            elif is_compound:
                likely_cause = "compound_partial_support"
            
            record = {
                "case_id": case.case_id,
                "domain": case.domain,
                "difficulty": getattr(case, "difficulty", "easy"),
                "failure_type": failure_type,
                "claim": case.claim_text,
                "expected_label": gold,
                "predicted_label": ensemble_res.label,
                "evidence_chunk_id": best_chunk_id,
                "evidence_text": best_chunk_text,
                "llm_verdict": llm_label,
                "heuristic_verdict": heur_res.label,
                "safety_gate_triggered": ensemble_res.safety_gate_triggered,
                "safety_gate_reason": ensemble_res.safety_gate_reason,
                "critical_entities": getattr(case, "critical_entities", []),
                "critical_values": getattr(case, "critical_values", []),
                "critical_dates": getattr(case, "critical_dates", []),
                "critical_units": critical_units,
                "value_mismatch_existed": is_value_mismatch,
                "entity_mismatch_existed": is_entity_mismatch,
                "date_mismatch_existed": is_date_mismatch,
                "evidence_related_but_non_supporting": is_related_non_supporting,
                "claim_was_compound": is_compound,
                "likely_failure_cause": likely_cause,
            }
            false_passes.append(record)
            
    # Write to JSON
    json_path = _REPO_ROOT / "reports" / "claim_grounding_false_pass_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(false_passes, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(false_passes)} false passes to {json_path}")

    # Generate Markdown Report
    md_lines = [
        "# Claim Grounding False Pass Diagnostic Analysis",
        "",
        "This report provides an in-depth audit of the false-pass failures made by the `ConservativeEnsembleVerifier` on our 105-case domain-agnostic claim grounding benchmark.",
        "",
        f"**Total False-Pass Cases Audited**: {len(false_passes)}",
        "",
        "## Executive Summary",
        "",
        "A false pass occurs when the verifier accepts a claim as supported (entailed) when the evidence actually contradicts it or contains insufficient details to confirm it. This represents the **highest safety risk** for GovRAG.",
        "",
        "Our audit reveals that while the `ConservativeEnsembleVerifier` blocks some clear value conflicts, the current implementation is **too permissive** because:",
        "1. It lacks deep entity alignment checking (allowing claims to swap related but distinct names).",
        "2. Value/date coverage gates only check if the string exists in the evidence, completely missing wrong values for the *same* attributes (e.g. timeout of 30 seconds vs 60 seconds).",
        "3. It misses unit and magnitude mismatches entirely.",
        "4. High-overlap lexical decoys easily bypass semantic entailment.",
        "",
        "---",
        "",
        "## False Pass Failure Cause Distribution",
        "",
        "| Likely Failure Cause Category | Count | Percentage | Primary Mitigation Strategy |",
        "| :--- | :---: | :---: | :--- |",
    ]

    causes = [r["likely_failure_cause"] for r in false_passes]
    cause_counts = {c: causes.count(c) for c in set(causes)}
    sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)

    cause_desc = {
        "value_mismatch_missed": "Critical values were missing or mismatched in context.",
        "date_mismatch_missed": "Critical date attributes did not align with evidence.",
        "unit_mismatch_missed": "Units (%, $, mg, etc.) or magnitudes did not match.",
        "entity_mismatch_missed": "Entities did not match or crucial entities were swapped.",
        "related_but_non_supporting": "High overlap text but lacks supporting predicate.",
        "compound_partial_support": "Compound claim accepted when only some clauses are supported.",
        "llm_overpermissive_no_deterministic_gate": "LLM was over-permissive and bypassed heuristic gates."
    }

    for cause, count in sorted_causes:
        pct = count / len(false_passes)
        md_lines.append(f"| `{cause}` | {count} | {pct:.1%} | {cause_desc.get(cause, 'Deterministic gate enforcement')} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Detailed Case Audit Registry",
        "",
    ])

    for r in false_passes:
        md_lines.extend([
            f"### Case ID: {r['case_id']} ({r['domain'].replace('_', ' ').title()})",
            "",
            f"- **Claim**: \"{r['claim']}\"",
            f"- **Evidence text**: \"{r['evidence_text']}\"",
            f"- **Gold Label**: `{r['expected_label']}` | **Predicted**: `{r['predicted_label']}`",
            f"- **Failure Type**: `{r['failure_type']}`",
            f"- **Likely Failure Cause**: `{r['likely_failure_cause']}`",
            f"- **Telemetry Context**:",
            f"  - Heuristic verdict: `{r['heuristic_verdict']}` | LLM verdict: `{r['llm_verdict']}`",
            f"  - Critical Entities: {r['critical_entities']}",
            f"  - Critical Values: {r['critical_values']}",
            f"  - Critical Dates: {r['critical_dates']}",
            f"  - Critical Units: {r['critical_units']}",
            "",
            "---",
        ])

    md_path = _REPO_ROOT / "reports" / "claim_grounding_false_pass_analysis.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved Markdown report to {md_path}")

if __name__ == "__main__":
    main()
