#!/usr/bin/env python3
"""Stress test runner for GovRAG failure analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stresslab.runners.run_case import run_case


def run_single_case(case_id: str, profile: str = "lan", dry_run: bool = True) -> None:
    """Run a single case and pretty-print the diagnosis."""
    try:
        result = run_case(case_id=case_id, profile=profile, dry_run=dry_run)

        print(f"\n{'='*80}")
        print(f"CASE: {result.case_id}")
        print(f"{'='*80}")

        # Case metadata
        print(f"\nQuery: {result.run.query}")
        print(f"Retrieved chunks: {len(result.run.retrieved_chunks)}")
        print(f"Final answer: {result.run.final_answer[:200]}...")

        # Diagnosis
        print(f"\n--- DIAGNOSIS ---")
        print(f"Primary Failure: {result.diagnosis.primary_failure.value}")
        print(f"Root Cause Stage: {result.diagnosis.root_cause_stage.value}")
        print(f"Should Have Answered: {result.diagnosis.should_have_answered}")
        print(f"Security Risk: {result.diagnosis.security_risk.value}")
        print(f"Confidence: {result.diagnosis.confidence}")

        if result.diagnosis.secondary_failures:
            print(f"Secondary Failures: {[f.value for f in result.diagnosis.secondary_failures]}")

        print(f"\n--- EVIDENCE ---")
        for evidence in result.diagnosis.evidence:
            print(f"  - {evidence[:100]}")

        print(f"\n--- RECOMMENDED FIX ---")
        print(f"{result.diagnosis.recommended_fix}")

        print(f"\n--- EXPECTED vs ACTUAL ---")
        expected_primary = result.run.metadata.get("expected_primary_failure", "?")
        print(f"Expected Primary Failure: {expected_primary}")
        print(f"Actual Primary Failure: {result.diagnosis.primary_failure.value}")
        print(f"Match: {expected_primary == result.diagnosis.primary_failure.value}")

        # Analyzer results summary
        print(f"\n--- ANALYZER RESULTS ---")
        for analyzer_result in result.diagnosis.analyzer_results:
            status_mark = "✓" if analyzer_result.status == "pass" else (
                "⚠" if analyzer_result.status == "warn" else (
                    "✗" if analyzer_result.status == "fail" else "⊘"
                )
            )
            print(f"{status_mark} {analyzer_result.analyzer_name}: {analyzer_result.status}")
            if analyzer_result.failure_type:
                print(f"   → {analyzer_result.failure_type.value}")

        # Retrieved chunks
        print(f"\n--- RETRIEVED CHUNKS ({len(result.run.retrieved_chunks)}) ---")
        for i, chunk in enumerate(result.run.retrieved_chunks[:3]):
            print(f"\n  [{i}] {chunk.source_doc_id} (score: {chunk.score:.3f})")
            print(f"      {chunk.text[:100]}...")

    except Exception as e:
        print(f"ERROR running case {case_id}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def main() -> None:
    """Run specified stress cases."""
    cases = [
        "parse_hierarchy_loss_ms20",
        "parse_table_corruption_ms39",
        "embedding_semantic_drift_duplicates",
        "retrieval_missing_critical_context_ms20",
        "abstention_required_private_fact",
    ]

    for case_id in cases:
        run_single_case(case_id, dry_run=True)


if __name__ == "__main__":
    main()
