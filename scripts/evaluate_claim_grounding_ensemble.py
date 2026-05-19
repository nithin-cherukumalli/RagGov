#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def main():
    report_path = Path("reports/claim_grounding_ensemble_report.json")
    if not report_path.exists():
        print(f"Error: {report_path} not found.")
        sys.exit(1)

    with open(report_path) as f:
        data = json.load(f)

    # Telemetry presence checks
    if "metrics" not in data:
        data = {"metrics": data} # Wrap it if it's the raw metrics dict
        
    metrics = data.get("metrics", data)
    
    # Extract key metrics
    fp_rate = metrics.get("false_pass_rate", 1.0)
    chunk_recall = metrics.get("evidence_chunk_recall", 0.0)
    safety_gates = metrics.get("safety_gate_downgrade_count", 0)
    disagreements = metrics.get("llm_supported_heuristic_contradicted_count", 0)

    print("=== Ensemble Acceptance Gate ===")
    print(f"False-Pass Rate: {fp_rate:.1f}%")
    print(f"Evidence Chunk Recall: {chunk_recall:.1f}%")
    print(f"Safety Gate Triggered Count: {safety_gates}")
    print(f"Verifier Disagreement Count: {disagreements}")

    # Acceptance conditions
    if fp_rate > 0.0:
        print("FAIL: False-pass rate must be 0.0 for safe-release.")
        sys.exit(1)
        
    if chunk_recall < 90.0: # Current baseline is >90%
        print("FAIL: Evidence chunk recall dropped below baseline (90.0%).")
        sys.exit(1)

    print("PASS: Conservative Ensemble meets safety requirements.")

if __name__ == "__main__":
    main()
