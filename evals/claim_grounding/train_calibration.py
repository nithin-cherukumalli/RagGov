"""
Train and evaluate claim-grounding calibration models.

Loads a labeled grounding dataset, computes calibration metrics for the
current verifier, and optionally fits and saves a calibration artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Resolve repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.claim_grounding.run_eval import load_dataset, _to_retrieved_chunks
from raggov.analyzers.grounding.evidence_layer import ClaimEvidenceBuilder, detect_claim_type
from raggov.analyzers.grounding.verifiers import HeuristicValueOverlapVerifier
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidateSelector
from raggov.calibration.claim_calibration import CalibrationMode

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute Brier score and Expected Calibration Error (ECE).
    """
    # Brier Score
    brier = np.mean((y_prob - y_true) ** 2)
    
    # ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin
        in_bin = np.logical_and(y_prob > bin_lower, y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)
            
    return {
        "brier_score": float(brier),
        "ece": float(ece),
        "reliability_bins": {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bin_boundaries": bin_boundaries.tolist()
        }
    }


def collect_features(dataset_path: Path) -> List[Dict[str, Any]]:
    """Runs the verifier on the dataset and collects features for training."""
    cases = load_dataset(dataset_path)
    
    config = {}
    verifier = HeuristicValueOverlapVerifier(config)
    selector = EvidenceCandidateSelector(config)
    builder = ClaimEvidenceBuilder(verifier, selector)
    
    data_points = []
    for case in cases:
        chunks = _to_retrieved_chunks(case.retrieved_chunks)
        record = builder._build_single(case.claim_text, 0, case.query, chunks)
        
        # Binary label for 'entailed' calibration
        is_entailed_gold = 1.0 if case.gold_label == "entailed" else 0.0
        
        data_points.append({
            "features": {
                "raw_score": record.raw_support_score,
                "label": record.verification_label,
                "claim_type": record.claim_type,
                "fallback_used": record.fallback_used,
                "verifier_mode": record.verification_method,
            },
            "y_true": is_entailed_gold
        })
    return data_points


def train_reliability_table(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes mean accuracy per (mode, fallback, label) bucket."""
    buckets: Dict[str, List[float]] = {}
    for pt in data:
        f = pt["features"]
        key = f"{f['verifier_mode']}:{str(f['fallback_used']).lower()}:{f['label']}"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(pt["y_true"])
        
        # Also contribute to generic bucket
        gen_key = f"default:{f['label']}"
        if gen_key not in buckets:
            buckets[gen_key] = []
        buckets[gen_key].append(pt["y_true"])
        
    return {k: sum(v)/len(v) for k, v in buckets.items()}


def main():
    parser = argparse.ArgumentParser(description="Train claim grounding calibration.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to labeled JSONL.")
    parser.add_argument("--output", type=Path, help="Path to save calibration JSON.")
    parser.add_argument("--mode", choices=["reliability_table", "temperature_scaling"], default="reliability_table")
    args = parser.parse_args()
    
    print(f"Collecting features from {args.dataset}...")
    data = collect_features(args.dataset)
    
    y_true = np.array([pt["y_true"] for pt in data])
    y_prob = np.array([pt["features"]["raw_score"] for pt in data])
    
    metrics_pre = compute_calibration_metrics(y_true, y_prob)
    print("\nPre-calibration metrics:")
    print(f"  Brier Score: {metrics_pre['brier_score']:.4f}")
    print(f"  ECE: {metrics_pre['ece']:.4f}")
    
    artifact = {
        "mode": args.mode,
        "params": {},
        "metadata": {
            "training_dataset": str(args.dataset),
            "n_samples": len(data),
            "pre_calibration_ece": metrics_pre["ece"]
        }
    }
    
    if args.mode == "reliability_table":
        table = train_reliability_table(data)
        artifact["params"]["table"] = table
        print(f"\nReliability table trained with {len(table)} buckets.")
        
    elif args.mode == "temperature_scaling":
        # Toy temperature scaling - just search for best T in [0.1, 5.0]
        best_t = 1.0
        min_ece = metrics_pre["ece"]
        for t in np.linspace(0.1, 5.0, 50):
            # Sigmoid scaling
            probs = []
            for pt in data:
                s = pt["features"]["raw_score"]
                if s <= 0: probs.append(0.0)
                elif s >= 1: probs.append(1.0)
                else:
                    logit = np.log(s / (1 - s))
                    probs.append(1 / (1 + np.exp(-logit / t)))
            
            ece = compute_calibration_metrics(y_true, np.array(probs))["ece"]
            if ece < min_ece:
                min_ece = ece
                best_t = float(t)
        
        artifact["params"]["temperature"] = best_t
        print(f"\nTemperature scaling trained: T={best_t:.4f}, ECE={min_ece:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nArtifact saved to {args.output}")

if __name__ == "__main__":
    main()
