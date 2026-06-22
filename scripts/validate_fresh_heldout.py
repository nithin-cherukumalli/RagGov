#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import Counter
import sys
import importlib.util

def normalize(text):
    if not text: return ""
    return re.sub(r'\s+', '', str(text).lower())

def load_jsonl(path):
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_existing_signatures():
    root = Path(__file__).resolve().parents[1]
    
    paths = [
        root / "evals" / "govrag_calib" / "govrag_calib_150.jsonl",
        root / "evals" / "govrag_calib" / "staging" / "raw" / "induced_candidates.jsonl",
        root / "evals" / "govrag_calib" / "staging" / "raw" / "starter_seed_intake.jsonl",
        root / "evals" / "govrag_calib" / "splits" / "heldout_v0_1.jsonl",
        root / "evals" / "govrag_calib" / "splits" / "heldout_candidate_v0_2.jsonl"
    ]
    
    seen_qa = set()
    seen_ids = set()
    
    for p in paths:
        for row in load_jsonl(p):
            # Source id for seeds vs case id for standard
            sid = row.get("source_id") or row.get("case_id")
            if sid:
                seen_ids.add(str(sid))
            
            q = normalize(row.get("query"))
            a = normalize(row.get("answer") or row.get("reference_answer"))
            if q and a:
                seen_qa.add(q + "|||" + a)
                
    return seen_ids, seen_qa

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_fresh_heldout.py <path_to_fresh_intake.jsonl>")
        sys.exit(1)
        
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Error: {in_path} not found.")
        sys.exit(1)
        
    seen_ids, seen_qa = get_existing_signatures()
    fresh_rows = load_jsonl(in_path)
    
    survivors = []
    
    for row in fresh_rows:
        sid = row.get("source_id") or row.get("case_id")
        q = normalize(row.get("query"))
        a = normalize(row.get("answer") or row.get("reference_answer"))
        
        is_dup = False
        if str(sid) in seen_ids:
            is_dup = True
        elif (q + "|||" + a) in seen_qa:
            is_dup = True
            
        if not is_dup:
            survivors.append(row)
            
    print(f"Loaded {len(fresh_rows)} fresh rows.")
    print(f"Dropped {len(fresh_rows) - len(survivors)} duplicates/overlaps.")
    print(f"Survived: {len(survivors)} rows.")
    
    if not survivors:
        print("No survivors left to validate.")
        return
        
    counts_by_type = Counter()
    counts_by_source = Counter()
    
    for s in survivors:
        ds = s.get("source_dataset", "unknown")
        ft = s.get("source_label") or s.get("expected_primary_failure") or "clean"
        counts_by_source[ds] += 1
        counts_by_type[ft] += 1
        
    print("\n--- Survivors by Source ---")
    for k, v in counts_by_source.items():
        print(f"{k}: {v}")
        
    print("\n--- Survivors by Label/Type ---")
    for k, v in counts_by_type.items():
        print(f"{k}: {v}")
        
    print("\n--- Heuristic Label Flags (Requires Human Review) ---")
    for s in survivors:
        lbl = s.get("source_label", "")
        ds = s.get("source_dataset", "")
        if ds == "ragtruth" and lbl in ["contradicted", "unsupported"]:
            sid = s.get("source_id") or s.get("case_id")
            print(f"FLAG: {sid} is a heuristic '{lbl}'. Please review manually before adding.")
            
    print("\n--- Validating Schema via add_calib_case ---")
    
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import add_calib_case
        ftypes = add_calib_case._failure_types()
        
        # We assume survivors are in starter_seed format and need to be mapped to Calib format
        # or they are already mapped.
        # If they are from pull_seed_intake.py they are in seed format. We will mock the validation 
        # but warn if they are missing required keys.
        
        for i, s in enumerate(survivors):
            # Try validating if it has Calib format keys
            if "case_id" in s:
                errs = add_calib_case.validate(s, set(), ftypes)
                hard = [e for e in errs if not e.startswith("WARN")]
                if hard:
                    print(f"Row {i} ({s.get('case_id')}) schema errors: {hard}")
            else:
                if i == 0:
                    print("Note: Fresh rows are in seed format, skipping strict calibration schema validation. Map to calibration schema first.")
                    break
    except ImportError:
        print("Warning: could not import add_calib_case.py")

    # Scoring Stub
    print("\n--- Provisional Scoring on Survivors (assuming mapped schema) ---")
    print("If rows are still in seed format, run induction/mapping first.")
    # Here the user would run the engine. Since we are just providing a stub:
    print("from raggov import DiagnosisEngine, RAGRun ... (stub execution)")

if __name__ == "__main__":
    main()
