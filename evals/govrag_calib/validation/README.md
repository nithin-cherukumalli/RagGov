# GovRAG-Calib-150 Validation

Validate the seed dataset with:

```bash
python scripts/validate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl
```

The validator checks JSONL parsing, required fields, enum values, unique `case_id` values, referenced claim/doc IDs, acceptable alternative failure labels, heldout locking, distribution summary, and production-gating honesty.

Warnings are expected while the dataset has fewer than 150 records or has below-target family counts.

Any record that implies `production_gating_eligible` is invalid. GovRAG-Calib-150 does not authorize production gating until reviewed calibration evidence and locked heldout results exist.
