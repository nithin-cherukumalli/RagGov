# GovRAG-Calib-150

GovRAG-Calib-150 is the calibration dataset scaffold for GovRAG diagnosis evaluation.

This dataset is not complete yet. The current `calib_150_seed.jsonl` file contains seed records only, derived from existing common benchmark cases with provenance metadata. Seed cases are calibration scaffolding, not statistical calibration evidence.

No production calibration claim is allowed from this dataset yet.

`production_gating_eligible` must remain false until GovRAG has a complete reviewed/adjudicated calibration set, a locked heldout split, confidence intervals, and documented calibration results.

## Target Distribution

The target size is 150 labeled cases:

- `clean_pass`: 15
- `retrieval`: 25
- `grounding`: 25
- `citation`: 20
- `sufficiency`: 15
- `version_validity`: 15
- `security_privacy`: 20
- `answer_quality`: 15

## Files

- `schema.json`: JSON Schema contract for GovRAG-Calib-150 JSONL records.
- `calib_150_seed.jsonl`: seed dataset records.
- `splits/README.md`: split policy.
- `validation/README.md`: validation policy.

## Validate

```bash
python scripts/validate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl
```

The validator reports errors for schema-breaking records and warnings for an incomplete dataset or below-target family counts.
