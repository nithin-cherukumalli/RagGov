# GovRAG-Calib Evaluation Results

This directory stores generated GovRAG-Calib evaluator reports.

Default evaluator outputs:

- `govrag_calib_eval_latest.json`
- `govrag_calib_eval_latest.md`

Generate them with:

```bash
python scripts/evaluate_govrag_calib.py evals/govrag_calib/calib_150_seed.jsonl
```

Reports are evaluation artifacts only. They must not be interpreted as calibrated confidence, production readiness, or production gating eligibility.

Every report must preserve:

- `calibration_status: "not_calibrated"`
- `production_gating_eligible: false`
- `confidence_intervals_available: false`
- `heldout_split_locked: false`
