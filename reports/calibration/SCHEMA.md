# Calibration Report Schema

This directory defines the report contract for calibration and reliability outputs.
It is a schema scaffold only: no metric values here are evidence, and no entry
implies production calibration.

`scripts/eval_report.py` should write reports that conform to this shape. Phase 3
calibration work should fill the metric fields after a locked, non-overlapping,
double-adjudicated heldout exists.

## Top-Level Contract

Required top-level fields:

- `schema_version`: currently `1`.
- `generated_at`: ISO-8601 timestamp or `null` in templates.
- `code_ref`: object with `git_commit`, `dirty_worktree`, and optional notes.
- `dataset`: object describing the evaluated dataset path, split, fingerprint,
  row count, and label status.
- `overall`: metrics aggregated across the report scope.
- `modes`: exactly the execution modes reported by the evaluator. Required keys
  for this contract are `default` and `native`.
- `notes`: free-form strings for caveats and degradation metadata.

## Metric Block

Every `overall`, per-mode, and per-failure-type block uses the same metric shape:

```json
{
  "n": null,
  "correct": null,
  "accuracy": null,
  "confidence_mean": null,
  "ece": null,
  "ace": null,
  "brier": null,
  "reliability_curve_bins": [],
  "bootstrap_ci": {
    "method": "bootstrap",
    "confidence_level": 0.95,
    "resamples": null,
    "accuracy": {"low": null, "high": null},
    "ece": {"low": null, "high": null},
    "brier": {"low": null, "high": null}
  },
  "calibration_status": "not_evaluated",
  "gating_eligible": false
}
```

Field meanings:

- `n`: evaluated examples in the block.
- `correct`: exact primary-failure matches in the block.
- `accuracy`: `correct / n`.
- `confidence_mean`: arithmetic mean of emitted confidence values used for
  calibration analysis.
- `ece`: expected calibration error.
- `ace`: adaptive calibration error.
- `brier`: Brier score for the selected confidence/accuracy event.
- `reliability_curve_bins`: ordered list of reliability bins.
- `bootstrap_ci`: bootstrap confidence intervals for selected metrics.
- `calibration_status`: one of `not_evaluated`, `insufficient_data`,
  `provisional_uncalibrated`, `calibrated`, or `invalid`.
- `gating_eligible`: must remain `false` unless calibration evidence and policy
  explicitly support production gating.

## Reliability Bin Shape

Each `reliability_curve_bins` item must use this shape:

```json
{
  "bin_index": 0,
  "confidence_low": 0.0,
  "confidence_high": 0.1,
  "n": null,
  "accuracy": null,
  "confidence_mean": null,
  "gap": null
}
```

## Failure-Type Coverage

Each mode must include a `per_failure_type` object with all 25 current
`FailureType` values:

- `STALE_RETRIEVAL`
- `SCOPE_VIOLATION`
- `CITATION_MISMATCH`
- `INCONSISTENT_CHUNKS`
- `INSUFFICIENT_CONTEXT`
- `UNSUPPORTED_CLAIM`
- `CONTRADICTED_CLAIM`
- `PROMPT_INJECTION`
- `SUSPICIOUS_CHUNK`
- `RETRIEVAL_ANOMALY`
- `PRIVACY_VIOLATION`
- `LOW_CONFIDENCE`
- `TABLE_STRUCTURE_LOSS`
- `HIERARCHY_FLATTENING`
- `METADATA_LOSS`
- `POST_RATIONALIZED_CITATION`
- `PARSER_STRUCTURE_LOSS`
- `CHUNKING_BOUNDARY_ERROR`
- `EMBEDDING_DRIFT`
- `RETRIEVAL_DEPTH_LIMIT`
- `RERANKER_FAILURE`
- `GENERATION_IGNORE`
- `INCOMPLETE_DIAGNOSIS`
- `CLAIM_EXTRACTION_FAILED`
- `CLEAN`

Failure types with too few real examples must still be present and marked
`insufficient_data` or `not_evaluated`; omission is not allowed because it hides
coverage gaps.

