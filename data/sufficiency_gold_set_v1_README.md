# Sufficiency Gold Set v1

This JSONL file is a developer calibration set for the GovRAG sufficiency analyzer. It is not used by runtime diagnostics.

Each line is one hand-labeled RAG retrieval case with a query, retrieved chunks, and a gold sufficiency label.

## Required Fields

- `example_id`: stable example identifier.
- `query`: user query.
- `retrieved_chunks`: chunks returned by retrieval.
- `gold_sufficiency_label`: one of `sufficient`, `partial`, `insufficient`, or `unknown`.

## Optional Fields

- `generated_answer`
- `gold_missing_evidence`
- `gold_covered_evidence`
- `gold_relevant_chunk_ids`
- `gold_irrelevant_chunk_ids`
- `gold_contradicting_chunk_ids`
- `gold_should_abstain`
- `gold_failure_stage`
- `gold_notes`

## Labeling Guidelines

- `sufficient`: retrieved chunks contain enough evidence to answer the query accurately, even if some irrelevant chunks are present.
- `partial`: some relevant evidence is present, but an exception, scope condition, supersession detail, contradiction, or other required evidence is missing or unresolved.
- `insufficient`: retrieved chunks do not contain the key evidence needed to answer safely.
- `unknown`: annotator cannot determine sufficiency from the example.

Use `gold_should_abstain=true` when a system should avoid answering because the context is insufficient, partial in a high-stakes way, or contradictory.

## Category Coverage (v1, 15 examples)

| Category | Examples |
|----------|----------|
| Fully sufficient (all evidence present) | suff_001, suff_004, suff_008, suff_014 |
| Scope mismatch (wrong domain applied) | suff_002, suff_007, suff_015 |
| Exception missing (main rule present, exception absent) | suff_003 |
| Noisy but answerable (relevant chunk buried in irrelevant) | suff_004 |
| Contradictory context (old and new GOs both present) | suff_005, suff_006 |
| Stale context (superseded order retrieved) | suff_009 |
| Multi-hop missing (dependent document absent) | suff_010 |
| Numeric missing (required number absent) | suff_011 |
| Date missing (effective date absent) | suff_012 |
| Authority missing (issuing authority absent) | suff_013 |

## Running Calibration

```bash
python tools/calibrate_sufficiency.py \
  --gold-set data/sufficiency_gold_set_v1.jsonl \
  --output data/calibration_report_v1.json \
  --llm mock
```

The report compares `term_coverage` and `requirement_aware` sufficiency modes. The checked-in v1 report uses the deterministic mock LLM so the requirement-aware path is measured instead of falling back to term coverage. Run without `--llm mock` to measure fallback behavior, or with a real configured provider to measure practical LLM performance.

## Interpreting the Report

The most important metric is **false-pass rate**: among gold `insufficient` or `partial` cases, how often did the analyzer avoid marking the case `insufficient`? This is dangerous because it allows downstream generation when evidence is incomplete.

Insufficient-context F1 is useful for threshold tuning, but do not treat it as validation until the gold set is larger and representative.

---

## Calibration Results (v1, 15 examples)

### V1 Advisory Status

Requirement-aware mode is the default advisory sufficiency mode. On the 15-example v1 gold set with the deterministic mock LLM, it reduced the dangerous false-pass rate from 0.818 to 0.000. This is good enough to expose likely insufficiency signals to downstream components, but it is **NOT FOR GATING**.

Advisory use means: the analyzer can recommend abstention or retrieval expansion, but it must not block generation by itself. Downstream components should weigh sufficiency alongside retrieval quality, grounding, attribution, parser validation, and user/product policy.

Critical caveat: requirement-aware mode over-flags in this seed set. Its false-fail rate is 1.000, meaning every gold-sufficient example was marked insufficient by the mock-LLM plus lexical-verifier path. That is acceptable for an advisory signal, but it is not acceptable for automated gating.

Calibration status: `preliminary_calibrated_v1`.

### Term Coverage Mode (baseline, min_coverage_ratio=0.3)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.133 |
| Insufficient Recall | 0.333 |
| Insufficient Precision | 1.000 |
| Insufficient F1 | 0.500 |
| **False-Pass Rate (DANGEROUS)** | **0.818** |
| False-Fail Rate | 0.000 |
| Best Threshold (from sweep) | 0.70 |
| Best Threshold F1 | 0.706 |

**Known Limitations:**
- Cannot detect missing evidence when query terms overlap with chunk text
- False-pass rate of 0.818 means 81.8% of insufficient/partial cases are missed
- Should NOT be used as a gating mechanism for generation
- Only useful as a cheap smoke test for gross retrieval failure (all chunks missing)
- Default threshold 0.3 is poorly calibrated; sweep suggests 0.70 is better, but even at 0.70 many structural failures pass through

### Requirement-Aware Mode (default advisory, mock LLM + lexical coverage)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.400 |
| Insufficient Recall | 1.000 |
| Insufficient Precision | 0.400 |
| Insufficient F1 | 0.571 |
| **False-Pass Rate (DANGEROUS)** | **0.000** |
| False-Fail Rate | 1.000 |

**Known Limitations:**
- Evidence requirements extracted by uncalibrated LLM judge or deterministic mock
- Coverage verified by lexical overlap only (not NLI)
- Lexical overlap thresholds (0.5, 0.25) are not empirically tuned
- Cannot detect contradictory or stale evidence
- No validation against ground truth beyond this gold set
- Falls back to term_coverage when an LLM client is not configured
- False-fail rate is high; use only as advisory signal

### Locking Status

- **Schema**: LOCKED (EvidenceRequirement, EvidenceCoverage, SufficiencyResult)
- **Harness**: LOCKED (calibration tool, metrics, report format)
- **Term Coverage Mode**: LOCKED as heuristic baseline, NOT recommended for gating
- **Requirement-Aware Mode**: LOCKED as v1 advisory default, NOT recommended for gating
- **Calibration Quality**: PRELIMINARY (15 examples, single labeler, no inter-annotator agreement)
- **Production Readiness**: advisory only (needs 50+ examples, multiple labelers, NLI verification before gating)

### What "Locked" Means

- The schema will not change without a migration path
- The calibration methodology is fixed
- Behavior is measured and documented
- Known limitations are explicit
- Downstream components CAN build against the schema
- Downstream components SHOULD NOT treat sufficiency decisions as authoritative
  unless they check `calibration_status` first

### Next Milestone: `calibrated_dev`

Requirements to advance from `preliminary_calibrated_v1` to `calibrated_dev`:
- 50+ gold examples
- Multiple labelers with measured inter-annotator agreement
- NLI-based coverage verification (replace lexical overlap)
- LLM requirement extraction evaluated against gold requirements
- Threshold re-swept on new gold set
