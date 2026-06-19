# Heldout Real v1 NLI Relabel Report

Date: 2026-06-19

Method status: `external_signal` / `llm_assisted_provisional`.

This is not gold, not calibration, and not a canonical dataset update. The script runs
`DiagnosisEngine` with `claim_grounding_verifier_policy=llm_entailment`, `KimiClient`,
and `model=moonshot-v1-8k`. Whole-answer labels are derived from per-claim NLI labels by
the most-severe-claim rule: contradicted > unsupported > clean; zero verifiable
claims remain `INSUFFICIENT_CONTEXT` for human review.

Input: `evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`

Relabeled staging output: `evals/govrag_calib/staging/raw/heldout_real_v1_relabeled.jsonl`

Human spot-audit worklist: `evals/govrag_calib/staging/raw/heldout_real_v1_nli_spot_audit_worklist.jsonl`

Rows relabeled: 75

## Mismapping Counts
- Source `CONTRADICTED_CLAIM` rows: 25 -> {'CLEAN': 1, 'CONTRADICTED_CLAIM': 10, 'INSUFFICIENT_CONTEXT': 6, 'UNSUPPORTED_CLAIM': 8}
- Source `CLEAN` rows: 50 -> {'CLEAN': 22, 'CONTRADICTED_CLAIM': 3, 'INSUFFICIENT_CONTEXT': 20, 'UNSUPPORTED_CLAIM': 5}; stayed CLEAN=22

## Source -> NLI Label Matrix
- `CLEAN` -> `CLEAN`: 22
- `CLEAN` -> `CONTRADICTED_CLAIM`: 3
- `CLEAN` -> `INSUFFICIENT_CONTEXT`: 20
- `CLEAN` -> `UNSUPPORTED_CLAIM`: 5
- `CONTRADICTED_CLAIM` -> `CLEAN`: 1
- `CONTRADICTED_CLAIM` -> `CONTRADICTED_CLAIM`: 10
- `CONTRADICTED_CLAIM` -> `INSUFFICIENT_CONTEXT`: 6
- `CONTRADICTED_CLAIM` -> `UNSUPPORTED_CLAIM`: 8

## Audit Sampling
- All source/NLI disagreements are included.
- Random agreeing rows included: 10 requested, deterministic seed `20260619`.

## Governance
- `label_source=llm_assisted_provisional`
- Gold/canonical dataset changed: no
- Dataset lock changed: no
- Thresholds/gates changed: no
- Human acceptance required before promotion.
