# Locked gold heldout + Phase C target list (verified)

**Date:** 2026-06-22. The heldout is relabeled, two-stage adjudicated (Kimi proposal → Codex
chunk-by-chunk adjudication → Opus verification), and LOCKED. First trustworthy numbers.

## Provenance (defensible — good for the README)
- Provisional benchmark labels were broken (25 CONTRADICTED rows; 0/25 held up).
- Kimi labeled all 75 rows into the v1 taxonomy (proposal only).
- Codex re-judged every row against the chunks; overruled Kimi on 8 (all verified correct by Opus —
  mostly multi-hop HotpotQA where support spans two chunks, which Kimi missed).
- Opus reproduced the numbers and spot-checked 4 of the 8 disagreements against the raw chunks: sound.
- File: `evals/govrag_calib/staging/raw/heldout_real_v1_gold.jsonl` (`label_source=gold_human_adjudicated_v1`).

## Numbers (native engine, mapped to v1 taxonomy)
- gold distribution: CLEAN 46 / UNSUPPORTED_CLAIM 25 / INSUFFICIENT_CONTEXT 4 (CONTRADICTED 0).
- v1 exact-match accuracy: 27/75 = **0.36**
- CLEAN false-positive rate: 30/46 = **0.65**  ← dominant weakness
- failure DETECTION rate (any flag on a real failure): 18/29 = **0.62**
- per-type exact: CLEAN 16/46, UNSUPPORTED 10/25, INSUFFICIENT 1/4.

## Open product decision (Nithin)
ALCE/QAMPARI list-answers: 19 rows labeled UNSUPPORTED under the strict faithfulness rule. Flipping
them to CLEAN would make the gold 65/6/4 (87% clean) and gut recall measurement. RECOMMEND keep
UNSUPPORTED (faithfulness-correct + keeps the benchmark useful). Confirm or override.

## Phase C — the 30 CLEAN false-positives by engine type (vs locked gold)
| n | engine type | analyzer | note |
|---|-------------|----------|------|
| 8 | INSUFFICIENT_CONTEXT | SufficiencyAnalyzer | scope_condition / missing_exception over-fires (has Calib TPs → scope the fix) |
| 5 | CONTRADICTED_CLAIM | ClaimGroundingAnalyzer | high-severity false alarm; gold has 0 CONTRADICTED |
| 5 | STALE_RETRIEVAL | RetrievalDiagnosis/TemporalSourceValidity | has Calib TPs → careful |
| 5 | UNSUPPORTED_CLAIM | ClaimGroundingAnalyzer | over-extraction on clean answers |
| 4 | PRIVACY_VIOLATION | PrivacyAnalyzer | high-severity false alarm on general QA; DON'T disable (core gov feature) — tighten the trigger |
| 1 | POST_RATIONALIZED_CITATION / PROMPT_INJECTION / SCOPE_VIOLATION | — | singletons |

## Phase C order (one pre-registered increment each, measured on locked gold + Calib + protected)
1. **High-severity false alarms first** (governance credibility): why does CONTRADICTED_CLAIM (×5)
   and PRIVACY_VIOLATION (×4) fire on clean answers? Tighten the trigger; do NOT disable privacy
   (it's a core feature on real privacy data — these 5/4 are just over-firing on general QA).
2. **SufficiencyAnalyzer scope-condition (×8)** — scope the fix to the over-firing sub-rule (it has
   Calib TPs, so don't demote the whole analyzer).
3. **STALE_RETRIEVAL (×5)** and **UNSUPPORTED over-extraction (×5)**.
- Target: CLEAN-FP from 0.65 toward ≤ 0.35 without dropping the 0.62 detection rate.

## Discipline
Each increment pre-registered; measured on the LOCKED gold + Calib 23/45 + protected 43/46; revert
on any regression. The gold is the ruler — never tune on it.
