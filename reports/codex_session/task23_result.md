# Task 23 result — source-assertion false-CLEAN repair — LANDED

**Date:** 2026-06-18. Prereg: `task23_prereg.md`. Implements the Codex sidekick report's
recommended first false-CLEAN patch.

## What changed
`src/raggov/analyzers/grounding/claims.py`: added `_SOURCE_ASSERTION_RE` (source/passage/
document/text/report/author + assertion verb: notes/states/reports/confirms/documents/says/
mentions/indicates/asserts/claims/observes/reaffirm…). When a sentence matches,
`HeuristicClaimExtractorV0` now treats it as substantive (`should_verify=True`), mirroring the
existing short-entity / entity-attribute hooks. Fabricated source-attribution suffixes are now
verified instead of silently skipped. No threshold/gate/policy change. List/short-answer recall
left out of scope (Task 24).

## Acceptance criteria
| # | Criterion | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Protected baseline | 41/46 | **41/46** | PASS |
| 2 | Calib scored primary | 23/45 (0.511) | **23/45 (0.511)** | PASS |
| 3 | Probe overall accuracy | 57/145 (0.393) | **80/145 (0.552)** | PASS (strict ↑) |
| 4 | Expected-CLEAN false positives | 17 | **17** (CLEAN-correct 13/30) | PASS (no rise) |
| 5 | `UNSUPPORTED_CLAIM → CLEAN` | 7 | **0** (target ≤2) | MET |

The +23 probe gain (not just the 7 CLEAN rows) is because the same fabricated suffix appears
across all 30 induced `UNSUPPORTED_CLAIM` rows; making it verifiable lets the grounding verifier
mark it unsupported, so `UNSUPPORTED_CLAIM` recall rose sharply (~2/30 → ~25/30) with no
expected-CLEAN precision cost.

## Tests
- `test_source_assertion_suffix_is_verifiable` and `test_source_topic_word_without_assertion_verb_is_not_promoted`
  (`test_claim_extractor.py`).
- `test_unsupported_source_assertion_suffix_is_not_silently_clean` (`test_grounding.py`):
  end-to-end — answer with a supported fact + unsupported source-assertion suffix →
  `ClaimGroundingAnalyzer` fails with `UNSUPPORTED_CLAIM`, not skip/CLEAN.
- Wider check: `tests/test_analyzers` + `tests/decision_policy` → 592 passed, 3 xfailed, 1
  pre-existing stale failure (`test_prompt_injection_warns_on_hits_below_risk_threshold`,
  unrelated English evidence-format drift).

## Honest note
This is a benchmark-shaped suffix; the win is real (silent-CLEAN on unsupported content is the
most dangerous failure) but partly reflects a single induced mutation pattern. Remaining
false-CLEAN: 8 (INSUFFICIENT_CONTEXT→CLEAN 6, CONTRADICTED→CLEAN 2) — addressed by Task 24
(list/short-answer recall) and Task 22 (contradiction promotion).
