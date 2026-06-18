# Task 22 result — CONTRADICTED_CLAIM recall — NO-GO (native mode), no code change

**Date:** 2026-06-18. Prereg: `task22_prereg.md`. Decision: **do not implement** in native
mode. Evidence below. (Two reverts beat one false fix.)

## Investigation (current engine state, post Task 23)
Of the 15 expected `CONTRADICTED_CLAIM` probe rows:
- 3 (rows 1,3,4): `ClaimGroundingAnalyzer` = fail+CONTRADICTED_CLAIM, but engine selects
  `UNSUPPORTED_CLAIM` via `_require_explicit_contradiction`.
- 3 (6,8,11): analyzer itself collapses mixed claims to fail+UNSUPPORTED.
- 2 (7,12): analyzer warn+CONTRADICTED → loses to fail-level sufficiency.
- 2 (2,9): warn+UNSUPPORTED → POST_RATIONALIZED_CITATION.
- 2 (5,15): analyzer skip → CLEAN (Codex flagged ambiguous).
- 3 (10,13,14): fail+UNSUPPORTED, no contradiction evidence (Codex flagged noisy labels).

## Why the guard is correct (claim-record inspection of rows 1/3/4)
The "contradicted" claims have **empty `value_conflicts` and populated `value_matches`** —
the answer values actually MATCH the evidence (6025=6025, rating 4.0=4.0, 15 min=15 min,
25=25). The heuristic labels them "contradicted" only because *chunks disagree among
themselves* ("contradictory evidence exists among top-k candidate chunks"), not because the
answer contradicts the source. `_has_explicit_contradiction` correctly refuses to treat the
default `explicit_contradiction` label as hard evidence without a real value/date/unit conflict
or permission-polarity flip. The migrated RAGTruth contradiction is semantic, with no
value/date anchor the native heuristic can verify.

## Load-bearing-guard experiment (read-only; no commit)
Monkeypatched `_has_explicit_contradiction → True` (promote every heuristic contradiction):

| Set | Baseline | Guard disabled | Net |
|---|---|---|---|
| Calib (45) | 23/45 (0.511) | **22/45 (0.489)** | **−1 (FAILS gate 2)** |
| Probe (145) | 80/145 (0.552) | 83/145 (0.572) | +3 only |
| Probe CONTRADICTED correct | 0 | 8 | +8 |
| Probe UNSUPPORTED correct | 25 | **20** | **−5** |
| Probe false-CONTRADICTED (non-contra rows) | 0 | **10** | **+10 dangerous** |

Promotion regresses Calib (revert trigger), trades 5 real UNSUPPORTED for noisy contradiction,
and creates 10 false contradictions — exactly the over-promotion the guard prevents and the
sidekick warned against.

## Conclusion / path forward
- Native heuristic mode cannot safely recover `CONTRADICTED_CLAIM` here; the guard stays.
- Real recall requires the **optional LLM/NLI entailment verifier**
  (`ClaimEntailmentVerifierV1` / `claim_grounding_verifier_policy=llm_entailment`), which is
  deliberately out of native scope, **and/or a human audit** of the migrated RAGTruth
  contradicted labels (Codex flagged rows 5/10/13/14/15 as noisy/ambiguous).
- No code, labels, thresholds, gates, or flags changed. Prereg + this result kept as the record.
