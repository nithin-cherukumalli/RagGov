# CLEAN false-positive audit — Opus verification of Sidekick-1 (2026-06-18)

Sidekick 1 (Antigravity) recommended "warn-level fallback escalation" as the safest +6-row fix.
I verified before acting. **The recommendation does not hold.**

## Verified breakdown of the 17 expected-CLEAN false positives
- **15 are FAIL-level winners** (genuine fail-level signals firing on CLEAN rows):
  INSUFFICIENT_CONTEXT 5 (scope marker — see Task 26, entangled), STALE_RETRIEVAL 3,
  CITATION_MISMATCH 2, UNSUPPORTED_CLAIM 2, CONTRADICTED_CLAIM 2, GENERATION_IGNORE 1.
- **Only 2 are WARN-level winners** (INCONSISTENT_CHUNKS) — the negation residuals already
  documented as irreducible (Task 19). Not 6.

## Why the warn-level lever is also unsafe
Across probe+Calib, a warn-level signal as primary is **never correct** (0 correct, 7 wrong on
probe; 0/0 Calib). But blanket "no fail-level winner → return CLEAN" is dangerous:
- 2 of the 7 warn-wrong rows are expected-CLEAN → would correctly become CLEAN (+2).
- **5 are expected non-CLEAN** (real failures the engine only weakly/warn-detected). Suppressing
  warn → CLEAN converts a visible wrong-type diagnosis into a **silent false-CLEAN** — the most
  dangerous error class, which this project explicitly works to reduce.

Net of blanket warn-suppression: +2 CLEAN exact-match, **+5 dangerous false-CLEAN**. Rejected.

## Conclusion — native-mode CLEAN precision has hit diminishing returns
The remaining CLEAN false positives are fail-level over-firing spread across many analyzers, each
entangled with its own true positives (proven for the INSUFFICIENT/scope bucket in Task 26: the
same mechanism produced 4 FPs and 3 TPs). There is no single safe high-recovery CLEAN-precision fix
left in native heuristic mode. Further gains require: (a) stronger verifiers / optional NLI
(value/entity/scope grounding), or (b) accepting a precision/recall trade — neither of which is a
quick policy patch.

## Honest next levers (not more native heuristic tweaks)
1. Real, non-overlapping heldout (pending the user's fresh-data pull) — to know the true number.
2. Optional NLI/entailment verifier path (out of native default) — for CONTRADICTED recall and
   answer/scope grounding.
3. Continue the honesty doc edits.

Sidekick value note: good at breadth/enumeration, but its ranked judgment ("+6 warn", "safest") was
wrong on both count and safety. Re-verification (this doc) is mandatory before acting on sidekick
recommendations.
