# Real CLEAN false-positives — diagnosis + NLI runbook (Phase 2)

**Date:** 2026-06-18.

## Diagnosis (why real CLEAN-FP = 0.76)
The 38 false positives on the 50 real clean answers are spread across many analyzers, not one:

| predicted (wrong) type | count | winning analyzer (top) |
|---|---|---|
| STALE_RETRIEVAL | 9 | TemporalSourceValidity / RetrievalDiagnosis |
| INSUFFICIENT_CONTEXT | 8 | SufficiencyAnalyzer |
| INCONSISTENT_CHUNKS | 7 | InconsistentChunksAnalyzer |
| UNSUPPORTED_CLAIM | 6 | ClaimGroundingAnalyzer |
| (tail: CONTRADICTED, CITATION, INJECTION, PRIVACY, SCOPE, GENERATION, ANOMALY) | 8 | various |

**Death by a thousand cuts.** Real clean RAG answers are long, multi-fact, entity/date/citation
rich, so every heuristic finds *something*. Confirmed: swapping the grounding verifier to NLI left
CLEAN-FP at 0.76 — because the FPs are mostly NOT from the grounding path.

## Implication
- No single per-analyzer fix recovers this, and per-analyzer tightening is entangled with that
  analyzer's true positives (proven in Task 26).
- The principled fix is a **cross-analyzer precision gate at the decision policy**: when the answer
  is **well-grounded** (claims entail the retrieved context) and there is no deterministic
  security/parser block, a single low-tier heuristic retrieval-health warning (stale / inconsistent
  / scope / citation) must NOT flip the verdict to a failure.
- That gate is only *trustworthy* when "well-grounded" comes from an **NLI/entailment verdict** —
  gating on the heuristic grounding signal (which is itself over-firing) would be circular. Hence
  the CLEAN-FP fix is **model-dependent** (the hybrid tier), not a native heuristic tweak.

## NLI runbook (run on YOUR machine — sandbox proxy-blocks api.groq.com)
```bash
export GROQ_API_KEY=...        # your key; never commit it. Rotate the one shared in chat.
PYTHONPATH=src:. python scripts/run_nli_heldout.py            # native vs llm-entailment on real heldout
```
This prints overall accuracy, CLEAN-FP rate, and per-type for NATIVE vs LLM-ENTAILMENT. With a real
model the entailment tier should improve UNSUPPORTED/grounding accuracy; CLEAN-FP will only move once
the grounded-clean policy gate (next increment) is added and keyed on the NLI verdict.

`scripts/groq_client.py` is the committed adapter (env-key only, no key in code). `--mock` runs the
wiring offline.

## Next engine increment (pre-register before coding)
"grounded-clean gate": if NLI says all answer claims are entailed/supported AND no
BLOCKING_DETERMINISTIC signal fired, suppress low-tier retrieval-health failures → CLEAN. Hard
gates: real-heldout CLEAN-FP drops materially; protected 43/46 unchanged; Calib ≥23/45; probe
overall not down; and **no real failure-type TP flips to CLEAN** (guard the dangerous direction).
Measured with `run_nli_heldout.py` (real model) + the standard guards.
