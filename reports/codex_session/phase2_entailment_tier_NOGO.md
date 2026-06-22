# Phase 2 entailment tier — NO-GO on real heldout (clean measurement)

**Date:** 2026-06-22. Measured on the locked real heldout (75 rows) with a 0%-fallback
local NLI run (`scripts/run_nli_heldout.py --provider local_nli`, model
cross-encoder/nli-deberta-v3-small). This supersedes all rate-limited cloud A/Bs.

## Numbers (clean, no fallback)
| arm | overall | CLEAN-FP | CONTRADICTED recall |
|-----|---------|----------|---------------------|
| NATIVE (heuristic) | 18/75 | 38/50 = **0.76** | 6/25 |
| LOCAL-NLI (T=0.6…0.9, identical) | 7/75 | 44/50 = **0.88** | 1/25 |

Threshold sweep 0.6 → 0.9 produced **byte-identical** output → the grounded-clean gate is not
the active variable; it barely fires.

## Why it fails (mechanism — this is the important part)
1. The small local NLI model labels paraphrased-but-faithful answer claims as `unsupported`
   (it misses paraphrase/entailment). So faithful answers now fail as **UNSUPPORTED_CLAIM**.
2. `UNSUPPORTED_CLAIM` is **not** in the gate's `_GROUNDED_CLEAN_SUPPRESSIBLE` set (which is
   retrieval-health types only). So the NLI tier *converts* retrieval-health false-positives into
   unsupported-claim false-positives that the gate is structurally unable to suppress → it is
   bypassed, and CLEAN-FP gets worse (0.76 → 0.88), not better.
3. The NLI verifier rarely emits `contradicted`, so CONTRADICTED_CLAIM recall collapses 6/25 → 1/25.
4. Corroborated by Codex's relabel: source-CLEAN rows flood to UNSUPPORTED_CLAIM under NLI.

## Verdict vs prereg (`phase2_grounded_clean_gate_v2_prereg.md`)
**FAIL.** Required: CLEAN-FP materially down (≤0.60) with CONTRADICTED recall flat. Got: CLEAN-FP
up, recall collapsed. The gate v2 loosening did not earn its place, and the entailment tier as
constituted (cloud = rate-limited to uselessness at 239 calls; local small model = too noisy)
does **not** improve real accuracy. Native heuristic (18/75) beats local NLI (7/75).

## What this does NOT say
- The gate is **native-safe**: in native/default mode it is inert (verified byte-identical:
  Calib 23/45, heldout native 18–19/75, protected pass). Keeping it costs nothing in production;
  it simply does not deliver in NLI mode.
- It does not rule out a *stronger* entailment model (e.g. deberta-v3-large / bart-large-mnli) or a
  redesigned gate that also suppresses NLI-induced UNSUPPORTED_CLAIM under a clean grounding verdict.
  Both are open, but neither is proven — and the zero-claim extraction gap caps NLI's reach anyway.

## Recommended pivot
The CLEAN-FP lever is **not** "add an entailment tier" with these models. Real accuracy lives in
(a) native-verifier precision on the retrieval-health analyzers that over-fire, and (b) the
claim-extraction gap (≈19/50 CLEAN rows extract zero claims → forced INSUFFICIENT_CONTEXT). Treat
Phase 2 as paused/NO-GO and either revert gate v2 to keep the tree honest, or leave it inert and
move to native precision + Phase 3 calibration.
