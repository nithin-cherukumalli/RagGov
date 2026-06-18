# Real heldout v1 — label-quality finding (decision needed from the architect)

**Date:** 2026-06-18. Independently found by Opus (3-row read) AND Codex S2 (25-row provisional
audit). This changes what "CONTRADICTED recall" means and must be resolved before optimizing it.

## Finding
The 25 RAGTruth `contradicted` rows in the real heldout are **long, mostly-faithful summaries**
where RAGTruth annotated a **span-level** hallucination (one bad fact among many correct ones).
Mapping that to a **whole-answer `CONTRADICTED_CLAIM` primary** is a granularity mismatch:
- Opus spot-read (Farkle rules, chimney cap, grill ribs): answers are faithful summaries; no
  whole-answer contradiction visible.
- S2 provisional audit (read-only, not accepted): 24/25 "actually fine/mislabeled", 1 "actually
  unsupported", 0 "clear contradiction".
- The engine returning `UNSUPPORTED_CLAIM` on these (spot parity) may be **more correct than the
  label**.

## Consequence
- **Do NOT optimize "CONTRADICTED recall (6/25)" against these labels** — that would be fitting to
  mismapped/span-level labels (the exact anti-pattern this project avoids).
- The **trustworthy real-heldout signal is CLEAN over-firing: 38/50 = 0.76 false-positive rate**,
  measured against reliable labels (HotpotQA/ALCE reference answers are faithful by construction;
  S2 CLEAN spot-check: 5 faithful / 5 needs-human-check). **CLEAN-FP is the #1 real target.**

## Decisions needed from the architect (only you can make these)
1. **Taxonomy granularity:** does `CONTRADICTED_CLAIM` mean "the answer contains any contradicted
   span" or "the answer is wholly/centrally contradictory"? RAGTruth labels are span-level; pick a
   definition and we re-map labels to it.
2. **Adjudicate the 25:** fill `contradiction_audit_worklist.md` (human). Likely most become
   `UNSUPPORTED_CLAIM` or stay `CLEAN` at the whole-answer level.
3. **Local NLI:** approve a local/offline HF NLI model (per `nli_provider_readiness.md`) so the
   hybrid entailment tier can actually run — this is the unblock for reducing CLEAN-FP (a clean
   answer that entails its context should not be flagged) and for span-level contradiction.

## Next engine work (Opus), once unblocked
- CLEAN-FP reduction via the entailment tier (deferred-to-NLI), measured on the real heldout.
- Offline NLI verifier scaffold (`local_nli` policy) per the provider-readiness spec — buildable
  now as a tested scaffold; real accuracy needs the model.

## Honest status line
The real heldout gave us truth: 0.24 overall, but its CONTRADICTED half is mislabeled and its CLEAN
half exposes a 76% false-positive rate. The number to drive down is CLEAN-FP; the contradiction
half needs relabeling, not engine-fitting.
