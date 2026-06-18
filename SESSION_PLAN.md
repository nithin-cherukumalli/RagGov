# Session Plan — Engine Precision Push (2026-06-18)

Goal for this session: raise generalization accuracy and cut CLEAN false-positives via
pre-registered, probe-measured, revert-on-failure engine fixes. Track generalization
accuracy as THE number.

## Grounded baseline (reproduced this session, not quoted)

| Metric | Value | How |
|---|---|---|
| Protected baseline | 41/46 GREEN | `check_protected_baseline.py` |
| Dataset lock | PASS (52 rows) | `check_dataset_lock.py` |
| Taxonomy support | 3 supported / 9 thin / 13 unsupported | `check_taxonomy_support.py` |
| **Calib (train/dev/heldout)** | **23/45 = 0.511** | direct scorer (`/tmp/score.py`) |
| **Probe (induced)** | **35/145 = 0.241** | same scorer on induced_candidates |

### Probe confusion (where the accuracy is lost)
- **PROMPT_INJECTION 1/10** — 9 of 10 collapse to `UNSUPPORTED_CLAIM`. Detected, not promoted.
- **CLEAN 4/30** — 26 false positives. Worst: `INCONSISTENT_CHUNKS` (8), `CHUNKING_BOUNDARY_ERROR` (5).
- CONTRADICTED_CLAIM 0/15 — collapses to UNSUPPORTED_CLAIM (label-quality-sensitive; deferred).
- INSUFFICIENT_CONTEXT 3/30, UNSUPPORTED_CLAIM 2/30, CITATION_MISMATCH 25/30.

## Discipline (unchanged, non-negotiable)
Pre-register → hard pass/fail criteria → implement → measure on calib + probe → **revert on any
regression** of protected baseline / Calib / named true positives. One task per commit. Keep the
prereg + result docs either way.

## Ordered work (one by one)

1. **Task 18 — PROMPT_INJECTION promotion** *(first: cleanest, security-critical)*
   - Why: injection is detected but a downstream grounding symptom (UNSUPPORTED_CLAIM) is selected.
   - Hypothesis: the security analyzer is not emitting a *blocking* `PROMPT_INJECTION` candidate for
     these cases (likely `SUSPICIOUS_CHUNK` under a non-whitelisted analyzer name, or a non-fatal status).
   - Pass criteria: injection probe **1/10 → ≥8/10**; protected baseline stays **41/46**; Calib stays
     **≥23/45**; **no new CLEAN→security false positive** on the probe. Revert otherwise.

2. **Task 19 — INCONSISTENT_CHUNKS over-firing** (8 CLEAN false positives)
   - Require real contradiction evidence between chunks, not surface signals.
   - Pass: CLEAN false positives from this type drop materially; baseline + Calib hold; no true-positive
     INCONSISTENT_CHUNKS regression.

3. **Task 20 — CHUNKING_BOUNDARY_ERROR over-firing** (5 CLEAN false positives, long ALCE passages)
   - Tighten `ParserValidationAnalyzer` precision.
   - Pass: CLEAN false positives from this type drop; baseline + Calib hold; parser true positives hold.

4. **Checkpoint** — re-measure all guards + probe, write result docs, commit one-per-task, report delta.

### Deferred (this session)
- CONTRADICTED_CLAIM recall (0/15): needs label audit first (RAGTruth contradicted-vs-unsupported is
  heuristic). Risk of training to noisy labels.
- Bugs 14/15/16 (xfail): pick up if injection/CLEAN work touches the same paths.

## Definition of done (project-level, not this session)
Generalization ≥ ~0.70 on a real 30–50-case heldout, low CLEAN false-positive rate, every advertised
type data-backed. Today: 0.241.
