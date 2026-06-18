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
| **Probe (induced)** | **43/145 = 0.297** (post Task 18; was 0.241) | same scorer on induced_candidates |

> Re-measure these at session start (reproduce, don't quote). The shim + PYTHONPATH
> setup is in `reports/codex_session/SESSION_HANDOFF.md` §4.

### Probe confusion (where the accuracy is lost)
- ~~PROMPT_INJECTION 1/10~~ → **9/10 (Task 18 LANDED)**. It was a *recall* gap
  (analyzer wasn't detecting multilingual injections), **not** a promotion problem —
  the handoff's framing was wrong; the decision policy was already correct. 1 romanized-Hindi
  payload still missed (documented evasion boundary).
- **CLEAN 4/30** — 26 false positives. Worst: `INCONSISTENT_CHUNKS` (8), `CHUNKING_BOUNDARY_ERROR` (5).
  NOTE: suppressing one over-firer often just lets the *next* one win, so overall CLEAN-correct
  may rise slowly even when a fix is correct (seen in Task 17: STALE FPs −60% but CLEAN-correct 3→4).
  Track **per-type false-positive count**, not only overall CLEAN-correct.
- CONTRADICTED_CLAIM 0/15 — collapses to UNSUPPORTED_CLAIM (label-quality-sensitive; deferred).
- INSUFFICIENT_CONTEXT 3/30, UNSUPPORTED_CLAIM 2/30, CITATION_MISMATCH 25/30.

## Discipline (unchanged, non-negotiable)
Pre-register → hard pass/fail criteria → implement → measure on calib + probe → **revert on any
regression** of protected baseline / Calib / named true positives. One task per commit. Keep the
prereg + result docs either way.

### Method that actually worked (Tasks 17 & 18) — follow it
1. **Instrument first.** Before hypothesizing, run the probe and print *which analyzer*
   emits the wrong failure_type on the target cases (status=fail + analyzer_name +
   failure_type). Task 17 and 18 both only succeeded after this — and 18 disproved the
   handoff's guess (recall, not promotion). Do not trust prior framing; measure.
2. **Find the narrowest gate.** Change the escalation/emission rule, not the whole
   analyzer (Task 17 attempts 1–2 reverted for being too blunt; v3 landed by gating
   one branch). No query/passage-text heuristics — structured signals only.
3. **Guards per fix (all must pass, else revert):** protected `41/46`; Calib `≥23/45`;
   named true positives unchanged; **`pytest tests/test_analyzers` green**; add a unit
   test + an end-to-end `diagnose()` test (both 17 & 18 did).
4. **A pinned baseline case may depend on the same heuristic you're tightening**
   (Task 17's `version_stale_not_cited_32`). If so, split the behavior; never edit the
   golden or loosen the criterion.

## Ordered work (one by one)

1. ~~**Task 18 — PROMPT_INJECTION**~~ ✅ **DONE / LANDED (2026-06-18)**. Was a *recall*
   gap, not promotion: added multilingual directive patterns to
   `analyzers/security/injection.py`. Probe injection **1/10 → 9/10**, overall **0.241 →
   0.297**, zero regressions, unit + e2e tests added. 1 romanized-Hindi payload still
   missed (documented evasion boundary). See `task18_result.md`. → **Start at Task 19.**

2. ~~**Task 19 — INCONSISTENT_CHUNKS over-firing**~~ ✅ **DONE / LANDED (2026-06-18)**.
   Root cause was the negation heuristic firing on a single incidental shared token near
   ANY negation. Rewrote `has_suspicious_negation_pair` to require a polarity-opposed
   multi-term proposition (≥2 shared content terms negated in one chunk, asserted in the
   other); dropped discourse markers. Negation-path CLEAN FPs **8→2** (2 residuals are
   lexically identical to the protected TP — need NLI). Overall **0.297→0.324**. See
   `task19_result.md`. Spawned **Task 21** (separate Jaccard duplicate path).

3. ~~**Task 20 — CHUNKING_BOUNDARY_ERROR over-firing**~~ ✅ **DONE / LANDED (2026-06-18)**.
   `_has_chunk_boundary_damage` ignored provenance; multi-hop passages from distinct docs
   were misread as split sentences. Now require the consecutive pair to share
   `source_doc_id`. CLEAN FPs **5→0**. Overall **0.324→0.372**. See `task20_result.md`.

4. ~~**Checkpoint**~~ ✅ guards re-verified (protected 41/46, lock PASS, taxonomy PASS,
   Calib 23/45, probe **54/145 = 0.372**); `tests/test_analyzers` 542 passed (1 pre-existing
   stale fail, 3 xfail). All session work committed locally (user pushes).

5. ~~**Task 21 — NCV Jaccard duplicate over-firing**~~ ✅ **DONE / LANDED (2026-06-18)**.
   Raised context-assembly duplicate threshold 0.85→0.97 (topical overlap ≠ duplicate chunk).
   CLEAN→INCONSISTENT_CHUNKS Jaccard path **3→0**; overall **0.372→0.393**. See `task21_result.md`.

### Next (not started this session)
- CONTRADICTED_CLAIM recall (0/15): needs label audit first (RAGTruth contradicted-vs-unsupported is
  heuristic). Risk of training to noisy labels.
- 2 irreducible negation-path INCONSISTENT residuals: need NLI/entailment, not lexical rules.
- Bugs 14/15/16 (xfail): pick up if injection/CLEAN work touches the same paths.

## Definition of done (project-level, not this session)
Generalization ≥ ~0.70 on a real 30–50-case heldout, low CLEAN false-positive rate, every advertised
type data-backed. **Probe today: 0.393** (start of session 0.241; +0.152 across Tasks 18–21).
CLEAN-correct 4/30 → 13/30.

---

## Context: external integrations, A2P, and realistic timeline (verified from code)

This is project-level context the engine-precision work doesn't change, but the next
person should know it (it shapes positioning and the "super package" goal).

### External packages (RAGAS / DeepEval / RAGChecker / RefChecker) — opt-in, unvalidated
- RagGov does **not** run them. `external_signal_bridge.py` is **bring-your-own-results**:
  you run those tools yourself and pass outputs in `run.metadata["external_evaluation_results"]`;
  the bridge maps provider+metric (ragas/deepeval faithfulness/relevancy low, ragchecker
  hallucination, refchecker bad claim_support) → internal signals, **only in
  `external-enhanced` mode**. `refchecker` is the one with a direct adapter (optional import,
  graceful native fallback).
- Default pipeline is fully **native/deterministic** and imports none of them.
- **Not yet validated**: no evidence external mode beats native (`calibration_status:
  not_calibrated`). Don't market as "powered by RAGAS/DeepEval" until wired turnkey + measured.

### Install seamlessness — core yes, externals no
- `pip install raggov` → pydantic/typer/rich only. Clean, light.
- `pip install raggov[external]` → ragas + deepeval + ragchecker + refchecker +
  sentence-transformers → pulls torch/transformers/spaCy + model downloads. Heavy,
  version-conflict-prone, **not seamless**. No end-to-end example exists for feeding external
  results into `external-enhanced` mode (only LangChain/LlamaIndex stubs in `examples/`).
  → Future work: a turnkey external-mode quickstart + a documented, version-pinned extras set.

### A2P (novel) — built but OFF by default, unmeasured
- `A2PAttributionAnalyzer` (Abduct-Act-Predict counterfactual attribution: causal chains,
  pinpointing, candidate scoring/selection) is registered but **disabled in native mode**
  (`enable_a2p` only true in non-native mode with the `a2p` provider). It contributes nothing
  on the default path and has **no isolated accuracy number**.
- Active novel analyzers (NCV verifier, taxonomy classifier, parser-validation,
  citation-faithfulness, claim-grounding, version-validity, retrieval-diagnosis) **run** but
  several **over-fire** — precision (this session's work), not existence, is the open question.
- Future work: measure A2P in isolation (does enabling it raise probe accuracy?) before
  claiming it as a differentiator.

### Realistic time to "super diagnostic package"
- **Engine precision:** ~4–6 pre-registered fixes (this session starts on them). Days each
  → a few focused weeks.
- **Real labeled data (the real bottleneck, human throughput):** ≥5–15 real cases per
  supported type + a real 30–50-case double-labeled heldout. Weeks of labeling.
- **External path wired + validated** and **A2P measured**: additional research-grade effort.
- **Calibration + release:** short once the above land.
- **Honest estimate:** *useful & honestly calibrated* (~0.70 on a real heldout) ≈ **1–2 focused
  months, dominated by data labeling**. *"Super / best-in-class with validated external
  ensembling"* is meaningfully more. No single change shortcuts it — but the path and the
  measurement are now fully in place.
