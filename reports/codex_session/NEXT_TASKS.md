# Codex Task Queue — Path to Genuinely-Useful v0.1

**Last updated:** 2026-06-12
**Current state:** Calib-50 primary 0.58, Heldout primary 0.733, Protected `(42, 46)` GREEN, all safety counters 0.

## How to use this queue

1. Work tasks **in order**. Each task is scoped so it lands cleanly without depending on later tasks.
2. For each task: write a **pre-registration** doc (`reports/codex_session/<task-id>_pre_registration.md`) listing hypothesis + hard acceptance criteria **before any code change**.
3. After the change: write a **result** doc (`reports/codex_session/<task-id>_result.md`) showing each criterion met or failed.
4. If any acceptance criterion fails → **revert**. Do not loosen criteria silently. The previous warn-promotion attempt is the model: a failed criterion was documented and the change was reverted (`reports/forensics_v0_1_warn_promotion_result.md`).
5. Hard invariants that must hold across every task:
   - Protected pin `(42, 46)` stays GREEN
   - Calib-50 `false_clean_count`, `dangerous_clean_miss_count`, `human_review_miss_count`, `false_security_count` all stay 0
   - Heldout primary accuracy never regresses below 0.733
   - `production_gating_eligible=False` remains until calibration is real
   - Locked heldout fingerprint `58261ecc...c9b15c76` unchanged
   - No golden label edits
   - No engine warn-promotion change (separate, gated follow-up — see Task 11)

---

## Phase A — Close the wrong_routing bucket (highest near-term leverage)

These are cases where the right signal is already in `analyzer_results` but the engine picks a different winner. Each task targets one concrete false-fire or priority mismatch.

### Task 1 — Fix `ParserValidationAnalyzer` missing-profile false-fire

**Problem.** When no parser/chunking profile is attached, `ParserValidationAnalyzer` emits `fail` with `TABLE_STRUCTURE_LOSS` on content that has no tables. This poisons primary selection for Calib-50 case 039 (steals STALE_RETRIEVAL) and several others.

**Hypothesis.** The "profile missing" path should `warn` (advisory) or `skip`, not `fail`, because absence of a profile is not evidence of structural loss. Only fire `fail` when an attached profile **and** parsed content together indicate loss.

**Scope.** Single file: `src/raggov/analyzers/parsing/<...>.py` (locate the analyzer that emits `parser_validation_profile_missing` evidence).

**Acceptance criteria.**
- Protected pin `(42, 46)` unchanged, composition unchanged.
- Calib-50 case 039 primary becomes `STALE_RETRIEVAL` (the prepared signal is now allowed to win).
- No protected case changes its primary_failure value.
- No new `false_clean_count`, `false_security_count`, `false_incomplete_count` on protected.
- Heldout primary accuracy ≥ 0.733.
- Calib-50 primary accuracy ≥ 0.58 (target: 0.60 with case 039 win).

**Out of scope.** Reworking the analyzer's positive-detection path. Only the false-positive path on missing profile.

---

### Task 2 — Promote `STALE_RETRIEVAL` over `INSUFFICIENT_CONTEXT` when both fire

**Problem.** Calib-50 case 038 has `StaleRetrievalAnalyzer=fail+STALE_RETRIEVAL` and `RetrievalDiagnosisAnalyzerV0=fail+INSUFFICIENT_CONTEXT`. Engine picks `INSUFFICIENT_CONTEXT`. But staleness is a stronger, more specific signal than insufficiency — INSUFFICIENT_CONTEXT means "not enough" while STALE_RETRIEVAL means "wrong version retrieved", which is a more precise diagnosis when both apply.

**Hypothesis.** In `src/raggov/decision_policy.py` (or the engine's priority table), `STALE_RETRIEVAL` should outrank `INSUFFICIENT_CONTEXT` when both fire at status=fail, because staleness specifies *which* of the retrieved chunks is the problem.

**Scope.** One priority/weight constant. No new analyzer code.

**Acceptance criteria.**
- Calib-50 case 038 primary becomes `STALE_RETRIEVAL`.
- No protected case changes its primary_failure.
- Heldout: no non-clean case changes its primary except in the direction of correctness.
- Heldout primary accuracy ≥ 0.733.
- Calib-50 primary accuracy ≥ 0.60 (after Task 1).
- Document the priority rationale in code comment + result doc.

**Out of scope.** Any other priority changes.

---

### Task 3 — Fix `CITATION_MISMATCH` vs `UNSUPPORTED_CLAIM` routing for missing-citation cases

**Problem.** Calib-50 cases 025, 029, 030 expect `CITATION_MISMATCH` but get `UNSUPPORTED_CLAIM`. Inspect each case: when an answer has claims with **no citation at all**, is that better classified as a citation problem (missing citation) or a grounding problem (no support)?

**Hypothesis.** Per the failure taxonomy:
- `CITATION_MISMATCH` = a citation exists but doesn't match the cited source
- `UNSUPPORTED_CLAIM` = no supporting evidence in retrieved chunks
- A separate failure type may be missing for "no citation at all". Check the taxonomy.

**Scope.** Read the taxonomy and the actual fixtures. May resolve in one of three ways:
- (a) `CitationFaithfulnessAnalyzerV0` should already fire on these — fix the predicate.
- (b) These cases are legitimately `UNSUPPORTED_CLAIM` and the goldens are wrong — file a labelling-review note (do NOT change goldens unilaterally).
- (c) A `MISSING_CITATION` failure type should be introduced.

**Acceptance criteria.**
- Pre-registration must explicitly choose (a), (b), or (c) **before** any code change, with rationale.
- If (a): at least 2 of cases {025, 029, 030} flip to `CITATION_MISMATCH`. Protected pin unchanged. Heldout ≥ 0.733.
- If (b): write a labelling-review note in `reports/codex_session/`. No code change.
- If (c): new enum value plus analyzer change. Full pre-registration. Protected pin unchanged.

**Out of scope.** Wholesale taxonomy redesign.

---

### Task 4 — Add `RETRIEVAL_DEPTH_LIMIT` analyzer (or repair existing path)

**Problem.** Calib-50 cases 004, 019 expect `RETRIEVAL_DEPTH_LIMIT` but get `UNSUPPORTED_CLAIM`. This failure mode is real and engineer-actionable: the retriever's top-K was too small, so the right chunk wasn't in the candidate set.

**Hypothesis.** A principled, pipeline-agnostic check: when `len(retrieved_chunks) <= k_floor` (configurable, default 5) **and** the answer's claims have low overlap with all retrieved chunks **and** the run's metadata indicates a `top_k` value, fire `RETRIEVAL_DEPTH_LIMIT`. If no `top_k` metadata, fire when retrieved set is suspiciously small.

**Scope.** New analyzer file `src/raggov/analyzers/retrieval/depth.py` (or extend `RetrievalDiagnosisAnalyzerV0`). Register in engine.

**Acceptance criteria.**
- Cases 004, 019 → `RETRIEVAL_DEPTH_LIMIT`.
- Protected pin unchanged.
- Heldout primary ≥ 0.733.
- No false fire on any case with `retrieved_chunks >= 5` and substantive answer alignment.
- Pre-registration must define the predicate in dataset-independent terms (no thresholds tuned to 004/019).

**Out of scope.** Reranker-related failures (Task 5).

---

### Task 5 — Add `RERANKER_FAILURE` detection

**Problem.** Calib-50 case 020 expects `RERANKER_FAILURE`. This is when the reranker demoted the relevant chunk below a less-relevant one.

**Hypothesis.** When `chunk.metadata` exposes `initial_rank` / `pre_rerank_score` and `final_rank` / `rerank_score`, and the chunk most textually aligned with the answer is significantly downranked (final_rank > initial_rank by a threshold) while a less-aligned chunk was upranked, fire `RERANKER_FAILURE`.

**Scope.** New analyzer or extension. Pipeline-agnostic: only fires when rerank metadata is present.

**Acceptance criteria.**
- Case 020 → `RERANKER_FAILURE`.
- Skips cleanly when no rerank metadata present (no false fire on Calib-50 cases without rerank info).
- Protected pin unchanged.
- Heldout primary ≥ 0.733.
- Pre-registration declares which metadata keys are recognised; no domain assumption.

**Out of scope.** Building a reranker. Detection only.

---

## Phase B — Engineer-facing surfaces (closes the "useful for AI engineers" goal)

### Task 6 — Extend "Why this verdict?" block to JSON output

**Problem.** `diagnose --format json` emits the raw Diagnosis dump but no synthesised provenance summary. Engineers piping JSON into dashboards or scripts can't easily extract "which analyzer voted".

**Hypothesis.** Add a `why_block` top-level key in the JSON output with the same four fields as the text-format renderer: `verdict_summary`, `voted_by` (list of analyzer names), `also_considered` (list of `{analyzer, failure_type, status}`), `inspect_next` (structured target dict).

**Scope.** `src/raggov/cli.py` JSON branch and `src/raggov/io/serialize.py` (or sibling).

**Acceptance criteria.**
- JSON output contains `why_block` key with the four fields above.
- New JSON test covers presence of `why_block` on a citation-mismatch fixture.
- No model field added (pure serialisation derivation).
- Rich + text formats unchanged.
- Protected pin unchanged.

---

### Task 7 — Add "Why this verdict?" block to rich (panel) output

**Scope.** `_diagnosis_panel` in `cli.py`. Reuse the same `_render_why_block` helper.

**Acceptance criteria.**
- Rich panel includes a "Why this verdict?" subsection with the same content as text format, styled appropriately.
- All `tests/cli/test_*` pass.
- Protected pin unchanged.

---

### Task 8 — Quickstart `examples/` with end-to-end LangChain integration

**Problem.** README documents capabilities but there's no copy-pasteable adoption path. A real engineer needs: "here's how I wire GovRAG into my LangChain pipeline in 20 lines".

**Scope.** New `examples/langchain_integration.py` (and a `examples/README.md`). Use the `RAGRun` model directly. Demonstrate:
- Converting LangChain `retriever.invoke()` output into `RAGRun.retrieved_chunks`
- Calling `DiagnosisEngine(...).diagnose(run)`
- Reading the why-block + `primary_failure`

Do **not** add LangChain as a runtime dependency in `pyproject.toml`. Document `pip install langchain` separately and gate the import in the example file.

**Acceptance criteria.**
- Example runs end-to-end on a stub retriever + stub answer.
- Documented in `README.md` Quickstart section.
- No new runtime dependency in `pyproject.toml`.

---

### Task 9 — Add `examples/llamaindex_integration.py`

Mirror Task 8 for LlamaIndex. Same constraints. Together with LangChain, these cover ~85% of real-world RAG pipelines.

---

## Phase C — Clean up pre-existing red tests

20 tests are currently red, pre-dating this session's work. They block any future "all green" claim.

### Task 10 — Audit and triage the 20 pre-existing red tests

**Scope.** For each of the 20 failing tests, classify as:
- **stale_test** — code drift, test needs update
- **dead_feature** — testing removed functionality, delete test
- **real_bug** — code is wrong, file a follow-up task

Write `reports/codex_session/red_test_triage.md` with one row per test: classification, rationale, follow-up action.

**Acceptance criteria.**
- All 20 tests classified.
- A second commit per classification: delete dead_feature tests, file follow-up tasks for real_bugs, update stale_tests (only if update is mechanical — no behaviour change).
- After Task 10, `tests/test_analyzers/ -q` should be either green or have a documented residual count with each remaining failure pointing to a follow-up task.

---

## Phase D — Engine-level cleanup (final, gated by analyzer maturity)

### Task 11 — Re-attempt warn-to-primary promotion removal

**Pre-condition.** Tasks 1–5 must land. Calib-50 false_clean_count must remain 0 without warn-promotion. The previous attempt regressed because analyzers under-fired; if Tasks 1–5 close that gap, this should now hold.

**Scope.** `src/raggov/engine.py:866-886` removal, `_secondary_failures` tightening, `human_review_required` section-4 adjustment. Re-introduce `tests/test_engine/test_warn_only_stays_clean.py`.

**Acceptance criteria.** Same as `reports/forensics_v0_1_warn_promotion_pre_registration.md` (unchanged). If any criterion fails → revert again and document why analyzers are still under-calibrated.

---

## Phase E — Calibration & release

### Task 12 — Generate first calibration evidence pack

Once Calib-50 primary ≥ 0.70 and Phase A/B/C/D land:
- Run 5 seeds, compute confidence intervals on primary/stage accuracy.
- Compute calibration curve (predicted confidence vs empirical accuracy).
- Write `reports/calibration_pack_v0_1.md`.
- Flip `calibration_status` from `not_calibrated` to `preliminary` (NOT `calibrated` — that requires heldout split rotation).
- `production_gating_eligible` stays `False` until heldout rotation.

### Task 13 — Update README + RELEASE_NOTES + ship v0.1-beta tag

- Update README claims to match measured numbers.
- Update RELEASE_NOTES with each task's contribution.
- Tag `v0.1-beta`.

---

## Discipline reminders

1. **Pre-registration before code.** Always. No exceptions.
2. **Hard criteria, no silent loosening.** If a criterion fails, revert.
3. **One task at a time.** Do not bundle.
4. **Pipeline/domain-agnostic.** No threshold tuned to a specific fixture.
5. **No golden label edits.** If a label looks wrong, write a review note for human adjudication.
6. **The 20% root-cause-accuracy bar is the bar.** Every change must improve it without false-clean / dangerous-miss regressions.
