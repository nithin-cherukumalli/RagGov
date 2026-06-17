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

## Phase A addendum — Tasks 3, 4, 5 reformulated as v2

> Tasks 3, 4, 5 (v1) were reverted on 2026-06-15 (commit `e759a12`) because each
> changed an analyzer in isolation without the matching decision-policy rank
> change, relied on query-string heuristics, or was never promoted by the engine.
> See `reports/codex_session/tasks_3_4_5_result.md`. The entries below are the v2
> reformulations with hard, pre-registered acceptance criteria. They are **queued,
> not yet implemented** — do not start until explicitly scheduled.

### Task 3-v2 — CITATION_MISMATCH: analyzer gate AND decision-policy rank

> ⚠ **BLOCKED (2026-06-17).** Feasibility check found the named cases (029, 030, 025)
> no longer carry the assumed goldens, and the gate does not match the real
> CITATION_MISMATCH cases (gc-024/025 are *cited-doc-not-retrieved*). Re-scope before
> pre-registration. See `reports/codex_session/v2_feasibility_blocker.md`. Criteria
> below preserved unchanged as the pre-registration record.

**Why v1 failed.** Changed analyzer gate only. Engine kept routing to
UNSUPPORTED_CLAIM via ClaimGroundingAnalyzer. New gate was also less specific
than the old one.

**Scope (both changes mandatory).**
1. Analyzer-level: refine the gate so CITATION_MISMATCH fires only when a
   citation points to a retrieved doc AND the cited content does not support
   the answer's specific value claim. Keep specificity ≥ the original gate.
2. Decision-policy-level: in src/raggov/decision_policy.py, promote
   CITATION_MISMATCH above UNSUPPORTED_CLAIM when both fire at status=fail
   AND a citation→retrieved-doc binding exists.

**Acceptance criteria.**
- ≥2 of {029, 030} flip to CITATION_MISMATCH; case 025 does not change
  (it has no citations — different root cause).
- Protected pin (42, 46) unchanged.
- All safety counters stay 0.
- Heldout primary ≥ 0.733.
- Calib-50 primary ≥ 0.62.
- New test: tests/test_engine/test_citation_mismatch_routing.py runs
  DiagnosisEngine end-to-end on cases 029, 030 and asserts primary_failure ==
  "CITATION_MISMATCH".

**Out of scope.** Any other decision-policy rank changes.

---

### Task 4-v2 — RETRIEVAL_DEPTH_LIMIT: pipeline-introspection only

> ⚠ **DEFERRED (2026-06-17).** Hits its own pre-registered deferral clause: no
> `top_k`/`k_floor`/saturation signal exists in the fixtures (`RAGRun.metadata` empty,
> chunk metadata `{}`; only `rank`/`score`). Named cases (004, 019) are also
> mislabeled now. See `reports/codex_session/v2_feasibility_blocker.md`. Criteria
> below preserved unchanged as the pre-registration record.

**Why v1 failed.** Predicate parsed cardinal numbers from query text. False-
fired on CLEAN cases 011, 012 and broke 023, 027. Heldout regressed.

**Pre-condition for landing.** Fixtures for cases 004, 019 must contain
pipeline-introspection signals (top_k in run.metadata, chunk-rank
saturation pattern, or configured k_floor). If those signals are not
present, this task is DEFERRED — there is no pipeline-agnostic way to detect
depth limit without them. Document the deferral and stop.

**Scope (both changes mandatory).**
1. Analyzer: predicate reads only from run.metadata / chunk.metadata /
   RetrievalEvidenceProfile. Never parses the query string for numbers.
2. Decision-policy rank: promote RETRIEVAL_DEPTH_LIMIT only when retrieval
   metadata confirms a saturation pattern.

**Acceptance criteria.**
- Cases 004, 019 flip to RETRIEVAL_DEPTH_LIMIT.
- Cases 011, 012 stay CLEAN (no false positive on numeric prose).
- Cases 023, 027 stay at their pre-change primary_failure.
- Protected pin (42, 46) unchanged.
- All safety counters stay 0.
- Heldout primary ≥ 0.733.
- Calib-50 primary ≥ 0.62.
- New test: tests/test_engine/test_retrieval_depth_limit_routing.py asserts
  end-to-end primary_failure on at least 004, 019, 011, 012.

**Out of scope.** Adding new metadata to fixtures to make the task land. If
the signal is not in the data, defer.

---

### Task 5-v2 — RERANKER_FAILURE: analyzer AND decision-policy rank

> ⚠ **DEFERRED (2026-06-17).** Not viable against current data: zero cases carry the
> `RERANKER_FAILURE` golden and no reranker metadata exists anywhere, so the required
> end-to-end test cannot be constructed. See
> `reports/codex_session/v2_feasibility_blocker.md`. Criteria below preserved
> unchanged as the pre-registration record.

**Why v1 failed.** Analyzer registered, but decision policy never promoted
RERANKER_FAILURE over UNSUPPORTED_CLAIM. Target case 020 unchanged.

**Scope (both changes mandatory).**
1. Analyzer reads reranker metadata from run.metadata or
   RetrievalEvidenceProfile (no query-string heuristics).
2. Decision-policy: promote RERANKER_FAILURE over UNSUPPORTED_CLAIM when
   reranker metadata is present AND indicates rank inversion or low
   confidence.

**Acceptance criteria.**
- Case 020 primary_failure becomes RERANKER_FAILURE.
- Protected pin (42, 46) unchanged.
- All safety counters stay 0.
- Heldout primary ≥ 0.733.
- Calib-50 primary ≥ 0.62.
- New test: tests/test_engine/test_reranker_failure_routing.py asserts
  end-to-end primary_failure on case 020 and on at least one negative case
  (reranker metadata absent → no RERANKER_FAILURE fired).

**Out of scope.** Re-tuning other analyzers' weights.

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

## Phase F — Follow-ups filed from Task 10 red-test triage

> These are `real_bug` rows from `reports/codex_session/red_test_triage.md`.
> Production was not changed in Task 10; each is a genuine routing/attribution
> gap surfaced by a golden-aligned analyzer test. Each needs its own
> pre-registration before any code change.

### Task 14 — STALE_RETRIEVAL over-promotes on irrelevant stale sources

**Symptom.** `tests/test_analyzers/test_version_validity_pipeline.py::test_stale_irrelevant_source_does_not_primary_fail` fails: a stale-but-irrelevant lease doc (2010) becomes `primary_failure=STALE_RETRIEVAL` even though the answer cites the fresh, active CEO doc (2024). The stale source is not answer-bearing, so it should not drive the primary verdict.

**Regression provenance (verified 2026-06-17).** This is a long-standing
pre-existing red, **not** caused by Task 2 (`a17f75c`). The test was run against
each commit's actual code: it fails identically at `38e50e5` (Task 1, the parent
of Task 2), and is already red at `64909ff` — before the STALE_RETRIEVAL feature
(`7b47abe`) even shipped. Task 2 is causally innocent, so the Task 3/4/5-v2
specificity-rank stack is safe to land on top of it; Task 14 is independent and
not a blocker for v2.

**Hypothesis.** The STALE_RETRIEVAL promotion does not gate on whether the stale
source is cited / answer-bearing. It should only primary-fail when the stale
source actually backs the answer. (Origin is the stale/version-validity detection
path, not the Task 2 rank change.)

**Scope (pre-register before coding).** Likely `src/raggov/decision_policy.py` and/or the stale/version-validity analyzer gate. Add an "is the stale source answer-bearing/cited?" condition.

**Acceptance criteria.**
- Case in `test_stale_irrelevant_source_does_not_primary_fail` stops returning STALE_RETRIEVAL.
- Task 2's target (Calib-50 case 038) still resolves to STALE_RETRIEVAL.
- Protected baseline unchanged (41/46 GREEN). Heldout primary ≥ 0.733. Calib-50 ≥ baseline.
- All safety counters stay 0.

---

### Task 15 — Stage attribution: incomplete answer should be GENERATION, not GROUNDING

**Symptom.** `tests/test_analyzers/test_answer_quality_confidence_metadata.py::test_quality_incomplete_38_has_generation_stage_candidate_if_supported` fails: engine agrees `primary_failure=UNSUPPORTED_CLAIM` but sets `root_cause_stage=GROUNDING`. An incomplete-answer case (case 38) is a generation-stage problem and should attribute to `GENERATION` with `AnswerQualityAnalyzer` as the selected analyzer.

**Hypothesis.** Stage attribution / analyzer selection for incomplete answers is dominated by a grounding-stage analyzer; AnswerQualityAnalyzer's generation-stage candidate is not promoted when it should be.

**Scope (pre-register before coding).** Engine stage-attribution / decision-policy. End-to-end DiagnosisEngine test on case 38.

**Acceptance criteria.**
- Case 38 → `root_cause_stage=GENERATION`, selected analyzer `AnswerQualityAnalyzer`, primary stays `UNSUPPORTED_CLAIM`.
- Protected baseline unchanged. Heldout primary ≥ 0.733. Calib-50 ≥ baseline. Safety counters 0.

---

### Task 16 — Specificity: case 41 should route to CONTRADICTED_CLAIM, not UNSUPPORTED_CLAIM

**Symptom.** `tests/test_analyzers/test_answer_quality_confidence_metadata.py::test_quality_ignores_context_41_has_generation_stage_candidate_if_supported` fails: golden expects `CONTRADICTED_CLAIM` (answer ignores/contradicts retrieved context) but engine routes to less-specific `UNSUPPORTED_CLAIM`.

**Hypothesis.** Same family as the v2 routing work: a more-specific failure (CONTRADICTED_CLAIM) is not promoted over UNSUPPORTED_CLAIM when contradiction signal is present. Sequence after Tasks 3-v2/5-v2 so the specificity-rank machinery is consistent.

**Scope (pre-register before coding).** Decision-policy specificity rank + analyzer contradiction signal. End-to-end DiagnosisEngine test on case 41.

**Acceptance criteria.**
- Case 41 → `primary_failure=CONTRADICTED_CLAIM`, `root_cause_stage=GENERATION`.
- No regression on UNSUPPORTED_CLAIM cases. Protected baseline unchanged. Heldout primary ≥ 0.733. Calib-50 ≥ baseline. Safety counters 0.

---

## Phase G — Generalization bugs (found by the fresh-data probe)

> Surfaced by `reports/codex_session/generalization_probe_v1.md`: the engine
> scores ~0.31 primary on fresh induced data vs ~0.62 on its own fixtures. These
> are the two clearest, most actionable gaps. Pre-register before any code change;
> validate against the induced candidate set, not just the existing fixtures.

### Task 17 — Over-firing on CLEAN inputs (false positives)

**Symptom.** On 20 induced CLEAN cases (HotpotQA gold answers, fully supported),
the engine returned CLEAN only 3/20; it flagged `INCONSISTENT_CHUNKS`,
`STALE_RETRIEVAL`, `INSUFFICIENT_CONTEXT` on healthy answers. A diagnosis tool
that cries wolf on working pipelines is not trustworthy.

**Hypothesis.** One or more analyzers fire on benign multi-passage contexts
(distractor chunks read as "inconsistent"; terse gold answers read as
"insufficient"). Likely a threshold/precision problem, not a routing one.

**Scope (pre-register first).** Identify which analyzers over-fire on CLEAN;
tighten precision **without** tuning to specific fixtures. Add a
false-positive-rate metric over the induced CLEAN set.

**Acceptance criteria.** CLEAN recall on the induced CLEAN set materially up
(target ≥ 0.7) with no new dangerous-miss; protected baseline unchanged; Calib-50
and Heldout primary not regressed.

---

### Task 18 — PROMPT_INJECTION detected but not promoted to primary

**Symptom.** On 10 injection cases, the injection analyzer fires ("Prompt
injection detected") but primary_failure came out `UNSUPPORTED_CLAIM` 9/10. A
security-relevant signal is detected and then buried.

**Hypothesis.** The decision policy does not rank `PROMPT_INJECTION` (security
stage) above grounding failures when both fire. Security should generally
out-rank downstream grounding symptoms.

**Scope (pre-register first).** Decision-policy specificity/priority for the
security stage. End-to-end test asserting primary == PROMPT_INJECTION on an
injection case, plus a negative (no injection → no false security primary).

**Acceptance criteria.** Injection cases route to PROMPT_INJECTION as primary;
no security false-positive on clean/non-injection cases; protected baseline
unchanged; Calib-50/Heldout not regressed; safety counters still 0.

---

## Discipline reminders

1. **Pre-registration before code.** Always. No exceptions.
2. **Hard criteria, no silent loosening.** If a criterion fails, revert.
3. **One task at a time.** Do not bundle.
4. **Pipeline/domain-agnostic.** No threshold tuned to a specific fixture.
5. **No golden label edits.** If a label looks wrong, write a review note for human adjudication.
6. **The 20% root-cause-accuracy bar is the bar.** Every change must improve it without false-clean / dangerous-miss regressions.
