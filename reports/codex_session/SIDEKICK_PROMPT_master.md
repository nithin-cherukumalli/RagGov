# Codex Sidekick MASTER Prompt — full groundwork for the next Opus session

Paste this to Codex. Everything here is READ-ONLY groundwork + data validation. Opus
implements all engine changes afterward. Append findings to a new dated section of
`reports/codex_session/codex_sidekick_session_plan.md` (the single ledger) — do not scatter
new report files except the data artifacts explicitly requested below.

## Role & non-negotiable rules
- You are Codex, sidekick to Claude Opus 4.8 on RagGov engine precision.
- Do NOT edit engine, analyzer, decision-policy, claims, label, fixture, threshold, gate, or
  production-flag files. No core code changes. Groundwork, measurement, and data validation only.
- Do NOT revert or modify Opus/user changes or dirty files.
- Reproduce every number yourself; never quote stale numbers from old reports.
- Keep `calibration_status=not_production_calibrated`, `production_gating_eligible=false`.
- Follow the ledger's "Non-Negotiable Rules" and "Closeout Format".

## Env (sandbox is Python 3.10; repo needs 3.11 shims)
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```
Reproduce and record the anchor baseline before anything else: protected `41/46`
(`scripts/check_protected_baseline.py`), Calib `23/45`, induced probe `~80/145=0.552`.
Build each `RAGRun` as `RetrievedChunk(chunk_id,text,source_doc_id=doc_id,score)`,
`final_answer=answer`, `cited_doc_ids=citations`; normalize citation dicts to their `doc_id`.

---

## PART A — Synthetic vs. production reality (HIGHEST PRIORITY)

The induced probe (`evals/govrag_calib/staging/raw/induced_candidates.jsonl`) is **synthetic
mutations** of clean cases. It is a better signal than fixtures but is NOT production data. Some
landed gains (esp. Task 23 source-assertion suffix) are partly artifacts of one repeated mutation
pattern. Opus needs to know which fixes are *production-real* vs *synthetic-inflated*.

Do all of:
1. **Inventory every real-labeled row available** (non-synthetic): canonical
   `govrag_calib_150.jsonl` (split train/dev/heldout), `splits/heldout_v0_1.jsonl`,
   `staging/raw/starter_seed_intake.jsonl`, and any RAGTruth/ALCE/HotpotQA-derived rows that carry
   ORIGINAL source labels (not mutated). Report counts per source and per expected type.
2. **Re-score the landed fixes (Tasks 18–23) on the non-synthetic rows only.** For each of: probe
   vs canonical-scored vs heldout, report per-type accuracy and, critically, how much of each
   fix's gain survives on non-synthetic data. Specifically isolate Task 23: count how many
   non-synthetic rows actually contain a "source notes/states/reports…" construction — i.e. is the
   source-assertion rule exercised by real data at all, or only by the mutation?
3. **Quantify the synthetic dependency** of the probe: how many induced rows differ from their
   clean source ONLY by the appended `"The source also notes this was formally reaffirmed…"`
   suffix? Report that count; it bounds how much of probe 0.552 is suffix-driven.
4. Recommend which landed fixes are safe to call "general" vs which are "probe-shaped, pending real
   data," with evidence. This directly informs honesty in the README/claims.

## PART B — Build the real 30–50 case heldout (the project's stated bar)

The definition of done is generalization ≥ ~0.70 on a REAL, double-labeled 30–50 case heldout.
Today's `heldout_v0_1.jsonl` is only 15 rows and optimistic. Groundwork (no label invention —
prepare, don't adjudicate):
1. From RAGTruth/ALCE/HotpotQA seed sources, assemble a CANDIDATE heldout pool of 40–60 rows with
   ORIGINAL labels, balanced across the 3 supported types (CLEAN, INSUFFICIENT_CONTEXT,
   CONTRADICTED_CLAIM) plus UNSUPPORTED_CLAIM/CITATION_MISMATCH where real labels exist. Output a
   draft file `evals/govrag_calib/staging/raw/heldout_candidate_v0_2.jsonl` (staging only, not
   canonical, not locked).
2. For each candidate, record: source id, original label, proposed RagGov expected_primary, and a
   one-line rationale. Flag every row where the migrated label is heuristic/ambiguous.
3. Produce a double-labeling worklist: which rows need human adjudication and why. Do NOT finalize
   labels yourself.
4. Validate shape with `scripts/add_calib_case.py` in validate-only mode; report rejects.
5. Do NOT append to canonical `govrag_calib_150.jsonl` or regenerate the lock/manifest — that is a
   human/Opus decision after adjudication. Note the exact steps Opus would run to do so.

## PART C — RAGTruth contradiction label audit (unblocks Task 22 / Task 16)

Task 22 was a documented NO-GO in native mode: `_require_explicit_contradiction` is load-bearing
(disabling it regresses Calib 23→22 and creates +10 false contradictions). The migrated RAGTruth
"contradicted" labels have matching values (no hard value/date conflict). Groundwork:
1. For all 15 expected-CONTRADICTED probe rows + the 11 Calib CONTRADICTED rows, classify each as:
   (a) hard conflict present (value/date/unit/entity mismatch the heuristic could detect),
   (b) semantic-only contradiction (needs NLI), (c) likely-mislabeled (really unsupported/clean).
   Give the evidence per row.
2. Inventory the optional NLI/entailment path: confirm exactly how to enable
   `claim_grounding_verifier_policy=llm_entailment` / `ClaimEntailmentVerifierV1`, what client
   interface it needs, and how it degrades when no client is present. Report whether enabling it in
   a TEST harness (not native default) could validate category (b) rows. No dependency installs.
3. Recommend: which CONTRADICTED rows are achievable in native mode (category a), which need NLI
   (b), which should be relabeled/dropped (c). This tells Opus whether Task 16 (case-41) is an
   (a)-type win or a (b)-type defer.

## PART D — Per-task engine groundwork (so Opus implements fast)

For EACH task below produce: exact failing assertion (`--runxfail` where relevant), full decision
trace (every candidate: analyzer, failure_type, stage, tier, weight, evidence_summary; the winner;
the suppression reason), the precise file+function+line of the change locus, the full regression
surface (every test/fixture asserting the same field), and a filled prereg (hypothesis, change
locus, hard criteria with exact test IDs + protected/Calib/probe gates, success criterion).

- **Task 15** (lowest risk): `test_quality_incomplete_38_has_generation_stage_candidate_if_supported`
  and `tests/test_pr5e_answer_quality.py::test_incomplete_answer_with_good_context_stage_generation`.
  Primary stays `UNSUPPORTED_CLAIM`; only stage GROUNDING→GENERATION via `AnswerQualityAnalyzer`.
  Find why the GENERATION candidate loses trace selection (`_suppress_citation_when_downstream_symptom`).
- **Task 14**: `test_stale_irrelevant_source_does_not_primary_fail`. Root cause confirmed:
  `evidence_profile.py` builds `stale_doc_ids` from ABSOLUTE age, flagging even the fresh cited
  `doc-ceo`. Find the populating function+lines; dump the protected case
  `version_stale_not_cited_32` (chunk relevance, citation, dated alternatives) so Opus can find a
  predicate that drops Task-14's irrelevant `doc-lease` and fresh `doc-ceo` WITHOUT breaking case 32.
  List every STALE_RETRIEVAL regression dependency.
- **Task 16**: `test_quality_ignores_context_41...`. Determine via PART C whether case 41 has a
  hard conflict (native-achievable) or is semantic-only (defer with Task 22). Provide the claim
  records for case 41.
- **Task 24** (false-CLEAN: list/short answers — higher CLEAN-precision risk): for the remaining
  `INSUFFICIENT_CONTEXT→CLEAN` (≈6) and short/list-answer rows, run a dry-run of candidate
  extractor predicates (comma-list ≥3 items; query-conditioned short factual answer) over BOTH the
  false-CLEAN slice (coverage) and the 30 expected-CLEAN rows (precision risk). Report the coverage
  vs. expected-CLEAN-firing trade-off so Opus can choose the smallest safe rule. Note: these are
  expected INSUFFICIENT_CONTEXT — verify whether better extraction even moves them toward the right
  label or just toward UNSUPPORTED; if the latter, flag that Task 24 may need sufficiency work, not
  extraction.

## PART E — Taxonomy honesty & calibration prep (no flag changes)
1. Re-run `scripts/check_taxonomy_support.py`; list the 13 zero-data types and `RERANKER_FAILURE`.
   Draft (do not apply) a quarantine note: which enum types must be excluded from any public/README
   capability claim until real data exists. Locate every README/docs line that currently implies
   broad type coverage.
2. Calibration prep groundwork only: enumerate exactly what a calibration pack would need (seeds,
   CI, calibration curve) and what blocks flipping `calibration_status` → `preliminary`. Change
   nothing.

## Closeout (per the ledger's Closeout Format)
Files inspected; changes made (None for code; list any staging data artifacts created under
`staging/`); method status; tests/scores run with results; protected files changed (no);
labels/thresholds/gates/flags changed (no); known limitations; and a single prioritized
"next implementation order for Opus" list. Hand back to Opus; do not implement engine changes.
