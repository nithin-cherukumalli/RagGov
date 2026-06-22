# Codex Sidekick Prompt — next assignment (Task 25 blast-radius + real-data honesty)

Paste to Codex. READ-ONLY groundwork + data validation only. Opus implements engine changes.
Append findings to a NEW dated section of `reports/codex_session/codex_sidekick_session_plan.md`.

## Rules (unchanged)
- No edits to engine/analyzer/decision-policy/claims/label/fixture/threshold/gate/flag files.
- REPRODUCE every number on the CURRENT committed code; do NOT quote earlier reports — they are
  stale (the engine changed under Tasks 23, 15, 16). Anchor to: protected effective **43/46**,
  Calib **23/45**, induced probe **80/145 = 0.552**.
- Keep `calibration_status=not_production_calibrated`, `production_gating_eligible=false`.

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```
Build each RAGRun: RetrievedChunk(chunk_id,text,source_doc_id=doc_id,score), final_answer=answer,
cited_doc_ids=citations (normalize citation dicts to doc_id).

== TASK A (PRIMARY): full blast-radius of wiring AnswerQualityAnalyzer into the default suite ==
Context: `AnswerQualityAnalyzer` is NOT in `DiagnosisEngine._default_analyzers()` today. Two strict
xfail tests (`test_quality_incomplete_38...`, `test_quality_ignores_context_41...`) require it to be
the *selected* analyzer at GENERATION stage. Opus will wire it in (Task 25) but needs the regression
surface first. Produce, in the ledger:
1. Exactly what `AnswerQualityAnalyzer.analyze()` emits, per signal type
   (`_answer_completeness_signals`, `_context_adherence_signals`, `_overconfidence_signals`):
   the failure_type, stage, status, and the precondition for each to fire.
2. A simulation over ALL 145 induced probe rows + all 45 scored Calib rows + the protected common
   cases: for each row, run the analyzer standalone (it consumes prior ClaimGrounding/Sufficiency
   results — replicate by building the full result list or instantiating it after the others) and
   record which rows it would fire on and with what failure_type/stage. Tabulate: how many rows
   gain a new AnswerQuality candidate, and of those, how many would change the SELECTED primary or
   stage if it were in the pool with normal tier/weight.
3. The decision-policy interaction: where would an AnswerQuality candidate rank
   (`_specificity_rank`, evidence tier) vs ClaimGroundingAnalyzer's UNSUPPORTED/CONTRADICTED? Will
   it actually WIN selection for cases 38/41, or also need a specificity/tier rule? Cite the exact
   functions/lines.
4. The complete regression surface: every test asserting `selected_analyzer`, `root_cause_stage`,
   or primary for UNSUPPORTED/CONTRADICTED/answer-quality cases (grep + list). Flag any that would
   change if AnswerQuality starts winning.
5. A filled prereg for Task 25 with hard criteria (protected 43/46; Calib >=23/45; probe overall
   >=80/145; both xfail tests flip; named list of tests that must stay green) and the smallest
   insertion point + any tier/specificity rule needed.

== TASK B (SECONDARY): production-vs-synthetic honesty + real heldout ==
The probe is synthetic mutations; we need real-data validation (the project's stated bar).
1. Re-score the landed fixes on NON-synthetic rows only (canonical Calib 45 + heldout_v0_1 15):
   per-type accuracy now, and specifically whether Task 23's source-assertion rule is exercised by
   ANY non-synthetic row (grep real answers for "source notes/states/reports…"). Report how much of
   probe 0.552 is the repeated suffix mutation (count induced rows whose only diff from their clean
   base is that suffix).
2. Assemble a CANDIDATE real heldout of 40–60 rows from original RAGTruth/ALCE/HotpotQA labels
   (NOT mutated), balanced across CLEAN / INSUFFICIENT_CONTEXT / CONTRADICTED_CLAIM / UNSUPPORTED /
   CITATION_MISMATCH, into `evals/govrag_calib/staging/raw/heldout_candidate_v0_2.jsonl` (staging
   only; do NOT touch canonical or the lock). For each row: source id, original label, proposed
   expected_primary, one-line rationale, and a flag if the label is heuristic/ambiguous. Validate
   shape with `scripts/add_calib_case.py --validate-only` (or equivalent); report rejects. Produce
   a double-labeling worklist; do NOT adjudicate yourself.

## Closeout (ledger format)
Files inspected; data artifacts created (staging only); changes to code (None); tests/scores run +
results on CURRENT code; protected/labels/gates changed (no); known limitations; and a single
"next implementation order for Opus". Do not implement engine changes.
