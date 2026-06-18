# Codex Sidekick Prompt — prepare the FRESH-data unblock (real heldout, no probe overlap)

Paste to Codex. READ-ONLY + prep-script + staging only. Opus implements engine changes.

## Why this task
The v0_2 heldout attempt is NOT usable: all 45 rows overlap `induced_candidates.jsonl` (same
RAGTruth-migrated rows), it's only 2 types, and the CONTRADICTED rows are the Task-22
native-undetectable ones. We need a GENUINELY held-out, fresh set. Hugging Face is blocked in this
sandbox, so the actual pull must run on the USER's machine. Your job is to make that turnkey and to
build the dedup/validation tooling so the result is clean.

## Rules
- No edits to engine/analyzer/policy/labels/gates/locked dataset. Prep scripts + staging only.
- Reproduce numbers on current code. Do not quote stale reports.
- Append findings to a NEW dated section of `reports/codex_session/codex_sidekick_session_plan.md`.

## Deliverables
1. **Exact puller runbook for the user.** Read `scripts/pull_seed_intake.py`. Produce the precise
   commands the user runs on THEIR machine (with HF access) to pull a FRESH slice — including any
   args to (a) target RAGTruth QA + ALCE + HotpotQA, (b) request a row count (~80–100 raw so ~40–60
   survive dedup), and (c) write to `evals/govrag_calib/staging/raw/fresh_intake_v1.jsonl`. Note any
   env/token requirements. If the script lacks a seed/offset to avoid re-pulling the SAME rows
   already in `starter_seed_intake.jsonl`, specify exactly what arg/patch the user needs (describe;
   do not edit engine code).
2. **A dedup + validation script** `scripts/validate_fresh_heldout.py` (NEW standalone script, not
   an engine file) that, given a fresh intake JSONL, (a) drops any row whose normalized query+answer
   or source id already appears in `govrag_calib_150.jsonl` OR `induced_candidates.jsonl` OR
   `starter_seed_intake.jsonl` (no training/probe overlap), (b) reports per-type + per-source counts
   of survivors, (c) flags rows whose migrated label is heuristic (RAGTruth contradicted/baseless)
   for human review, (d) validates schema via `add_calib_case.py` validate-only. It must NOT write
   to canonical or the lock.
3. **A scoring stub**: a function/CLI that scores the engine on the validated fresh heldout (build
   RAGRun as in prior sims) and prints per-type + overall — so the moment the user drops fresh data,
   we get the first honest, non-overlapping production-generalization number. Clearly label it
   provisional until labels are double-adjudicated.
4. **Honesty note in the ledger**: state plainly that until the user runs the pull, NO valid
   production-generalization number exists (the probe is synthetic; v0_2 overlapped the probe).

## Closeout (ledger format)
Files inspected; new prep scripts created (paths); code/labels/gates changed (no); the exact
user runbook; known limitations; next step. Hand back to Opus.
