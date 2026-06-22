# Codex Sidekick Prompt — build a REAL held-out generalization set (corrected)

Paste to Codex. READ-ONLY + staging-data only. The previous v0_2 attempt just concatenated the 55
existing seed cases (which overlap training content) — that is NOT a usable heldout. Redo it
properly using the REAL-labeled rows already in the repo. No Hugging Face needed (it's blocked in
sandbox anyway).

## Rules
- No edits to engine/analyzer/policy/claims/label/fixture/threshold/gate/flag files or the locked
  canonical dataset/manifest. Output staging files only.
- Reproduce numbers on CURRENT committed code. Anchors: protected effective 43/46, Calib 23/45,
  probe 80/145=0.552. Do NOT quote older reports.
- Append findings to a NEW dated section of `reports/codex_session/codex_sidekick_session_plan.md`.

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```

## Goal
A genuinely held-out, real-labeled evaluation set of 40–60 rows that does NOT overlap training
content, so Opus can measure production generalization (the project's stated ≥0.70 bar) instead of
the synthetic probe.

## Steps
1. **Identify real (non-synthetic) source rows.** In `induced_candidates.jsonl` and any seed/raw
   files, separate rows by `label_source`: keep only `benchmark_migrated` (real RAGTruth labels) and
   genuine clean ALCE/HotpotQA base rows. EXCLUDE `synthetic_mutation` rows and any row whose answer
   contains the suffix "The source also notes this was formally reaffirmed at a later international
   summit." (synthetic artifact). Report counts per source and per expected type.
2. **De-overlap with training.** For each candidate, check its source content does not duplicate a
   canonical `govrag_calib_150.jsonl` row (compare normalized query+answer or source doc id). Drop
   overlaps. Report how many survive.
3. **Curate 40–60 balanced rows** across the 3 supported types (CLEAN, INSUFFICIENT_CONTEXT,
   CONTRADICTED_CLAIM) plus UNSUPPORTED_CLAIM / CITATION_MISMATCH. Write to
   `evals/govrag_calib/staging/raw/heldout_real_v0_3.jsonl`. Each row MUST include: source id,
   original source label, proposed `expected_primary_failure`, a one-line rationale, and
   `label_confidence` (high/medium/low) + an `ambiguous` flag where the migrated label is heuristic.
4. **Validate shape** with `scripts/add_calib_case.py` validate-only mode; report any rejects. Do
   NOT append to canonical or regenerate the lock.
5. **Score current engine on this real heldout** (build RAGRun as before) and report per-type
   accuracy + overall — this is the first honest production-generalization number. Clearly label it
   provisional (labels not yet double-adjudicated).
6. **Double-labeling worklist:** list rows needing human adjudication (all `ambiguous` + all
   `CONTRADICTED_CLAIM`, since RAGTruth contradiction labels are heuristic). Do NOT adjudicate.

## Closeout (ledger format)
Files inspected; staging artifacts created; code/labels/gates changed (no); the provisional real-
heldout accuracy with per-type breakdown; known limitations; next step. Hand back to Opus.
