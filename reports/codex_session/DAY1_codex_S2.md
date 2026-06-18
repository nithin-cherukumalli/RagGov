# Day 1 — Codex (Sidekick 2): calibration report schema + LLM-labeling harness scaffold

READ-ONLY on engine/analyzer/policy/labels/gates/canonical dataset. You may create standalone
`scripts/` data tools and `reports/calibration/` scaffolding only. Append findings to a dated
section of `reports/codex_session/codex_sidekick_session_plan.md`. Reproduce numbers on current code.

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```

## Task A — fix the calibration report schema (so Phase 3 just fills it)
Create `reports/calibration/SCHEMA.md` + an empty `reports/calibration/template.json` defining,
per failure type: n, accuracy, confidence_mean, ECE, ACE, Brier, reliability-curve bins, bootstrap
CI, calibration_status, gating_eligible. Overall block + per-mode (default/native). This is the
contract S1's `eval_report.py` writes into and Phase 3 calibration extends. No numbers yet.

## Task B — LLM-assisted labeling harness scaffold (the Phase-1 critical tool)
Build `scripts/llm_label_heldout.py` (standalone; does NOT call an LLM in the sandbox — define the
interface + a deterministic mock judge so it runs offline; real judges plug in where keys exist).
Per row it must:
1. Accept K independent judge callables (default mock); collect K verdicts (expected_primary +
   short rationale) per row.
2. **Majority vote** for the label; record per-row inter-judge **agreement** as a provisional
   `label_confidence` (SOTA: majority of checkers best matches human annotation).
3. Emit a **human spot-audit worklist**: every row where judges disagree OR the voted label is
   CONTRADICTED_CLAIM (RAGTruth contradiction is heuristic) OR agreement < threshold.
4. Tag every output row `label_source=llm_assisted_provisional`, never `gold`; write a
   `LABEL_CHANGELOG` stub entry. Validate shape via `add_calib_case.py` validate-only.
5. NEVER write to the canonical dataset or the lock; staging only.

This is where the trust comes from later — labels are provisional, disagreements go to humans,
agreement becomes confidence. Document those guarantees in the script header.

## Task C — confirm the fresh-data runbook is current
Re-read `scripts/pull_seed_intake.py` and the existing `SIDEKICK_PROMPT_fresh_data_unblock.md`
runbook; correct any drift so the user can pull ~80–100 fresh, non-`starter_seed` rows in one go.

## Closeout (ledger format)
Files inspected; scripts/scaffolding created (paths); the labeling guarantees (majority + audit +
provisional); protected/labels/gates changed (no); next step. HAND BACK — Opus owns when/whether
labels are accepted.
