# Sidekick 2 (Codex) — claims / honesty & taxonomy audit

READ-ONLY. No engine/analyzer/policy/label/gate edits. Append findings to a NEW dated section of
`reports/codex_session/codex_sidekick_session_plan.md`. Reproduce numbers on current committed code.

## Why
We have established: the induced probe is SYNTHETIC (30/145 are one suffix mutation; 0 non-synthetic
rows exercise the Task-23 rule), and NO valid production-generalization number exists yet (every
local "real" heldout overlaps the probe). The project's trustworthiness depends on public artifacts
not overstating this. Audit, do not edit.

## Deliverables (a precise, line-referenced edit list for Opus — no edits)
1. **Overstatement scan.** Grep README.md, docs/**, RELEASE_NOTES*, and any *.md that makes
   capability claims, for: accuracy/percent numbers, "calibrated", "production", "best-in-class",
   "state-of-the-art", "robust", per-failure-type coverage claims, and any quoted probe/heldout
   score. For each hit: file+line, the claim, and whether it is (a) accurate, (b) stale (e.g. quotes
   old 0.62/0.733 or pre-fix numbers), or (c) overstated vs the synthetic-probe / no-real-heldout
   reality. Propose the corrected wording.
2. **Taxonomy honesty.** Re-run `scripts/check_taxonomy_support.py`. Cross-check the 3 supported /
   9 thin / 13 unsupported types against any doc/README that implies all 25 are diagnosable. List
   every type advertised without ≥5 real cases (esp. RERANKER_FAILURE) and where it's claimed.
3. **Flag-state verification.** Confirm in code/config that `calibration_status` is
   `not_production_calibrated` and `production_gating_eligible=false`, and that nothing in this
   session changed them. Cite the exact files/lines.
4. **Honest-numbers table.** Produce one table Opus can paste into README/handoff: current Calib
   (23/45=0.511), probe (80/145=0.552, SYNTHETIC), per-type probe accuracy, and an explicit
   "production generalization: UNKNOWN — pending fresh non-overlapping heldout" line.

## Closeout (ledger format)
Files inspected; changes (None); the line-referenced edit list; taxonomy + flag verification;
honest-numbers table; protected/labels/gates changed (no); next step.
