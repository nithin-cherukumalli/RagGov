# Day 1 — Antigravity (Sidekick 1): seeded multi-run measurement wrapper

READ-ONLY on engine/analyzer/policy/labels/gates. You may create ONE standalone script under
`scripts/` and append findings to a dated section of
`reports/codex_session/codex_sidekick_session_plan.md`. Reproduce all numbers on current committed
code — do NOT quote earlier reports (they drift).

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```

## Anchors to confirm first (must match before doing anything else)
- protected effective 43/46 (`python scripts/check_protected_baseline.py`)
- Calib 23/45 (train/dev/heldout split)
- induced probe: 80/145 in DEFAULT config, 82/145 in `mode=native` — report BOTH, label the mode.

## Task: `scripts/eval_report.py` (standalone, read-only measurement)
Opus is committing a canonical scorer today (`scripts/raggov_score.py` with a `score_file(path,
mode, splits)` API). IMPORT it — do not write a second scoring path. Your wrapper must:
1. Run the scorer across **N seeds** (default 5) for both `mode="default"` and `mode="native"`,
   over: canonical Calib (train/dev/heldout) and the induced probe.
2. Emit a per-type table: type, n, correct, accuracy, and a `confidence_mean` column read from
   `diagnosis` if present (placeholder/None is fine today — schema must exist).
3. Emit overall accuracy + per-mode + seed variance (min/max/mean) so we can see determinism.
4. Write JSON + markdown to `reports/calibration/eval_report_<date>.json|md` using the schema
   Codex (S2) fixes today.
5. Verify your wrapper matches Opus's `score_file` exactly on 3 named spot cases (gc-001, a probe
   CITATION row, a probe CLEAN row) — print the 3 side-by-side.

Do NOT compute ECE/Brier yet (no calibrated confidence exists). Just accuracy + variance + the
confidence-column plumbing, so Phase 3 can drop calibration in without reshaping the report.

## Closeout (ledger format)
Files inspected; the one script created (path); anchors reproduced (with mode labels); seed
variance; the 3-case parity check vs Opus scorer; protected/labels/gates changed (no); next step.
HAND BACK — Opus re-verifies your numbers before they enter the record.
