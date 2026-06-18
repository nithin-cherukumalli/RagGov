# Phase 2 — Antigravity (S1): wire the REAL heldout into the seeded report + NLI A/B harness

READ-ONLY on engine/analyzer/policy/labels/gates. Extend `scripts/eval_report.py` only; append
findings to a dated ledger section. Reproduce on current committed code.

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```

## Anchors (confirm first)
Calib 23/45; synthetic probe 80/145 default / 82 native; **real heldout v1
(`evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`) 18/75 = 0.24 default.** This real number
is now THE metric.

## Tasks
1. **Add the real heldout as a first-class scored set** in `eval_report.py` (import `score_file`
   from `raggov_score`; do NOT write a new scoring path). Report per-type accuracy AND a dedicated
   **CLEAN false-positive rate** (of the 50 CLEAN rows, how many got any non-CLEAN label) — that is
   our #1 trust metric now.
2. **Build an NLI A/B comparison mode**: run the real heldout with `claim_grounding_verifier_policy`
   in (a) native heuristic and (b) the entailment policy Opus is adding (flag-gated, e.g.
   `config={"claim_grounding_verifier_policy":"llm_entailment"}` with a mock/local verifier when no
   LLM). Emit a before/after table: overall, per-type, CLEAN-FP rate, CONTRADICTED recall. Opus will
   read this to judge whether the NLI tier helps without breaking native.
3. **Determinism + variance** across 3 seeds as before.
4. Write to `reports/calibration/eval_report_<date>.json|md` (Codex's schema). Spot-parity 2 cases
   vs `raggov_score`.

Do NOT compute ECE yet (Phase 3). Surface the `confidence` column (still uncalibrated).

## Closeout
Files inspected; eval_report changes; the real-heldout + NLI A/B tables; parity check;
protected/labels/gates changed (no); next step. HAND BACK — Opus re-verifies before acting.
