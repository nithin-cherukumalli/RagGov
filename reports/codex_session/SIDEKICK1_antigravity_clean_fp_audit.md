# Sidekick 1 (Antigravity) — CLEAN false-positive per-mechanism audit

READ-ONLY. No engine/analyzer/policy/label/gate edits. Append findings to a NEW dated section of
`reports/codex_session/codex_sidekick_session_plan.md`. Reproduce numbers on current committed code
(anchors: protected effective 43/46, Calib 23/45, probe 80/145).

## Env
```
mkdir -p /tmp/shim
printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
export PYTHONPATH=/tmp/shim:src:.
```

## Scope (do NOT touch INSUFFICIENT_CONTEXT — Opus is fixing that bucket now)
The 30 expected-CLEAN induced rows currently yield 17 false positives. Opus owns the
`CLEAN→INSUFFICIENT_CONTEXT` bucket. You audit the OTHER buckets:
`CLEAN→STALE_RETRIEVAL` (~3), `CLEAN→UNSUPPORTED_CLAIM` (~2), `CLEAN→CITATION_MISMATCH` (~2),
`CLEAN→INCONSISTENT_CHUNKS` (~2), `CLEAN→CONTRADICTED_CLAIM` (~2), `CLEAN→GENERATION_IGNORE` (~1).

For EACH such row produce: the query (short), the winning analyzer + its failure_type + the exact
evidence string it fired on, the decision-trace selection_reason, and a one-line judgment of whether
the firing is (a) a genuine precision bug (analyzer over-fires on benign content), (b) a
plausible-but-debatable label, or (c) a synthetic-mutation artifact. Group by mechanism and rank the
buckets by how many rows a single narrow fix could recover and how safe it looks. Do NOT propose
code — just the ranked, evidenced map so Opus can pick the next precision task.

## Closeout (ledger format)
Files inspected; changes (None); per-row table; ranked mechanism map; protected/labels/gates changed
(no); next recommended precision target for Opus.
