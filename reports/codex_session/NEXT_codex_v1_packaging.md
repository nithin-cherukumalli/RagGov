# Codex task — v1 packaging verification + examples (read-only / docs only)

Goal: get RagGov to a clean, demoable v0.1. You do verification + example/doc generation on the
Mac; do NOT touch engine/analyzer/policy code (those are Opus-only, verified against the protected
suite). Report findings; Opus assembles the final README.

Setup: `cd /Users/nitin/Desktop/RagGov && source .venv/bin/activate && export PYTHONPATH=src:.`

## 1. Install + demo sanity
- `pip install -e .` — confirm it installs clean; report any dependency errors.
- Run the demo on both examples and PASTE the output:
  - `raggov diagnose examples/quickstart/clean_answer.json`
  - `raggov diagnose examples/quickstart/unsupported_answer.json`
  (expected: CLEAN and UNSUPPORTED_CLAIM respectively).

## 2. Full test-suite triage (this is the trustworthy-state check)
- `pytest -q` — report totals. For EVERY failure, classify it as **pre-existing** vs **new** by
  also running it on a clean `git stash` of any uncommitted changes. Known pre-existing (do NOT
  treat as regressions): `test_prompt_injection_warns_on_hits_below_risk_threshold` (2 hits vs 1),
  the `tests/stresslab/test_subtle_benchmark_regressions_real.py` subtle cases, and the
  `test_cli_diagnose_*` cases that fail with `CliRunner has no attribute isolated_filesystem`
  (a Click/test-harness version mismatch, not our bug). Flag anything else as a real regression.

## 3. Reproduce the benchmark numbers
- `python scripts/gen_benchmark_report.py` — confirm it prints `CLEAN-FP 22/46=0.48 |
  detection 18/29=0.62 | exact 35/75=0.47` and writes BENCHMARK.md. Report if anything differs.

## 4. Build the remaining quickstart examples (NEW data files only — no code)
Create realistic small RAGRun JSON files (schema below) so the demo covers all 6 classes, and run
`raggov diagnose` on each to confirm the engine returns the intended class. Keep them tiny + clearly
synthetic. Put them in `examples/quickstart/`:
- `prompt_injection.json` — a retrieved chunk containing "ignore previous instructions ..." and an
  answer that obeys it.
- `stale_retrieval.json` — chunk with an old dated value, answer repeating the stale value.
- `insufficient_context.json` — query needs info the chunks don't contain; answer hedges/guesses.
RAGRun schema (CLI-validated):
```json
{"query": "...", "final_answer": "...",
 "retrieved_chunks": [{"chunk_id":"c0","source_doc_id":"d1","score":0.9,"text":"..."}],
 "cited_doc_ids": ["d1"]}
```
Report, per file: the class the CLI returned vs intended. (If the engine disagrees, that's useful
signal — note it, don't tune anything.)

## 5. Draft ARCHITECTURE.md (prose only)
~1 page: the analyzer pipeline (retrieval-health, grounding/claim-verification, security stages) ->
typed/tiered signals -> decision policy with audit trace. Use `src/raggov/engine.py`,
`decision_policy*.py`, and `reports/codex_session/taxonomy_v1.md` as sources. No code changes.

## Hard rules
- No engine/analyzer/policy/dataset edits. Examples + ARCHITECTURE.md + reports only.
- Report raw command output; Opus re-verifies before anything is final.
- Don't tune the engine to make an example come out "right" — if it misclassifies a clean synthetic
  case, that's a real finding for the next increment.
