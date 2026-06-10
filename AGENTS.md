# GovRAG Agent Rules

This repo is a research-grade RAG diagnosis and governance system. GovRAG diagnoses failures in high-trust production RAG systems by identifying what failed, which claim failed, which pipeline stage caused it, what evidence supports the diagnosis, what alternative explanations exist, and what should be fixed next.

The goal is not to remove every heuristic immediately. The goal is to make every heuristic, signal, external package output, fallback, and confidence claim explicit, accountable, testable, and eventually calibrated.

## Quick Reference

- Install dev: `pip install -e ".[external,llm]"`
- Run tests: `python -m pytest tests/ -q --tb=short`
- Run single test: `python -m pytest tests/path/to/test.py::test_func -v`
- CLI demo: `raggov diagnose examples/native_diagnose_demo_run.json`

## Architecture Rules

1. Native mode is primary. External providers are optional signal enhancers, not defaults and not source of truth.
2. External packages are signal providers. RAGAS, DeepEval, RAGChecker, RefChecker, NLI models, LLM judges, and rerankers may provide signals, but GovRAG must normalize, compare, and explain them.
3. Never invent confidence scores. Confidence must come from evidence, calibration, statistical estimation, sampling, or clearly labeled heuristic status.
4. Label heuristics explicitly. Regex, keyword matching, token overlap, fixed thresholds, and rule-based scoring are `heuristic_baseline` or `practical_approximation`.
5. No silent fallbacks. If an external provider, LLM, model, or stronger method fails, surface degradation in metadata.
6. Prioritize claim grounding: extraction, claim-to-context evidence, entailment labels, citation support, and provenance before meta layers such as NCV, Layer6, A2P, or Semantic Entropy.
7. A signal is not a diagnosis. Scores and external outputs feed the diagnosis layer; they do not overwrite native diagnosis.

## Method Status

Use one of:

- `research_faithful`
- `practical_approximation`
- `heuristic_baseline`
- `external_signal`
- `calibrated_statistical`
- `experimental_unvalidated`

Do not claim a paper is implemented unless the actual mechanism is implemented. Semantic Entropy requires multiple generations, semantic clustering, and entropy over meaning groups. A2P requires abduct-act-predict counterfactual reasoning. RefChecker requires triplet-level extraction and verification. RAGChecker requires modular claim-level diagnostics. Conformal prediction requires calibration data. Regex-only prompt injection detection is a heuristic baseline.

## Failure Taxonomy

Key `FailureType` values: `CLEAN`, `CITATION_MISMATCH`, `UNSUPPORTED_CLAIM`, `CONTRADICTED_CLAIM`, `STALE_RETRIEVAL`, `SCOPE_VIOLATION`, `PROMPT_INJECTION`, `INSUFFICIENT_CONTEXT`, `RETRIEVAL_ANOMALY`.

Key `FailureStage` values: `PARSING`, `CHUNKING`, `EMBEDDING`, `RETRIEVAL`, `RERANKING`, `CONTEXT_ASSEMBLY`, `GROUNDING`, `SUFFICIENCY`, `GENERATION`, `CITATION`, `SECURITY`, `CONFIDENCE`.

See `docs/FAILURE_TAXONOMY.md` for full mapping of failure types to stages and fixes.

## Working Conventions

Before editing, read only files named in the prompt, direct imports if needed, and targeted tests if needed. List inspected files and reasons in the response.

For code changes:

- Prefer minimal patches.
- Preserve public APIs, test structure, and fixtures unless asked to change them.
- Run targeted tests for the changed component.
- Do not rewrite architecture unless requested.
- Do not add a new analyzer if strengthening an existing evidence substrate solves the problem.
- Do not read reports, research notes, old prompts, logs, archived files, large eval outputs, or benchmark dumps unless requested.

## Dependencies

Core dependencies: `pydantic>=2.0`, `typer>=0.9`, `rich>=13.0`.

Heavy dependencies such as `sentence-transformers`, `ragas`, `deepeval`, `ragchecker`, `refchecker`, and `spacy` must remain optional extras. Native mode must work without them.

## Avoid

Do not load the full repo by default, treat external evaluator output as ground truth, invent confidence values, hide fallback/provider failures, present heuristics as research-faithful, use token overlap as final citation faithfulness, use regex-only prompt injection detection as final security diagnosis, add thresholds without tests or calibration hooks, blindly average external scores, claim Semantic Entropy/A2P/RefChecker/RAGChecker without the actual mechanism, or add meta-diagnosis before strengthening claim-level evidence.

## Entry Points

- SDK: `from raggov import diagnose, RAGRun, Diagnosis`
- CLI: `raggov diagnose <run.json>`
- Engine: `DiagnosisEngine(config={...})` with analyzers passed explicitly

## Validation

Before claiming completion, run targeted tests; run `tests/test_engine.py` for orchestration changes; run relevant `tests/test_analyzers/` tests for analyzer changes; confirm heuristic labels, visible degradation metadata, no fake confidence scores, and native mode without external providers.

## Agent Workspace Harness

This repo includes a lightweight audit-only harness for future coding agents. Run it before and after edits that may affect benchmark integrity, workspace safety, thresholds, reports, launch readiness, or production gating.

Protected baseline:

- native common benchmark: `41/46`
- external-enhanced common benchmark: `41/46`
- `false_clean_count = 0`
- `false_security_count = 0`
- `false_incomplete_count = 0`
- citation `5/5`, grounding `7/7`, sufficiency `5/5`, version_validity `5/5`
- `production_gating_eligible = false`
- calibration status: `not_production_calibrated`

Required preflight before edits:

- Run `python scripts/workspace_audit.py`.
- Run `python scripts/harness_preflight.py`.
- If common-suite context is explicitly needed, run `python scripts/harness_preflight.py --run-common`.
- Stop and ask before changing benchmark, fixture, golden, threshold, gate, or production-gating files.

Required post-edit validation:

- Run targeted tests for the changed component.
- Run `python scripts/harness_post_edit_validation.py`.
- For benchmark or readiness work only, add `--run-common` or `--run-launch`.
- Preserve `false_clean_count`, `false_security_count`, and `false_incomplete_count` unless the user explicitly asks otherwise.

Golden fixtures and benchmark labels:

- Do not change `expected_primary_failure`, `expected_stage`, `expected_root_cause`, expected fix categories, golden outputs, or fixture cases without explicit user instruction.
- If a benchmark result regresses, stop, report the regression, and diagnose the implementation path before touching labels.

Thresholds and production gating:

- Do not change analyzer thresholds or launch-readiness gates in unrelated PRs.
- Do not enable `production_gating_eligible`.
- Do not claim calibration from harness outputs; `harness/protected_baseline.json` is a reference only and not a product logic input.

Reports and degraded external mode:

- Reports may be generated additively, but must not be rewritten to hide failures, Not Ready status, provider runtime degradation, or known blockers.
- External-enhanced mode may be degraded; surface that as warning/degradation metadata rather than treating provider output as source of truth.

Change reporting:

- List files created or modified.
- List commands run and results.
- State whether protected files changed.
- State whether benchmark labels, thresholds, gates, or `production_gating_eligible` changed.

## Response Format

After coding tasks, respond with:

```text
Files inspected:
- <file>: <why>

Changes made:
- <change>

Method status:
- <status>

Fallback/degradation behavior:
- <behavior>

Tests run:
- <command and result>

Known limitations:
- <limitations>

Next recommended step:
- <one concrete next step>
```

Be surgical. Do not produce long project essays after small code changes.
