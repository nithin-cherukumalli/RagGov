# No Silent Fallback Policy

## Rule

No silent fallback is allowed.

If any component falls back from:

- LLM to heuristic
- structured verifier to deterministic verifier
- external provider to native fallback
- semantic method to lexical method
- calibrated method to uncalibrated method

then that fallback must be exposed in outputs, reports, logs, or metadata.

## Current Code Reality

Fallback behavior already exists in multiple places:

- `src/raggov/engine.py`
  records `missing_external_providers`, `external_adapter_errors`, `degraded_external_mode`, and `fallback_heuristics_used`

- `src/raggov/analyzers/grounding/support.py`
  falls back among verifier modes and records fallback state on claim records

- `src/raggov/analyzers/sufficiency/sufficiency.py`
  can fall back from requirement-aware handling to deterministic term coverage

- `src/raggov/analyzers/verification/ncv.py`
  records `fallback_heuristics_used` node by node

- `src/raggov/analyzers/citation_faithfulness/analyzer.py`
  records external verifier fallback in report fields

- `src/raggov/analyzers/version_validity/analyzer.py`
  records `age_based_fallback_used`

## Required Output Behavior

Fallback exposure should appear in at least one of:

- structured result fields
- report limitations
- `run.metadata`
- diagnosis warnings
- operator-visible logs

Preferred runtime surfaces:

- diagnosis-level `fallback_heuristics_used`
- analyzer-level `fallback_used`
- analyzer-level `fallback_heuristics_used`
- summary-level trust notes
- decision-trace fallback context

## Forbidden Pattern

Do not replace a missing advanced method with a weaker one and then present the output as if nothing changed.
