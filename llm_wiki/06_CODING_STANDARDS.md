# Coding Standards

## Architectural Standards

1. Preserve diagnosis-first architecture.
2. Prefer substrate hardening over policy-layer patches.
3. Add structured evidence before adding new scores.
4. Keep evidence contracts in `src/raggov/models/` explicit and typed.
5. Do not hide fallback behavior.
6. Treat context engineering as part of implementation quality.

## Change Discipline

- Do not patch weak analyzer signals by growing `decision_policy.py`.
- Do not add benchmark-specific `if/else` overrides to meta layers.
- Do not add new analyzer classes unless explicitly approved.
- Do not repurpose `warn` or `skip` semantics casually.

## Context Hygiene Standards

- keep architecture language consistent across code and wiki
- do not let research-branded names drift away from implementation truth
- when changing a contract, update all relevant readers and docs in the same PR
- when changing semantics, update the glossary or registry if needed
- do not leave hidden meaning in tests only

## Evidence Standards

- If a module produces evidence, output structured fields when possible.
- If a module interprets evidence, preserve provenance to upstream analyzers.
- If a module uses thresholds, document:
  - the threshold
  - why it exists
  - whether it is calibrated
  - whether it is advisory only

## Naming Standards

- Use “heuristic”, “proxy”, “practical approximation”, or “experimental” when true.
- Avoid “semantic”, “faithfulness”, “calibrated”, “counterfactual”, or paper-branded names unless the implementation warrants it.
- If a legacy name remains, document the gap instead of pretending it does not exist.
