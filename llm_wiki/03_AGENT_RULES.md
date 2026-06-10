# Agent Rules

## Global Rules

All agents working on GovRAG must follow these rules:

1. Work from code, not memory.
2. Treat GovRAG as an advanced prototype.
3. Do not claim production-grade behavior unless code and evaluation prove it.
4. Do not describe heuristics as calibrated or research-faithful unless explicitly true in code.
5. Do not add new analyzers without explicit approval.
6. Prefer hardening existing substrates over adding surface area.
7. No silent fallback is allowed.
8. Meta layers may not invent evidence.
9. Decision-policy edits require a separate plan PR first.
10. If uncertain, write: `verify in code before relying on this`.
11. Keep the wiki synchronized with the implementation.
12. Prefer high-quality context engineering over local patching.

## Evidence Discipline

Agents must distinguish:

- evidence producers
- evidence interpreters
- policy selectors

Evidence producers may create structured evidence.
Evidence interpreters may aggregate or explain evidence.
Policy selectors may choose among evidence-backed candidates.
None of these may fabricate missing substrate evidence.

## Context Engineering Rules

Agents must maintain strong working context for GovRAG.

Required behavior:

1. Read the relevant substrate files before changing them.
2. Read the nearest model contracts in `src/raggov/models/`.
3. Read the downstream consumers before changing upstream evidence shape.
4. Update the matching wiki files when behavior or meaning changes.
5. Keep terminology stable across code, tests, reports, and wiki.

Context engineering means:

- using the real architecture map
- preserving causal relationships between modules
- minimizing ambiguity in system identity
- preventing documentation drift
- making future AI work safer, narrower, and more reliable

## Wiki Update Rule

The wiki must be updated when any of the following changes:

- analyzer order
- analyzer classification
- fallback behavior
- trust metadata
- evidence contract shape
- architectural priority
- PR rules
- agent workflow

## Dangerous Assumptions To Avoid

- Do not assume paper names imply faithful implementation.
- Do not assume scores imply probabilities.
- Do not assume external evaluator signals are root-cause diagnoses.
- Do not assume `warn` means safe to ignore.
- Do not assume `CLEAN` means trustworthy if critical evidence is missing.

## Required Review Targets Before Any Non-Trivial PR

- `src/raggov/engine.py`
- `src/raggov/decision_policy.py`
- `src/raggov/models/diagnosis.py`
- the substrate file being changed
- the nearest tests and stresslab coverage
