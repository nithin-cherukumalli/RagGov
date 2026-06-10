# Project Memory Status

## Repository Snapshot For PR-000A (COMPLETED)

This wiki was validated and corrected against the current repository state.

High-confidence repository facts:

- core orchestration is centered in `src/raggov/engine.py`
- final primary-failure selection is centered in `src/raggov/decision_policy.py`
- diagnosis assembly is typed through `src/raggov/models/diagnosis.py`
- parser validation, grounding, sufficiency, retrieval diagnosis, citation faithfulness, version validity, and security are the evidence-producing backbone
- NCV, Layer6, A2P, semantic entropy, and decision policy are meta interpretation or selection layers
- complete substrate file reference now available in `llm_wiki/21_SUBSTRATE_FILES.md`

## PR-000A Changes (Completed)

1. **02_ARCHITECTURE_MAP.md**: Updated analyzer order to match actual 5-layer execution order in `DiagnosisEngine._default_analyzers()`
2. **04_ANALYZER_REGISTRY.md**: Added missing analyzers:
   - `ContradictionAnalyzer`
   - `FreshnessAnalyzer`
   - `RelevanceAnalyzer`
   - Updated notes on `RetrievalEvidenceProfilerV0` and `VersionValidityAnalyzerV1`
3. **Created 21_SUBSTRATE_FILES.md**: Complete reference of all substrate implementation files

## Memory Warnings

- verify current local modifications before relying on any unstaged runtime behavior
- verify in code before relying on exact analyzer thresholds
- verify in code before relying on exact test coverage
- update this folder when project meaning changes, not only when files are added

## Ongoing Maintenance Rule

This folder must stay operationally current.

That means:

- architecture changes require architecture doc updates
- analyzer trust changes require registry updates
- fallback changes require fallback policy updates
- workflow changes require agent rule and PR protocol updates

If those updates do not happen, future AI work will accumulate confusion instead of reducing it.

## Recommended Next PR

`PR-001: Honest analyzer classification and trust metadata`

Target intent:

- unify analyzer method classification
- unify fallback exposure
- unify trust metadata across reports and diagnosis outputs

Current state:

- analyzer results now carry normalized trust metadata in runtime output
- diagnosis summaries now surface fallback and trust notes
- decision traces now carry fallback context
- wiki now accurately reflects actual analyzer inventory and file structure
