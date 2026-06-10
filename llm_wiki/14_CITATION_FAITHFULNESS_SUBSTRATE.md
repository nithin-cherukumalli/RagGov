# Citation Faithfulness Substrate

Core files:

- `src/raggov/analyzers/citation_faithfulness/analyzer.py`
- `src/raggov/analyzers/grounding/citation_faithfulness.py`
- `src/raggov/models/citation_faithfulness.py`

## Current Honest Description

Current citation faithfulness is a practical approximation.

It uses:

- claim grounding outputs
- retrieval evidence
- optional structured LLM or RefChecker-backed signals

It does not prove genuine model reliance on cited evidence.

## Evidence-Producing Components

- `CitationFaithfulnessAnalyzerV0`
- `CitationFaithfulnessProbe`
- `CitationMismatchAnalyzer` only as a supporting retrieval provenance signal

## Authority Rule

In native mode, `CitationFaithfulnessAnalyzerV0` is the primary citation failure authority.
`CitationMismatchAnalyzer` may support provenance mismatch diagnosis, but it should not outrank grounding-stage citation evidence when that evidence exists.

## Main Risks

- doc-level citation can overstate support
- external citation verification is optional and advisory
- fallback to native rollups must remain visible

## Rule

Do not call current citation analysis research-faithful.
