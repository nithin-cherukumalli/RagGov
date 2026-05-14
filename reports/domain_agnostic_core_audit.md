# Domain-Agnostic Core Audit

Date: 2026-05-13

## Verdict

The audit found real government/policy leakage in default grounding and sufficiency paths. The highest-risk items were fixed in this pass:

- Claim extraction no longer requires `G.O.`, circular, notification, order, or gazette terms to treat an answer sentence as claim-worthy.
- Generic value extraction no longer emits GO numbers by default.
- Default claim typing no longer returns `go_number`, `eligibility`, or `policy_rule`.
- Triplet extraction now defaults to `GenericRuleTripletExtractorV0`; government triplets are explicitly `GovernmentPolicyTripletExtractorV0`.
- LLM triplet fallback is generic.
- Sufficiency requirement extraction now prompts for generic RAG evidence requirements.
- Version/source validity now accepts generic metadata fields such as `publication_date`, `updated_at`, `version`, and `current_version`.

## Findings By Risk

Critical:

- `src/raggov/analyzers/grounding/claims.py`: default claim-worthiness included government publication terms and `G.O.`. Fixed by replacing the default pattern with generic factual, value, version, relationship, procedural, and requirement signals.
- `src/raggov/analyzers/grounding/value_extraction.py`: default extractor emitted GO numbers. Fixed by moving GO extraction to `extract_government_policy_value_mentions`.
- `src/raggov/analyzers/grounding/evidence_layer.py`: default taxonomy used `go_number`, `eligibility`, and `policy_rule`. Fixed with generic assertion labels.

High:

- `src/raggov/analyzers/grounding/triplets.py`: enabled triplet extraction defaulted to policy rules. Fixed by adding `GenericRuleTripletExtractorV0` and making government extraction explicit.
- `src/raggov/analyzers/sufficiency/sufficiency.py`: requirement extraction prompt framed the task as government-document RAG. Fixed with generic requirement types.
- `src/raggov/models/diagnosis.py`: sufficiency requirement schema was policy/legal-flavored. Fixed by adding generic requirement types while preserving legacy aliases.
- `src/raggov/analyzers/version_validity/analyzer.py`: source validity under-supported common non-government metadata. Fixed with generic metadata handling.

Medium:

- `src/raggov/analyzers/sufficiency/sufficiency.py`: scope detection still uses a US-state location list for one structured heuristic. It now also includes generic qualifier terms, but full generic location/scope extraction remains open.
- `src/raggov/analyzers/confidence/semantic_entropy.py`: ambiguity detection is still tuned to policy/return query wording. Not changed in this pass because it is a confidence sidecar, not core grounding.
- `src/raggov/analyzers/grounding/verifiers.py`: one rationale still says "eligibility" for generic overgeneralization beyond an explicit `only` constraint. Behavior is useful; wording should be generalized next.
- Existing tests and stresslab fixtures are government-heavy. Kept, and supplemented with non-government domain tests.

Low:

- `stresslab/ingest/parse_go_order.py` is intentionally government-specific and remains outside the core path.
- `department` remains in version-validity records as optional metadata and is not used as a decision signal.
- Government PDFs and fixtures remain as government benchmark material.

## Default Core Path After Refactor

No government/policy regex is required in default core grounding. Government-specific triplets and GO value extraction remain available only through explicit non-default calls/configuration.

## Open Follow-Ups

- Rename the remaining overgeneralization rationale in `verifiers.py` to remove eligibility-specific wording.
- Generalize confidence ambiguity detection beyond policy/returns language.
- Add calibrated semantic/entity extraction for generic scope and location signals.
