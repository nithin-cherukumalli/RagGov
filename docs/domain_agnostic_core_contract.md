# Domain-Agnostic Core Contract

GovRAG core diagnoses universal RAG failure modes. The default package must not require government or policy document conventions to extract claims, select evidence, compare values, judge sufficiency, classify source validity, or choose a primary failure.

## Core May Use

- Lexical overlap between query, claims, answer, and retrieved context.
- Semantic relevance when a configured provider is available.
- Entity overlap and named anchor overlap.
- Value overlap and value conflict checks.
- Date, number, unit, range, duration, currency, percentage, version, and identifier comparison.
- Citation existence, citation target validity, and cited-source support.
- Source lifecycle states and generic lifecycle metadata.
- Claim support, contradiction, unsupported, partial, and uncertain labels.
- Generic sufficiency requirements.
- Retrieval precision, retrieval coverage, and context utilization signals.
- Answer completeness against the query and retrieved context.
- Uncertainty and ambiguity signals.
- External evaluator signals from providers such as RAGAS, DeepEval, RAGChecker, RefChecker, cross-encoders, or configured LLM judges.

## Core May Not Depend On

- Government order patterns.
- Scheme names or scheme-specific language.
- Department, ministry, mandal, district, AP, India, or government metadata names as required signals.
- GO/G.O./G.O.Rt/G.O.Ms identifiers.
- Policy-only terminology as default claim-worthiness logic.
- Legal or government citation formats as default citation logic.
- Beneficiary or eligibility language as default sufficiency requirement logic.
- Authority hierarchy terms as default requirement logic.

## Allowed Generic Value Types

- `number`
- `percentage`
- `currency`
- `date`
- `duration`
- `measurement`
- `unit`
- `version`
- `identifier`
- `name_entity`
- `quantity`
- `range`
- `ordinal_rank`

## Allowed Generic Sufficiency Requirement Types

- `required_entity`
- `required_value`
- `required_date_or_time`
- `required_condition_or_scope`
- `required_exception_or_limitation`
- `required_comparison_baseline`
- `required_step_or_procedure`
- `required_causal_support`
- `required_source_or_citation`

## Allowed Source Lifecycle States

- `current`
- `stale`
- `expired`
- `superseded`
- `deprecated`
- `withdrawn`
- `draft`
- `not_yet_effective`
- `unknown`

Core analyzers may map these to existing internal model names for backward compatibility, but default diagnosis evidence must remain generic.

## Government Logic

Government-specific logic may remain only as optional, explicitly named non-core behavior, such as `GovernmentPolicyTripletExtractorV0` or government-specific value extractors. It must be disabled by default and must not be required for generic grounding, citation, sufficiency, retrieval, or version-validity diagnosis.
