# GovRAG-Calib Adjudication Guide

Adjudication improves label trust without changing product behavior to chase evaluator accuracy. Do not edit labels simply because GovRAG currently predicts something different.

## Label Status Values

- `seed`: initial scaffold label, synthetic/manual draft, migrated benchmark label, or uncertain reviewed case.
- `reviewed`: checked by a human or careful agent pass for schema consistency and obvious label quality.
- `adjudicated`: resolved after comparing evidence, alternatives, and ambiguity notes.
- `heldout_locked`: locked heldout case. Do not use this status until heldout policy is formally applied.

## When To Downgrade `reviewed` To `seed`

Downgrade when:

- evidence is plausible but ambiguous
- expected failure family is uncertain
- clean-pass context contains distracting evidence that could reasonably trigger another family
- expected stage is not confidently separable from another stage
- provenance does not fully explain a synthetic transformation

Downgrading is preferred over forcing labels to improve accuracy.

## When To Add Acceptable Alternatives

Add `acceptable_alternative_failures` when the same record evidence reasonably supports more than one diagnosis:

- retrieval versus sufficiency
- grounding versus citation
- stale retrieval versus contradicted claim
- low confidence versus metadata loss
- security-relevant sufficiency versus privacy/security review

Do not add an alternative merely because the evaluator produced it. The alternative must be defensible from the record evidence.

## When To Require Human Review

Set `expected_human_review_required: true` when:

- answer could affect safety, privacy, security, eligibility, money, legal status, or operational decisions
- case has ambiguous but consequential evidence
- diagnosis is expected to be non-clean and label confidence is not high
- security/privacy/adversarial evidence is present
- version validity affects current policy or access decisions

Do not use human review as a substitute for a missing label. Use notes to describe ambiguity.

## When Not To Change Labels

Do not change labels just to improve evaluator accuracy. Keep or downgrade labels when:

- GovRAG misses evidence but the label is clear
- a model limitation is being intentionally measured
- exact stage remains disputed
- fixing the case would hide an RC or production blocker

## Documenting Ambiguity

Use `notes` and `provenance.transformation_notes` to record:

- the competing labels considered
- why the primary label was kept
- why alternatives were added
- why a reviewed case was downgraded to seed
- what evidence would be needed for adjudication

Ambiguity documentation must not imply calibration or production readiness.
