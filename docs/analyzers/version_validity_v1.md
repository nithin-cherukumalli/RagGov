# Temporal Source Validity Analyzer V1

`TemporalSourceValidityAnalyzerV1` diagnoses whether retrieved and cited source
documents appear active and temporally usable for the query based on available
lifecycle metadata. `VersionValidityAnalyzerV1` remains as a compatibility name.

This is a practical approximation. It is not a research-faithful VersionRAG
implementation and is not domain-specific temporal reasoning.

## What It Does

- Inspects retrieved document IDs and cited document IDs.
- Reads available corpus, chunk, retrieval, citation, and lineage metadata.
- Supports generic lifecycle fields such as `effective_date`, `expiry_date`,
  `valid_from`, `valid_to`, `version_id`, `supersedes`, `superseded_by`,
  `replaces`, `replaced_by`, `deprecated_by`, `withdrawn_by`, and `status`.
- Flags documents as active, stale by age, superseded, amended or revised,
  replaced, deprecated, withdrawn, expired, not yet effective, or metadata
  missing.
- Exposes `evidence_paths` on document and claim records for every decision.
- Builds claim-level source validity records when a citation faithfulness
  report is available.
- Reports dependency visibility:
  - `retrieval_evidence_profile_used`
  - `citation_faithfulness_report_used`
  - `lineage_metadata_used`
  - `age_based_fallback_used`

## What It Does Not Do

- It does not run an LLM judge.
- It does not perform full domain-specific temporal reasoning.
- It does not prove component-level applicability.
- It does not modify retrieval evidence or citation faithfulness internals.
- It is not a research-faithful VersionRAG implementation.
- It does not encode concepts such as government orders, circulars, statutes,
  medical guidelines, or API versions in the core analyzer; those belong in
  optional adapters.

## Why Age-Based Stale Retrieval Is Insufficient

Age is only a weak heuristic. A recent document can be superseded, withdrawn,
or not yet effective. An old document can remain authoritative if no later
lineage event changes it. For that reason, age-based staleness is reported as
unknown risk with an explicit heuristic freshness warning, not invalidity.

## Difference From StaleRetrievalAnalyzer

`StaleRetrievalAnalyzer` checks whether retrieved documents exceed an age
threshold. `TemporalSourceValidityAnalyzerV1` checks richer temporal metadata
when available: status, effective/valid-from date, expiry/valid-to date,
supersession, replacement, deprecation, amendment/revision, and withdrawal
lineage. It only uses age fallback when no lineage metadata exists.

## Use Of Retrieval And Citation Reports

The analyzer uses `retrieval_evidence_profile` to make dependency use explicit
and to align with the existing retrieval evidence pipeline. It uses
`citation_faithfulness_report` to create claim-source validity records and
identify high-risk claims when cited documents are invalid.

## Why Gating Is Disabled

`recommended_for_gating` is `False` because V1 is uncalibrated and depends on
metadata completeness. Missing or partial lineage can produce unknowns, and
amendments can require clause-level analysis.

## Future Upgrade Path

- V2 can add structured document lineage graphs and clause-level amendment
  mapping.
- V2 can add optional domain adapters for corpus-specific applicability
  concepts while keeping the core metadata/lineage based.
- V3 can add calibrated temporal validity benchmarks and human-labeled
  claim-source temporal applicability data.
- Research-faithful VersionRAG-style modes should only be claimed once the
  implementation matches the method assumptions and evaluation protocol.
