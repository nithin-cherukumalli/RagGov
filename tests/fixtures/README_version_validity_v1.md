# Temporal Source Validity v1 Fixtures

These JSONL fixtures validate TemporalSourceValidityAnalyzerV1 rule behavior
only.

They are not calibration data, not domain-specific temporal proof, and not
production gating evidence. They do not replace human-labeled review, document
lineage extraction, or corpus-specific temporal reasoning.

The cases cover active, stale-by-age, superseded, withdrawn, expired,
not-yet-effective, amended, missing-metadata, citation-linked invalidity, and
skip behavior. They are regression fixtures for the current practical
approximation.

Future v2 work needs document lineage extraction. Future v3 work needs a
human-labeled validity dataset and evaluation against domain-specific
applicability judgments.
