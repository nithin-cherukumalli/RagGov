# GovRAG Failure Taxonomy

GovRAG is a diagnosis-first RAG governance framework.
This taxonomy is the project vocabulary for locating failures by pipeline stage and recommended fix.

Status:

- implementation-aware
- useful as a reference
- not a scientific ontology
- verify exact runtime behavior in code before relying on edge-case semantics

## Core Rule

Do not collapse all RAG failures into "hallucination."
GovRAG treats failures as stage-specific engineering defects with different fixes.

## Failure Map

| Failure Type | Originating Stage | Common Cause | Downstream Symptom | Likely Fix |
| --- | --- | --- | --- | --- |
| `METADATA_LOSS` | `PARSING` | parser dropped source metadata, page, or provenance fields | later retrieval/grounding ambiguity | repair parser profile and metadata normalization |
| `TABLE_STRUCTURE_LOSS` | `PARSING` | table flattened or cells detached from headers | fabricated row or column facts | preserve table structure and table provenance |
| `HIERARCHY_FLATTENING` | `PARSING` | section nesting destroyed | answer binds clause to wrong scope | preserve header/parent-child structure |
| `INSUFFICIENT_CONTEXT` | `RETRIEVAL` or `SUFFICIENCY` | governing evidence not retrieved or not enough support present | abstention should have happened | broaden retrieval or improve abstention logic |
| `SCOPE_VIOLATION` | `RETRIEVAL` | off-topic chunks retrieved | answer supported by irrelevant text | tighten retrieval scope |
| `STALE_RETRIEVAL` | `RETRIEVAL` | outdated or superseded source selected | old policy/version cited as current | improve freshness/version filtering |
| `CITATION_MISMATCH` | `GROUNDING` with retrieval provenance support | cited source does not support the claim or citation span is phantom | answer appears sourced but is not | harden claim-grounding and citation-faithfulness checks |
| `UNSUPPORTED_CLAIM` | `GROUNDING` | answer claim has no evidence in retrieved context | plausible but ungrounded answer | strengthen retrieval recall or generation constraints |
| `CONTRADICTED_CLAIM` | `GROUNDING` | answer conflicts with evidence | direct factual conflict | improve grounding enforcement and contradiction handling |
| `INCONSISTENT_CHUNKS` | `RETRIEVAL` | retrieved corpus contains conflicting chunks | unstable or contradictory answer | deduplicate or rank better |
| `PROMPT_INJECTION` | `SECURITY` | instruction-like malicious content in context | model follows hostile instructions | sanitize or filter retrieved content |
| `PRIVACY_VIOLATION` | `SECURITY` | sensitive/private data surfaced | answer exposes disallowed information | block response and harden filters |
| `SUSPICIOUS_CHUNK` | `SECURITY` | poisoning-like steering chunk | anomalous answer bias | inspect corpus and chunk provenance |
| `RETRIEVAL_ANOMALY` | `SECURITY` or `RETRIEVAL` | abnormal retrieval pattern suggests manipulation/noise | irrelevant or adversarial evidence set | inspect retrieval telemetry and corpus health |
| `LOW_CONFIDENCE` | `CONFIDENCE` | mixed, weak, or unstable evidence | diagnosis should be treated cautiously | inspect upstream evidence before trusting policy layer |
| `INCOMPLETE_DIAGNOSIS` | `UNKNOWN` | critical substrate missing or skipped | tool cannot justify clean or specific diagnosis | restore required analyzers and fallback visibility |
| `CLEAN` | `UNKNOWN` | no substantial failure found | no immediate diagnosis | verify in code before treating as production-safe |

## Operating Rules

- Claim grounding is the highest-leverage substrate to harden first.
- Citation faithfulness depends on claim/evidence quality.
- Meta layers such as NCV, Layer6, and A2P may interpret evidence, not invent it.
- A failure label is only trustworthy when the evidence path is visible.

