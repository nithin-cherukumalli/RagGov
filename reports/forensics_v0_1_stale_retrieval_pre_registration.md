# Pre-Registered Fix: Relative-Recency STALE_RETRIEVAL Detection

**Date pre-registered:** 2026-06-12
**Filed before any code change.** Code change will only be applied if all
acceptance criteria below hold on the locked heldout
(`heldout_v0_1`, fingerprint `58261ecc...c9b15c76`), the protected baseline
pin (42, 46), and the Calib-50 safety counters.

## The gap

After Codex analyzer calibration landed (Calib-50 primary 0.48 → 0.54),
14 cases remain wrong on Calib-50. The dominant cluster is 4
version-validity cases (`038`, `039`, `040`, `041`) whose expected primary
failure is `STALE_RETRIEVAL` but GovRAG returns `UNSUPPORTED_CLAIM` /
`TABLE_STRUCTURE_LOSS` / etc.

The existing `StaleRetrievalAnalyzer` only handles two paths:
1. `RetrievalEvidenceProfile.stale_doc_ids` (pre-computed externally)
2. Absolute corpus age (timestamps > `max_age_days`)

Neither path fires on these cases because:
- No retrieval evidence profile is attached.
- No `corpus_entries` block is provided — temporal info lives on
  `chunk.metadata` (`effective_date`, `valid_until`, `as_of`, etc.).

## What a real RAG engineer does

A senior engineer triaging "Why did the model return the 2025 rate for a
2026 question?" inspects chunk metadata, finds the older and newer chunks
both retrieved, sees that the answer textually matches the older one, and
concludes: *retrieval returned a stale chunk and the generator picked it.*

This is the **relative-recency** check. It is pipeline-agnostic
(only reads `chunk.metadata` keys that are standard across LangChain /
LlamaIndex / Haystack / raw retrievers), and domain-agnostic (tax, legal,
HR, security policy — anything time-bound). It is not a Calib-50 heuristic.

## The rule

Fire `STALE_RETRIEVAL` when ALL of the following hold:

1. At least 2 retrieved chunks expose a recognised temporal-metadata key
   in `chunk.metadata`. Recognised keys (configurable, lower-cased):
   `effective_date`, `valid_until`, `valid_from`, `as_of`, `published_at`,
   `updated_at`, `date`.
2. Among chunks with parseable dates, identify the chunk whose text has
   the highest token-overlap with the answer (the "answer-aligned chunk").
3. There exists another retrieved chunk whose effective date is **strictly
   newer** than the answer-aligned chunk's date by at least
   `min_staleness_days` days (default 30).
4. The two chunks are NOT identical-date duplicates (which would mean the
   ordering is incidental, not a freshness conflict).

Evidence emitted names the answer-aligned (stale) chunk, the newer chunk,
the date delta, and the recognised metadata key. Stage = `RETRIEVAL`.

## What this is NOT

- It is not "fire whenever any chunk is old" (that's the absolute-age check
  the legacy path already does and which is too noisy in practice).
- It is not "fire whenever multiple dates exist" — it requires actual
  evidence the answer used the older chunk.
- It is not a token-similarity threshold tuned to pass any specific case.
  The threshold is dataset-independent (max token-overlap chunk wins).
- It does not depend on any specific date format beyond ISO-8601 prefix
  parsing (`YYYY-MM-DD` and `YYYY` both accepted).

## Acceptance criteria (pre-registered, hard)

On the **protected common baseline** (full 46-case suite, both modes):

| Metric | Required |
|---|---|
| pin count | `(42, 46)` unchanged |
| composition | no protected case changes its primary_failure |
| `false_clean_count` | 0 |
| `false_security_count` | 0 |
| `false_incomplete_count` | 0 |

On the **locked heldout** (`heldout_v0_1`, 15 cases):

| Metric | Required |
|---|---|
| `primary_failure_accuracy` | ≥ 0.667 (current) |
| `false_clean_count` | 0 |
| `dangerous_clean_miss_count` | 0 |
| `human_review_miss_count` | 0 |
| Non-version-validity cases | no primary_failure changes (strong form) |

On the **Calib-50** (full 50):

| Metric | Required |
|---|---|
| `false_clean_count` | 0 |
| `dangerous_clean_miss_count` | 0 |
| `human_review_miss_count` | 0 |
| `primary_failure_accuracy` | ≥ 0.54 (current) |
| Case 032 (POST_RATIONALIZED_CITATION, identical dates) | must NOT change |
| Cases 039, 040, 041 (training) | primary becomes `STALE_RETRIEVAL` |

If **any** criterion fails → revert (delete the new method, revert the
analyzer wiring, delete the new tests). Document the failure in a
result file. No silent loosening of criteria.

## Out of scope

- The other 10 wrong Calib-50 cases (RETRIEVAL_DEPTH_LIMIT,
  RERANKER_FAILURE, CITATION_MISMATCH confusion, etc.) stay wrong for
  this step.
- No engine, decision-policy, or other-analyzer change.
- No golden label change.
- No threshold tuning of the existing StaleRetrievalAnalyzer paths.

## Provenance

- Calib-50 forensic snapshot: `/tmp/calib_v2.{json,md}`
  (`primary_failure_accuracy=0.54`).
- Heldout snapshot before fix: `reports/codex_session/heldout_after.json`
  (`primary_failure_accuracy=0.667`).
- Protected baseline before fix: GREEN at `(42, 46)`.

## Rationale: why this is genuinely useful, not test-driven

A relative-recency staleness check is one of the most common manual
inspection steps a RAG engineer runs. Any production-grade RAG pipeline
that stores documents with effective dates needs this signal. The check
generalises to:
- legal policy updates (effective_from / superseded_by),
- tax / regulatory rates (changes by tax year),
- product specs (model_year, version_release_date),
- security configurations (policy effective_date),
- HR / org records (role tenure dates).

The chunk-metadata keys we accept are the same ones standard RAG
frameworks already populate. An engineer who never touches GovRAG-Calib
benefits directly: their retrieved chunks already have these dates, and
our analyzer now actually reads them.
