# Tasks 14/15/16 — instrumentation groundwork (deferred, not yet implemented)

**Date:** 2026-06-18. Root causes captured from read-only `diagnose()` traces so a focused next
pass can implement with prereg + revert-on-failure. Not implemented this session: each is a
decision-policy / profile change entangled with protected cases, and rushing it at the tail of a
long session would risk the protected baseline (the project's trust anchor).

## Task 14 — stale irrelevant source should not primary-fail
Test: `test_stale_irrelevant_source_does_not_primary_fail` (strict xfail).
Case: query "Who is the CEO?", answer cites fresh `doc-ceo` (2024, status active); `doc-lease`
(2010, "office lease terms", score 0.2) is an uncited, query-irrelevant distractor.

Trace findings (current engine):
- `StaleRetrievalAnalyzer._from_profile` fails STALE on **both** `doc-lease` AND `doc-ceo`,
  because `profile.stale_doc_ids` is built from **absolute age** (>180 days from 2026) — so even
  the freshest, cited, active source is "stale."
- The NCV profile already knows `chunk-lease` is `noisy_chunk_ids` and query-relevance is "1/2".

Why a single gate is insufficient:
- A relevance gate alone drops `doc-lease` but **keeps `doc-ceo`** (it is query-relevant), so STALE
  still fires. Passing the test also requires `doc-ceo` to not be flagged stale — i.e.
  relative-recency (Task 17-v3 logic) applied in the **profile's `stale_doc_ids` computation**
  (`evidence_profile.py`), not just the version-validity analyzer.

Risk: protected case `version_stale_not_cited_32` is an uncited-but-genuinely-stale case that flows
through the same stale paths. The separator is query-relevance + presence of a strictly newer
retrieved alternative. Implement as: stale doc drives primary only if (a) query-relevant (≥1
non-noisy chunk) AND (b) a strictly newer dated alternative was retrieved; the freshest/active doc
is never itself "stale." Pre-register against both this test and `version_stale_not_cited_32`.

## Task 15 — incomplete-answer stage attribution
Test: `...test_quality_incomplete_38_has_generation_stage_candidate_if_supported` (strict xfail).
Primary is correctly `UNSUPPORTED_CLAIM`; only `root_cause_stage` is wrong (`GROUNDING`, expected
`GENERATION`). Selected analyzer `ClaimGroundingAnalyzer`; reason `_suppress_citation_when_downstream_symptom`.
`AnswerQualityAnalyzer` has a GENERATION-stage candidate that does not win trace selection. Fix is
stage/analyzer attribution only — keep primary `UNSUPPORTED_CLAIM`. Lower risk; pure trace selection.

## Task 16 — case-41 specificity (CONTRADICTED_CLAIM)
Test: `...test_quality_ignores_context_41...` (strict xfail). Expects primary `CONTRADICTED_CLAIM`;
current `UNSUPPORTED_CLAIM`. **Same family as Task 22**, which this session proved is a native-mode
NO-GO: the `_require_explicit_contradiction` guard is load-bearing (disabling it regresses Calib
23→22 and creates false contradictions). Case 41's `ClaimGroundingAnalyzer` shows `total=1,
contradicted=1` — single-claim; check whether it carries a hard value/date conflict
(`_has_explicit_contradiction`). If not, Task 16 is bounded by the same native-mode limit as Task 22
and should wait for the optional LLM/NLI verifier. Do **not** weaken the guard to flip one case.

## Recommended next-session order
1. Task 15 (lowest risk: pure stage/analyzer trace attribution).
2. Task 14 (profile relative-recency + relevance gate; prereg against protected case 32).
3. Task 16 only if case 41 has hard contradiction evidence; otherwise defer with Task 22.
