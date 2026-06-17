# Red-Test Triage — Task 10

**Date:** 2026-06-17
**Baseline:** e759a12 (post Tasks 1, 2; Tasks 3/4/5 reverted)
**Scope:** `tests/test_analyzers/` — the documented "20 pre-existing red tests" (Phase C).

## Method

Ran `pytest -q tests/test_analyzers` against `e759a12` + the Tasks 6–9 / docs
commits (none of which touch analyzer or engine routing, so the failing set is
unchanged from the documented baseline). Found **exactly 20 failures**, matching
the inventory assumption (within ±5 tolerance). Each row below is classified as
one of `stale_test`, `dead_feature`, or `real_bug`.

Production code was **not** modified during this task. `real_bug` rows are filed
as follow-up Tasks 14–16 in `NEXT_TASKS.md`; they are **not** fixed here.

## Triage table

| test_node_id | classification | rationale | follow_up_action |
|---|---|---|---|
| test_evidence_layer.py::test_detect_claim_type_go_number | stale_test | Production claim-type taxonomy was redesigned to a defined enum (`numeric/temporal/policy_rule/eligibility/obligation/entity_attribute/...`); test still asserts the old label `general_factual`. `detect_claim_type` is deterministic → `entity_attribute`. | Mechanical: expected `general_factual` → `entity_attribute`. |
| test_evidence_layer.py::test_detect_claim_type_go_number_ms_variant | stale_test | Same taxonomy rename; deterministic output `entity_attribute`. | Mechanical: `general_factual` → `entity_attribute`. |
| test_evidence_layer.py::test_detect_claim_type_numeric_percentage | stale_test | Old label `value_assertion` renamed; deterministic output `numeric`. | Mechanical: `value_assertion` → `numeric`. |
| test_evidence_layer.py::test_detect_claim_type_numeric_currency | stale_test | Same; deterministic output `numeric`. | Mechanical: `value_assertion` → `numeric`. |
| test_evidence_layer.py::test_detect_claim_type_date_or_deadline_month | stale_test | Old label `date_time_assertion` renamed; deterministic output `temporal`. | Mechanical: `date_time_assertion` → `temporal`. |
| test_evidence_layer.py::test_detect_claim_type_date_or_deadline_keyword | stale_test | Same; deterministic output `temporal`. | Mechanical: `date_time_assertion` → `temporal`. |
| test_evidence_layer.py::test_detect_claim_type_eligibility | stale_test | Old umbrella `requirement_or_condition` split into `eligibility`/`obligation`; eligibility prose → `eligibility`. | Mechanical: `requirement_or_condition` → `eligibility`. |
| test_evidence_layer.py::test_detect_claim_type_policy_rule | stale_test | Same split; "must submit" obligation prose → `obligation`. | Mechanical: `requirement_or_condition` → `obligation`. |
| test_evidence_layer.py::test_detect_claim_type_general_factual | stale_test | `general_factual` renamed to `entity_attribute`. | Mechanical: `general_factual` → `entity_attribute`. |
| test_evidence_layer.py::test_detect_claim_type_go_takes_priority_over_numeric | stale_test | Priority preserved but label renamed; deterministic output `numeric`. | Mechanical: `value_assertion` → `numeric`. |
| test_evidence_layer.py::test_claim_evidence_builder_sets_claim_type | stale_test | Builder propagates the new taxonomy; deterministic output `numeric`. | Mechanical: `value_assertion` → `numeric`. |
| test_triplet_verification.py::test_aggregation_all_entailed | stale_test | `VerificationResult` gained a required `support_label` field; test uses the old constructor signature. Assertions read `label`/`raw_score` only, so `support_label` is purely a constructor-arg sync. | Mechanical: add `support_label` mirroring `label`. |
| test_triplet_verification.py::test_aggregation_one_contradicted | stale_test | Same constructor-signature drift. | Mechanical: add `support_label`. |
| test_triplet_verification.py::test_aggregation_one_unsupported | stale_test | Same. | Mechanical: add `support_label`. |
| test_triplet_verification.py::test_triplet_verification_flow | stale_test | Same; `VerificationResult` built in a mock return value. | Mechanical: add `support_label`. |
| test_triplet_verification.py::test_triplet_verification_fallback_on_extraction_failure | stale_test | Same. | Mechanical: add `support_label`. |
| test_version_validity_pipeline.py::test_version_validity_decision_trace_explains_downstream_claim_failure | stale_test | `selection_reason` wording was reworded ("downstream of invalid source lifecycle evidence" → "make downstream claim-level symptoms secondary"). No behavior change, just message text. | Mechanical: update expected substring. |
| test_version_validity_pipeline.py::test_stale_irrelevant_source_does_not_primary_fail | real_bug | Product invariant: a stale source that is *irrelevant* to the answer should not become the primary failure. Engine now returns `STALE_RETRIEVAL` for an irrelevant stale lease doc while the answer cites the fresh CEO doc. Likely an over-broad STALE_RETRIEVAL promotion (interacts with Task 2). Not a string drift. | **Task 14** — do not weaken test, do not change behavior here. |
| test_answer_quality_confidence_metadata.py::test_quality_incomplete_38_has_generation_stage_candidate_if_supported | real_bug | Engine agrees primary is `UNSUPPORTED_CLAIM` but attributes `root_cause_stage=GROUNDING`; an incomplete-answer case should attribute to `GENERATION`. Stage-attribution gap, golden-aligned. | **Task 15** — file follow-up; no change here. |
| test_answer_quality_confidence_metadata.py::test_quality_ignores_context_41_has_generation_stage_candidate_if_supported | real_bug | Golden expects `CONTRADICTED_CLAIM` (answer ignores/contradicts context); engine routes to less-specific `UNSUPPORTED_CLAIM`. Specificity-rank gap (sibling of the v2 routing work). | **Task 16** — file follow-up; no change here. |

## Summary

- **stale_test:** 17 (11 evidence-layer taxonomy rename, 5 triplet constructor-signature sync, 1 version-validity message wording) — fixed mechanically in commit `test(cleanup)`.
- **dead_feature:** 0 — no removed functionality; nothing deleted.
- **real_bug:** 3 — filed as Tasks 14, 15, 16 in `NEXT_TASKS.md`. Production untouched.

## Residual after cleanup

After the mechanical stale-test updates, `tests/test_analyzers/` has **3 residual
failures**, each pointing to a filed Task (14/15/16). These are intentional,
documented residuals — production routing/stage behavior, deferred to the v2 and
follow-up routing work, not test bugs.

## Guardrails (verified)

- No production code modified in this task.
- Protected baseline re-checked after each commit: **41/46 GREEN** (unchanged).
- Calib-50 and Heldout primary numbers cannot move (no engine/analyzer change).
- Failure count = 20, within ±5 of the documented inventory → assumption holds.
