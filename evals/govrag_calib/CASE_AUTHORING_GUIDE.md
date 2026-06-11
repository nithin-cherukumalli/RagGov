# GovRAG-Calib Case Authoring Guide

GovRAG-Calib records are JSONL calibration scaffolding for diagnosis evaluation. They are not production calibration evidence, and they must not be used to claim calibrated confidence, confidence intervals, or production gating eligibility.

## Required Fields

Every record must include:

- `case_id`
- `domain`
- `source_suite`
- `source_case_id`
- `query`
- `retrieved_chunks`
- `answer`
- `claims`
- `citations`
- `expected_primary_failure`
- `expected_stage`
- `expected_first_failing_node`
- `expected_root_cause`
- `expected_fix_category`
- `expected_affected_claim_ids`
- `expected_affected_doc_ids`
- `expected_human_review_required`
- `acceptable_alternative_failures`
- `failure_family`
- `difficulty`
- `adversarial`
- `security_relevant`
- `metadata_requirements`
- `labeler`
- `label_status`
- `label_confidence`
- `notes`
- `calibration_split`
- `calibration_status`
- `provenance`

Use `calibration_split: "unset"` until split policy is explicitly applied. Do not use `heldout_locked` before a formal heldout lock step.

## Failure Families

- `clean_pass`: answer is supported, citations are valid when present, no diagnosis should fire.
- `retrieval`: retrieval depth, scope, ranking, embedding, or noisy retrieval is the primary issue.
- `grounding`: answer makes unsupported or contradicted claims against available retrieved context.
- `citation`: cited document or chunk does not support the answer, is phantom, or is post-rationalized.
- `sufficiency`: retrieved context is not enough to answer safely or completely.
- `version_validity`: answer relies on stale, superseded, expired, or missing freshness/version metadata.
- `security_privacy`: prompt injection, privacy disclosure, suspicious chunk, or security-relevant poisoning.
- `answer_quality`: generation completeness, ambiguity handling, low-confidence behavior, or answer composition issue.

## Clean Cases

Clean cases must be genuinely supported by retrieved chunks. Include enough context to support every claim and citation. Do not hide a failure in a clean case. For clean cases:

- use `expected_primary_failure: "CLEAN"`
- use `expected_stage: "UNKNOWN"`
- keep `expected_affected_claim_ids` and `expected_affected_doc_ids` empty unless a local workflow intentionally tracks supported items
- set `expected_human_review_required: false`
- avoid noisy distractor chunks that can reasonably be interpreted as retrieval anomalies

## Retrieval Cases

Retrieval cases should make the retrieval fault visible in the record:

- missing required document, scope, jurisdiction, or appendix
- relevant chunk buried by irrelevant chunks
- old/current version ranking issue
- embedding drift or reranker failure

Record the relevant and missing/incorrect document IDs in notes or provenance. Use acceptable alternatives when the same evidence could reasonably be diagnosed as grounding or sufficiency.

## Grounding Cases

Grounding cases require a concrete answer claim and retrieved evidence that does not support it or contradicts it. Include affected claim IDs and affected document IDs. Do not label a case grounding if the real issue is missing context and no available chunk could settle the claim.

## Citation Cases

Citation cases require explicit `citations` unless the case is intentionally testing missing citations, in which case document that in `notes`. The cited `doc_id` and `chunk_id` should exist unless the expected label is a phantom citation. Explain whether the issue is wrong document, wrong chunk, post-rationalized citation, or missing required citation.

## Sufficiency Cases

Sufficiency cases should show why the available context is inadequate:

- missing appendix, table, exception, approval state, safety data, or eligibility variable
- context says an answer depends on unavailable information
- answer should abstain or request more evidence

Security-relevant sufficiency cases should set `security_relevant: true` and `expected_human_review_required: true`, but they do not automatically become `SECURITY` stage.

## Version-Validity Cases

Version-validity cases need explicit metadata requirements such as `effective_date`, `valid_until`, `version`, `as_of`, or lifecycle status. Include old and current evidence when available. Do not rely on vague "current" wording without metadata.

## Security/Privacy Cases

Security/privacy cases should include visible security evidence:

- prompt injection or instruction override
- sensitive personal, financial, medical, or credential data
- suspicious chunk payload or answer steering
- privacy or security review requirement

Do not label a case security just because retrieval is noisy. Security labels require security/privacy evidence, not only ranking or retrieval anomaly evidence.

## Answer-Quality Cases

Answer-quality cases target generation behavior:

- incomplete list despite complete context
- ignores explicit exception
- overconfident answer to ambiguous context
- fails to state uncertainty

Use `GENERATION` stage only when retrieval and evidence are sufficient but the answer composition is wrong.

## Avoid Near-Duplicates

Do not add cases that differ only by entity names or small wording changes. New cases should vary at least one substantive dimension:

- domain
- failure family
- evidence shape
- metadata requirement
- citation pattern
- security or human-review risk
- retrieval topology

## Provenance

Every record must explain where it came from:

- `source`: common benchmark, stresslab, manual, synthetic, public dataset mapping, or other source
- `created_at`: creation date
- `derived_from`: original case or scenario
- `transformation_notes`: what changed and whether the case is synthetic/manual

Never present synthetic or seed labels as calibrated truth.

## No Production Calibration Claim

GovRAG-Calib is not production-calibrated until it has enough reviewed/adjudicated cases, a locked heldout split, confidence interval computation, and documented calibration results. Until then:

- `production_gating_eligible` must remain false
- `calibration_status` in evaluator output must remain `not_calibrated`
- confidence intervals must not be claimed
