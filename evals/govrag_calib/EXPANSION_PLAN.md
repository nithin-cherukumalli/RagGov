# GovRAG-Calib Expansion Plan

This plan prepares future Calib-75, Calib-100, and Calib-150 work. It does not add cases, create splits, lock heldout data, compute confidence intervals, or enable production gating.

## Current State

- Current dataset size: 50 cases
- Current split: all cases remain `unset`
- Current calibration status: seed/review scaffolding only
- Production gating: false
- Heldout split: not created

## Target Distribution

Final GovRAG-Calib-150 target:

| Failure family | Target |
| --- | ---: |
| `clean_pass` | 15 |
| `retrieval` | 25 |
| `grounding` | 25 |
| `citation` | 20 |
| `sufficiency` | 15 |
| `version_validity` | 15 |
| `security_privacy` | 20 |
| `answer_quality` | 15 |

## Calib-75 Target

Calib-75 should add 25 cases while preserving family balance and avoiding near-duplicates.

Suggested minimum counts by Calib-75:

- `clean_pass`: 8
- `retrieval`: 12
- `grounding`: 12
- `citation`: 10
- `sufficiency`: 8
- `version_validity`: 8
- `security_privacy`: 10
- `answer_quality`: 7

Before adding cases, resolve known Calib-50 safety/human-review semantics and label ambiguity.

## Calib-100 Target

Calib-100 should add another 25 cases and increase coverage of:

- multi-document retrieval
- citation-specific failures
- temporal/version metadata
- safety-critical sufficiency
- security/privacy evidence
- answer-quality generation-stage cases

Suggested minimum counts by Calib-100:

- `clean_pass`: 10
- `retrieval`: 17
- `grounding`: 17
- `citation`: 13
- `sufficiency`: 10
- `version_validity`: 10
- `security_privacy`: 13
- `answer_quality`: 10

## Calib-150 Target

Calib-150 completes the planned target distribution. Do not treat the dataset as production calibration merely because it reaches 150 cases. It still requires adjudication, split assignment, heldout locking, and statistical reporting.

## Review And Adjudication Workflow

1. Draft cases as `label_status: seed` and `calibration_status: seed` or `reviewed`.
2. Validate JSONL structure and evidence references.
3. Run the evaluator and inspect evidence diagnostics.
4. Review false clean, dangerous clean miss, human-review miss, and security-stage miss cases.
5. Add acceptable alternatives only when record evidence supports them.
6. Downgrade uncertain reviewed cases to `seed`.
7. Promote to `reviewed` only after label evidence is clear.
8. Promote to `adjudicated` only after a second pass resolves ambiguity.

## Split Policy Timing

Do not assign train/calibration/heldout splits during Calib-75 or early Calib-100 expansion unless the team explicitly starts split planning. Keep `calibration_split: "unset"` while labels are still being repaired and adjudicated.

## Heldout Lock Policy

Heldout data should be locked only after:

- enough cases exist across all families
- labels are reviewed/adjudicated
- near-duplicates are removed
- evaluator semantics are stable
- cases are not being edited to chase metric results

Once a case is `heldout_locked`, it should not be edited except through a documented human review process.

## Why Heldout Is Not Created Yet

The current dataset is still seed/review scaffolding. It has unresolved human-review semantics, immature label adjudication, and no statistical calibration workflow. Creating a heldout split now would lock unstable labels and make future calibration less trustworthy.
