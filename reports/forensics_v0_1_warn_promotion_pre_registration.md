# Pre-Registered Fix: Drop Warn-to-Primary Promotion in Engine

**Date pre-registered:** 2026-06-12
**Filed before any code change.** Code change will only be applied if
acceptance criteria measured on the locked heldout (`heldout_v0_1`, fingerprint
`58261ecc...c9b15c76`) hold.

## Forensic finding

Calib-50 root-cause-family accuracy: 20%. The dominant failure pattern, per
per-case forensics, is:

| Bucket | Wrong-family cases (of 17) | Root cause |
|---|---|---|
| `missing_analyzer` (or wrong specific failure type) | 9 | Expected `primary_failure` value is never emitted by any current analyzer, or is emitted by the right analyzer but with the wrong specific label. |
| `wrong_routing` (strong + weak) | 5 | The right `primary_failure` value IS emitted by some analyzer, but the engine picks a different winner. |
| **`over_classification`** | **3** | **Clean cases where every analyzer is at `warn` (none `fail`), and the engine promotes one `warn` to primary failure anyway.** |

The `over_classification` bucket has the clearest, smallest-scope, lowest-risk
fix. It is the only bucket we will touch in this step.

## The buggy code path

`src/raggov/engine.py:866-886`:

```python
if primary == FailureType.CLEAN:
    ranked_warnings = self._ranked_warn_failure_types(results, result_weights)
    if ranked_warnings:
        warning_failure = ranked_warnings[0][0]
        warning_result  = self._best_result_for_failure(
            results, result_weights, warning_failure, statuses={"warn"},
        )
        if warning_result is not None:
            primary = warning_failure
            primary_result = warning_result
            ...
            decision_trace["selection_reason"] = (
                "No fail-level policy winner was available; falling back to an "
                "allowed warn-level signal."
            )
```

The decision policy at `decision_policy.py:217` correctly returns `CLEAN` when
no analyzer has `status="fail"`. The engine then OVERRIDES it by elevating the
highest-ranked `warn`-status analyzer's `failure_type` to primary.

This means on a clean RAG output where multiple analyzers raise advisory
warnings — which is normal because warnings are not necessarily wrong, just
non-fatal — GovRAG reports a confident `primary_failure` that no analyzer
actually believed was a failure.

For a real engineer this is the single worst behavior of a diagnostic tool:
a confident wrong answer on a clean input. It also inflates the "I detected
something" surface area, making GovRAG look more capable than it is.

## Hypothesis

Removing (or weakening) the warn-to-primary promotion will:

1. Recover 3 Calib-50 clean_pass cases (011, 012, 013) → return `CLEAN`.
2. Not regress any `fail`-based diagnosis, because the buggy path only fires
   when the policy already returned `CLEAN`.
3. Improve Calib-50 `primary_failure_accuracy` from 0.48 to ~0.54 (+3/50).
4. Make `human_review_required` on those 3 cases return `False`, matching
   their golden labels.

## What we will measure (pre-registered)

On `heldout_v0_1` (15 family-balanced cases):

| Metric | Required after fix | Notes |
|---|---|---|
| `false_clean_count` | 0 | A clean case must not slip past a real failure. |
| `dangerous_clean_miss_count` | 0 | Security-relevant cases must still escalate. |
| `human_review_miss_count` | 0 | Don't lose escalation on cases that need it. |
| `security_stage_miss_count` | 0 | |
| 2 clean_pass cases in heldout (013, 015) | must return `CLEAN` | Already do today — must not change. |
| `primary_failure_accuracy` on heldout | ≥ 0.600 (current) | No regression on the locked slice. |
| `stage_accuracy` on heldout | ≥ 0.400 (current) | No regression. |
| 13 non-clean heldout cases | each case's primary_failure value must not change from its current value | Strong-form regression check: the fix only affects the CLEAN branch, so no non-clean case should move. |

On the **protected common benchmark** (full 46-case suite, both modes):

| Metric | Required |
|---|---|
| count | `(42, 46)` unchanged (with current acceptable-alternative for case 20) |
| `false_clean_count` | 0 |
| `false_security_count` | 0 |
| `false_incomplete_count` | 0 |
| Composition | unchanged from current pin-green state |

On **Calib-50** (full 50):

| Metric | Required |
|---|---|
| `false_clean_count` | 0 (must not regress) |
| `dangerous_clean_miss_count` | 0 (must not regress) |
| `human_review_miss_count` | 0 (must not regress) |
| `primary_failure_accuracy` | ≥ 0.48 (current) |
| Per-case: the 7 currently-correct clean_pass-or-low-impact cases | unchanged |

If **any** of the above fails, the change is reverted and we re-design.

## Out of scope for this step

- We will NOT add new analyzers (the 9 `missing_analyzer` cases stay wrong).
- We will NOT change the `wrong_routing` selection priority (the 5 cases stay wrong).
- We will NOT modify any analyzer threshold.
- We will NOT change golden labels.
- We will NOT replay the stashed routing edits.

This step targets only the engine's `warn`-promotion logic.

## Two implementation options to choose between

**Option A — Drop the warn-promotion entirely.**
`warn`-status analyzers stay as visible `analyzer_results` entries with a
`failure_type` set, but the diagnosis-level `primary_failure` stays `CLEAN`.
An engineer can still inspect each `warn` in the diagnosis output. The CLI
will show "no failure detected; N analyzers raised advisory warnings".

**Option B — Promote warn-only diagnoses to `LOW_CONFIDENCE` (an existing enum value)** instead of to the warn's specific failure type. This is the
safety-net version: the diagnosis surfaces "I am not confident in CLEAN" but
does not falsely name a specific failure family.

Both preserve the integrity guarantee. Option A is more honest; Option B is
more cautious. Decision will be made after seeing the heldout impact of each
on the same `heldout_v0_1` slice. If both pass acceptance criteria, prefer A
(simpler, more honest).

## Rationale for why this is genuinely useful, not a castle

This fix is not "adjusting a threshold to pass a test." It is removing an
existing piece of code that confabulates a primary failure when none exists.
The piece of code being removed has no principled justification: a `warn`
from one analyzer is not evidence that the failure family that analyzer
named is the *root cause* of any problem — it is at most evidence that the
analyzer is *concerned*.

A real engineer using GovRAG on a clean RAG output today gets a confident,
wrong primary_failure pointing at the wrong place. After this fix they get
either CLEAN (Option A) or LOW_CONFIDENCE (Option B) with the warning
signals still visible for inspection. Either is strictly more honest.

## Provenance

- Forensic data: `/tmp/calib_forensic.json` (Calib-50 native run, 2026-06-12).
- Locked heldout: `evals/govrag_calib/splits/heldout_v0_1.{json,jsonl}` with
  SHA256 `58261ecc57247f9fe80e114d126e914107c52bbe598acf851e4b616bc9b15c76`.
- Pin status before fix: GREEN at `(42, 46)` per
  `reports/baseline_pin_v0_1_alpha_public_migration.md`.
- Calib-50 status before fix: `primary_failure_accuracy=0.48`,
  `stage_accuracy=0.42`, safety counters all 0.
- Heldout status before fix: `primary_failure_accuracy=0.60`,
  `stage_accuracy=0.40`, safety counters all 0.
