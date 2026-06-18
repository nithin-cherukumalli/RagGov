# Task 16 pre-registration — case-41 permission-contradiction specificity

**Date:** 2026-06-18 · written BEFORE code.

## Distinction from Task 22 (why this one is safe)
Task 22 was a native-mode NO-GO: those rows had matching values / no real conflict, so promoting
them regressed Calib and invented false contradictions. Case 41 is the opposite — a genuine,
human-obvious contradiction: context "Policy: No blue shirts." vs answer "you can wear anything you
want." It fails only because `_claim_has_textual_contradiction`'s permission vocabulary is too
narrow (recognizes "allowed/permitted/prohibited" but not "can wear anything" vs "No blue shirts").

## Change (one, narrow)
Extend `_claim_has_textual_contradiction` (`decision_policy_support.py`) with a third disjunct:
the claim asserts BROAD/unrestricted permission ("wear/do/use … anything|whatever|any", "no
restrictions/rules/dress code", "anything you want") AND the evidence states an explicit
RESTRICTION ("no <X>", "not allowed", "prohibited", "must/may not", "only … allowed", "required
to"). This only matters for claims the verifier already labeled `contradicted`, so blast radius is
tiny.

## Read-only precision check (BEFORE)
Predicate fires on case 41 = True; fires on **0 of 145** induced probe rows. Confirms it will not
broadly promote contradictions.

## Hard acceptance criteria
1. Protected baseline stays **43/46** (current effective). *(revert)*
2. Calib scored primary stays **≥ 23/45**. *(revert)*
3. Probe overall stays **≥ 80/145**; no new CLEAN→CONTRADICTED or UNSUPPORTED→CONTRADICTED false
   positives on probe. *(revert)*
4. `tests/test_analyzers` + `tests/decision_policy` green (modulo known pre-existing stale fail). *(revert)*

**Success:** `test_quality_ignores_context_41_has_generation_stage_candidate_if_supported` flips
xfail→pass (primary `CONTRADICTED_CLAIM`).

If any gate fails → revert, keep prereg as record (this would confirm even precise broadening is
unsafe and Task 16 defers with Task 22 to the optional NLI path).
