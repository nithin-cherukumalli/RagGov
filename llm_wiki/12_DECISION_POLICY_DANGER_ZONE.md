# Decision Policy Danger Zone

## Primary Warning

`src/raggov/decision_policy.py` is a danger zone.

It must not be casually modified.
It must not be used to patch weak analyzer signals through more overrides.
It must not grow through benchmark-specific `if/else` rules.

## Why It Is Dangerous

This file:

- classifies evidence tiers
- builds ranked decision candidates
- suppresses advisory and some meta candidates
- applies special-case winner overrides
- influences final root-cause selection more than any other single file after analyzers run

Small policy edits can create large behavior changes while hiding the real problem: weak substrate evidence.

## Allowed Reason To Change It

Only change this file when:

- substrate evidence is already trustworthy enough
- the policy bug is clearly isolated
- the change is justified as policy, not analyzer repair
- a plan PR exists first

## Required Review Context Before Any Change

- `src/raggov/engine.py`
- `src/raggov/models/diagnosis.py`
- `src/raggov/analyzers/verification/ncv.py`
- `src/raggov/analyzers/taxonomy_classifier/layer6.py`
- `src/raggov/analyzers/attribution/a2p.py`

## Preferred Alternative

If diagnosis quality is weak, prefer fixing the substrate file that generated the weak or ambiguous evidence.
