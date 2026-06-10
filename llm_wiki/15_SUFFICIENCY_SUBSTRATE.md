# Sufficiency Substrate

Core files:

- `src/raggov/analyzers/sufficiency/sufficiency.py`
- `src/raggov/analyzers/sufficiency/claim_aware.py`
- `src/raggov/models/diagnosis.py` for `SufficiencyResult`

## Current Honest Description

Sufficiency is an advisory substrate, not a production generation gate.

The code supports:

- requirement-aware sufficiency logic
- deterministic term-coverage fallback
- claim-aware sufficiency using prior results

## Important Truths

- thresholds are not globally calibrated
- some calibration language in code is still preliminary
- requirement extraction failure can trigger heuristic fallback

## Rule

Sufficiency must not silently degrade from structured reasoning to term overlap.
