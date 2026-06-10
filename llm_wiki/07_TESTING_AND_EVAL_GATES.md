# Testing And Eval Gates

## Required Mindset

GovRAG becomes credible through evaluation and calibration, not by stronger prose.

## Existing Test / Eval Surfaces Visible In Repo

- `tests/`
- `stresslab/`
- CLI commands in `src/raggov/cli.py`:
  - `stresslab-suite`
  - `stresslab-diagnosis`
  - `providers doctor`
  - `calibrate`

## Minimum PR Gates

Each PR should define:

1. Unit tests to add or update.
2. Relevant stresslab or evaluation command.
3. Expected diagnosis behavior before and after change.
4. Whether fallback metadata changed.

## Native Phase 1 Gate

Before credibility work, native mode must satisfy a concrete common-suite gate:

- citation category: `5/5`
- grounding category: `4+/5`
- sufficiency category: `3+/5`
- zero false negatives for `CITATION_MISMATCH`
- zero false negatives for `CONTRADICTED_CLAIM`

This gate now belongs in launch-readiness logic.
Do not replace it with a vague overall pass-rate story.

## Calibration Honesty

Important code truth:

- `mode == "calibrated"` is not natively available in `src/raggov/engine.py`
- many reports still mark `recommended_for_gating=False`
- many analyzers still mark `uncalibrated`

Therefore:

- do not treat passing tests as calibration
- do not treat benchmark fit as scientific validation
- do not convert heuristic scores into probabilities in docs or code

## What To Prefer Testing Next

1. claim grounding claim-level reliability
2. citation faithfulness consistency against grounding records
3. retrieval diagnosis rollup correctness under mixed evidence
4. NCV fallback surfacing
5. decision policy suppression and override behavior
