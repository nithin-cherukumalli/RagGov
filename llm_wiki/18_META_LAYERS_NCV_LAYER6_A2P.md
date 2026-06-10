# Meta Layers: NCV, Layer6, A2P

## Rule

Meta layers must not invent evidence.
They may only interpret, aggregate, or explain evidence produced by core substrates.

## NCV

File:

- `src/raggov/analyzers/verification/ncv.py`

Current reading:

- practical architecture upgrade
- evidence aggregation when reports exist
- explicit heuristic fallback when reports do not
- not research-faithful NCV

## Layer6

File:

- `src/raggov/analyzers/taxonomy_classifier/layer6.py`

Current reading:

- taxonomy remapper over prior analyzer outputs
- uses score bands and rule heuristics
- should not be treated as authoritative root-cause evidence

## A2P

File:

- `src/raggov/analyzers/attribution/a2p.py`

Current reading:

- architecture for attribution and fix recommendation
- can be evidence-backed in claim-level modes
- still contains deterministic and legacy fallback behavior
- should be treated as experimental interpretation, not first-order evidence

## Practical Consequence

If claim grounding or retrieval evidence is weak, these layers can only produce weakly justified explanations.
They must not be used to compensate for weak citation or grounding substrate quality.
