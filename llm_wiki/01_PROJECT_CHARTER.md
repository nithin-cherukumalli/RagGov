# Project Charter

## Identity

GovRAG is a diagnosis-first RAG governance framework.

The project goal is to build diagnosis infrastructure for high-trust production RAG systems.
The current codebase does not yet represent a production-grade system.
The native diagnosis path is the primary product path.
External-enhanced mode is optional and must remain visibly degradable.

## What GovRAG Must Do

GovRAG must:

1. Diagnose failures, not just score outputs.
2. Attribute likely root cause across pipeline stages.
3. Expose evidence and uncertainty.
4. Detect security and adversarial risks.
5. Recommend actionable engineering fixes.
6. Avoid silent fallback.
7. Avoid fake research claims.
8. Become credible through evaluation and calibration.

## What GovRAG Is Not

GovRAG is not:

- a generic evaluator wrapper
- a leaderboard scorer
- a calibrated scientific instrument
- a research-faithful implementation of every paper named in the repo

## Current Architectural Reading

From the actual runtime path in `src/raggov/engine.py`, the system is organized around:

- evidence-producing substrate analyzers
- evidence rollups and cross-checks
- meta interpretation layers
- final policy selection

The project direction is diagnosis-first, not score-first.
The current implementation direction is also native-first, not external-first.

## Current Truth Statement

The current system is an advanced prototype.

Evidence for that statement in code:

- many analyzers explicitly declare `recommended_for_gating=False`
- many reports declare `uncalibrated`
- multiple analyzers include legacy heuristic fallback paths
- `mode == "calibrated"` is not implemented natively in `src/raggov/engine.py`

## Strategic Priority

Claim grounding is the highest-leverage substrate to harden first.

Reason:

- citation faithfulness depends on claim evidence
- sufficiency interpretation is strengthened by claim evidence
- retrieval diagnosis consumes claim evidence and other upstream reports
- NCV consumes upstream reports and falls back heuristically when they are absent
- Layer6 and A2P reinterpret upstream results rather than producing first-order evidence
- final diagnosis policy quality is bounded by substrate quality

Meta-layer improvements must not be prioritized before claim/evidence substrate hardening.
