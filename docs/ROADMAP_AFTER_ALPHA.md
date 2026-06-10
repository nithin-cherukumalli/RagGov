# Roadmap After v0.1-alpha

## v0.1-rc Tasks

- Triage and repair full pytest failures without xfail/skip shortcuts.
- Keep the common benchmark at or above the protected `41/46` baseline while improving exact matches.
- Improve subtle suite behavior without changing golden labels to match current behavior.
- Preserve zero false-clean, false-security, and false-incomplete counts.
- Keep launch readiness blockers visible and classified as alpha, RC, or production blockers.
- Keep production gating disabled until calibrated evidence exists.

## Full Pytest Triage

Prioritize failures by user-facing risk:

- triplet verification flow and fallback behavior
- version-validity decision trace behavior
- answer-quality generation-stage attribution
- evidence-layer claim-type regressions
- pinpointing first-failing-node mismatches

Do not weaken tests, add skips, or change labels to make full pytest pass.

## Subtle Suite Improvement

The subtle suite should move from advisory/RC status toward release-candidate quality by:

- improving unsupported versus contradicted claim separation
- improving citation mismatch versus grounding mismatch separation
- improving ambiguous-query and low-confidence handling
- improving near-miss retrieval diagnosis
- preserving the rule that subtle failures must not return `CLEAN`

## Calibration Dataset Expansion

Production calibration requires:

- labeled diagnosis samples across major failure types
- labeled claim-grounding and citation-support examples
- labeled retrieval and sufficiency examples
- confidence interval computation
- explicit calibration artifacts
- documented sampling assumptions and limitations

Until this exists, `production_gating_eligible` must remain false.

## Answer-Quality And Confidence Metadata

Improve answer-quality diagnosis by:

- separating answer completeness from claim grounding
- preserving generation-stage candidates when grounding evidence is downstream
- making confidence metadata explicit and uncalibrated by default
- avoiding invented confidence scores
- ensuring every confidence claim has provenance

## External Provider Stabilization

External packages should remain optional extras. Stabilization work should:

- make provider runtime availability explicit
- surface missing package, missing model, missing API key, and network degradation reasons
- normalize external output into advisory signals
- prevent external scores from overwriting native diagnosis
- add integration coverage only where runtime dependencies are available

## Harness Hardening

The harness should continue to protect:

- native common benchmark pass count
- external-enhanced common benchmark pass count
- false-clean, false-security, and false-incomplete counts per mode
- production gating disabled state
- generated report visibility
- benchmark and golden label integrity

Future hardening should improve report classification without deleting or hiding generated artifacts.

## Production-Calibrated Tasks

Before production-calibrated status:

- gather enough labeled validation samples
- compute confidence intervals
- document calibration status per analyzer
- validate production-gating policy on held-out data
- prove provider degradation is visible
- run security and privacy review
- keep `recommended_for_gating` false until calibration evidence supports it

## Final Cleanup

Cleanup should happen only after alpha freeze decisions are recorded:

- classify generated reports as current, archived, or disposable
- archive or remove reports only with explicit approval
- keep protected report evidence visible until replaced by newer evidence
- avoid deleting tests, fixtures, labels, thresholds, or launch gates
