# GovRAG v0.1-alpha Release

## What v0.1-alpha Is

`v0.1-alpha` is the first honest release-freeze target for GovRAG's native RAG diagnosis path. It is an alpha-quality research and engineering checkpoint, not a production-calibrated product release.

This release is intended for local evaluation, integration experiments, and continued hardening of claim grounding, citation support, retrieval diagnosis, sufficiency, security, and provenance metadata.

## What It Can Diagnose Today

GovRAG can currently surface and explain these failure families in native mode:

- claim grounding failures such as unsupported and contradicted claims
- citation mismatch and citation support failures
- stale or lifecycle-invalid retrieval sources when metadata is available
- insufficiency and missing-context symptoms
- retrieval anomalies, retrieval misses, noisy retrieval, and scope issues
- parser/chunking/metadata warnings when parser profiles are available or missing
- prompt injection, suspicious chunks, and privacy/security risks
- optional external evaluator degradation without treating external outputs as source of truth

Native mode remains primary. External providers are advisory signal enhancers only.

## Protected Benchmark Status

Protected alpha baseline:

- common benchmark native: `41/46`
- common benchmark external-enhanced: `41/46`
- `false_clean_count = 0`
- `false_security_count = 0`
- `false_incomplete_count = 0`
- citation: `5/5`
- grounding: `7/7`
- sufficiency: `5/5`
- version validity: `5/5`
- `production_gating_eligible = false`
- calibration status: `not_production_calibrated`

The common benchmark is not 100%. The current alpha target protects the `41/46` baseline and zero false-clean/security/incomplete counters.

## Alpha Gates

The v0.1-alpha gate is satisfied when:

- native common benchmark remains at least `41/46`
- external-enhanced common benchmark remains at least `41/46`
- `false_clean_count == 0`
- `false_security_count == 0`
- `false_incomplete_count == 0`
- advisory external signals do not become primary failures alone
- retrieval anomalies do not become security failures without security evidence
- degraded or unavailable providers expose visible reasons
- calibrated confidence is absent unless a calibration artifact exists
- `production_gating_eligible == false`
- launch readiness reports `v0.1-alpha-clean Ready`

## How To Run Launch Readiness

```bash
python scripts/launch_readiness.py
```

Expected alpha status:

```text
Status: v0.1-alpha-clean Ready
```

Production status must remain:

```text
production_readiness_status: Not Ready
```

## How To Run The Common Benchmark

```bash
python scripts/evaluate_common_failures.py --suite common
python scripts/evaluate_common_failures.py --suite common --mode external-enhanced
```

Expected protected alpha result for both commands:

```text
Benchmark completed: 41/46 passed
```

## Production Gating Interpretation

`production_gating_eligible = false` is intentional for v0.1-alpha.

It means GovRAG does not yet have enough labeled calibration data, confidence intervals, and validated gating evidence to make production blocking decisions. This alpha may support investigation and diagnosis workflows, but it must not be presented as production-calibrated or used as an automated production gate.

## Known Limitations

- Full pytest still fails and remains an RC blocker.
- The subtle benchmark suite remains advisory/RC-level and is not an alpha blocker.
- External providers such as RAGAS, DeepEval, RefChecker, and RAGChecker may be degraded or unavailable.
- External-enhanced mode must surface degradation instead of treating external output as source of truth.
- Calibration is incomplete.
- Confidence claims remain uncalibrated unless explicitly backed by a calibration artifact.
- Several analyzers remain `heuristic_baseline` or `practical_approximation`.
- Semantic Entropy, A2P, RefChecker, RAGChecker, and conformal prediction must not be claimed as research-faithful unless the full mechanism is implemented.

## RC Blockers

- Full pytest failure triage and repair.
- Common benchmark exact-match improvement beyond protected alpha baseline.
- Subtle suite improvement.
- Answer-quality and confidence metadata cleanup.
- Remaining pinpointing mismatches.
- Mark registration and test warning cleanup.

## Production Blockers

- Labeled calibration dataset expansion.
- Confidence intervals and production calibration artifacts.
- External provider runtime stabilization.
- Production-gating policy validation.
- Evidence-backed calibration for any blocking decision.
- Security and privacy review for production deployment.

## Degraded External Providers

External provider degradation is allowed for v0.1-alpha only when visible in metadata and reports. Degraded providers must not silently pass and must not become native truth.

Current degraded/advisory areas include:

- `deepeval`
- `ragas`
- `refchecker_claim`
- `refchecker_citation`
- `ragchecker`

## Calibration Status

v0.1-alpha is not production calibrated.

No production-calibrated confidence, production blocking eligibility, or calibrated gating claim should be made from this release.
