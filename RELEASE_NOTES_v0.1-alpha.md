# GovRAG v0.1-alpha-clean Release Notes

Release name: GovRAG v0.1-alpha-clean

## Summary

GovRAG v0.1-alpha-clean is the first alpha freeze for the native RAG diagnosis path. It is intended for local evaluation, integration experiments, and continued hardening of diagnosis evidence, provenance, and release gates.

This release is not production-calibrated.

## What Is Included

- Native RAG failure diagnosis for grounding, citation, sufficiency, retrieval, version validity, parser/chunking, and security-related failures.
- Optional external-enhanced mode as an advisory signal path.
- Visible degradation metadata for missing or unavailable external providers.
- Launch readiness classification that separates alpha blockers from RC and production blockers.
- Minimal alpha release gate script at `scripts/check_v0_1_alpha_release.py`.
- Alpha release documentation at `docs/V0_1_ALPHA_RELEASE.md`.
- Post-alpha roadmap at `docs/ROADMAP_AFTER_ALPHA.md`.

## Protected Benchmark Status

- common benchmark native: `41/46`
- common benchmark external-enhanced: `41/46`
- `false_clean_count = 0`
- `false_security_count = 0`
- `false_incomplete_count = 0`
- `production_gating_eligible = false`
- launch readiness status: `v0.1-alpha-clean Ready`
- production readiness status: `Not Ready`

The common benchmark is not expected to be `46/46` for this alpha freeze.

## Safety Gates

The alpha safety gate requires:

- native common benchmark at or above `41/46`
- external-enhanced common benchmark at or above `41/46`
- zero false-clean, false-security, and false-incomplete counts
- production gating disabled
- launch readiness reporting `v0.1-alpha-clean Ready`
- degraded external providers remaining visible and advisory

## Known Limitations

- Full pytest still fails and is an RC blocker.
- The subtle suite remains advisory/RC-level.
- External providers may be degraded or unavailable.
- Calibration is incomplete.
- Confidence metadata is not production calibrated.
- External provider output is not source of truth.

## RC Blockers

- Full pytest failure triage and repair.
- Common benchmark improvement beyond the protected alpha baseline.
- Subtle suite improvement.
- Answer-quality and confidence metadata hardening.
- Pinpointing mismatch repair.
- Test warning and mark registration cleanup.

## Production Blockers

- Labeled calibration dataset expansion.
- Confidence intervals and calibration artifacts.
- Production-gating policy validation.
- External provider runtime stabilization.
- Security and privacy review.
- Evidence-backed calibration before any production gating.

## Validation

Run:

```bash
python scripts/evaluate_common_failures.py --suite common
python scripts/evaluate_common_failures.py --suite common --mode external-enhanced
python scripts/launch_readiness.py
python scripts/check_v0_1_alpha_release.py
pytest -q tests/harness
pytest -q tests/decision_policy
pytest -q tests/test_analyzers/test_grounding.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --collect-only -q tests/stresslab
```

Expected alpha results:

- both common benchmark modes report `41/46`
- false counts remain `0/0/0`
- `production_gating_eligible` remains `false`
- launch readiness reports `v0.1-alpha-clean Ready`
- production remains `Not Ready`
