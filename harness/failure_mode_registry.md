# GovRAG Failure Mode Registry

This registry is a harness reference for coding agents. It names common failure modes, likely owners, protected files, and safe first actions. It is not a benchmark label source and production logic must not depend on it.

| ID | Category | Likely owner | First diagnostic command |
| --- | --- | --- | --- |
| `retrieval_depth_or_scope_regression` | retrieval | retrieval evidence emission | `python scripts/evaluate_common_failures.py` |
| `grounding_claim_support_regression` | grounding | claim grounding analyzer | `PYTHONPATH=src:. pytest -q tests/analyzers/grounding tests/test_analyzers/test_grounding_integration.py` |
| `citation_evidence_regression` | citation | citation analyzer | `PYTHONPATH=src:. pytest -q tests/test_analyzers/test_citation_faithfulness_v0.py` |
| `sufficiency_context_regression` | sufficiency | sufficiency analyzer | `PYTHONPATH=src:. pytest -q tests/domain_agnostic/test_sufficiency_core.py` |
| `version_validity_staleness_regression` | version_validity | version validity analyzer | `PYTHONPATH=src:. pytest -q tests/domain_agnostic/test_version_validity_core.py` |
| `parser_chunking_contract_regression` | parser_chunking | parser/chunking pipeline | `PYTHONPATH=src:. pytest -q tests/stresslab/test_chunking.py tests/stresslab/test_ingest.py` |
| `answer_quality_metadata_regression` | answer_quality | answer quality analyzer | `PYTHONPATH=src:. pytest -q tests/test_analyzers/test_answer_quality_confidence_metadata.py` |
| `confidence_calibration_regression` | confidence | confidence analyzer | `PYTHONPATH=src:. pytest -q tests/test_analyzers/test_confidence.py tests/models` |
| `security_prompt_or_retrieval_anomaly_regression` | security | security analyzer | `PYTHONPATH=src:. pytest -q tests -k "security or anomaly"` |
| `decision_policy_selection_regression` | decision_policy | decision policy | `PYTHONPATH=src:. pytest -q tests/decision_policy tests/engine tests/test_engine.py` |
| `external_provider_runtime_degradation` | external_provider_runtime | external provider bridge | `PYTHONPATH=src:. pytest -q tests/evaluators tests/engine/test_external_signal_bridge.py` |
| `calibration_overclaim` | calibration | calibration readiness | `PYTHONPATH=src:. python scripts/launch_readiness.py` |
| `benchmark_integrity_damage` | benchmark_integrity | benchmark governance | `python scripts/harness_preflight.py --strict` |
| `workspace_integrity_risk` | workspace_integrity | coding agent workspace | `python scripts/workspace_audit.py` |
| `report_integrity_regression` | report_integrity | reporting harness | `python scripts/workspace_audit.py` |

## Safe Actions

- Preserve protected benchmark labels, expected stages, fix categories, thresholds, launch-readiness gates, and production gating.
- Prefer additive audit reports and targeted regression tests.
- Surface external-provider degradation and heuristic status instead of hiding failures.

## Unsafe Actions

- Editing golden labels to match current output.
- Enabling `production_gating_eligible`.
- Marking heuristic, proxy, or advisory signals as calibrated.
- Rewriting reports to hide Not Ready status, degraded external runtime, or known benchmark regressions.
