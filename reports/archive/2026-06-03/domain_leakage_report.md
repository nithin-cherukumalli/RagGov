# Domain Leakage Report

Date: 2026-05-13

## Final Verdict

GovRAG core is substantially more domain-agnostic after this pass, but the project should not yet claim full general OSS domain-agnostic maturity.

The default core grounding path no longer requires government or policy regexes. Generic mode passes the new non-government benchmark with false CLEAN = 0 and no government-specific evidence strings. Government-specific logic remains available, but it is now explicitly named or non-default.

## Government-Specific Logic That Remains

- `GovernmentPolicyTripletExtractorV0` in `src/raggov/analyzers/grounding/triplets.py`.
- Backward-compatible alias `RuleBasedPolicyTripletExtractorV0` for existing government benchmark tests and callers.
- `extract_government_policy_value_mentions` in `src/raggov/analyzers/grounding/value_extraction.py`.
- Government-focused stresslab ingestion in `stresslab/ingest/parse_go_order.py`.
- Government-heavy PDFs and fixtures under `tests/Data`, `stresslab/cases/fixtures`, and `stresslab/cases/golden`.
- Some legacy test names and comments still mention policy/GO cases because they validate explicit government behavior.

## Government Logic In Default Core Path

No government/policy regex is required in default core grounding after this refactor.

Default behavior now uses:

- Generic claim-worthiness signals.
- Generic claim type labels.
- Generic value extraction.
- Generic triplet extraction when triplet extraction is enabled.
- Generic sufficiency requirement types.
- Generic source lifecycle metadata.

Known residual wording risk:

- `src/raggov/analyzers/grounding/verifiers.py` still has an eligibility-specific rationale for overgeneralizing beyond an explicit `only` constraint.
- `src/raggov/analyzers/confidence/semantic_entropy.py` still has a policy/returns-oriented ambiguity heuristic.

These are not required for default grounding extraction, but they should be generalized before a strong OSS domain-agnostic claim.

## Non-Government Domains Tested

- Software documentation.
- Healthcare guidelines.
- Finance and insurance disclosures.
- Product manuals and support content.
- Scientific paper summaries.
- General enterprise knowledge-base content.

The benchmark includes retrieval miss, retrieval noise, unsupported claim, contradicted claim, citation mismatch, stale/deprecated source, insufficient context, incomplete answer, and weak grounding cases for each domain.

## Benchmark Results

Non-government benchmark:

- Cases: 54
- Pass rate: 100%
- False CLEAN: 0
- Government logic used in evidence: 0
- Output files: `reports/domain_agnostic_benchmark_report.json`, `reports/domain_agnostic_benchmark_report.md`

Government/common benchmark:

- Command: `python scripts/evaluate_common_failures.py --suite common --mode external-enhanced`
- Cases: 46
- Passed: 32
- Pass rate: 69.6%
- False CLEAN: 0 for `external-enhanced`
- False SECURITY: 0 for `external-enhanced`
- Output file: `reports/common_failure_coverage_matrix.md`

## Blocks To Full General OSS Claim

- The non-government benchmark is deterministic and synthetic. It is useful as a regression guard but not enough for a calibrated public claim.
- The common benchmark still has failures outside the core leakage work.
- Confidence ambiguity and one grounding rationale still contain domain-flavored wording.
- There is no calibrated cross-domain gold set with independent labels.
- Generic entity/scope extraction is still heuristic and lexical.
- External evaluator integrations remain optional and dependency-dependent.
