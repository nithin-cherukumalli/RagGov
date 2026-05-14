# Domain-Agnostic Benchmark Report

- Cases: 54
- Pass rate: 100%
- False CLEAN: 0
- Government logic used: 0
- Domains: enterprise_kb, finance_insurance, healthcare_guideline, product_manual_support, scientific_paper, software_docs

| Case | Domain | Expected | Actual | Passed |
| --- | --- | --- | --- | --- |
| software_docs_retrieval_miss | software_docs | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| software_docs_retrieval_noise | software_docs | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| software_docs_unsupported_claim | software_docs | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| software_docs_contradicted_claim | software_docs | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| software_docs_citation_mismatch | software_docs | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| software_docs_stale_deprecated_source | software_docs | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| software_docs_insufficient_context | software_docs | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| software_docs_incomplete_answer | software_docs | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| software_docs_weak_grounding | software_docs | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| healthcare_guideline_retrieval_miss | healthcare_guideline | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| healthcare_guideline_retrieval_noise | healthcare_guideline | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| healthcare_guideline_unsupported_claim | healthcare_guideline | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| healthcare_guideline_contradicted_claim | healthcare_guideline | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| healthcare_guideline_citation_mismatch | healthcare_guideline | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| healthcare_guideline_stale_deprecated_source | healthcare_guideline | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| healthcare_guideline_insufficient_context | healthcare_guideline | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| healthcare_guideline_incomplete_answer | healthcare_guideline | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| healthcare_guideline_weak_grounding | healthcare_guideline | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| finance_insurance_retrieval_miss | finance_insurance | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| finance_insurance_retrieval_noise | finance_insurance | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| finance_insurance_unsupported_claim | finance_insurance | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| finance_insurance_contradicted_claim | finance_insurance | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| finance_insurance_citation_mismatch | finance_insurance | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| finance_insurance_stale_deprecated_source | finance_insurance | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| finance_insurance_insufficient_context | finance_insurance | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| finance_insurance_incomplete_answer | finance_insurance | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| finance_insurance_weak_grounding | finance_insurance | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| product_manual_support_retrieval_miss | product_manual_support | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| product_manual_support_retrieval_noise | product_manual_support | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| product_manual_support_unsupported_claim | product_manual_support | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| product_manual_support_contradicted_claim | product_manual_support | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| product_manual_support_citation_mismatch | product_manual_support | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| product_manual_support_stale_deprecated_source | product_manual_support | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| product_manual_support_insufficient_context | product_manual_support | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| product_manual_support_incomplete_answer | product_manual_support | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| product_manual_support_weak_grounding | product_manual_support | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| scientific_paper_retrieval_miss | scientific_paper | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| scientific_paper_retrieval_noise | scientific_paper | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| scientific_paper_unsupported_claim | scientific_paper | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| scientific_paper_contradicted_claim | scientific_paper | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| scientific_paper_citation_mismatch | scientific_paper | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| scientific_paper_stale_deprecated_source | scientific_paper | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| scientific_paper_insufficient_context | scientific_paper | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| scientific_paper_incomplete_answer | scientific_paper | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| scientific_paper_weak_grounding | scientific_paper | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| enterprise_kb_retrieval_miss | enterprise_kb | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| enterprise_kb_retrieval_noise | enterprise_kb | RETRIEVAL_ANOMALY | RETRIEVAL_ANOMALY | yes |
| enterprise_kb_unsupported_claim | enterprise_kb | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
| enterprise_kb_contradicted_claim | enterprise_kb | CONTRADICTED_CLAIM | CONTRADICTED_CLAIM | yes |
| enterprise_kb_citation_mismatch | enterprise_kb | CITATION_MISMATCH | CITATION_MISMATCH | yes |
| enterprise_kb_stale_deprecated_source | enterprise_kb | STALE_RETRIEVAL | STALE_RETRIEVAL | yes |
| enterprise_kb_insufficient_context | enterprise_kb | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| enterprise_kb_incomplete_answer | enterprise_kb | INSUFFICIENT_CONTEXT | INSUFFICIENT_CONTEXT | yes |
| enterprise_kb_weak_grounding | enterprise_kb | UNSUPPORTED_CLAIM | UNSUPPORTED_CLAIM | yes |
