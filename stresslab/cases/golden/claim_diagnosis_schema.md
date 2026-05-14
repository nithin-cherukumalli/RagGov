# Claim Diagnosis Gold Schema

The claim diagnosis harness accepts both historical and current gold-set wrappers:

- Legacy wrapper: `version`, `cases`
- Current wrapper: `evaluation_status`, `examples`

Internally, `evaluation_status` maps to `version` and `examples` maps to `cases`.
No examples are skipped during migration or loading.

## Case Fields

Each case/example requires:

- `case_id`
- `query`
- `retrieved_chunks`
- `final_answer`
- `expected_claims`
- `expected_sufficient` or `expected_sufficiency`
- `expected_primary_stage` or `expected_stage`
- `expected_fix_category`

Optional case fields include:

- `category`
- `cited_doc_ids`
- `corpus_entries`
- `metadata`

## Expected Claim Fields

Each expected claim requires:

- `claim_text`
- `expected_claim_label` or `expected_label`

Optional claim-axis labels include:

- `expected_citation_validity`
- `expected_freshness_validity`
- `expected_a2p_primary_cause`

The harness reports claim label accuracy, sufficiency accuracy, A2P primary cause
accuracy, primary stage accuracy, fix category partial accuracy, false clean count,
claim-label breakdowns, and per-case mismatches with expected and actual payloads.
