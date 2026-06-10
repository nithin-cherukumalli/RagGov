# Claim Grounding False-Pass Analysis

False-pass cases analyzed: 20

## Category Breakdown

| Category | Count |
| :--- | ---: |
| `llm_overpermissive_no_deterministic_gate` | 8 |
| `related_but_non_supporting` | 5 |
| `value_mismatch_missed` | 3 |
| `entity_mismatch_missed` | 2 |
| `lexical_decoy` | 2 |

## Case Audit

### `sd-003`

- Domain: `software_docs`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The SDK supports retries in version 2.4.
- Evidence chunk: `chunk-sd-002`
- Evidence text: The SDK supports asynchronous processing of API calls.
- Critical entities: ['SDK']
- Critical values: ['2.4']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `sd-007`

- Domain: `software_docs`
- Difficulty: `easy`
- Failure type: `insufficient_missing_entity`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `supported`
- Safety gate: `False` / `None`
- Claim: The base URL for the API is api.example.com.
- Evidence chunk: `chunk-sd-005`
- Evidence text: The API endpoint requires authentication via bearer token.
- Critical entities: ['base URL']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `True`
- Compound claim: `False`
- Likely failure cause: `related_but_non_supporting`

### `sd-008`

- Domain: `software_docs`
- Difficulty: `easy`
- Failure type: `contradicted_entity`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `insufficient_evidence`
- Safety gate: `False` / `None`
- Claim: The --silent flag disables all verbose logging in the CLI.
- Evidence chunk: `chunk-sd-006`
- Evidence text: Use the --quiet or -q flags to suppress standard console output.
- Critical entities: ['silent flag']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `True`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `entity_mismatch_missed`

### `sd-012`

- Domain: `software_docs`
- Difficulty: `easy`
- Failure type: `contradicted_value`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `insufficient_evidence`
- Safety gate: `False` / `None`
- Claim: The maximum request body size is limited to 50 MB.
- Evidence chunk: `chunk-sd-009`
- Evidence text: Requests with payloads larger than 10 megabytes will fail with HTTP 413.
- Critical entities: ['body size']
- Critical values: ['50']
- Critical dates: []
- Critical units: ['mb']
- Value mismatch existed: `True`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `value_mismatch_missed`

### `hc-007`

- Domain: `healthcare_guidelines`
- Difficulty: `easy`
- Failure type: `insufficient_missing_entity`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: Treatment A causes headaches, nausea, and double vision.
- Evidence chunk: `chunk-hc-006`
- Evidence text: Side effects of treatment A commonly include mild headaches and nausea.
- Critical entities: ['double vision']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `True`
- Compound claim: `False`
- Likely failure cause: `related_but_non_supporting`

### `hc-012`

- Domain: `healthcare_guidelines`
- Difficulty: `easy`
- Failure type: `contradicted_value`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `insufficient_evidence`
- Safety gate: `False` / `None`
- Claim: A high fever is defined as a temperature of 39.5 degrees Celsius or higher.
- Evidence chunk: `chunk-hc-010`
- Evidence text: Medical guidelines classify high fevers at 38.5 C and above.
- Critical entities: ['high fever']
- Critical values: ['39.5']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `True`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `value_mismatch_missed`

### `hc-013`

- Domain: `healthcare_guidelines`
- Difficulty: `easy`
- Failure type: `insufficient_missing_entity`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: Medicine A may cause mild drowsiness.
- Evidence chunk: `chunk-hc-011`
- Evidence text: Medicine A shows an excellent safety profile with minimal systemic interactions.
- Critical entities: ['drowsiness']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `True`
- Compound claim: `False`
- Likely failure cause: `related_but_non_supporting`

### `fn-003`

- Domain: `finance_insurance`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The annual interest rate on the checking account is 3.5%.
- Evidence chunk: `chunk-fn-002`
- Evidence text: Checking accounts offer standard online banking privileges.
- Critical entities: ['checking account']
- Critical values: ['3.5']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `fn-007`

- Domain: `finance_insurance`
- Difficulty: `easy`
- Failure type: `insufficient_missing_entity`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: Routine dental cleanings are covered fully without a copay.
- Evidence chunk: `chunk-fn-005`
- Evidence text: Insurance covers standard pediatric dental hygiene visits.
- Critical entities: ['copay', 'routine dental']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `True`
- Compound claim: `False`
- Likely failure cause: `related_but_non_supporting`

### `fn-009`

- Domain: `finance_insurance`
- Difficulty: `hard`
- Failure type: `lexical_decoy`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `supported`
- Safety gate: `False` / `None`
- Claim: The premium savings account offers a 4.5% interest rate.
- Evidence chunk: `chunk-fn-007`
- Evidence text: We offer 4.5% interest on standard savings; premium account holders receive 5.5% interest.
- Critical entities: ['premium savings']
- Critical values: ['4.5']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `lexical_decoy`

### `pm-003`

- Domain: `product_manuals`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The charging voltage of the device is 9V.
- Evidence chunk: `chunk-pm-002`
- Evidence text: The device features a USB-C input connector.
- Critical entities: ['charging voltage']
- Critical values: ['9']
- Critical dates: []
- Critical units: ['v']
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `pm-013`

- Domain: `product_manuals`
- Difficulty: `easy`
- Failure type: `insufficient_missing_entity`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The phone features IP68 water resistance rating.
- Evidence chunk: `chunk-pm-010`
- Evidence text: Keep the phone away from direct exposure to water or heavy dampness.
- Critical entities: ['IP68']
- Critical values: []
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `True`
- Compound claim: `False`
- Likely failure cause: `related_but_non_supporting`

### `sp-003`

- Domain: `scientific_papers`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The trial experienced a withdrawal rate of 5%.
- Evidence chunk: `chunk-sp-002`
- Evidence text: We recorded primary endpoints for all enrolled members.
- Critical entities: ['withdrawal rate']
- Critical values: ['5']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `sp-005`

- Domain: `scientific_papers`
- Difficulty: `easy`
- Failure type: `contradicted_value`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `insufficient_evidence`
- Safety gate: `False` / `None`
- Claim: The study achieved a statistically significant p-value of less than 0.001.
- Evidence chunk: `chunk-sp-003`
- Evidence text: Calculated outcomes achieved statistical significance with p < 0.05.
- Critical entities: ['p-value']
- Critical values: ['0.001']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `True`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `value_mismatch_missed`

### `sp-007`

- Domain: `scientific_papers`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The drop-out rate of cohorts was 12%.
- Evidence chunk: `chunk-sp-005`
- Evidence text: Adherence scores remained high across all patient populations.
- Critical entities: ['drop-out']
- Critical values: ['12']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `lr-009`

- Domain: `legal_regulatory_general`
- Difficulty: `hard`
- Failure type: `citation_like_mismatch`
- Expected / predicted: `contradicted` -> `entailed`
- LLM / heuristic verdicts: `supported` / `insufficient_evidence`
- Safety gate: `False` / `None`
- Claim: Corporate mergers are governed by Section 12.
- Evidence chunk: `chunk-lr-008-a`
- Evidence text: Section 12 regulates tax implications of corporate restructuring.
- Critical entities: ['mergers']
- Critical values: ['12']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `True`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `entity_mismatch_missed`

### `ed-007`

- Domain: `education_general_kb`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The population of city Z is 250,000 citizens.
- Evidence chunk: `chunk-ed-006`
- Evidence text: City Z is an industrial center situated along the riverbank.
- Critical entities: ['population']
- Critical values: ['250,000']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `ed-009`

- Domain: `education_general_kb`
- Difficulty: `hard`
- Failure type: `lexical_decoy`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The population of City X is 2 million people.
- Evidence chunk: `chunk-ed-007`
- Evidence text: City X was established 2 centuries ago and spans 200 square kilometers.
- Critical entities: ['population']
- Critical values: ['2']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `lexical_decoy`

### `gp-003`

- Domain: `government_policy`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The maximum grant from the Chief Minister's Relief Fund is Rs. 5 lakhs per family.
- Evidence chunk: `chunk-gp-002`
- Evidence text: The Chief Minister's Relief Fund provides assistance to persons affected by natural calamities.
- Critical entities: ['CMRF']
- Critical values: ['5']
- Critical dates: []
- Critical units: []
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`

### `gp-007`

- Domain: `government_policy`
- Difficulty: `easy`
- Failure type: `insufficient_missing_value`
- Expected / predicted: `unsupported` -> `entailed`
- LLM / heuristic verdicts: `supported` / `unknown`
- Safety gate: `False` / `None`
- Claim: The response time limit for RTI is 30 days.
- Evidence chunk: `chunk-gp-005`
- Evidence text: Public authorities must handle applications promptly upon receipt.
- Critical entities: ['RTI response']
- Critical values: ['30']
- Critical dates: []
- Critical units: ['day']
- Value mismatch existed: `False`
- Entity mismatch existed: `False`
- Date mismatch existed: `False`
- Related but non-supporting: `False`
- Compound claim: `False`
- Likely failure cause: `llm_overpermissive_no_deterministic_gate`
