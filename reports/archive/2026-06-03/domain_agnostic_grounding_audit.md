# Claim Grounding Core: Domain-Agnostic Audit Report

This report presents a thorough domain-leakage audit of the GovRAG claim grounding core engine to guarantee that the default extraction and verification pipelines operate purely on domain-agnostic foundations and are free of hardcoded assumptions from the government or policy domain.

---

## 1. Audit Scope & Methodology
We scanned all Python source files in the grounding analyzer layer (`src/raggov/analyzers/grounding/`) to locate occurrences of terms such as:
- `government` / `gov`
- `policy` / `act` / `rule` / `order` / `circular` / `notification`
- specific pattern-matching regexes (e.g. `G.O. Ms No.`)

---

## 2. Findings & Classification

All discovered occurrences were audited and classified into three groups:

### Group A: Truly Generic Core (Default Path)
These logic paths are active in default configurations and are completely domain-agnostic:

1. **`policy_rule` Claim Classification / Matching** (`claims.py` & `evidence_layer.py`):
   - *Pattern*: Checks for terms like `policy`, `regulation`, `order`, `rule`, `applies`.
   - *Domain Relevance*: **Truly Generic**. Used to capture rule-based/procedural claims in financial terms, software version validity policies, medical guidelines, and product manuals.
2. **`HeuristicValueOverlapVerifier`** (`verifiers.py`):
   - *Pattern*: Parses numbers, percentages, dates, and durations.
   - *Domain Relevance*: **Truly Generic**. Evaluates any numeric or date conflicts universally.
3. **`ConservativeEnsembleVerifier`** (`verifiers.py`):
   - *Pattern*: Safety gates comparing LLM & Heuristic verifiers.
   - *Domain Relevance*: **Truly Generic**. Works universally for checking facts-coverage and verifier alignment across all industries and disciplines.

### Group B: Government-Specific but Explicitly Non-Default (Decoupled)
These are specialized patterns and classes that only activate under non-default developer configurations:

1. **`extract_government_policy_value_mentions`** (`value_extraction.py`):
   - *Pattern*: Matches Indian government order IDs: `g.o. ms no.`.
   - *Core Gating*: **Strictly Non-Default**. It is gated under the `include_government_policy_extra=False` argument and is not called by the default verifier loop.
2. **`GovernmentPolicyTripletExtractorV0`** (`triplets.py`):
   - *Pattern*: Extracts triplets based on government predicate verbs and government order subjects.
   - *Core Gating*: **Strictly Non-Default**. This entire triplet extraction module is disabled by default (`enable_triplet_extraction: False`). Even if triplet extraction is enabled, it defaults to the domain-agnostic `GenericRuleTripletExtractorV0` unless `triplet_extractor_mode` is set to `"government_policy_v0"`.

### Group C: Core Leakage (Must Be Removed)
- *Finding*: **None**. 
- *Verification*: The core default grounding engine requires absolutely zero government-specific extraction or parameters to run.

---

## 3. Generalization Guarantees & Design Assurances

1. **Zero Hardcoded Government Logic**: No default grounding analyzer configuration relies on or triggers any government-specific regexes.
2. **Safe Fallbacks**: Optional paths like specialized triplet extraction gracefully degrade to standard, domain-agnostic rule-based extractors on non-government domains.
3. **Clean Telemetry**: Verification reports record domain-neutral metrics and indicators.
