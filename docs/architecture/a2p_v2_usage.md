# A2P V2 Usage Guide

## Overview

A2P v2 implements world-class counterfactual claim-level attribution with:
- Multi-hypothesis candidate generation (10 cause types)
- Transparent heuristic scoring (uncalibrated, with explicit basis)
- Primary/secondary cause selection
- Evidence_for/evidence_against reasoning
- Modular architecture (5 focused files)

## Modular Architecture

```
src/raggov/analyzers/attribution/
  ├── a2p.py          # Main orchestrator (857 lines)
  ├── trace.py        # Trace extraction (219 lines)
  ├── candidates.py   # Candidate generation (466 lines)
  ├── scoring.py      # Transparent scoring (186 lines)
  └── selection.py    # Primary/secondary selection (186 lines)
```

## Configuration

### Enable A2P v2

To use A2P v2 mode, set `use_a2p_v2: True` in analyzer config:

```python
from raggov.analyzers.attribution.a2p import A2PAttributionAnalyzer

analyzer = A2PAttributionAnalyzer(
    config={
        "use_a2p_v2": True,  # Enable v2 mode
        "prior_results": prior_analyzer_results,
    }
)

result = analyzer.analyze(run)
```

### V1 vs V2 Output

**V1 Output** (claim_attributions):
```python
result.claim_attributions: list[ClaimAttribution]
# - Simple primary_cause string
# - Candidate causes list
# - Single A/A/P per claim
```

**V2 Output** (claim_attributions_v2):
```python
result.claim_attributions_v2: list[ClaimAttributionV2]
# - primary_cause + secondary_causes
# - Full candidate_causes with scores
# - evidence_summary (concise)
# - recommended_fix (composite)
# - Transparent score_basis
# - Uncalibrated status (explicit)
```

## Candidate Cause Types

A2P v2 generates up to 10 candidate causes per failed/risky claim:

1. **insufficient_context_or_retrieval_miss** - Missing evidence in corpus/retrieval
2. **weak_or_ambiguous_evidence** - Evidence present but quality too low
3. **generation_contradicted_retrieved_evidence** - Generation drifted from context
4. **stale_source_usage** - Evidence from outdated sources
5. **citation_mismatch** - Phantom citations (cited docs not retrieved)
6. **post_rationalized_citation** - Citations attached after generation
7. **verification_uncertainty** - Fallback verifier used
8. **prompt_injection_or_adversarial_context** - Security compromise detected
9. **retrieval_noise** - (Future)
10. **unknown** - Fallback when no candidates apply

## Risky Claims

V2 attributes **risky claims** (entailed but governance issues):
- Citation invalid despite entailment
- Stale source despite entailment
- Post-rationalized citation risk
- Verification uncertainty

These are flagged for remediation even when textually supported.

## Scoring

Transparent heuristic scoring (uncalibrated):

```
Score components:
- +0.35 for strong direct analyzer failure
- +0.25 for claim label match
- +0.20 for supporting signals
- +0.10 for affected chunk overlap
- -0.20 for evidence_against items

Score capped at [0.0, 1.0]
```

Every score includes explicit `score_basis`:
```
"Heuristic score 0.75: +0.35 from 2 analyzers; +0.25 from claim match; -0.10 from 1 contradicting analyzer"
```

## Primary/Secondary Selection

Selection algorithm:
1. **Rank** by composite score (60% heuristic + 20% stage severity + 10% evidence balance + 10% analyzer support)
2. **Primary** = highest-ranked candidate
3. **Secondary** = remaining candidates above 0.30 threshold
4. **Special rule**: Preserve citation/freshness as secondary when generation is primary

## Example Usage

```python
# Configure engine with A2P v2
from raggov.engine import diagnose_rag_run

result = diagnose_rag_run(
    run=my_rag_run,
    config={
        "a2p_config": {
            "use_a2p_v2": True,
        }
    }
)

# Access v2 attributions
for attribution in result.claim_attributions_v2:
    print(f"Claim: {attribution.claim_text}")
    print(f"Primary: {attribution.primary_cause}")
    print(f"Secondary: {attribution.secondary_causes}")
    print(f"Recommended fix: {attribution.recommended_fix}")

    # Inspect scored candidates
    for candidate in attribution.candidate_causes:
        print(f"  - {candidate.cause_type}: {candidate.heuristic_score}")
        print(f"    Score basis: {candidate.score_basis}")
        print(f"    Evidence for: {candidate.evidence_for}")
        print(f"    Evidence against: {candidate.evidence_against}")
```

## Backward Compatibility

V1 and V2 coexist in the same analyzer:
- `use_a2p_v2: False` (default) → Uses v1 claim-level mode or deterministic mode
- `use_a2p_v2: True` → Uses v2 multi-hypothesis mode

Both modes populate their respective fields:
- v1: `claim_attributions`
- v2: `claim_attributions_v2`

## Next Steps (Phases 7-11)

Remaining work:
- **Phase 7**: LLM-assisted A2P mode (schema-constrained)
- **Phase 8**: Update evaluation harness for multi-cause metrics
- **Phase 9**: Add comprehensive tests (12+ scenarios)
- **Phase 10**: Run full evaluation suite
- **Phase 11**: Complete architecture documentation

## Important Constraints

✅ **V2 does NOT conflate**:
- Claim textual support ≠ citation validity
- Claim textual support ≠ freshness validity
- Primary cause ≠ only cause (secondary causes preserved)

✅ **V2 ensures**:
- No silent fallbacks (fallback_used always tracked)
- No vague LLM prompts (structured trace first, LLM optional in Phase 7)
- Transparent scoring (heuristic with explicit basis)
- Uncalibrated status (never pretends to be calibrated confidence)
