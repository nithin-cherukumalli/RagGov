# A2P V2 Implementation Status & Plan

## Completion Status: Phase 6 Complete (Modular Implementation)

**Completed:**
- ✅ Phase 0: Diagnosis and understanding
- ✅ Phase 1: Rich attribution models (CandidateCause, ClaimAttributionV2)
- ✅ Phase 2: Modular structure refactoring
- ✅ Phase 3: Candidate generation (8 candidate types in candidates.py)
- ✅ Phase 4: Transparent scoring (scoring.py)
- ✅ Phase 5: Primary/secondary selection (selection.py)
- ✅ Phase 6: V2 mode wired into main analyzer (a2p.py)

**Next Steps (Phases 7-11):**

### Remaining Priorities

**Core implementation complete!** Modular A2P v2 now functional with:
- 5 focused modules (~1,914 lines total)
- 8 candidate cause generators
- Transparent heuristic scoring
- Primary/secondary selection
- V1/V2 backward compatibility

**Remaining work:**

1. **Phase 7: LLM-Assisted Mode** (~300 lines)
   - Schema-constrained LLM attribution as alternative to heuristic mode
   - Structured prompt using AttributionTrace
   - JSON schema for CandidateCause validation
   - Fallback to heuristic if LLM fails

2. **Phase 8: Evaluation Harness Updates** (~200 lines)
   - Multi-cause evaluation metrics
   - Secondary cause tracking
   - V2-specific golden test cases

3. **Phase 9: Comprehensive Tests** (~400 lines)
   - Unit tests for trace extraction
   - Unit tests for candidate generation
   - Unit tests for scoring
   - Unit tests for selection
   - Integration tests for v2 mode
   - Edge case coverage (12+ scenarios)

4. **Phase 10: Full Evaluation Run**
   - Run claim diagnosis harness with v2
   - Baseline comparison v1 vs v2
   - Validate multi-cause accuracy

5. **Phase 11: Documentation** (~300 lines)
   - Architecture deep-dive
   - API reference
   - Migration guide v1→v2
   - Example workflows

### Modular Implementation Complete

The modular A2P v2 implementation provides:

1. **trace.py** - Structured evidence aggregation
   - AttributionTrace dataclass with all diagnostic signals
   - extract_attribution_trace() - comprehensive extraction from prior analyzers

2. **candidates.py** - Multi-hypothesis generation
   - identify_claims_needing_attribution() - failed + risky claims
   - generate_candidate_causes() - 8 candidate types per claim
   - Evidence_for/evidence_against per candidate

3. **scoring.py** - Transparent heuristic scoring
   - score_candidate() - explicit scoring rules
   - score_all_candidates() - batch scoring + ranking
   - Uncalibrated status (never pretends to be confidence)

4. **selection.py** - Primary/secondary cause selection
   - select_primary_and_secondary_causes() - composite ranking
   - build_evidence_summary() - concise evidence
   - build_composite_fix_recommendation() - multi-cause fixes

5. **a2p.py** - Orchestration with v1/v2 modes
   - _claim_level_mode_v2() - new multi-hypothesis mode
   - Backward compatibility with v1
   - Configuration via use_a2p_v2 flag

### Recommended Approach

**Option A: Incremental Implementation**
- Implement one phase per session
- Test after each phase
- Maintain backward compatibility

**Option B: Focused V2 Release**
- Complete Phases 3-5 first (core logic)
- Test with existing harness
- Add LLM mode (Phase 7) later
- Full test suite (Phase 9) after core works

**Option C: Simplified V2 MVP**
- Implement only insufficient_context and generation_contradicted candidates
- Skip LLM mode initially
- Focus on primary cause selection
- Defer secondary causes

## Current Code Location

**Models:**
- `/src/raggov/models/diagnosis.py`
  - CandidateCause (lines 99-121)
  - ClaimAttributionV2 (lines 124-141)
  - AnalyzerResult.claim_attributions_v2 (line 161)

**Attribution Modules:**
- `/src/raggov/analyzers/attribution/trace.py` (219 lines)
  - AttributionTrace dataclass
  - extract_attribution_trace()

- `/src/raggov/analyzers/attribution/candidates.py` (466 lines)
  - identify_claims_needing_attribution()
  - generate_candidate_causes()
  - 8 candidate generator functions

- `/src/raggov/analyzers/attribution/scoring.py` (186 lines)
  - score_candidate()
  - score_all_candidates()

- `/src/raggov/analyzers/attribution/selection.py` (186 lines)
  - select_primary_and_secondary_causes()
  - build_evidence_summary()
  - build_composite_fix_recommendation()

- `/src/raggov/analyzers/attribution/a2p.py` (857 lines)
  - A2PAttributionAnalyzer
  - _claim_level_mode_v2() - main v2 orchestration
  - _claim_level_mode() - v1 (backward compatibility)
  - _deterministic_mode() - legacy
  - _llm_mode() - legacy

## Key Design Decisions Made

1. **Backward Compatibility**: claim_attributions (v1) and claim_attributions_v2 coexist
2. **Trace-First**: All evidence extracted before attribution
3. **Transparent Scores**: Heuristic only, clearly marked uncalibrated
4. **Multi-Hypothesis**: Primary + secondary causes supported
5. **Risky Claims**: Entailed but citation/freshness invalid cases included

## Critical Implementation Notes

From requirements:
- Must not conflate textual support with citation/freshness
- Must preserve stale/citation as secondary when contradiction is primary
- Must make fallback visible (no silent failures)
- Must use structured trace, not vague LLM prompts
- Must generate evidence_for and evidence_against per candidate

## Effort Tracking

**Completed (Phases 0-6):**
- Phase 0-2: Diagnosis, models, refactoring (~3 hours)
- Phase 3-4: Candidate generation, scoring (~4 hours)
- Phase 5-6: Selection, integration (~2 hours)
- **Subtotal**: ~9 hours

**Remaining (Phases 7-11):**
- Phase 7: LLM mode (~2 hours)
- Phase 8: Evaluation updates (~1.5 hours)
- Phase 9: Comprehensive tests (~3 hours)
- Phase 10: Full evaluation run (~1 hour)
- Phase 11: Documentation (~2 hours)
- **Subtotal**: ~9.5 hours

**Total Estimated Effort**: ~18.5 hours for world-class implementation

## Recommendation

Given the complexity, I recommend:
1. Complete this task across 2-3 focused sessions
2. Start with Option B (Focused V2 Release)
3. Implement Phases 3-5 first
4. Test with existing claim diagnosis harness
5. Add comprehensive tests once core works
6. Add LLM mode last

This ensures high-quality implementation without rushing.
