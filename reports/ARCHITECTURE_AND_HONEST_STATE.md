# RagGov — Architecture & Honest State (for the architect)

Written 2026-06-18 by the engineering pass. Purpose: let you make decisions with full sight of
what each stage actually does, what is real vs heuristic vs test-shaped, and what it takes to make
this world-class. No flattery.

---

## 0. The one-line truth

RagGov is a **root-cause attribution layer** for RAG failures: given a `RAGRun` (query, chunks,
answer, citations) it returns a structured `Diagnosis` (primary failure, stage, security risk,
fix). The **architecture is genuinely good and novel** (evidence-tiered decision policy, A2P causal
attribution, layered analyzers, an honest discipline harness). The **substance is early**: most
semantic analyzers are lexical heuristics, nothing is calibrated, and the only evaluation set is
synthetic. It is a strong research preview, not yet a trustworthy product.

---

## 1. The pipeline, stage by stage (what actually runs)

Order is `DiagnosisEngine._default_analyzers()`. Each analyzer returns an `AnalyzerResult`
(status pass/warn/fail, failure_type, stage, evidence, method_status). The decision policy then
picks ONE primary.

### Layer 1 — Intake gate
- **ParserValidationAnalyzer** — structural parse/chunk validation. Uses a parser profile if
  present, else a text-only fallback (dangling-word + same-document boundary check — hardened in
  Task 20). **Method: structural heuristic.** Emits TABLE_STRUCTURE_LOSS / METADATA_LOSS /
  CHUNKING_BOUNDARY_ERROR — all taxonomy-**unsupported** (0 real cases). Treat as experimental.
- **SufficiencyAnalyzer** — "is the retrieved context enough?" Requirement extraction needs an LLM;
  with no LLM it **falls back to term-coverage** (you see the warnings). Pattern detectors:
  missing_critical_requirement, missing_exception, missing_scope_condition. **Method: lexical
  heuristic, NOT NLI.** Emits INSUFFICIENT_CONTEXT. The scope detector is coarse (Task 26).

### Layer 2 — Claim / evidence extraction (the substrate everything downstream needs)
- **ClaimGroundingAnalyzer** — extracts atomic claims from the answer (HeuristicClaimExtractorV0 in
  native mode — Task 23 widened its recall; or an LLM extractor), then verifies each claim against
  chunks. Native verifier = `HeuristicValueOverlapVerifier`, which the code itself labels
  **"not NLI", "top-k lexical/value aggregation."** Labels claims entailed / unsupported /
  contradicted / abstain. Emits UNSUPPORTED_CLAIM / CONTRADICTED_CLAIM. **This is the single most
  load-bearing analyzer, and in native mode it is lexical, not entailment.** That is the root cause
  of the Task 22 contradiction no-go.
- **ClaimAwareSufficiencyAnalyzer** — sufficiency conditioned on extracted claims.

### Layer 3 — Retrieval health
- **ScopeViolationAnalyzer** — query/entity scope mismatch (lexical). SCOPE_VIOLATION.
- **StaleRetrievalAnalyzer** — age + relative-recency + (Task 14) query-relevance gate.
  STALE_RETRIEVAL. **Heuristic.**
- **CitationMismatchAnalyzer** — phantom citation (cited doc not in retrieved set) is **deterministic
  and reliable**; broader mismatch is heuristic. CITATION_MISMATCH.
- **InconsistentChunksAnalyzer** — cross-chunk contradiction via a negation heuristic (Task 19:
  now requires a polarity-opposed proposition). **NOT NLI.** INCONSISTENT_CHUNKS (taxonomy-unsupported).
- **CitationFaithfulnessAnalyzerV0** — does the answer match what it cited? overlap heuristic.
  CITATION_MISMATCH / POST_RATIONALIZED_CITATION.
- **TemporalSourceValidityAnalyzerV1** — lifecycle metadata (expired/withdrawn/superseded) + age.
  STALE_RETRIEVAL. Self-labeled **practical_approximation.**
- **RetrievalDiagnosisAnalyzerV0** — aggregates retrieval reports. **heuristic_baseline.**
- **RetrievalAnomalyAnalyzer** — score z-score outliers / duplicates. RETRIEVAL_ANOMALY / SUSPICIOUS_CHUNK.

### Layer 4 — Security
- **PromptInjectionAnalyzer** — structural + multilingual regex (Task 18) + lexical semantic-intent
  + optional cross-encoder. **The strongest native analyzer** (detection is structural, not
  semantic-judgment), but explicitly **not a complete defense.** PROMPT_INJECTION.
- **PoisoningHeuristicAnalyzer** — score anomaly + answer steering. heuristic.
- **PrivacyAnalyzer** — PII/medical disclosure patterns. PRIVACY_VIOLATION.

### Layer 5 — Attribution + confidence
- **Layer6TaxonomyClassifier** — meta classifier over prior signals. heuristic.
- **SemanticEntropyAnalyzer** — uncertainty via a **deterministic label-entropy proxy (self-labeled
  "not research-faithful sampling")** or LLM. LOW_CONFIDENCE.
- **NCVPipelineVerifier** (optional; default on in external-enhanced) — a node-chain that aggregates
  prior results into a pipeline-health verdict. Self-labeled **"practical_approximation, not
  research-faithful NCV."** It **over-fires** (drove Task 21 and CLEAN-FP audit findings).
- **A2PAttributionAnalyzer** (optional) — answer-to-passage causal attribution; deterministic +
  optional LLM. The **most novel/differentiating** component. Legacy fallback is heuristic.
- **AnswerQualityAnalyzer** — generation-stage signals (incompleteness, context-adherence,
  overconfidence). **NOT wired into the default suite** (needs prior_results plumbing) — why the
  Task 15/16 xfail tests still partially fail.

### External signals (mode = external-enhanced, opt-in)
`external_signal_bridge` maps **bring-your-own** RAGAS / DeepEval / RAGChecker / RefChecker results
into internal signals — **advisory tier only.** RagGov does not run those tools; you pass their
outputs in `run.metadata`. Native mode imports none of them.

### The brain — decision policy
Every analyzer result is classed into an **evidence tier**:
`BLOCKING_DETERMINISTIC > STRUCTURED_DIAGNOSTIC > HEURISTIC_SUPPORTING > EXTERNAL_ADVISORY`.
Within tier, a **specificity rank** orders failure types, plus guards (e.g.
`_require_explicit_contradiction`, which is load-bearing — disabling it regresses Calib). Output:
one `primary_failure`, a `root_cause_stage`, and a full `decision_trace` (why this won, what was
suppressed). **This explainability layer is a real strength and a product differentiator.**

---

## 2. Are this session's fixes test-fitting or genuine? (direct answer)

**They are genuine mechanism fixes, not test-gaming — with one honest asterisk.** Evidence:
- Every fix was measured on the **145-row held-out probe**, not the unit tests, and gated on the
  protected baseline (43/46) and Calib (23/45) not regressing.
- Three attempts were **reverted or declared no-go** (Task 22 contradiction, Task 24 list-answer,
  Task 26 scope) precisely because they would have been test-fitting or traded one error for
  another. A test-gaming process does not voluntarily revert net-positive changes.

| Fix | Mechanism | Generalizes? |
|---|---|---|
| 18 multilingual injection | real DE/ES/FR patterns | yes (to those languages) |
| 19 INCONSISTENT polarity | requires real opposed proposition | yes |
| 20 chunk-boundary same-doc | structured provenance | yes |
| 21 Jaccard 0.97 | near-identity duplicate | yes |
| 14 stale relevance+recency gate | structured relevance + dates | yes |
| 15 incomplete→GENERATION stage | narrow, principled | yes |
| 16 permission contradiction | vocabulary extension, fires 0/145 elsewhere | yes (narrow) |
| **23 source-assertion extraction** | **mechanism general; metric inflated** | **mechanism yes; the +0.16 probe jump rides ONE synthetic suffix repeated 30×** |

**The asterisk that matters most:** the probe is **synthetic mutations of clean cases**, and
Sidekick-2 confirmed **0 non-synthetic rows exercise the Task-23 rule** and **30/145 probe rows are
one repeated suffix**. So "the fixes generalize to *any* RAG run" is **NOT proven** — only "they
generalize across these mutation families." Production generalization is **UNKNOWN** until a real,
non-overlapping heldout exists. The improvement is real engineering on a synthetic yardstick.

---

## 3. The real problems (ranked, blunt)

1. **Nothing is calibrated.** All scores/confidence are uncalibrated; the gate is correctly off.
   You cannot make trust claims without calibration data.
2. **The evaluation is synthetic.** No real heldout. The headline 0.55 probe number is on mutated
   data; production accuracy is unknown. This is the #1 blocker to credibility.
3. **Native semantic analyzers are lexical, not NLI.** Contradiction, insufficiency, and grounding
   are the hardest jobs and are done with term/value overlap. This is why Tasks 22/26 were no-gos:
   the heuristic literally cannot see semantic contradiction, and its FPs and TPs are the *same
   mechanism* (entangled — you can't fix one without losing the other).
4. **Data sparsity / taxonomy honesty.** Only 3 of 25 failure types have ≥5 real cases; 13 have
   zero. The taxonomy is aspirational beyond CLEAN / CONTRADICTED_CLAIM / INSUFFICIENT_CONTEXT.
5. **NCV over-fires.** The node verifier is a practical approximation and inflates STALE/INCONSISTENT.
6. **AnswerQualityAnalyzer isn't wired in** — a real taxonomy/implementation gap.
7. **Heuristic ceiling reached.** The remaining native precision gains all trade against recall or
   safety (verified in the CLEAN-FP audit). Further native tweaking is low-value.

---

## 4. How to make it solid and world-class (the plan)

**Phase A — earn a real number (unblocks everything).**
- Run the fresh-data pull (tooling ready: `scripts/validate_fresh_heldout.py` dedups vs probe +
  training). Build a 40–60 row real, double-adjudicated heldout. Grow each supported type to ≥5
  real cases. Track generalization accuracy as THE metric.

**Phase B — add a real semantic tier (the biggest quality lever).**
- Make the **optional LLM/NLI entailment verifier** (`ClaimEntailmentVerifierV1`,
  `claim_grounding_verifier_policy=llm_entailment`) a first-class tier above native heuristics, with
  visible degradation when absent. This is what fixes contradiction/insufficiency/grounding — the
  things native heuristics cannot do. Keep native as the cheap deterministic floor.

**Phase C — calibrate.**
- With real data: per-type confidence, calibration curves, decision thresholds, confidence
  intervals across seeds. Then flip `calibration_status` → `preliminary`, and only later
  `production_gating_eligible` for the types that earn it.

**Phase D — leverage external signals properly.**
- External (RAGAS/DeepEval/RefChecker) stays **corroborating evidence**, never source-of-truth.
  The tiered policy already does this. Productize: "native says X; RAGAS agrees/disagrees" raises
  or lowers confidence — a genuinely useful triangulation no single OSS tool offers.

---

## 5. Productization — features beyond open source

The OSS core is the analyzers + taxonomy. The **defensible product** is the layer on top:
1. **Calibrated, per-type confidence + governance gating** ("safe to auto-serve?" / "block & route
   to human"). Nobody ships calibrated RAG-failure attribution.
2. **Causal attribution (A2P) with an evidence chain** — not "the answer is wrong" but "the
   generator dropped step 3 that chunk-7 contained" — the explainability moat.
3. **Regression guard for *your* pipeline** — track diagnosis drift across deploys; alert when your
   RAG's failure profile changes. CI for RAG quality.
4. **Triangulated multi-signal mode** — native + external + NLI fused with calibrated weights.
5. **Audit/governance trails** — every diagnosis carries its decision trace; "why was this blocked"
   is a first-class, exportable artifact for regulated domains.
6. **Hosted API + adapters** (LangChain/LlamaIndex examples already exist).

The honest sequencing: you cannot sell #1–#2 until Phases A–C are real. The architecture is ready
for them; the data and calibration are not.
