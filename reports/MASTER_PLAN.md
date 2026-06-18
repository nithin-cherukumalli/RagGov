# RagGov — Master Plan to Trustworthy RC → Production (FINAL)

Owner: Opus (main builder). Sidekicks: Antigravity (S1), Codex (S2).
Decided constraints: **hybrid semantic tier** (native floor + optional LLM/NLI), **LLM-assisted
labeling with human spot-audit** (labels are provisional, never gold), day-by-day to an RC
milestone, then a production extension. Grounded in 2025–26 SOTA (claim-response entailment is the
strongest faithfulness signal; LLM judges are overconfident → calibrate + majority-vote + audit).

> Calendar note: the stated "June 5" precedes today (June 18), so this plan is written as **Day 1…N
> from the day we start**, RC at ~Day 21, production extension after. Map Days to your calendar.

---

## 0. Principles carried forward (non-negotiable — this is why it's trustworthy)
1. **Pre-registration before any engine change** — hypothesis + hard pass/fail criteria written first.
2. **Hard gates, revert on failure** — protected baseline (43/46), Calib (≥23/45), named TPs,
   and (new) calibration metrics must not regress. Two reverts beat one false fix.
3. **Measure on a held-out set, never the unit tests.** Tune on train/dev; the real heldout is
   LOCKED and never peeked (SOTA: tune on one set, evaluate on another).
4. **Honesty first** — heuristic is labeled heuristic; uncalibrated is labeled uncalibrated; no
   claim outruns the data. `calibration_status`/`gating` flags only move when earned, per-type.
5. **One task per commit. Sidekick outputs are ALWAYS re-verified by Opus** (they have been wrong
   on numbers/recommendations every round — great for breadth, not for final judgment).
6. **Leverage the in-repo harness** (`stresslab` runners, `evaluate_*`, `check_*` guards) as the
   measurement spine; build the calibration loop on top of it, don't reinvent.

## 1. Roles (who does what, and the hard boundary)
- **Opus (main builder) — owns all trust-bearing code:** engine, analyzers, decision policy, the
  NLI tier integration, calibration math, gating logic, every pre-registration and revert decision,
  final measurement of record. Nothing trust-bearing ships without Opus measuring it.
- **Antigravity (S1) — measurement & instrumentation:** run the harness across seeds, produce
  per-row/per-type tables, ECE/Brier computations, regression-surface enumeration, scoring stubs,
  dedup/validation scripts. READ-ONLY on engine/policy/labels/gates. Output = evidence, not edits.
- **Codex (S2) — data & docs pipeline:** fresh-data pull runbooks, LLM-assisted labeling scripts
  (multi-checker + agreement + spot-audit lists), schema validation, taxonomy/claims/honesty audits,
  calibration-report scaffolding. READ-ONLY on engine/policy/labels/gates; may create standalone
  `scripts/` data tools and staging artifacts only.
- **Boundary rule:** only Opus edits `src/raggov/**`, `decision_policy*`, labels, thresholds, gates,
  the locked dataset/manifest. Sidekicks prepare; Opus decides and commits.

---

## 2. PHASE PLAN (Day-by-Day to RC)

### Phase 0 — Calibration/eval loop on the harness (Day 1–2)
Goal: a repeatable, seeded evaluation that emits accuracy + calibration (ECE/Brier) per type, so
every later change is measured the same way.
- **Opus D1:** define the canonical scorer as a committed harness entry (replace the throwaway
  `/tmp` scorers): build `RAGRun` from a case, run `diagnose`, compare primary; fix the documented
  drift in `evaluate_govrag_calib.py` OR wrap the working direct scorer as a script. Record config
  mode explicitly (default vs native — they differ: 0.552 vs 0.566). Pre-reg: numbers reproduce
  (protected 43/46, Calib 23/45, probe 0.552 default).
- **S1 D1:** build a seeded multi-run harness wrapper that reports per-type accuracy + a confidence
  column placeholder; verify it matches Opus's scorer exactly on 3 spot cases.
- **S2 D1:** scaffold a `reports/calibration/` report format (per-type accuracy, ECE, Brier, n,
  CI) — empty but schema-fixed.
- **Opus D2:** add a `confidence` field path through the pipeline if absent (diagnostic confidence,
  explicitly uncalibrated) so calibration has something to calibrate. Pre-reg + guards.
- **Gate to exit Phase 0:** one command reproduces accuracy + (placeholder) calibration per type;
  all existing guards green.

### Phase 1 — Real, non-overlapping data + LLM-assisted labels (Day 3–9, parallel with Phase 2)
Goal: a real heldout that does NOT overlap training/probe, with provisional LLM-assisted labels and
a human spot-audit — the thing that lets us claim generalization at all.
- **USER (you) D3:** run the fresh-data pull (HF is blocked in sandbox) per the runbook in
  `SIDEKICK_PROMPT_fresh_data_unblock.md` → drop `fresh_intake_v1.jsonl` (~80–100 raw).
- **S2 D3–5:** `scripts/validate_fresh_heldout.py` (already built) dedups vs canonical+probe+seeds;
  then **multi-checker LLM labeling**: 2–3 independent LLM judges per row → majority vote → label +
  inter-judge agreement as a provisional confidence; emit a **human spot-audit worklist** of all
  disagreements + all CONTRADICTED rows (RAGTruth contradiction is heuristic). NEVER auto-finalize.
- **USER + Opus D6:** human spot-audit the disagreement/CONTRADICTED worklist (cheap: only the hard
  rows). Lock a real heldout v1 (40–60 rows, ≥5 real per supported type where possible) into a
  versioned staging file with `LABEL_CHANGELOG` entry. **Heldout is locked and never tuned on.**
- **S1 D7:** score current engine (default AND native) on the locked real heldout → **the first
  honest generalization number.** Per-type table. Label provisional (LLM-assisted labels).
- **Gate:** a locked, de-overlapped real heldout exists with documented label provenance and a known
  generalization number. This is the alpha→RC data gate.

### Phase 2 — Hybrid semantic (NLI) tier — the trust lever (Day 5–14, overlaps Phase 1)
Goal: make contradiction/grounding/sufficiency trustworthy by adding an optional claim-response
entailment tier above the native heuristics, degrading visibly when no model is present.
- **S1 D5:** inventory the existing hooks (`ClaimEntailmentVerifierV1`, `conservative_ensemble`,
  `refchecker` adapter, `claim_grounding_verifier_policy`) and write the exact wiring map +
  regression surface. (Verify — its inventories have been right before here.)
- **Opus D6–9:** implement the **claim-response entailment verifier** as a first-class tier:
  per claim, NLI(premise=retrieved evidence, hypothesis=claim) → entailment/contradiction/neutral.
  Use it to set claim labels when a model/NLI is configured; native heuristic remains the fallback
  with a visible `method_status=heuristic_baseline`. Pre-reg per change; gate on protected/Calib +
  no native-mode regression (the tier is OFF by default in native).
- **Opus D10–12:** route the entailment verdict into the decision policy as **STRUCTURED_DIAGNOSTIC
  with a hard signal** so `_require_explicit_contradiction` promotes a real NLI-backed contradiction
  (this is the principled fix for the Task 22 no-go — contradiction becomes trustworthy *because*
  it's entailment-backed, not lexical). Pre-reg; measure CONTRADICTED recall on probe + real heldout
  in NLI mode; native mode unchanged.
- **S1 D9–13:** measure each NLI change on probe + real heldout, both modes; produce confusion
  tables; flag any native regression immediately.
- **Methodology guardrails (from SOTA):** use **majority/ensemble of checkers**, not a single judge;
  record neutral≠contradiction; localize unfaithful claims (token/claim level) for the trace.
- **Gate:** with NLI on, CONTRADICTED and grounding accuracy rise materially on the real heldout
  with NO native-mode regression; native remains the deterministic floor.

### Phase 3 — Calibration (Day 12–18)
Goal: turn raw signals into **calibrated per-type confidence** with measured ECE/Brier — the step
that lets a developer "bet money."
- **S1 D12:** compute baseline **ECE, Brier, reliability curves per failure type** across ≥5 seeds
  on train/dev (NOT heldout). (SOTA metrics: ECE/ACE/MCE/Brier/NLL; couple with F1.)
- **Opus D13–16:** implement a calibration layer — temperature/Platt/isotonic per type on the
  diagnostic confidence; **bias-correct for imperfect judge sensitivity/specificity** (the
  LLM-assisted labels are imperfect — account for it). Confidence becomes calibrated probability,
  stored with `calibration_status` per type. Pre-reg; revert if ECE doesn't improve or accuracy drops.
- **S1 D16–18:** validate calibration on the LOCKED real heldout (held-out ECE/Brier) — the honest
  calibration number. Confidence intervals via bootstrap.
- **Gate:** per-type calibrated confidence with heldout ECE below an agreed threshold (e.g. <0.10)
  for the 3 supported types; flip `calibration_status → preliminary` ONLY for those types.

### Phase 4 — RC hardening (Day 16–21)
Goal: a coherent, honest RC a developer can trust for the supported scope.
- **Opus D17–18:** per-type gating logic — `production_gating_eligible` becomes true *only* for
  types that pass heldout accuracy + ECE thresholds; everything else stays advisory. Wire
  `AnswerQualityAnalyzer` properly if it clears its regression surface (else keep deferred — honest).
- **S2 D17–19:** **regression guard / drift harness** — a harness mode that snapshots the diagnosis
  profile and fails CI if a deploy shifts it (productizable: "CI for RAG quality").
- **Opus D19:** finalize the **decision trace as the explainability artifact** (export JSON: primary,
  stage, evidence chain, suppressed candidates, calibrated confidence, method_status).
- **S2 D20:** apply the full honesty edit list (README/docs) to match the earned RC claims exactly;
  taxonomy: quarantine the 13 zero-data types from any capability claim.
- **Opus D21:** RC tag `v0.2-rc` — run all guards + probe + real heldout + calibration; write the RC
  honesty report (what's trustworthy: the 3 calibrated types; what's not: everything else).
- **RC definition of done:** real heldout generalization ≈0.70 on supported types, calibrated
  confidence with heldout ECE <0.10 on those types, NLI tier trustworthy, native floor intact, every
  claim data-backed, full explainable trace, regression guard live.

---

## 3. PRODUCTION EXTENSION (post-RC, ~Weeks 4–8)
- **W4 Governance & audit:** exportable audit trail per diagnosis (trace + calibrated confidence +
  provenance); "why was this blocked" first-class. Regulated-domain ready.
- **W4–5 External triangulation (productized):** fuse native + NLI + bring-your-own
  RAGAS/DeepEval/RefChecker with calibrated weights; "native says X, RAGAS agrees → confidence ↑".
  Differentiator no single OSS tool offers.
- **W5–6 Hosted API + adapters:** stable API; LangChain/LlamaIndex adapters (examples exist);
  per-type confidence + gating in the response.
- **W6–7 Data flywheel:** every customer run + correction feeds (with consent) the labeled set;
  scheduled re-calibration; drift alerts. This is the moat.
- **W7–8 Grow taxonomy coverage:** move thin types (UNSUPPORTED, STALE, CITATION, SCOPE) to
  supported with real data + NLI; calibrate them; expand the gated scope type-by-type.
- **Paid differentiators (beyond OSS):** calibrated per-type gating, A2P causal explainability,
  drift/regression guard for the customer's own pipeline, triangulated multi-signal confidence,
  governance/audit exports, hosted SLAs.

---

## 4. Risks & how the plan defends against them
- **LLM-label unreliability** → multi-checker majority + human spot-audit + bias-corrected
  calibration; heldout labels provenance-tagged; calibration accounts for judge error.
- **Overfitting to synthetic/probe** → real LOCKED heldout, tune-elsewhere discipline, never peek.
- **NLI cost/latency/privacy** → hybrid: native floor always works offline; NLI optional with
  visible degradation; allow local NLI model for privacy.
- **Sidekick error** → Opus re-verifies every number/recommendation before acting (proven necessary).
- **Scope creep across 25 types** → ship trust for 3 types first; expand only as data+calibration earn it.
- **Sandbox limits** → HF pull on user machine; re-create the 3.11 shim each session; commit working
  scorers into the repo so we stop depending on `/tmp`.

## 5. The single sentence to hold onto
Ship **calibrated trust for three failure types backed by real data and entailment-grade evidence**,
with a deterministic native floor and an honest banner for everything else — then expand the gated
scope type-by-type. That is a product a developer can bet on; a 25-type uncalibrated heuristic is not.

Sources informing methodology: RAGChecker (NeurIPS 2024), RefChecker (arXiv 2405.14486),
LettuceDetect 2025, "Overconfidence in LLM-as-a-Judge" (arXiv 2508.06225), "Calibrating LLM Judges"
(arXiv 2512.22245), "How to Correctly Report LLM-as-a-Judge Evaluations" (arXiv 2511.21140).
