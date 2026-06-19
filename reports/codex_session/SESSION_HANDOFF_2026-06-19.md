# RagGov — Session Handoff for the NEXT chat (read this first)

Last updated: 2026-06-19. This chat got very large; everything the next session needs is here.
Read order: this file → `reports/MASTER_PLAN.md` → `reports/ARCHITECTURE_AND_HONEST_STATE.md` →
`reports/codex_session/codex_sidekick_session_plan.md` (the sidekick ledger).

---

## 1. What RagGov is (one line)
A root-cause **diagnosis layer** for RAG failures: given a `RAGRun` (query, retrieved chunks,
answer, citations) it returns a structured `Diagnosis` (primary failure, stage, security risk, fix,
decision trace). It does root-cause attribution, not answer-quality scoring.

## 2. The brutally honest current state (do not overstate)
- **Real generalization = 0.24** on the first non-synthetic, non-overlapping heldout (75 rows).
  The old synthetic probe (0.552) overstated by ~2.3×.
- **#1 problem: CLEAN false-positive rate = 0.76** — 38 of 50 faithful answers get a false failure.
  It is "death by a thousand cuts": stale(9) + insufficient(8) + inconsistent(7) + unsupported(6) +
  tail, across MANY analyzers. No single fix recovers it.
- The semantic spine is **lexical heuristics, uncalibrated, not NLI**. `calibration_status:
  not_calibrated`, `production_gating_eligible: false`. Keep them that way until earned.
- **The CONTRADICTED labels in the real heldout are mismapped** — RAGTruth span-level annotations
  mapped to whole-answer CONTRADICTED. Confirmed by 4 independent judges (engine, Codex audit, live
  Groq, live Kimi). Do NOT optimize "CONTRADICTED recall" against them.

## 3. The plan we are following (FINAL): `reports/MASTER_PLAN.md`
Phase 0 eval loop ✅ → Phase 1 real data ✅(provisional) → **Phase 2 hybrid NLI tier ✅(infra) /
⚠️(no measured gain yet)** → Phase 3 calibration (next) → Phase 4 RC → production.
Decisions locked: **hybrid semantic tier** (native floor + optional LLM/NLI), **LLM-assisted
labeling with human spot-audit** (labels provisional, never gold).

## 4. What happened THIS session (chronological, all committed locally)
Engine precision (measured on the synthetic probe, gated on protected 43/46 + Calib 23/45):
- **Task 18** multilingual prompt-injection recall 1/10→9/10.
- **Task 19** INCONSISTENT_CHUNKS: require polarity-opposed proposition (8→2 FP).
- **Task 20** CHUNKING_BOUNDARY: gate on same source_doc_id (5→0 FP).
- **Task 21** NCV Jaccard duplicate threshold 0.85→0.97 (3→0 FP).
- **Task 23** source-assertion claim extraction recall (UNSUPPORTED→CLEAN 7→0; probe 0.393→0.552
  but this gain is synthetic-suffix-driven — honest caveat).
- **Task 14** stale irrelevant source: gate on query-relevance + newer dated alt (xfail→pass).
- **Task 15** incomplete-answer stage GROUNDING→GENERATION (fixed protected case 38).
- **Task 16** permission-contradiction specificity (case 41 primary now CONTRADICTED).
- **Task 22 NO-GO** (documented): native-mode contradiction recall can't be recovered; the
  `_require_explicit_contradiction` guard is load-bearing (disabling regresses Calib + invents false
  contradictions).
- **Task 24 / Task 26 deferred/reverted**: list/short-answer recall and the scope-marker CLEAN fix
  are entangled (FPs share the TP mechanism) — reverted on the pre-registered gate. Two reverts beat
  one false fix.

Foundation + hybrid tier this session:
- **Canonical scorer** `scripts/raggov_score.py` (`score_file`/`build_run`) — single source of truth,
  replaces /tmp scripts. Reproduces Calib 23/45, probe 80/145 default / 82 native. Tested.
- **First real heldout** `evals/govrag_calib/staging/raw/heldout_real_v1.jsonl` (75 rows, 0 overlap;
  user ran the HF pull). Number: 0.24. LOCKED — never tune on it.
- **Hybrid NLI tier wired** (Phase 2):
  - increment 1: entailment-grade contradiction counts as "explicit" in the policy (promotes) —
    `_ENTAILMENT_GRADE_METHODS` in `decision_policy_support.py`.
  - increment 2: offline `local_nli` verifier (`cross-encoder/nli-deberta-v3-small`) in
    `verifiers.py`, lazy load + heuristic fallback, policy `claim_grounding_verifier_policy="local_nli"`.
  - increment 3: **grounded-clean gate** in `decision_policy.py` — if an entailment-grade grounding
    verdict finds the answer clean (no contradicted/unsupported) and the winner is a low-tier
    retrieval-health heuristic with no blocking signal → CLEAN. Native-safe (inactive without NLI).
- **LLM providers** (run on the USER's machine; sandbox proxy-blocks them):
  - Groq: `scripts/groq_client.py` (.env loader) + `raggov.connectors.groq_client.GroqLLMClient` (SDK).
  - Kimi/Moonshot: `scripts/kimi_client.py` (OpenAI-compatible, `KIMI_API_KEY` in .env, model
    `moonshot-v1-8k` works; `kimi-k2.5` 400s on some params).
  - Diagnostics: `scripts/{groq,kimi}_nli_diagnose.py`; A/B: `scripts/run_nli_heldout.py --provider {groq,kimi,mock}`.

## 5. The two big findings from the last sidekick runs (CRITICAL for next steps)
1. **NLI A/B (Kimi, with the grounded-clean gate live): NO gain.** Native 18/75 (CLEAN-FP 0.76) vs
   Kimi-NLI 16/75 (CLEAN-FP 0.78). The gate barely fired because on real data NLI rarely declares an
   answer fully clean (it finds some "unsupported" claims, or extracts none), and several false
   primaries are `INCOMPLETE_DIAGNOSIS` (NOT in the suppressible set). **The gate is correct but too
   strict / too narrow to help as-is.** Also 215s for 75 rows, 42% of claim checks hit fallback
   (rate limits) → the NLI signal was degraded.
2. **Provisional relabel (Codex):** of 25 source-CONTRADICTED rows, NLI re-derives only 10
   CONTRADICTED, 8 UNSUPPORTED, 6 INSUFFICIENT, 1 CLEAN; of 50 source-CLEAN, only 22 stay CLEAN
   (20→INSUFFICIENT, 5→UNSUPPORTED, 3→CONTRADICTED). BUT 101/239 claim checks used fallback →
   provisional, NOT accepted. Files: `heldout_real_v1_relabeled.jsonl`,
   `heldout_real_v1_nli_spot_audit_worklist.jsonl`, `reports/calibration/heldout_real_v1_nli_relabel_report.md`.

## 6. NEXT STEPS (prioritized for the next chat)
**A. Make the grounded-clean gate actually work (Opus, the real CLEAN-FP lever).**
   - Loosen from "zero unsupported" to a calibrated rule: suppress a low-tier retrieval-health
     failure if (no contradicted claim) AND (entailed fraction of verifiable claims ≥ threshold),
     not strict-zero-unsupported. Add `INCOMPLETE_DIAGNOSIS` + `GENERATION_IGNORE`? NO — those are
     generation-stage; think before adding. Pre-register; guard the dangerous direction (no real
     failure → CLEAN). Measure on real heldout with a CLEAN, no-rate-limit NLI run.
   - This is blocked on a reliable NLI run (see B).
**B. Make NLI runs reliable + cheap (sidekick, user machine).** Implement async/batched claim
   verification + only-verify-risky-claims (skip claims a cheap check already entails) so the A/B
   isn't 42% fallback. Re-run the A/B clean. (S1's own recommendation.)
**C. Accept real labels (human + Opus).** Adjudicate the relabel worklist after a clean (no-fallback)
   NLI re-run; lock a trustworthy heldout. Then the 0.24 number and the gate effect are measured
   against truth, not noise.
**D. Phase 3 calibration** once A–C land: per-type ECE/Brier on train/dev, calibrate confidence,
   flip `calibration_status→preliminary` ONLY for types that earn it. Report schema:
   `reports/calibration/SCHEMA.md` + `template.json`.

## 7. Discipline (non-negotiable — this is the product's trust)
Pre-register before any engine change (`taskN_prereg.md`); hard pass/fail criteria; **revert on any
regression** of protected 43/46, Calib ≥23/45, named TPs, real-heldout CLEAN; measure on the
**locked real heldout** (never the unit tests, never tune on heldout); honesty (heuristic=heuristic,
uncalibrated=uncalibrated); one task per commit; **Opus re-verifies every sidekick number** (they
have been wrong on judgment every round — great for breadth, not final calls).

## 8. The 3-actor model
- **Opus (main builder):** all engine/policy/calibration/trust-bearing code; every pre-reg + revert.
- **Antigravity (S1) & Codex (S2):** read-only measurement, data, docs; run on the USER's machine so
  they CAN call the LLM (sandbox can't). Prompts live in `reports/codex_session/NEXT_*.md` and
  `SIDEKICK_PROMPT_*.md`. They prepare; Opus verifies + commits.

## 9. Environment gotchas (the sandbox will bite you)
- Python 3.11 shim required (sandbox is 3.10), and **/tmp gets wiped between turns — recreate it**:
  ```
  mkdir -p /tmp/shim
  printf 'import datetime as _dt\nif not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc\n' > /tmp/shim/sitecustomize.py
  printf 'from tomli import *\nfrom tomli import load, loads\n' > /tmp/shim/tomllib.py
  pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
  export PYTHONPATH=/tmp/shim:src:.
  ```
- **HF, Groq, Kimi are all proxy-blocked (403) in the sandbox.** LLM/data work runs on the user's
  machine. `.env` (gitignored) holds `groq_api` and `KIMI_API_KEY`. Rotate the keys shared in chat.
- `rm -f .git/index.lock` before commits (mounted-folder lock). Commits are local; user pushes.
- `staging/raw/*` is gitignored (heldout, relabel, fresh_intake live there locally).
- `workspace_audit.py` / `harness_preflight.py` fail due to pre-existing dirty-tree/archive
  deletions — that's cleanup-state noise, not a regression. Use the protected-baseline + scorer.

## 10. Key files
- Scorer/eval: `scripts/raggov_score.py`, `scripts/eval_report.py`, `reports/calibration/`
- LLM/NLI: `scripts/groq_client.py`, `scripts/kimi_client.py`, `scripts/run_nli_heldout.py`,
  `src/raggov/analyzers/grounding/verifiers.py` (LocalNLIClaimVerifier), `support.py`
- Policy: `src/raggov/decision_policy.py` (grounded-clean gate), `decision_policy_support.py`
  (`_ENTAILMENT_GRADE_METHODS`, `grounded_clean_override`, `_require_explicit_contradiction`)
- Data: `evals/govrag_calib/staging/raw/{heldout_real_v1,heldout_real_v1_relabeled}.jsonl`,
  `scripts/validate_fresh_heldout.py`, `scripts/pull_seed_intake.py --fresh-preset`
- Guards: `scripts/check_protected_baseline.py` (43/46), `check_dataset_lock.py`, `check_taxonomy_support.py`
- Plan/state: `reports/MASTER_PLAN.md`, `reports/ARCHITECTURE_AND_HONEST_STATE.md`, this file.

## 11. The one sentence to hold onto
The infra is real (hybrid NLI tier, 3 providers, grounded-clean gate, canonical scorer, real
heldout) and everything is honest; the next win is making the grounded-clean gate fire usefully on a
reliable NLI run to drop real CLEAN-FP, then calibrate — NOT more native heuristic tweaks, NOT
optimizing the mislabeled CONTRADICTED rows.
