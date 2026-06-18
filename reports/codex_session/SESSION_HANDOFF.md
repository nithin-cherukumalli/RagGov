# RagGov — Session Handoff (read this first)

**Last updated:** 2026-06-17
**Purpose:** everything a new session needs to continue without re-deriving context.
Read this, then `FAILED_APPROACHES.md`, then `foundation_stabilization_plan.md`.

---

## 1. What RagGov is, in one line

A diagnosis layer for RAG pipelines: given a `RAGRun` (query, retrieved chunks,
answer, citations) it returns a structured `Diagnosis` naming the primary failure,
the stage, security risk, and recommended fix. It is **not** an answer-quality
scorer — it does root-cause attribution.

## 2. Honest maturity (do not overstate)

- **Foundation: done.** Dataset locked + governed, CI enforces guardrails,
  taxonomy honestly tiered, data pipeline built.
- **Product quality: early.** Calib scored primary **0.511** (45 live cases);
  **~0.25–0.31 on fresh data** (generalization probe). The engine **over-fires**
  (flags healthy answers). Only **3 of 25** failure types have ≥5 real cases.
  Heldout is **3 cases** (not yet meaningful).
- Roughly **one-third of the way** to "trustworthy." Remaining work is concentrated
  in (a) engine precision and (b) real data growth.
- `calibration_status: not_calibrated`, `production_gating_eligible: False`. Keep it
  that way until earned.

## 3. The discipline (non-negotiable — this is why the project is trustworthy)

1. **Pre-registration before any engine code change.** Write a `taskN_prereg.md`
   with a hypothesis + hard pass/fail criteria BEFORE editing code.
2. **Hard criteria, revert on failure.** If protected baseline, Calib, or a named
   true-positive regresses → revert, keep the prereg as the record. Two reverts
   beat one false "fix". (See `FAILED_APPROACHES.md` for 6 worked examples.)
3. **No query-string / passage-text heuristics** for pipeline failure modes. Read
   structured metadata (`run.metadata`, `chunk.metadata`, records, profiles).
4. **Analyzer change ⇒ check the decision policy too.** Firing in isolation ≠ being
   selected by the engine.
5. **Never edit a golden label** to make code pass. Flag for human adjudication.
6. **One task per commit. Never bundle.**
7. **Measure on fresh data, not just fixtures** (the 0.62→0.31 gap only showed on
   the probe).

## 4. Environment gotchas (the sandbox will bite you otherwise)

- **Python:** code needs 3.11 (`datetime.UTC`, `tomllib`); sandbox default is 3.10.
  A local-only shim makes it run (NOT committed):
  ```bash
  mkdir -p /tmp/shim && cat > /tmp/shim/sitecustomize.py <<'EOF'
  import datetime as _dt
  if not hasattr(_dt,"UTC"): _dt.UTC=_dt.timezone.utc
  EOF
  cat > /tmp/shim/tomllib.py <<'EOF'
  from tomli import *
  from tomli import load, loads
  EOF
  pip install -q pytest pydantic typer rich numpy httpx tomli --break-system-packages
  ```
  Then always run with `PYTHONPATH=/tmp/shim:src:.`
- **Hugging Face is BLOCKED in the sandbox** (proxy 403). The data puller
  (`scripts/pull_seed_intake.py`) must be run on the user's machine. Do not try to
  route around it.
- **git index.lock** sometimes persists on the mounted folder — `rm -f .git/index.lock`
  before commits. File deletion needed one-time approval already granted.
- **git identity** is set locally to `ch Nithin <nithin.ch@techbharat.ai>`.
- **Commits are local only** — the user pushes (`git push origin main`) themselves.

## 5. How to measure (the guard for every change)

```bash
export PYTHONPATH=/tmp/shim:src:.
python scripts/check_protected_baseline.py        # MUST stay 41/46 GREEN
python scripts/check_dataset_lock.py              # dataset unchanged w/o changelog
python scripts/check_taxonomy_support.py          # tiers in sync with data
```
For Calib + generalization probe, the project's own `evaluate_govrag_calib.py` is
**stale** (validates the old rich schema, crashes on live format — a known drift).
Use a direct scorer instead: build a `RAGRun` per case
(`RetrievedChunk(chunk_id,text,source_doc_id=doc_id,score)`, `final_answer=answer`,
`cited_doc_ids=[citations]`), call `diagnose()`, compare `primary_failure`.
- Calib scored split = `govrag_calib_150.jsonl` rows with `split in {train,dev,heldout}`
  minus the 7 placeholders → baseline **23/45 = 0.511**.
- Probe = `evals/govrag_calib/staging/raw/induced_candidates.jsonl` (gitignored;
  regenerate with the puller + `induce_cases.py` if absent).

## 6. What was accomplished (this + prior sessions)

- **Tasks 6–9** committed (CLI why_block, LangChain/LlamaIndex examples).
- **Task 10** red-test triage: 17 stale tests fixed, 3 real bugs filed (14/15/16),
  3 marked `xfail(strict)`.
- **Tasks 3/4/5-v2** reformulated then found **blocked** — case IDs in their specs
  no longer match the relabeled dataset; signals (top_k/reranker) absent. See
  `v2_feasibility_blocker.md`. Do NOT implement as written.
- **P0** foundation: `DATASET_MANIFEST.json` + `check_dataset_lock.py`,
  `DATASET_GOVERNANCE.md`, `LABEL_CHANGELOG.md`, `dataset_quality_audit.md`.
- **P1** `taxonomy_support_tiers.json` + `check_taxonomy_support.py` (3 supported /
  9 thin / 13 unsupported).
- **P3** packaging fix (requires-python >=3.11).
- **CI** (`.github/workflows/ci.yml`) enforces all guardrails.
- **Data pipeline**: `pull_seed_intake.py` (RAGTruth/HotpotQA/ALCE/prompt-injections),
  `induce_cases.py` (clean→CLEAN+variants, labelled→mapped), `add_calib_case.py`
  (validate+append with immutable ids). `AUTHORING_GUIDE.md`, `DATA_SOURCES.md`,
  `staging/README.md`.
- **Generalization probe** (`generalization_probe_v1.md`): engine 0.31 on fresh
  data; filed Tasks 17 (CLEAN over-firing) and 18 (injection not promoted).
- **Task 17 LANDED** (v3, after 2 reverts): relative-recency gate on STALE
  escalation; STALE false positives −60% (5→2), zero regression.

## 7. NEXT SCOPE OF WORK (prioritized)

### Priority 1 — Engine precision (biggest lever for "useful"; ~4–6 fixes)
Over-firing is what makes the tool untrustworthy. Each is a pre-registered,
probe-measured change like Task 17-v3.
- **Task 18 — PROMPT_INJECTION promotion** (recommended first; likely cleanest).
  Injection is *detected* but not promoted to primary (1/10 on probe). Fix the
  decision policy so the security stage out-ranks downstream grounding symptoms.
  Add an end-to-end test (injection → primary PROMPT_INJECTION; clean → no false
  security primary).
- **INCONSISTENT_CHUNKS over-firing** (8/30 on CLEAN). Find the analyzer; require
  real contradiction evidence between chunks, not surface signals.
- **CHUNKING_BOUNDARY_ERROR over-firing** (5/30 on CLEAN, esp. long ALCE passages).
  `ParserValidationAnalyzer` — tighten precision (Task 1 already touched it once).
- **Citation gate over-eagerness** — UNSUPPORTED_CLAIM cases routed to
  CITATION_MISMATCH (overlaps Task 16 / Task 3-v2 specificity work).
- Pre-existing real bugs already filed: **Task 14** (STALE over-promotes on
  irrelevant source), **Task 15** (incomplete-answer stage attribution),
  **Task 16** (case-41 specificity). All are `xfail` tests today.

### Priority 2 — Real data growth (the long pole; needs the user)
- User runs `pull_seed_intake.py` on their machine → drop JSONL in `staging/raw/`.
- Merge real RAGTruth/ALCE cases via `add_calib_case.py` (review labels first;
  RAGTruth subtype contradicted-vs-unsupported is heuristic — audit `notes`).
- Grow each supported type to ≥5 real cases; build a real **30–50 case heldout**,
  double-labelled. Update `LABEL_CHANGELOG.md` + regenerate manifest/tiers each time.
- The 145 induced candidates stay a **probe**, not merged wholesale (avoid synthetic
  domination).

### Priority 3 — Taxonomy honesty
- The **13 zero-data failure types** (incl. `RERANKER_FAILURE`): get data or
  quarantine from public claims. `check_taxonomy_support.py` already flags them.

### Priority 4 — Calibration & release (only after accuracy rises)
- Calibration pack: 5 seeds, confidence intervals, calibration curve.
- Flip `calibration_status` → `preliminary`; align README numbers; tag `v0.1-beta`.

### The bar / definition of done
Generalization accuracy **≥ ~0.70 on a real ~30–50-case heldout**, low CLEAN
false-positive rate, every advertised type data-backed. Track generalization
accuracy as THE number (≈0.25–0.31 today).

## 8. Key files map

- Discipline/history: `FAILED_APPROACHES.md`, `task*_prereg.md`, `task*_result.md`
- Plan: `foundation_stabilization_plan.md`, `NEXT_TASKS.md` (Tasks 14–18 in Phases F/G)
- Dataset: `evals/govrag_calib/govrag_calib_150.jsonl` (canonical, locked),
  `DATASET_MANIFEST.json`, `DATASET_GOVERNANCE.md`, `LABEL_CHANGELOG.md`,
  `taxonomy_support_tiers.json`
- Data pipeline: `scripts/pull_seed_intake.py`, `scripts/induce_cases.py`,
  `scripts/add_calib_case.py`, `evals/govrag_calib/{AUTHORING_GUIDE,DATA_SOURCES}.md`,
  `staging/README.md`
- Guards: `scripts/check_protected_baseline.py`, `check_dataset_lock.py`,
  `check_taxonomy_support.py`
- Engine: `src/raggov/engine.py`, `src/raggov/decision_policy.py`,
  `src/raggov/analyzers/**`, `src/raggov/models/diagnosis.py` (FailureType enum)

## 9. Open caveats / traps

- `evaluate_govrag_calib.py` and `schema.json` describe a **richer/older** case
  format than the live dataset uses — both are drifted; don't trust them as-is.
- The dataset was silently relabeled between sessions once (the cause of the v2
  block). The lock check now prevents recurrence — keep using it.
- Calib "0.62" / heldout "0.733" numbers quoted in older docs are **not reproducible**
  on the current data; the honest current numbers are 0.511 / ~0.25–0.31.
- 2 STALE false positives remain after Task 17 (documented residual).
