# RagGov — Reset Roadmap (the 6-week plan to a credible flagship)

**Goal:** turn RagGov from a sprawling, accuracy-chasing research engine into a focused, honest,
demoable open-source tool that (a) a senior engineer / international hiring manager respects, (b)
earns GitHub stars, and (c) has a clear paid-feature path later. Written 2026-06-22.

## 0. The one-line repositioning (this changes everything)
Not "a generic RAG failure-diagnosis engine" (crowded: RAGAS, TruLens, DeepEval — and you'd compete
on accuracy you can't yet prove). Instead:

> **RagGov — a governance & failure-attribution layer for RAG in compliance-sensitive, on-prem
> deployments.** It tells you *which stage failed and why* (retrieval / grounding / security), with
> calibrated confidence and a full audit trail.

That niche is uncrowded and it's *yours* — it matches lived experience (AP government, air-gapped,
audit-grade RAG). Lean the README, demo, and examples into governance/security/auditability.

## 1. Honest current state (so we don't kid ourselves)
- Real heldout accuracy: **0.36** (was 0.24) after demoting one pure-FP analyzer. CLEAN-FP **0.58**.
- The heldout's 25 "failure" rows are **mislabeled** → we can't yet measure failure recall at all.
- The engine is ~25 uncalibrated heuristics; at least one had zero true-positives. Likely more dead weight.
- The entailment-tier (NLI) direction is a proven NO-GO with the models available.

## 2. The plan (≈6 weeks at ~10–15 hrs/week)

### Phase A — Fix the ruler (Week 1) ← we are here, tooling is built
- Run `scripts/label_heldout_gold.py` with your Kimi key (strong model as a one-time labeler, NOT
  the runtime engine). It proposes a v1-taxonomy label + rationale + confidence per row and writes
  a worklist of disagreements/low-confidence rows.
- **Human spot-check the worklist** (every disagreement + low-confidence). 75 rows is an evening.
- Promote the adjudicated labels to a LOCKED gold heldout. Re-run the scorer against it.
- Deliverable: a heldout you can trust. Until this exists, every other number is noise.

### Phase B — Adopt the reduced v1 taxonomy (Week 1–2)
- Collapse the engine's ~25 primary types to the 6 in `taxonomy_v1.md` (map legacy → v1 at the
  output boundary; keep internals, change what the product reports/measures).
- Re-score on the locked gold heldout to get honest per-type precision/recall for the 6.

### Phase C — Keep/cut the analyzers (Week 2–3)
- For each analyzer, measure TP vs FP on Calib + gold heldout (the triage method already used to
  kill InconsistentChunksAnalyzer). Rule: an analyzer that can't show net-positive on real data is
  demoted to advisory (primary-ineligible) — exactly the `_PRIMARY_INELIGIBLE_ANALYZERS` mechanism.
- Target: CLEAN-FP ≤ ~0.35 and a real, defensible recall number on the 6 types.

### Phase D — Calibrate confidence (Week 3–4)
- For the surviving types, fit per-type confidence (ECE/Brier) on train/dev so each diagnosis ships
  an honest confidence. Flip `calibration_status` to `preliminary` ONLY for types that earn it.
- This is the difference between "a heuristic guessed" and "82% confidence, here's why".

### Phase E — Package for humans (Week 4–5)
- README: 30-second problem statement, an animated demo GIF, `pip install raggov`, 3 copy-paste
  examples, and an explicit **"what it does / does NOT do"** section (honesty = credibility).
- One-command demo (`raggov diagnose example.json`) + a tiny hosted/Colab demo.
- Governance framing throughout: audit trail, on-prem, no-data-leaves-network, confidence + citations.
- CI green, a clean test subset, a LICENSE, CHANGELOG, and a short ARCHITECTURE.md.

### Phase F — Distribute (Week 5–6, ongoing)
- Write-up post: "I built a RAG failure auditor, found my own benchmark was mislabeled, and what I
  learned." The honesty + the governance angle is the hook.
- Show HN / r/LocalLLaMA / r/MachineLearning / LinkedIn. Tag the RAG-eval conversation.
- Stars follow real pain + dead-simple trial + a good story — not more analyzers.

## 3. The paid-feature path (later, only after v1 lands)
- Hosted dashboard over the OSS core; batch auditing of production RAG logs; governance/compliance
  reports (the thing enterprises actually pay for); private connectors. Keep the core OSS; sell the
  governance/observability layer on top.

## 4. What NOT to do (lessons already paid for)
- No more analyzers chasing accuracy on the broken benchmark.
- No entailment/NLI tier with free rate-limited or small local models (proven NO-GO).
- Don't optimize against the mislabeled CONTRADICTED rows.
- One change per increment, pre-registered, measured on the LOCKED gold heldout, reverted on regression.

## 5. Immediate next action
Run on your Mac (Kimi key in `.env`):
```
PYTHONPATH=src:. python scripts/label_heldout_gold.py --limit 5     # sanity check first
PYTHONPATH=src:. python scripts/label_heldout_gold.py               # full 75-row pass
```
Then adjudicate `staging/raw/heldout_real_v1_gold_worklist.jsonl`. That single evening unblocks the
entire rest of the roadmap.
