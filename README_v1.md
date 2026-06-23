<!-- Proposed v1 README (reset positioning). Review, then replace README.md when happy. -->

# RagGov — a failure-attribution & governance layer for RAG

**RagGov diagnoses *why* a RAG answer went wrong — which stage failed (retrieval, grounding, or
security) — with a full decision trace and an honest, uncalibrated confidence.** It's built for
**compliance-sensitive, on-prem deployments** where "the model said something wrong" isn't good
enough — you need to know *where* the pipeline broke and *why*, with an audit trail.

It is **not** an answer-quality scorer and it does **not** call an LLM at inference time — it runs
fully offline, which is the point for air-gapped and governed environments.

```bash
pip install -e .
raggov diagnose examples/quickstart/unsupported_answer.json
```
```
Primary failure   UNSUPPORTED_CLAIM
Root cause stage  GROUNDING
Security risk     NONE
Why this verdict? The answer asserts facts not present in any retrieved chunk.
```

## What it detects (v1 taxonomy)
Six classes, chosen because they are reliably labelable and detectable on real data:

| class | stage | meaning |
|-------|-------|---------|
| `CLEAN` | — | answer is faithful and supported by the retrieved chunks |
| `PROMPT_INJECTION` | security | a chunk/query tries to hijack the model, or the answer obeys it |
| `STALE_RETRIEVAL` | retrieval | retrieved chunks are an outdated/superseded version |
| `INSUFFICIENT_CONTEXT` | retrieval | chunks don't contain enough to answer |
| `UNSUPPORTED_CLAIM` | grounding | answer asserts facts in none of the chunks (fabrication) |
| `CONTRADICTED_CLAIM` | grounding | answer directly conflicts with a chunk |

## Honest evaluation (this is the part most repos skip)
Numbers on a **maintainer-adjudicated real heldout** (75 runs from RAGTruth / HotpotQA / ALCE),
native/offline mode — see [`BENCHMARK.md`](BENCHMARK.md), reproducible with one command:

- **CLEAN false-positive rate: 0.48** (faithful answers wrongly flagged — the metric that matters
  most for trust; lower is better)
- **Failure detection rate: 0.62** (real failures that get flagged)
- **Exact-class accuracy: 0.47**

### The benchmark story (why you can trust the above)
The first thing this project found was that **its own migrated benchmark was mislabeled** — 25 rows
tagged `CONTRADICTED_CLAIM`, of which **zero** held up under re-adjudication. So step one was fixing
the ruler: a two-stage labeling protocol (strong-LLM proposal → independent chunk-by-chunk
adjudication → maintainer verification) produced a locked gold set. Only then were the engine
numbers measured. Every precision improvement since (CLEAN-FP **0.65 → 0.48**) came from removing
analyzers/signals that had **zero true-positives on real data**, each change gated against a
protected regression suite. **No tuning on the heldout.**

## What it does NOT do (read this before using)
- It is **uncalibrated heuristics**, not a trained classifier. Confidence is advisory, not a probability.
- It is **not production-gating-ready** — use it as a diagnostic/observability aid, not an automated blocker.
- The benchmark is **small (75 rows)** and maintainer-labeled. Numbers are directional.
- `CONTRADICTED_CLAIM` detection in offline mode is weak by design (the heuristic can't reliably tell
  "contradicted" from "unsupported" without an entailment model — and the bundled local NLI option
  was measured to *hurt*, so it's off by default).

## How it works
A pipeline of stage-specific analyzers (retrieval health, grounding/claim-verification, security)
each emit typed, tiered signals; a decision policy selects the primary failure with an auditable
trace ("why this verdict / also considered"). Optional LLM/NLI tiers exist but are **off by default**
and not required. See `reports/codex_session/RESET_ROADMAP.md` and `taxonomy_v1.md` for design notes.

## Roadmap
- [x] Fix the benchmark (locked, adjudicated gold)
- [x] Reduce to a reliable 6-class taxonomy
- [x] Precision pass: CLEAN-FP 0.65 → 0.48, zero regressions
- [ ] Per-class confidence calibration (preliminary)
- [ ] Grounding-stage recall (UNSUPPORTED/STALE) without LLM dependence
- [ ] Hosted governance/observability dashboard (paid layer; core stays OSS)

## Status
Research preview / `v0.1`. Honest, offline, and improving in small guarded increments. Issues and
PRs welcome — especially real RAG failure cases for the benchmark.
