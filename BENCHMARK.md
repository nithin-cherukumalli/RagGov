# RagGov Benchmark — Real Heldout (locked gold)

Reproducible: `PYTHONPATH=src:. python scripts/gen_benchmark_report.py`. Native (heuristic)
mode, no LLM. Evaluated against `heldout_real_v1_gold.jsonl` — 75 real RAG runs from RAGTruth /
HotpotQA / ALCE, relabeled into the 6-class v1 taxonomy via a two-stage protocol (strong-LLM
proposal -> independent chunk-by-chunk adjudication -> human/maintainer verification).

## Headline (native, no LLM)
- **CLEAN false-positive rate: 22/46 = 0.48** (lower is better — faithful answers wrongly flagged)
- **Failure detection rate: 18/29 = 0.62** (real failures that get flagged)
- **Exact-class accuracy: 35/75 = 0.47** (correct v1 class incl. CLEAN)

## Gold label distribution

- CLEAN: 46
- UNSUPPORTED_CLAIM: 25
- INSUFFICIENT_CONTEXT: 4

## Per-class exact accuracy
| class | correct / total |
|-------|-----------------|
| CLEAN | 24/46 |
| INSUFFICIENT_CONTEXT | 1/4 |
| UNSUPPORTED_CLAIM | 10/25 |

## Honesty notes (read these)
- The engine is **uncalibrated heuristics**, not an ML classifier; confidence is advisory.
- The benchmark is small (75 rows) and the labels are maintainer-adjudicated, not a community
  gold standard. Treat numbers as directional, not authoritative.
- The original migrated labels were mislabeled (25 rows tagged CONTRADICTED; 0 held up under
  re-adjudication). Fixing the benchmark was step one — see `reports/codex_session/`.
- CLEAN false-positive reduction (0.65 -> 0.48 on this set) came from removing analyzers/
  signals that had zero true-positives on real data, each change guarded against a protected
  regression suite. No tuning on the heldout.
