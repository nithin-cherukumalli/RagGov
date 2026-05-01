# Claim-Grounding Evaluation Harness

## Why this dataset exists

GovRAG verifies whether generated answer claims are supported by retrieved
document chunks. Today, that verification is done by
`HeuristicValueOverlapVerifier` — a heuristic that compares numeric and
keyword overlap between claim text and candidate evidence chunks.

Heuristics are fast and interpretable, but they are **not calibrated**. Without
a labeled dataset we cannot answer:

- What fraction of entailed claims is the verifier correctly marking as
  entailed?
- At what rate does it silently accept fabricated or contradicted claims?
- How does the heuristic compare to a structured LLM verifier or a future NLI
  model?

This harness exists to answer those questions. It is **not** a benchmark for
comparing GovRAG against other RAG systems. It is a **project calibration tool**
for comparing different verifier implementations against the same gold labels.

---

## Why false-pass rate matters more than false-fail rate

In a high-trust RAG system serving government policy queries, the two failure
modes have very different consequences:

| Failure mode | Definition | Consequence |
|---|---|---|
| **False pass** | Gold label = unsupported or contradicted, predicted = entailed | A wrong answer is silently accepted and potentially acted on |
| **False fail** | Gold label = entailed, predicted = unsupported/contradicted | A correct answer is flagged for review — annoying but safe |

A false pass in a policy context can mean a citizen receives incorrect
information about their entitlements, a deadline, or a legal obligation. A
false fail just means the system is overly cautious — it adds review friction
but does not cause harm.

**The primary calibration goal is to minimize false-pass rate**, even at the
cost of a higher false-fail rate.

---

## Dataset schema

Each line in `seed_cases.jsonl` is a JSON object conforming to
`schema.py::ClaimGroundingCase`. Key fields:

| Field | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique identifier (`cgc-001`, etc.) |
| `query` | `str` | Original user query |
| `answer` | `str` | Full generated answer |
| `claim_text` | `str` | The specific extracted claim being evaluated |
| `retrieved_chunks` | `list[ChunkRecord]` | Chunks retrieved for this query |
| `cited_doc_ids` | `list[str]` | Document IDs cited in the answer |
| `gold_label` | `entailed \| unsupported \| contradicted` | Human-verified label |
| `gold_supporting_chunk_ids` | `list[str]` | Chunks that genuinely support the claim |
| `gold_contradicting_chunk_ids` | `list[str]` | Chunks whose content contradicts the claim |
| `claim_type` | enum | `numeric`, `date_or_deadline`, `go_number`, etc. |
| `atomicity_status` | enum | `atomic`, `compound`, `unclear` |
| `error_type` | enum or null | Root cause when `gold_label ≠ entailed` |
| `notes` | `str` or null | Free-text annotation |

### Error type vocabulary

| Value | Meaning |
|---|---|
| `retrieval_miss` | The supporting chunk was never retrieved |
| `context_ignored` | The chunk was retrieved but the model ignored it |
| `value_error` | A number, date, or identifier is wrong |
| `stale_source_error` | The answer relies on outdated information |
| `citation_error` | The wrong document was cited for a true claim |
| `generation_hallucination` | The model invented a fact not in any chunk |
| `insufficient_context` | Not enough context to support or refute |

---

## How to add cases

1. Write a new JSON object following the schema.
2. Give it a unique `case_id` (e.g. `cgc-026`).
3. Set `gold_label` to `entailed`, `unsupported`, or `contradicted`.
4. If `gold_label ≠ entailed`, set `error_type` to the closest root cause.
5. Add the line to `seed_cases.jsonl`.
6. Run the harness to verify it loads without errors:

```bash
python evals/claim_grounding/run_eval.py
```

### Coverage guidelines

Aim for at least:
- 40% `entailed` cases (correct answers)
- 30% `unsupported` cases (information not in chunks)
- 30% `contradicted` cases (information in chunks but answer is wrong)

Within the failure cases, try to cover each `error_type` at least twice.

---

## How to run the evaluation

```bash
# Basic run (prints report to stdout):
python evals/claim_grounding/run_eval.py

# Save outputs:
python evals/claim_grounding/run_eval.py \
    --json-out evals/claim_grounding/reports/latest.json \
    --md-out evals/claim_grounding/reports/latest.md

# Against a custom dataset:
python evals/claim_grounding/run_eval.py --dataset path/to/my_cases.jsonl
```

---

## Metrics reference

| Metric | Formula | Notes |
|---|---|---|
| `overall_accuracy` | correct / total | Macro accuracy across all labels |
| `false_pass_rate` | FP(entailed) / N(unsupported+contradicted) | **Primary risk metric** |
| `false_fail_rate` | FN(entailed) / N(entailed) | Over-rejection rate |
| `contradiction_detection_rate` | TP(contradicted) / N(contradicted) | Ability to catch conflicting facts |
| `evidence_chunk_recall` | ∩(predicted_support, gold_support) / gold_support | How well the verifier locates supporting chunks |
| `fallback_rate` | fallback_used / total | Fraction using fallback heuristic |

Per-label precision, recall, and F1 are also computed for `entailed`,
`unsupported`, and `contradicted`.

---

## This is not a benchmark

This harness is a **calibration tool**, not a benchmark:

- The seed cases are synthetic and hand-written. They do not represent a
  statistically valid sample of production queries.
- The cases are designed to stress specific failure modes, not to reflect the
  distribution of real GovRAG queries.
- Do not use these metrics to compare GovRAG against other RAG systems.
- Use these metrics to compare **verifier implementations** and to detect
  regression when the verifier logic changes.

When the system is stable, a separate benchmark dataset with real queries and
blind annotation should be created.
