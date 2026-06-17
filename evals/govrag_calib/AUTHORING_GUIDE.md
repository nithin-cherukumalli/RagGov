# How to grow the calibration dataset (without hand-writing hundreds of cases)

This is the practical playbook for adding real, trustworthy cases to
`dataset_v1`. Read it before authoring. It exists because the single biggest
thing standing between RagGov and "trustworthy diagnosis" is data, not code.

## The mindset: you do NOT hand-write every case

Nobody builds an eval set by typing hundreds of examples from scratch. Real
datasets are built in **three tiers**, cheapest-to-richest:

### Tier 1 — Seed (hand-crafted anchors) · ~5-10 per failure type
A small set of carefully labelled cases per failure type, written by a human who
understands the taxonomy. This is the only heavy manual part, and it's bounded:
~5 × 14 active types ≈ **70-140 cases total**, not thousands. These are the
"anchors" everything else is checked against. Draw the *inputs* (query + context)
from real sources (below) rather than inventing them, so they reflect reality.

### Tier 2 — Induced (synthetic mutation) · scalable
Take a **clean, real** question + context + correct answer, then deliberately
break **one** pipeline stage. The label is then known *by construction*. This is
how RagGov's existing 45 cases were made, and it's how you get volume cheaply.
Per-type recipes:

| To create… | Start from a CLEAN case and… |
|---|---|
| `INSUFFICIENT_CONTEXT` | drop the chunk(s) that contain the answer |
| `STALE_RETRIEVAL` | swap the controlling chunk for an older-version copy |
| `CITATION_MISMATCH` | change the answer's citation to a doc not in retrieval |
| `UNSUPPORTED_CLAIM` | add a sentence to the answer with no support in any chunk |
| `CONTRADICTED_CLAIM` | edit the answer to state the opposite of a chunk |
| `SCOPE_VIOLATION` | replace retrieved chunks with on-domain-but-off-question ones |
| `PROMPT_INJECTION` | inject an instruction-like / exfiltration string into a chunk |
| `RETRIEVAL_DEPTH_LIMIT` | keep only low-rank near-misses; **and add `top_k`/`k_floor` to chunk/run metadata** so it's detectable from introspection, not phrasing |

Caveat: induced labels are exact but the distribution is artificial. Keep induced
cases out of (or a minority of) the **heldout** split so heldout reflects reality.

### Tier 3 — Captured (production traces) · gold standard
If you have any running RAG pipeline, log real `RAGRun`s, sample the ones that
failed, and have a human label the root cause. This is the most valuable data and
the only way to know your real-world accuracy. A handful of these is worth dozens
of induced cases.

## How other packages source data (so you can borrow real inputs)

- **RAGTruth / RAGBench / FaithBench / HaluEval** — public benchmarks with
  human-annotated RAG hallucination/faithfulness labels. Pull real query +
  passages + responses from these as raw material.
- **Natural Questions, HotpotQA, MS MARCO, TriviaQA, BEIR, TREC-RAG** — large
  open Q/A + passage corpora. Great source of *clean* cases to then induce
  failures from (Tier 2).
- **ARES** — generates synthetic queries with an LLM, trains lightweight judges,
  then uses a **small human-labelled set + PPI (prediction-powered inference)** to
  statistically correct the synthetic estimate. This is the scalable path to
  calibration and is exactly what RagGov's reserved `calibrated` mode anticipates.
- **RAGAS** — LLM-as-judge, *no* human labels. Useful but it answers "was the
  answer good", which is the question RagGov deliberately does **not** stop at.
- **RAGChecker** — claim-level entailment checking.

The practical recipe: **clean inputs from a public Q/A dataset (or your own
corpus) → induce failures (Tier 2) → anchor with a few hand-labelled seeds (Tier
1) → add real production traces when you have them (Tier 3).**

## Labelling rubric (so labels are consistent and defensible)

Pick the **first / most-specific** broken stage, in this order (mirrors the
engine's specificity rank):

1. **Security** — injection / privacy / poisoning in the context → `PROMPT_INJECTION`, `PRIVACY_VIOLATION`
2. **Retrieval** — wrong/missing/old/shallow evidence → `INSUFFICIENT_CONTEXT`, `STALE_RETRIEVAL`, `SCOPE_VIOLATION`, `RETRIEVAL_DEPTH_LIMIT`, `RETRIEVAL_ANOMALY`
3. **Grounding** — answer vs evidence mismatch → `CITATION_MISMATCH`, `UNSUPPORTED_CLAIM`, `CONTRADICTED_CLAIM`
4. **Generation** — model ignored good context → `GENERATION_IGNORE`
5. **Clean** — nothing broken → `CLEAN`

For the **heldout** split specifically: have **two** people label independently
and adjudicate disagreements; record inter-annotator agreement (Cohen's kappa).
Single-annotator labels are fine for train/dev, not for the split you report
numbers on.

## Targets to aim for

- **≥5 real cases per supported failure type** (the floor enforced by
  `check_taxonomy_support.py`); aim for 15-20 so accuracy means something.
- **Heldout ≥ ~30-50 cases, frozen**, mostly non-synthetic. The current 3-case
  heldout cannot support any headline number.
- Track the synthetic:real ratio; don't let synthetic dominate heldout.

## Workflow to add a case (enforced, safe)

1. Copy `templates/calib_case_template_v1.json` and fill it (real format).
2. Validate: `python scripts/add_calib_case.py mycase.json` (checks schema,
   immutable-ID, enum, references; refuses placeholders).
3. Append: `python scripts/add_calib_case.py mycase.json --append` (assigns the
   next `gc-0NN`, adds it to the canonical file).
4. Re-lock: `python scripts/check_dataset_lock.py --regenerate`.
5. Record it: add a `LABEL_CHANGELOG.md` entry (date, id, type, source, who).
6. Re-tier: `python scripts/check_taxonomy_support.py --regenerate`.
7. Re-run the eval and record the new accuracy **with its sample size**.

## Known cleanup item (flag for a human)

`schema.json` and `templates/calib_case_template.json` describe a *richer* format
(`claims`, `provenance`, `failure_family`, `calibration_split`, …) than the live
dataset actually uses (`split`, `label_source`, `source_type`, citations as plain
doc-id strings). Use `calib_case_template_v1.json` (matches the live format). The
old schema/template should be reconciled or retired in a future pass.
