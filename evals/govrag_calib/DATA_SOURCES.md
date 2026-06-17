# Where to source real calibration data

Curated, mapped to RagGov failure types and the three sourcing tiers in
`AUTHORING_GUIDE.md`. **Check each dataset's license before redistributing**; for
induced (Tier 2) cases you transform the inputs, which is usually fine, but don't
commit raw third-party corpora into this repo without confirming terms.

## Starter pack (if you only grab 4)

1. **RAGTruth** — real, human span-level hallucination labels → grounding seeds.
2. **HotpotQA** — clean Q + passages with supporting-fact labels → induce retrieval/grounding failures.
3. **ALCE** — citation-quality benchmark → CITATION_MISMATCH seeds.
4. **deepset/prompt-injections** — adversarial strings → PROMPT_INJECTION.

That alone can produce 100+ cases across most supported types.

## Tier 1 / 3 — real labelled RAG failure corpora (best seeds)

| Source | Where | Feeds these failure types |
|---|---|---|
| **RAGTruth** (query + docs + answer + span hallucination labels; "Conflict" = contradiction, "Baseless Info" = unsupported) | https://github.com/ParticleMedia/RAGTruth · HF mirror: https://huggingface.co/datasets/jakobsnel/RAGTruth_Xtended | `UNSUPPORTED_CLAIM`, `CONTRADICTED_CLAIM` |
| **RAGBench** (100k, TRACe labels: relevance/utilization/adherence/completeness; industry domains incl. user manuals) | https://huggingface.co/datasets/rungalileo/ragbench (also `galileo-ai/ragbench`) | `INSUFFICIENT_CONTEXT`, `UNSUPPORTED_CLAIM`, `SCOPE_VIOLATION` |
| **ALCE** (ASQA / QAMPARI / ELI5; citation correctness) | https://github.com/princeton-nlp/ALCE | `CITATION_MISMATCH`, `POST_RATIONALIZED_CITATION` |
| **HaluEval** (QA hallucination labels) | https://github.com/RUCAIBox/HaluEval | `UNSUPPORTED_CLAIM`, `CONTRADICTED_CLAIM` |

## Tier 2 — clean Q/A + passages (induce failures from these)

| Source | Where (HF id) | Good for inducing |
|---|---|---|
| **HotpotQA** (multi-hop, supporting facts labelled — you know which chunk is needed) | `hotpot_qa` | `INSUFFICIENT_CONTEXT`, `RETRIEVAL_DEPTH_LIMIT` (drop/derank the needed chunk) |
| **Natural Questions** (real Google queries + Wikipedia) | `google-research-datasets/natural_questions` | `STALE_RETRIEVAL`, `SCOPE_VIOLATION`, `UNSUPPORTED_CLAIM` |
| **MS MARCO** (real Bing queries + passages) | `ms_marco` | retrieval-stage failures generally |
| **SQuAD v2** (context + Q/A incl. unanswerable) | `rajpurkar/squad_v2` | `INSUFFICIENT_CONTEXT`, should-not-answer / `CLEAN` |
| **BEIR** (many heterogeneous retrieval corpora) | https://github.com/beir-cellar/beir | `SCOPE_VIOLATION`, `RETRIEVAL_ANOMALY`, depth |

## Security / poisoning

| Source | Where | Feeds |
|---|---|---|
| **deepset/prompt-injections** | https://huggingface.co/datasets/deepset/prompt-injections | `PROMPT_INJECTION` |
| **PoisonedRAG** (corpus-poisoning attack texts for RAG) | https://github.com/sleeepeer/PoisonedRAG | `SUSPICIOUS_CHUNK` (poisoning) |
| **AI4Privacy PII masking** | `ai4privacy/pii-masking-200k` | `PRIVACY_VIOLATION` |

## Domain-specific (RagGov's apparent focus: govt orders + software)

| Source | Where | Note |
|---|---|---|
| **TechQA** (IBM technical support QA) | search HF `techqa` / IBM release | matches the existing `software` domain cases |
| **CUAD** (legal/contract clause QA) | https://www.atticusprojectai.org/cuad · HF `cuad` | policy/legal `SCOPE_VIOLATION`, `CITATION_MISMATCH` |
| **PrivacyQA / PolicyQA** | search HF | policy domain + `PRIVACY_VIOLATION` |
| **Your own corpus** (e.g. Indian G.O. / gazette docs from data.gov.in) | your pipeline / data.gov.in | **highest value** — Tier 3 production traces in your real domain |

## How to hand it to me

Once you've pulled any of these, the fastest hand-off is:
- a small JSONL/CSV with `query`, the retrieved/passage `text`(s), and the
  reference answer, **or**
- the raw dataset files in a folder you point me at.

I'll then help script the Tier-2 induction (clean → failure variants) into the
live case format and run them through `scripts/add_calib_case.py`. I will **not**
invent gold labels — for real (non-induced) cases the label is yours to confirm.

(Links verified June 2026 where marked with a full URL; HF ids may need the exact
namespace confirmed at download time.)
