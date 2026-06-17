# Seed intake — deliver raw source material here

You deliver a **normalized seed-intake JSONL** (one object per line) into
`evals/govrag_calib/staging/raw/`. That folder is **gitignored** — full
third-party passage text is never committed. I transform these into the live case
format (inducing Tier-2 failures where needed) and commit only short per-case
excerpts with source attribution.

A puller **script that produces this JSONL** is preferred (reproducible); a static
hand-exported JSONL is also fine.

## Seed-intake schema (one JSON object per line)

```json
{
  "source_dataset": "ragtruth | hotpotqa | alce | prompt_injections | <other>",
  "source_id": "original id from that dataset",
  "domain": "wiki | software | finance | policy | health | ...",
  "query": "the question asked",
  "passages": [
    {"doc_id": "d1", "text": "FULL passage text (required)", "rank": 1, "score": null}
  ],
  "reference_answer": "gold answer (clean sources) OR the model's answer (labelled sources)",
  "kind": "clean | labelled_failure",
  "source_label": null,
  "supporting_doc_ids": ["d1"],
  "license": "the source dataset license string",
  "notes": ""
}
```

Field rules:
- **kind = "clean"** → a correct answer fully supported by `passages`. I induce
  failure variants from it (and keep one as a `CLEAN` case). For these, fill
  `supporting_doc_ids` with the doc(s) that actually contain the answer — that's
  what lets induction be precise (e.g. HotpotQA supporting facts).
- **kind = "labelled_failure"** → the source already says this answer is wrong;
  set `source_label` to one of: `unsupported`, `contradicted`, `citation`,
  `prompt_injection`, `poisoning`, `privacy`. I map it to the RagGov failure type.
  `reference_answer` here is the (flawed) model answer being judged.
- `passages` must contain the **full** text and a stable `doc_id` per passage.
- `license` is required so committed derivatives can be attributed.

## Target counts for the starter batch (~60-75 raw items → ~150 cases)

| source | kind | items | yields (after induction) |
|---|---|---|---|
| **RAGTruth** | labelled_failure | 30 (≈15 baseless→unsupported, ≈15 conflict→contradicted) | ~30 real grounding cases |
| **HotpotQA** | clean | 20 (with `supporting_doc_ids`) | ~80 (1 clean + 3 induced each: insufficient / depth / scope/contradicted) |
| **ALCE** (ASQA/ELI5) | labelled_failure or clean | 10 | ~10-20 citation cases |
| **deepset/prompt-injections** | labelled_failure (`prompt_injection`) | 10 strings | ~10 injection cases (I splice into clean contexts) |

That comfortably clears the ≥5 floor for grounding, retrieval, citation, and
security types, and leaves enough to build a ~30-case heldout.

## Hand-off

Drop the JSONL (and/or the puller script) in `staging/raw/` and tell me. I'll run
induction + `add_calib_case.py`, then we re-lock, log, and re-run calibration.
I will not invent gold labels: induced labels are by-construction; labelled-source
labels come from the source; any genuinely ambiguous real case I'll flag for your
adjudication rather than guess.
