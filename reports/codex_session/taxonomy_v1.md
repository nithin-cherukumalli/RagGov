# RagGov v1 — reduced failure taxonomy

The engine defines ~25 `FailureType`s. No labeler (human or LLM) and no heuristic engine can be
reliable across 25 fine-grained buckets, and trying to was a big source of the CLEAN false-positive
problem. v1 commits to **6 buckets** that are (a) distinct, (b) reliably labelable by a strong
model + human, and (c) realistically detectable. Everything we ship/measure uses these.

## The 6 v1 labels
| v1 label | stage | meaning |
|----------|-------|---------|
| `CLEAN` | — | Answer faithful + adequately supported by the chunks. |
| `PROMPT_INJECTION` | SECURITY | Query/chunk tries to hijack the model, or the answer obeys injected instructions. |
| `STALE_RETRIEVAL` | RETRIEVAL | Retrieved chunks are an outdated/superseded version; answer reflects stale info. |
| `INSUFFICIENT_CONTEXT` | RETRIEVAL | Chunks don't contain enough to answer; answer is incomplete/guesses. |
| `UNSUPPORTED_CLAIM` | GROUNDING | Answer asserts facts in NONE of the chunks (fabrication), not directly contradicting. |
| `CONTRADICTED_CLAIM` | GROUNDING | Answer directly conflicts with the chunks. |

## Precedence (when multiple apply)
`PROMPT_INJECTION` > `CONTRADICTED_CLAIM` > `UNSUPPORTED_CLAIM` > `STALE_RETRIEVAL` >
`INSUFFICIENT_CONTEXT` > `CLEAN`. Security first; then "answer is wrong" before "retrieval was
thin"; CLEAN only when nothing else fires.

## How the legacy 25 map down (for migrating labels + engine output)
- SCOPE_VIOLATION, RETRIEVAL_ANOMALY, RETRIEVAL_DEPTH_LIMIT, EMBEDDING_DRIFT, RERANKER_FAILURE →
  **INSUFFICIENT_CONTEXT** (all "retrieval didn't bring the right/enough evidence").
- CITATION_MISMATCH, POST_RATIONALIZED_CITATION, INCONSISTENT_CHUNKS → **UNSUPPORTED_CLAIM** (the
  answer's support is not actually in the cited/retrieved evidence). NOTE: INCONSISTENT_CHUNKS is
  already demoted to primary-ineligible (pure-FP); it does not surface as a primary at all.
- SUSPICIOUS_CHUNK → **PROMPT_INJECTION** (security-stage).
- PRIVACY_VIOLATION → kept as a SECURITY sub-signal but OUT of the v1 primary taxonomy for now
  (too rare to label/measure reliably; revisit once volume exists).
- Parser/chunking/structure types (TABLE_STRUCTURE_LOSS, HIERARCHY_FLATTENING, METADATA_LOSS,
  PARSER_STRUCTURE_LOSS, CHUNKING_BOUNDARY_ERROR) → not in v1 primary taxonomy; advisory only.
- LOW_CONFIDENCE, INCOMPLETE_DIAGNOSIS, CLAIM_EXTRACTION_FAILED, GENERATION_IGNORE, UNKNOWN →
  meta/operational, never a product-facing primary; map to the best of the 6 or to CLEAN.

## Why this is the right move for credibility
A tool that reliably classifies 6 well-defined RAG failure modes with calibrated confidence is
honest, demoable, and starrable. "Detects 25 failure types" that it cannot actually measure is the
opposite. We can always grow the taxonomy back once each new type earns its place on real data.
