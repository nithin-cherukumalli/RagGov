# Citation Faithfulness Analyzer V0

`CitationFaithfulnessAnalyzerV0` is a practical-approximation analyzer that
checks whether cited sources appear to support claim-level evidence records.
It consumes existing claim grounding outputs, an optional retrieval evidence
profile, run-level citation IDs, and retrieved chunks.

## What It Does

- Builds a `CitationFaithfulnessReport` for a `RAGRun`.
- Creates one `ClaimCitationFaithfulnessRecord` per available claim evidence
  record.
- Marks missing citations, phantom cited documents, unsupported citations,
  contradicted citations, and fully or partially supported citations.
- Records dependency visibility:
  - `claim_grounding_used`
  - `retrieval_evidence_profile_used`
  - `legacy_citation_fallback_used`
- Runs after claim grounding and retrieval evidence consumers in the default
  diagnosis pipeline.

## What It Does Not Do

- It does not run an LLM judge.
- It does not modify or rerun claim grounding.
- It does not modify retrieval evidence profiling.
- It does not perform post-rationalization or counterfactual citation probes.
- It does not prove the model genuinely relied on the cited source.

## Difference From CitationMismatchAnalyzer

`CitationMismatchAnalyzer` checks document-level provenance consistency: whether
the answer cites document IDs that are absent from retrieved documents.

`CitationFaithfulnessAnalyzerV0` is claim-level. It asks whether the cited
document or chunk overlaps the evidence already identified as supporting or
contradicting a specific claim. A citation can be non-phantom and still be
unfaithful for a claim if support comes from a different retrieved source.

## Research-Faithfulness Boundary

V0 is not a research-faithful RefChecker, RAGChecker, or Wallat-style citation
faithfulness implementation. It does not run NLI, counterfactual ablations,
source-removal probes, or calibrated citation attribution. Its labels are
heuristic interpretations of existing GovRAG evidence substrates.

## Why Gating Is Disabled

`recommended_for_gating` is `False` because V0 is uncalibrated and inherits the
limitations of upstream claim grounding and retrieval evidence. It is useful
for diagnosis and reporting, not for blocking production answers.

## Future Upgrade Path

- V1 can add calibrated scoring over claim grounding and retrieval evidence
  without changing the report schema.
- V1 can add stronger claim-to-cited-source matching, including chunk-level
  citation extraction when available.
- V2 can add explicit counterfactual probes that remove or swap cited sources
  and measure whether the answer or support judgment changes.
- Later research-faithful modes can be introduced only when the implementation
  actually matches the selected method's assumptions and evaluation protocol.
