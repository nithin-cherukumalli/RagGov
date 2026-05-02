# Citation Faithfulness v0 Fixtures

These fixtures validate `CitationFaithfulnessAnalyzerV0` regression behavior
only. They are synthetic cases designed to keep v0 behavior stable across code
changes.

They are not human-labeled calibration data. They do not prove citation
faithfulness, and they do not prove that a model genuinely relied on a cited
source.

The fixtures also do not prove post-rationalized citation detection. V0 only
uses existing claim grounding records, retrieval evidence profiles, legacy
citation IDs, and retrieved chunks.

Future v1 work needs human-labeled claim-citation support data for calibration
and evaluation. Future v2 work needs counterfactual post-rationalization tests,
such as cited-source removal or source-swap probes, before making stronger
claims about model reliance.
