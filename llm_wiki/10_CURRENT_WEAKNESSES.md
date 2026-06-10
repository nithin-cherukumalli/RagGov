# Current Weaknesses

## Advanced Prototype Warning

GovRAG has a strong architecture and promising taxonomy, but it is not yet production-validated.

## Highest-Risk Weakness

The highest-risk weakness is mixed-quality substrate evidence feeding increasingly aggressive meta interpretation.

Why this matters:

- claim grounding quality affects almost everything downstream
- citation faithfulness and citation probes reuse grounding outputs
- retrieval diagnosis rolls up upstream evidence instead of recomputing truth
- NCV contains explicit heuristic fallback branches
- Layer6 and A2P can over-interpret upstream symptoms
- decision policy performs strong winner-selection and suppression logic

## Concrete Weaknesses Found In Code

- `src/raggov/decision_policy.py`
  high-coupling priority and override logic can mask substrate weakness instead of fixing it

- `src/raggov/analyzers/confidence/semantic_entropy.py`
  deterministic path is only a proxy, not full semantic entropy

- `src/raggov/analyzers/taxonomy_classifier/layer6.py`
  score-band and rule heuristics can look more authoritative than they are

- `src/raggov/analyzers/attribution/a2p.py`
  includes legacy heuristic fallback modes under a research-branded frame

- `src/raggov/analyzers/retrieval/evidence_profile.py`
  core retrieval substrate is still heavily heuristic and uncalibrated

- `src/raggov/analyzers/sufficiency/sufficiency.py`
  thresholding and requirement extraction remain approximate

- `src/raggov/analyzers/version_validity/analyzer.py`
  age-based warning logic is explicitly only a practical approximation

## Priority

Harden claim grounding first.
Then harden citation faithfulness, sufficiency, retrieval diagnosis, and version validity before expanding meta layers.
