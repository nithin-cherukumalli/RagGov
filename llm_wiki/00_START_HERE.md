# GovRAG LLM Wiki

This folder is the engineering operating system for GovRAG.

GovRAG is a diagnosis-first RAG governance framework.

It is not primarily a scoring tool.
It is not a RAGAS clone.
It is not a DeepEval clone.
It is not a generic wrapper over external evaluators.

The current system is an advanced prototype.
It has a strong architecture and promising taxonomy, but it is not yet production-validated.

This wiki is a living control surface, not a static documentation dump.
If the code changes and the wiki is not updated, the wiki becomes untrustworthy.

Read in this order:

1. [01_PROJECT_CHARTER.md](./01_PROJECT_CHARTER.md)
2. [02_ARCHITECTURE_MAP.md](./02_ARCHITECTURE_MAP.md)
3. [11_SUBSTRATE_ORDER.md](./11_SUBSTRATE_ORDER.md)
4. [04_ANALYZER_REGISTRY.md](./04_ANALYZER_REGISTRY.md)
5. [08_NO_SILENT_FALLBACK_POLICY.md](./08_NO_SILENT_FALLBACK_POLICY.md)
6. [12_DECISION_POLICY_DANGER_ZONE.md](./12_DECISION_POLICY_DANGER_ZONE.md)
7. [10_CURRENT_WEAKNESSES.md](./10_CURRENT_WEAKNESSES.md)
8. [03_AGENT_RULES.md](./03_AGENT_RULES.md)
9. [09_PR_PROTOCOL.md](./09_PR_PROTOCOL.md)

Non-negotiable truths:

- Claim grounding is the highest-leverage substrate to harden first.
- Native mode is the primary product path and the first mode that must become reputable.
- External-enhanced mode is opt-in and must always surface degradation honestly.
- Meta layers must not invent evidence.
- No silent fallback is allowed.
- Research-inspired is not research-faithful.
- Scores are not probabilities unless explicitly validated.
- `src/raggov/decision_policy.py` is a danger zone.
- Context engineering quality is a first-class project requirement.
- The wiki must be updated as the code evolves, or it becomes operationally harmful.

Real core runtime files:

- `src/raggov/engine.py`
- `src/raggov/decision_policy.py`
- `src/raggov/cli.py`
- `src/raggov/taxonomy.py`
- `src/raggov/external_signal_bridge.py`
- `src/raggov/evaluators/registry.py`
- `src/raggov/models/`

Use this wiki to answer:

1. What failed?
2. Where in the RAG pipeline did it fail?
3. What evidence supports that diagnosis?
4. What alternative explanations were considered?
5. What fix should an engineer apply?
6. How trustworthy is the diagnosis?

Maintenance rule:

- every PR that changes architecture, analyzer behavior, fallback behavior, trust metadata, or operator workflow must update the relevant `llm_wiki/` files in the same PR unless there is a documented reason not to
