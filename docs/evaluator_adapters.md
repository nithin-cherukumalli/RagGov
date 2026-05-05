# External Signal Adapter Layer

GovRAG can optionally consume signals from established external tools — RAGAS, DeepEval, cross-encoders, NLI models, and others — without depending on them at runtime.

## Design Principles

### GovRAG owns diagnosis

External tools provide **evidence signals**. They do not decide root cause, pipeline health, or gating outcome. GovRAG is solely responsible for:

- NCV (Node-wise Consistency Verification)
- Retrieval diagnosis rollups
- A2P scores
- Final AnalyzerResult status

External adapters feed *advisory* inputs into that pipeline, nothing more.

### External validation ≠ local calibration

RAGAS faithfulness of 0.85 is not equivalent to a GovRAG `pass`. External metrics are computed by different models under different assumptions. They have not been calibrated against GovRAG's labeled evaluation sets. Every `ExternalSignalRecord` carries:

```
calibration_status = "uncalibrated_locally"
recommended_for_gating = False
```

These fields are immutable defaults. Adapters must not override them.

### All external dependencies are optional

No adapter package (RAGAS, DeepEval, sentence-transformers, etc.) is a mandatory install. The evaluator layer uses lazy imports — nothing is imported at module load time. If a dependency is missing, the adapter's `is_available()` returns `False` and the registry surfaces a clear error result.

Optional installs:

```bash
pip install ragas
pip install deepeval
```

If packaging extras are configured by the project release, the equivalent install pattern is:

```bash
pip install "govrag[eval]"   # installs ragas, deepeval, ragchecker
pip install "govrag[claim]"  # installs refchecker
pip install "govrag[llm]"    # installs openai, anthropic
```

### Missing adapters are always visible

Silence is not allowed. When an adapter is unavailable, the registry returns:

```python
ExternalEvaluationResult(
    succeeded=False,
    missing_dependency=True,
    error="<adapter name>: <dependency> not installed.",
)
```

Callers can inspect this result and decide how to handle it. The pipeline does not pretend the adapter ran successfully.

### Diagnosis Modes

To control how external adapters are consumed, GovRAG configuration supports a `mode` parameter:

- `external-enhanced` (default): Attempts to use specified external signals (like structured LLM tools). If dependencies are missing or the LLM is not configured, it gracefully degrades to heuristics and populates `missing_external_providers`.
- `native`: Force-disables all external signals. Analyzers fall back completely to internal heuristic scoring.
- `calibrated`: (Reserved for future ARES PPI-corrected outputs)

---

## Architecture

```
src/raggov/evaluators/
├── __init__.py           # Public re-exports
├── base.py               # Enums, models, protocols
├── registry.py           # ExternalEvaluatorRegistry
├── claim/
│   ├── __init__.py
│   └── structured_llm.py # LLM-based claim verifier (skeleton)
├── citation/
│   ├── __init__.py
│   └── structured_llm.py # LLM-based citation verifier (skeleton)
└── retrieval/
    ├── __init__.py
    ├── cross_encoder.py  # Cross-encoder relevance scorer (skeleton)
    ├── ragas_adapter.py  # RAGAS adapter (skeleton)
    └── deepeval_adapter.py # DeepEval adapter (skeleton)
```

## Retrieval Adapter Usage

RAGAS and DeepEval adapters are available as optional retrieval/context signal providers:

```python
from raggov.evaluators.retrieval import (
    DeepEvalRetrievalSignalProvider,
    RagasRetrievalSignalProvider,
)

ragas_provider = RagasRetrievalSignalProvider()
deepeval_provider = DeepEvalRetrievalSignalProvider()

ragas_result = ragas_provider.evaluate(run)
deepeval_result = deepeval_provider.evaluate(run)

run.metadata.setdefault("external_evaluation_results", []).extend(
    [ragas_result, deepeval_result]
)
```

`RetrievalDiagnosisAnalyzerV0` consumes these results only as evidence signals. NCV consumes them only through retrieval diagnosis/profile outputs. RAGAS and DeepEval never decide the final diagnosis directly.

For tests or enterprise integrations, callers may provide a local metric runner:

```python
provider = RagasRetrievalSignalProvider(
    {"metric_runner": lambda run: {"context_precision": 0.7, "context_recall": 0.4}}
)
```

The runner output is preserved as `ExternalEvaluationResult.raw_payload`, and each normalized metric is also preserved in `ExternalSignalRecord.raw_payload`.

### RAGAS Metrics

| RAGAS metric | GovRAG signal type | Diagnostic interpretation |
|---|---|---|
| `context_precision` | `retrieval_context_precision` | Low value is advisory evidence for retrieval noise |
| `context_recall` | `retrieval_context_recall` | Low value is advisory evidence for retrieval miss |
| `claim_recall` | `claim_recall` | Low value is advisory evidence for retrieval miss |
| `faithfulness` | `faithfulness` | Advisory grounding signal; does not override GovRAG claim/citation verification |

### DeepEval Metrics

| DeepEval metric | GovRAG signal type | Diagnostic interpretation |
|---|---|---|
| `contextual_relevancy` | `retrieval_contextual_relevancy` | Low value is advisory evidence for retrieval noise |
| `contextual_precision` | `retrieval_contextual_precision` | Poor value is advisory evidence for rank failure/rank failure unknown |

### RAGChecker Metrics

RAGChecker provides comprehensive diagnostics for both retrieval and generation quality:

| RAGChecker metric | GovRAG signal type | Diagnostic interpretation |
|---|---|---|
| `context_precision` | `retrieval_context_precision` | Low value is advisory evidence for retrieval noise |
| `context_utilization` | `context_utilization` | Low value suggests retrieved evidence may have been ignored by generation |
| `retrieval_context_recall` | `retrieval_context_recall` | Low value is advisory evidence for retrieval miss |
| `claim_recall` | `claim_recall` | Low value is advisory evidence for retrieval miss |
| `faithfulness` | `faithfulness` | Advisory grounding evidence; does not alone force a diagnosis |
| `hallucination` | `hallucination` | High value is advisory evidence for claim support/generation issues |
| `claim_support` | `claim_support` | Advisory evidence for claim support |

*Note: Metrics like `claim_recall` and `retrieval_context_recall` require reference material. If `run.metadata["reference_answer"]` or `run.metadata["gold_claims"]` is missing, the adapter safely emits a `missing_reference_input` signal rather than failing.*

### RefChecker Metrics

RefChecker adapters provide advisory claim and citation verification signals:

| Metric | GovRAG signal type | Diagnostic interpretation |
|---|---|---|
| `claim_check` | `claim_support` | Evidence for claim support status (entailed/contradicted/unsupported/unclear) |
| `citation_check` | `citation_support` | Evidence for citation support status (supports/contradicts/does_not_support/citation_missing/unclear) |

RefChecker signals also preserve fine-grained **triplet-level evidence** (subject, predicate, object) in the `raw_payload["triplets"]` field where available.

### Retrieval Adapter Limitations

- RAGAS and DeepEval are optional dependencies.
- Tests must mock metric outputs and must not require API keys.
- Adapter output is `uncalibrated_locally`.
- `recommended_for_gating` remains `False`.
- External metrics can strengthen evidence but cannot localize root cause without GovRAG diagnosis logic.
- Faithfulness-style external metrics support grounding/citation investigation but do not replace GovRAG claim grounding or citation faithfulness analyzers.

### Key types (`base.py`)

| Type | Role |
|---|---|
| `ExternalEvaluatorProvider` | Enum identifying the tool (ragas, deepeval, cross_encoder, …) |
| `ExternalSignalType` | Enum for the signal kind (faithfulness, retrieval_relevance, pii, …) |
| `ExternalSignalRecord` | One signal from one adapter, with metadata and limitations |
| `ExternalEvaluationResult` | Top-level result from one adapter run |
| `ExternalSignalProvider` | Protocol every adapter must implement |
| `ClaimVerifierAdapter` | Specialized protocol with `verify_claims()` |
| `CitationVerifierAdapter` | Specialized protocol with `verify_citations()` |
| `RetrievalSignalProvider` | Specialized protocol with `score_relevance()` |

### Registry (`registry.py`)

`ExternalEvaluatorRegistry` manages the lifecycle:

```python
registry = ExternalEvaluatorRegistry()
registry.register(RAGASAdapter())
registry.register(CrossEncoderRelevanceAdapter())

results = registry.evaluate_enabled(
    run,
    enabled_providers=["ragas", "cross_encoder_relevance"],
    strict_mode=False,   # capture exceptions; set True to propagate
)
```

`evaluate_enabled` rules:
- Adapter not registered → silently skipped (caller controls which adapters to enable).
- Adapter registered but `is_available()` is False → `missing_dependency=True` result returned.
- Adapter raises in non-strict mode → exception message captured in `error` field.
- Adapter raises in strict mode → exception propagates to caller.

---

## Adding a New Adapter

1. Create a module under the appropriate subfolder (`claim/`, `citation/`, or `retrieval/`).
2. Implement the `ExternalSignalProvider` protocol (or a specialized variant).
3. Use **lazy imports** — wrap the third-party import in `is_available()` or inside `evaluate()`.
4. Return `missing_dependency=True` when the import fails.
5. Never set `recommended_for_gating=True` or override `calibration_status`.
6. Register in the application entry-point:

```python
from raggov.evaluators.registry import ExternalEvaluatorRegistry
from raggov.evaluators.retrieval.ragas_adapter import RAGASAdapter

registry = ExternalEvaluatorRegistry()
registry.register(RAGASAdapter())
```

---

## Current Adapter Status

| Adapter | Location | Status |
|---|---|---|
| `StructuredLLMClaimVerifier` | `claim/structured_llm.py` | Active — first-class in `external-enhanced` mode |
| `StructuredLLMCitationVerifier` | `citation/structured_llm.py` | Active — first-class in `external-enhanced` mode |
| `CrossEncoderRelevanceAdapter` | `retrieval/cross_encoder.py` | Active — first-class in `external-enhanced` mode |
| `RagasRetrievalSignalProvider` | `retrieval/ragas_adapter.py` | Active — optional signal provider; install via `govrag[eval]` |
| `DeepEvalRetrievalSignalProvider` | `retrieval/deepeval_adapter.py` | Active — optional signal provider; install via `govrag[eval]` |
| `RAGCheckerSignalProvider` | `retrieval/ragchecker_adapter.py` | Active — optional signal provider; install via `govrag[eval]` |
| `RefCheckerClaimSignalProvider` | `claim/refchecker_adapter.py` | Active — optional signal provider; install via `govrag[claim]` |
| `RefCheckerCitationSignalProvider` | `citation/refchecker_adapter.py` | Active — optional signal provider; install via `govrag[claim]` |

Adapters with optional third-party dependencies return `missing_dependency=True` until their dependencies are installed and wired. Missing adapters are never treated as clean evidence.
