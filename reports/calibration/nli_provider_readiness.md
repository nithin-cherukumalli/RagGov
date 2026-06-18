# NLI Provider Readiness Note

Source files inspected:
- `src/raggov/analyzers/grounding/verifiers.py`
- `src/raggov/analyzers/grounding/support.py`
- `src/raggov/evaluators/claim/refchecker_adapter.py`
- `src/raggov/evaluators/claim/structured_llm.py`
- `src/raggov/engine.py`
- `src/raggov/config.py`

## Exact verifier interface

The native grounding analyzer consumes an `EvidenceVerifier`:

```python
def verify(
    self,
    claim: str,
    query: str,
    candidates: list[EvidenceCandidate],
    metadata: dict[str, Any] | None = None,
) -> VerificationResult
```

`ClaimEntailmentVerifierV1` is the explicit entailment interface. It adapts
`verify(...)` into:

```python
def verify_entailment(
    self,
    *,
    claim_text: str,
    source_sentence: str,
    top_k_candidates: list[EvidenceCandidate],
    cited_doc_ids: list[str],
    cited_chunk_ids: list[str],
    claim_type: str,
    numbers: list[str],
    dates: list[str],
    entities: list[str],
    atomicity_status: str,
    query: str,
    metadata: dict[str, Any] | None = None,
) -> VerificationResult
```

Provider outputs must be `VerificationResult` with:
- `label`: one of `entailed`, `unsupported`, `contradicted`, `abstain`.
- `support_label`: one of `supported`, `contradicted`, `insufficient_evidence`, `unverifiable`, `skipped`.
- `raw_score`: provider score/confidence, uncalibrated unless a calibrator exists.
- `evidence_chunk_id`, `supporting_chunk_ids`, `contradicting_chunk_ids`, `neutral_chunk_ids`.
- `rationale`, `verifier_name`, optional warnings/limitations/fallback metadata.
- `confidence_status`: should stay `unavailable` or `uncalibrated_heuristic_proxy` unless calibrated.

`EvidenceCandidate` inputs provide chunk id, source doc id, chunk text, preview,
lexical/anchor/value overlap scores, retrieval score, rerank score, and candidate reason.

## How to enable current providers

In `ClaimGroundingAnalyzer.__init__`, provider selection is:
- `claim_grounding_verifier_policy` first, otherwise `claim_verifier`.
- Mode fallback uses `claim_grounding_verifier_policy` or `claim_verifier_mode`, defaulting to `conservative_ensemble`.

Current config options:

```python
DiagnosisEngine(config={
    "claim_grounding_verifier_policy": "llm_entailment",
    "llm_client": client,
})
```

or:

```python
DiagnosisEngine(config={
    "claim_grounding_verifier_policy": "conservative_ensemble",
    "llm_client": client,
})
```

or external-adapter style:

```python
DiagnosisEngine(config={
    "claim_verifier": "structured_llm",
    "llm_client": client,  # or "llm_fn": callable
})
```

RefChecker selection:

```python
DiagnosisEngine(config={
    "claim_verifier": "refchecker",
    "enabled_external_providers": ["refchecker_claim"],
})
```

## Existing provider behaviors

`LLMClaimEntailmentVerifierV1`:
- Requires `config["llm_client"]`.
- Client must expose `.chat(prompt)` or `.complete(prompt)`.
- Prompt requests strict JSON with `support_label`, supporting/contradicting/neutral chunk ids, reason, warnings, confidence.
- On invoke or parse/repair failure, it visibly falls back to `HeuristicValueOverlapVerifier`.
- Fallback sets `fallback_used=True`, `fallback_from="llm_entailment_verifier"`, `fallback_to="heuristic_top_k_verifier"`, and warning labels such as `llm_entailment_invoke_failed:*`.

`ConservativeEnsembleVerifier`:
- Requires `llm_client`.
- Runs `LLMClaimEntailmentVerifierV1` plus `HeuristicValueOverlapVerifier`.
- Safety-gates LLM-supported outputs when heuristic/value/date/entity/compound checks disagree or are missing.
- Emits `verifier_policy="conservative_ensemble"`, `llm_label`, `heuristic_label`, `safety_gate_*`, and `verifier_disagreement` metadata.

`StructuredLLMClaimVerifierAdapter`:
- Accepts `llm_client` with `.chat/.complete` or `llm_fn(prompt)`.
- Returns `ExternalSignalRecord`, not gold labels.
- Labels normalize to `entailed`, `contradicted`, `unsupported`, `unclear`.
- Signal metadata is `method_type="external_signal_adapter"`, `calibration_status="uncalibrated_locally"`, `recommended_for_gating=False`.

`RefCheckerClaimSignalProvider`:
- Optional dependency; `is_available()` imports `refchecker`.
- Readiness also checks spaCy and `en_core_web_sm`.
- Native runtime is not implemented unless a configured runner/mock result is supplied.
- `verify_claims(claims, context)` uses `config["claim_runner"]` if present and otherwise returns `[]`.
- Signals remain advisory, uncalibrated locally, and not recommended for gating.

## Local/offline NLI options

No local Hugging Face NLI provider is currently wired into the code path. Opus can add one by implementing `ClaimEntailmentVerifierV1` or `EvidenceVerifier` with an offline model such as:
- `cross-encoder/nli-deberta-v3-small`
- `typeform/distilbert-base-uncased-mnli`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

Implementation sketch:
- Load tokenizer/model lazily in `__init__`.
- For each `EvidenceCandidate`, score premise=`candidate.chunk_text`, hypothesis=`claim_text`.
- Map model logits to `supported` / `contradicted` / `insufficient_evidence`.
- Return best supporting or contradicting chunk ids and raw probability as `raw_score`.
- Keep `confidence_status="unavailable"` or `uncalibrated_heuristic_proxy` until calibrated.
- On import/model-load/inference failure, return or fall back with visible `fallback_used`, `error_info`, `fallback_from`, and `fallback_to`.

## Visible degradation when absent

Absent/misconfigured providers are visible in multiple places:
- `ClaimGroundingAnalyzer` sets `_external_verifier_error`, then appends evidence like `External claim verifier unavailable: ...`.
- Missing `llm_client` for `llm_entailment` or `conservative_ensemble` falls back to `HeuristicValueOverlapVerifier`.
- `DiagnosisEngine` records `missing_external_providers`, `external_provider_readiness`, `external_adapter_errors`, `degraded_external_mode`, and `fallback_heuristics_used` in run metadata.
- RefChecker missing package/config reports readiness statuses such as `package_missing`, `spacy_missing`, `spacy_model_missing`, or `runtime_execution_not_configured`.

## Spec for Opus

The clean integration point for a real NLI provider is a new class implementing
`ClaimEntailmentVerifierV1.verify_entailment(...)`, plus a config branch in
`ClaimGroundingAnalyzer` such as `claim_grounding_verifier_policy="local_nli"`.
The class must return `VerificationResult`, not external labels, and must surface
all fallback/degradation metadata.
