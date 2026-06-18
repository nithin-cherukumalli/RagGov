#!/usr/bin/env python3
"""Build Phase 2 human-audit worklists from the staged real heldout.

Outputs reports only. Does not edit the canonical dataset, lock, labels, gates,
engine, analyzer, or policy files.
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "evals/govrag_calib/staging/raw/heldout_real_v1.jsonl"
WORKLIST = ROOT / "reports/calibration/contradiction_audit_worklist.md"
NLI_NOTE = ROOT / "reports/calibration/nli_provider_readiness.md"


CONTRADICTION_READS: dict[str, tuple[str, str]] = {
    "13392": ("actually fine/mislabeled", "Answer appears supported by Farkle Ten instructions; no explicit not-X conflict found."),
    "15893": ("actually fine/mislabeled", "Answer steps are present across chimney-cap passages; no explicit contradiction found."),
    "13701": ("actually fine/mislabeled", "Answer follows rib-grilling/smoking procedure in context; no explicit contradiction found."),
    "13368": ("actually fine/mislabeled", "Answer paraphrases Bible/words passage; no explicit contradiction found."),
    "15700": ("actually fine/mislabeled", "Blood-flow sequence matches passage; no explicit contradiction found."),
    "15043": ("actually fine/mislabeled", "Prune preparation answer mirrors passage text."),
    "13084": ("actually fine/mislabeled", "Parkinson symptom list appears supported by retrieved text."),
    "17239": ("actually fine/mislabeled", "iTunes/Apple Music radio steps match passage."),
    "13214": ("actually fine/mislabeled", "Rump-roast oven instructions match passage."),
    "15941": ("actually fine/mislabeled", "Strut/shock distinction is supported by passage."),
    "16676": ("actually fine/mislabeled", "Answer abstains; source contains related advice, but there is no contradiction."),
    "15699": ("actually fine/mislabeled", "Blood-flow sequence matches passage; no explicit contradiction found."),
    "15821": ("actually fine/mislabeled", "iPhone charging/troubleshooting steps are supported by passage."),
    "17183": ("actually fine/mislabeled", "Hard-boil-egg steps appear supported by cooking/video context."),
    "17329": ("actually fine/mislabeled", "Tibetan Terrier pet traits appear supported by context."),
    "15363": ("actually fine/mislabeled", "Baking-soda underarm odor guidance appears supported."),
    "12561": ("actually fine/mislabeled", "Pulmonary embolism symptom list appears supported."),
    "15279": ("actually unsupported", "Answer extrapolates airport rental workflow/purchase details beyond the passages; not an explicit contradiction."),
    "16495": ("actually fine/mislabeled", "Answer abstains; source contains prostate screening decision guidance, but no contradiction."),
    "15487": ("actually fine/mislabeled", "Answer abstains despite source including hickey-giving guidance; no contradiction."),
    "13480": ("actually fine/mislabeled", "Roasted Brussels sprouts steps appear supported."),
    "16392": ("actually fine/mislabeled", "Answer abstains; source gives selective breeding contrast but no contradiction."),
    "11925": ("actually fine/mislabeled", "Weather forecast answer mirrors passage values."),
    "16200": ("actually fine/mislabeled", "Leave-email guidance is supported by passage."),
    "17324": ("actually fine/mislabeled", "Chinese broccoli trimming/cooking steps are supported by passages."),
}


CLEAN_SPOT_READS: dict[str, tuple[str, str]] = {
    "5ab345db55429969a97a8122": ("faithful", "Longs Drugs passage says locations are in Hawaii; Warren Bryant passage identifies him as CEO."),
    "5adf732a5542993a75d264e9": ("faithful", "Sue Donahue passage says Donahue replaced Kelli Ward."),
    "5ac2a912554299218029dae8": ("faithful", "Wolfhounds formed in 1985; Hole formed in 1989 in Courtney Love passage."),
    "5a7272eb5542997f827839d7": ("faithful", "Hunger Games passage identifies Katniss as 16-year-old and Catching Fire continues that story."),
    "5ac2c3545542990b17b1548b": ("faithful", "Korea under Japanese rule ended at conclusion of World War II; Chang Ucchin born under that rule."),
    "ALCE-data/qampari_eval_gtr_top100:481__wikitables_simple__dev": ("needs human check", "List answer is long; sampled context supports science-fiction-magazine topic but full list needs manual verification."),
    "ALCE-data/qampari_eval_gtr_top100:439__wikidata_intersection__dev": ("needs human check", "List answer is long; requires checking every film against script/producer constraints."),
    "ALCE-data/qampari_eval_gtr_top100:772__wikidata_simple__dev": ("needs human check", "List answer names German Type U 31 examples; likely supported but all items need manual verification."),
    "ALCE-data/qampari_eval_gtr_top100:336__wikidata_comp__dev": ("needs human check", "List of countries is long and duplicate-heavy; manual verification needed."),
    "ALCE-data/qampari_eval_gtr_top100:479__wikitables_simple__dev": ("needs human check", "Bog-body list answer needs item-by-item support check."),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_line(text: str, width: int = 220) -> str:
    return textwrap.shorten(" ".join(str(text or "").split()), width=width, placeholder=" ...")


def block(text: str, limit: int = 1800) -> str:
    compact = "\n".join(line.rstrip() for line in str(text or "").splitlines()).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "\n...[truncated for audit sheet]"


def passage_text(row: dict[str, Any], *, clean: bool = False) -> str:
    chunks = row.get("retrieved_chunks") or []
    citations = {str(c) for c in row.get("citations") or []}
    selected = []
    if clean and citations:
        selected = [c for c in chunks if str(c.get("doc_id")) in citations]
    if not selected:
        selected = chunks
    parts = []
    for chunk in selected:
        label = chunk.get("chunk_id") or chunk.get("doc_id") or "chunk"
        doc = chunk.get("doc_id") or "unknown_doc"
        parts.append(f"[{label} | doc={doc}]\n{block(chunk.get('text', ''), 1400)}")
    return "\n\n".join(parts)


def contradiction_section(rows: list[dict[str, Any]]) -> tuple[str, Counter[str]]:
    contradiction_rows = [r for r in rows if r.get("expected_primary_failure") == "CONTRADICTED_CLAIM"]
    counts: Counter[str] = Counter()
    lines = ["## CONTRADICTED_CLAIM Human Audit Worklist", ""]
    for index, row in enumerate(contradiction_rows, start=1):
        source_id = str(row.get("source_id"))
        classification, rationale = CONTRADICTION_READS.get(
            source_id,
            ("actually unsupported", "No explicit not-X conflict found by provisional text-only read."),
        )
        counts[classification] += 1
        lines.extend(
            [
                f"### C{index:02d}. source_id={source_id}",
                "",
                f"- Provisional S2 read: **{classification}**",
                f"- Provisional rationale: {rationale}",
                "- Human label: [ ] yes genuine contradiction  [ ] no unsupported/not contradicted  [ ] unsure",
                "- Human notes:",
                "",
                "**Query**",
                "",
                f"> {one_line(row.get('query'), 500)}",
                "",
                "**Answer**",
                "",
                f"> {block(row.get('answer'), 1200)}",
                "",
                "**Cited / retrieved passage text**",
                "",
                "```text",
                passage_text(row),
                "```",
                "",
                "**Human prompt**",
                "",
                "> Is this a genuine contradiction, where the answer asserts X and the source says not-X? Mark yes/no/unsure above.",
                "",
            ]
        )
    return "\n".join(lines), counts


def clean_section(rows: list[dict[str, Any]]) -> tuple[str, Counter[str]]:
    clean_rows = [r for r in rows if r.get("expected_primary_failure") == "CLEAN"]
    selected: list[dict[str, Any]] = []
    for dataset in ("hotpotqa", "alce"):
        selected.extend([r for r in clean_rows if r.get("source_dataset") == dataset][:5])
    counts: Counter[str] = Counter()
    lines = ["## CLEAN Spot-Check Worklist", ""]
    for index, row in enumerate(selected, start=1):
        source_id = str(row.get("source_id"))
        classification, rationale = CLEAN_SPOT_READS.get(
            source_id,
            ("needs human check", "Not manually classified in this scaffold."),
        )
        counts[classification] += 1
        lines.extend(
            [
                f"### K{index:02d}. source_id={source_id}",
                "",
                f"- Source dataset: `{row.get('source_dataset')}`",
                f"- Provisional S2 read: **{classification}**",
                f"- Provisional rationale: {rationale}",
                "- Human label: [ ] faithful  [ ] not faithful  [ ] unsure",
                "- Human notes:",
                "",
                "**Query**",
                "",
                f"> {one_line(row.get('query'), 500)}",
                "",
                "**Reference answer**",
                "",
                f"> {block(row.get('answer'), 1200)}",
                "",
                "**Cited / retrieved passage text**",
                "",
                "```text",
                passage_text(row, clean=True),
                "```",
                "",
                "**Human prompt**",
                "",
                "> Is this CLEAN reference answer fully faithful to the cited/retrieved passage text? Mark faithful/not faithful/unsure above.",
                "",
            ]
        )
    return "\n".join(lines), counts


def write_worklist(rows: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    contradiction_md, contradiction_counts = contradiction_section(rows)
    clean_md, clean_counts = clean_section(rows)
    header = [
        "# Phase 2 Human-Audit Worklist",
        "",
        "Source: `evals/govrag_calib/staging/raw/heldout_real_v1.jsonl`.",
        "",
        "This is a human audit sheet. S2 classifications are provisional text-only reads, not accepted labels.",
        "",
        "Summary:",
        f"- CONTRADICTED rows reviewed: {sum(contradiction_counts.values())}",
        f"- CONTRADICTED provisional counts: {dict(contradiction_counts)}",
        f"- CLEAN rows spot-checked: {sum(clean_counts.values())}",
        f"- CLEAN provisional counts: {dict(clean_counts)}",
        "",
    ]
    WORKLIST.parent.mkdir(parents=True, exist_ok=True)
    WORKLIST.write_text("\n".join(header) + contradiction_md + "\n" + clean_md + "\n", encoding="utf-8")
    return contradiction_counts, clean_counts


def write_nli_note() -> None:
    note = """# NLI Provider Readiness Note

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
"""
    NLI_NOTE.parent.mkdir(parents=True, exist_ok=True)
    NLI_NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    rows = load_rows(HELDOUT)
    contradiction_counts, clean_counts = write_worklist(rows)
    write_nli_note()
    print(f"Wrote {WORKLIST}")
    print(f"Wrote {NLI_NOTE}")
    print(f"contradiction_counts={dict(contradiction_counts)}")
    print(f"clean_counts={dict(clean_counts)}")


if __name__ == "__main__":
    main()

