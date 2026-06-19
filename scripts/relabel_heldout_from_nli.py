#!/usr/bin/env python3
"""LLM-assisted provisional relabeling of the staged real heldout from NLI claims.

Governance contract:
  - Reads the staged heldout only; does not write gold/canonical dataset files.
  - Runs DiagnosisEngine with claim grounding verifier policy `llm_entailment`.
  - Derives whole-answer labels from per-claim NLI labels:
      any contradicted -> CONTRADICTED_CLAIM
      else any unsupported -> UNSUPPORTED_CLAIM
      else any verifiable entailed claim -> CLEAN
      else no verifiable claims -> INSUFFICIENT_CONTEXT for human review
  - Writes provisional staging outputs and calibration reports for human audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kimi_client import KimiClient  # noqa: E402
from raggov.analyzers.grounding.support import ClaimGroundingAnalyzer  # noqa: E402
from raggov.engine import DiagnosisEngine  # noqa: E402
from raggov_score import _load_rows, build_run  # noqa: E402


DEFAULT_INPUT = ROOT / "evals/govrag_calib/staging/raw/heldout_real_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "evals/govrag_calib/staging/raw/heldout_real_v1_relabeled.jsonl"
DEFAULT_AUDIT = ROOT / "evals/govrag_calib/staging/raw/heldout_real_v1_nli_spot_audit_worklist.jsonl"
DEFAULT_REPORT = ROOT / "reports/calibration/heldout_real_v1_nli_relabel_report.md"
DEFAULT_LEDGER = ROOT / "reports/calibration/real_heldout_label_quality_finding.md"
FORBIDDEN_OUTPUT_NAMES = {
    "govrag_calib_150.jsonl",
    "calib_150_seed.jsonl",
    "DATASET_LOCK.json",
    "DATASET_MANIFEST.json",
    "LABEL_CHANGELOG.md",
}


def _refuse_protected_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name in FORBIDDEN_OUTPUT_NAMES:
        raise ValueError(f"refusing to write protected file: {resolved}")
    protected_fragments = (
        "evals/govrag_calib/govrag_calib_150.jsonl",
        "evals/govrag_calib/calib_150_seed.jsonl",
        "DATASET_LOCK.json",
        "LABEL_CHANGELOG.md",
    )
    if any(fragment in str(resolved) for fragment in protected_fragments):
        raise ValueError(f"refusing to write protected path: {resolved}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _refuse_protected_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _source_label(row: dict[str, Any]) -> str:
    return str(
        row.get("original_source_label")
        or row.get("source_label")
        or row.get("expected_primary_failure")
        or "UNKNOWN"
    )


def _claim_dict(claim: Any) -> dict[str, Any]:
    if hasattr(claim, "model_dump"):
        return claim.model_dump(mode="json")
    if isinstance(claim, dict):
        return dict(claim)
    return {
        "claim_text": getattr(claim, "claim_text", None),
        "label": getattr(claim, "label", None),
    }


def _count_labels(claims: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for claim in claims:
        label = str(claim.get("label") or "").strip().lower()
        if label:
            counts[label] += 1
    for key in ("entailed", "unsupported", "contradicted", "abstain"):
        counts.setdefault(key, 0)
    return dict(sorted(counts.items()))


def _derive_label(counts: dict[str, int]) -> tuple[str, str]:
    if counts.get("contradicted", 0) > 0:
        return "CONTRADICTED_CLAIM", "any_claim_contradicted"
    if counts.get("unsupported", 0) > 0:
        return "UNSUPPORTED_CLAIM", "any_claim_unsupported"
    if counts.get("entailed", 0) > 0:
        return "CLEAN", "all_verifiable_claims_entailed_or_abstain"
    return "INSUFFICIENT_CONTEXT", "no_verifiable_claims_extracted"


def _compact_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for claim in claims:
        compact.append(
            {
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("claim_text"),
                "label": claim.get("label"),
                "support_label": claim.get("support_label"),
                "verification_method": claim.get("verification_method"),
                "fallback_used": bool(claim.get("fallback_used", False)),
                "supporting_chunk_ids": claim.get("supporting_chunk_ids") or [],
                "contradicting_chunk_ids": claim.get("contradicting_chunk_ids") or [],
                "neutral_chunk_ids": claim.get("neutral_chunk_ids") or [],
                "evidence_reason": claim.get("evidence_reason"),
                "skip_reason": claim.get("skip_reason"),
            }
        )
    return compact


def _relabel_row(index: int, row: dict[str, Any], engine: DiagnosisEngine) -> dict[str, Any]:
    diagnosis = engine.diagnose(build_run(row))
    claims = [_claim_dict(claim) for claim in diagnosis.claim_results]
    counts = _count_labels(claims)
    nli_label, derivation_reason = _derive_label(counts)
    source = _source_label(row)
    return {
        "row_index": index,
        "case_id": row.get("case_id"),
        "source_dataset": row.get("source_dataset"),
        "source_id": row.get("source_id"),
        "original_source_label": source,
        "source_label": source,
        "nli_derived_label": nli_label,
        "nli_derivation_reason": derivation_reason,
        "per_claim_label_counts": counts,
        "per_claim_nli": _compact_claims(claims),
        "label_source": "llm_assisted_provisional",
        "agreement_with_source": nli_label == source,
        "human_review_required": nli_label != source
        or derivation_reason == "no_verifiable_claims_extracted",
        "engine_primary_failure": diagnosis.primary_failure.value,
        "engine_degraded": bool(getattr(diagnosis, "degraded", False)),
        "engine_fallback_heuristics_used": list(getattr(diagnosis, "fallback_heuristics_used", []) or []),
    }


def relabel_rows(
    rows: list[dict[str, Any]],
    engine: DiagnosisEngine,
    *,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        output = _relabel_row(index, row, engine)
        relabeled.append(output)
        if checkpoint_path is not None:
            _write_jsonl(checkpoint_path, relabeled)
        print(
            f"[{index}/{len(rows)}] {output['case_id']} "
            f"{output['source_label']} -> {output['nli_derived_label']} "
            f"{output['per_claim_label_counts']}",
            flush=True,
        )
    return relabeled


def _mismapping_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contradicted = Counter(
        row["nli_derived_label"]
        for row in rows
        if row["source_label"] == "CONTRADICTED_CLAIM"
    )
    clean = Counter(row["nli_derived_label"] for row in rows if row["source_label"] == "CLEAN")
    return {
        "source_CONTRADICTED_CLAIM_total": sum(contradicted.values()),
        "source_CONTRADICTED_CLAIM_by_nli": dict(sorted(contradicted.items())),
        "source_CLEAN_total": sum(clean.values()),
        "source_CLEAN_by_nli": dict(sorted(clean.items())),
        "source_CLEAN_stay_CLEAN": clean.get("CLEAN", 0),
    }


def build_audit_worklist(
    rows: list[dict[str, Any]],
    *,
    random_agrees: int,
    seed: int,
) -> list[dict[str, Any]]:
    disagrees = [row for row in rows if not row["agreement_with_source"]]
    agrees = [row for row in rows if row["agreement_with_source"]]
    rng = random.Random(seed)
    sampled_agrees = rng.sample(agrees, k=min(random_agrees, len(agrees)))
    worklist: list[dict[str, Any]] = []
    for reason, selected in (
        ("nli_label_disagrees_with_source", disagrees),
        ("random_agree_spot_check", sampled_agrees),
    ):
        for row in selected:
            worklist.append(
                {
                    "audit_reason": reason,
                    "row_index": row["row_index"],
                    "case_id": row["case_id"],
                    "source_dataset": row.get("source_dataset"),
                    "source_id": row.get("source_id"),
                    "source_label": row["source_label"],
                    "nli_derived_label": row["nli_derived_label"],
                    "per_claim_label_counts": row["per_claim_label_counts"],
                    "per_claim_nli": row["per_claim_nli"],
                    "human_decision": None,
                    "human_notes": "",
                }
            )
    return worklist


def _format_counts(counts: dict[str, Any]) -> str:
    contradicted = counts["source_CONTRADICTED_CLAIM_by_nli"]
    clean = counts["source_CLEAN_by_nli"]
    return (
        f"- Source `CONTRADICTED_CLAIM` rows: {counts['source_CONTRADICTED_CLAIM_total']} -> "
        f"{contradicted}\n"
        f"- Source `CLEAN` rows: {counts['source_CLEAN_total']} -> {clean}; "
        f"stayed CLEAN={counts['source_CLEAN_stay_CLEAN']}\n"
    )


def write_report(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    rows: list[dict[str, Any]],
    counts: dict[str, Any],
    model: str,
    audit_seed: int,
) -> None:
    _refuse_protected_output(path)
    label_pairs = Counter((row["source_label"], row["nli_derived_label"]) for row in rows)
    pair_lines = "\n".join(
        f"- `{source}` -> `{derived}`: {count}"
        for (source, derived), count in sorted(label_pairs.items())
    )
    text = f"""# Heldout Real v1 NLI Relabel Report

Date: {datetime.now(UTC).date().isoformat()}

Method status: `external_signal` / `llm_assisted_provisional`.

This is not gold, not calibration, and not a canonical dataset update. The script runs
`DiagnosisEngine` with `claim_grounding_verifier_policy=llm_entailment`, `KimiClient`,
and `model={model}`. Whole-answer labels are derived from per-claim NLI labels by
the most-severe-claim rule: contradicted > unsupported > clean; zero verifiable
claims remain `INSUFFICIENT_CONTEXT` for human review.

Input: `{input_path.relative_to(ROOT)}`

Relabeled staging output: `{output_path.relative_to(ROOT)}`

Human spot-audit worklist: `{audit_path.relative_to(ROOT)}`

Rows relabeled: {len(rows)}

## Mismapping Counts
{_format_counts(counts)}
## Source -> NLI Label Matrix
{pair_lines}

## Audit Sampling
- All source/NLI disagreements are included.
- Random agreeing rows included: 10 requested, deterministic seed `{audit_seed}`.

## Governance
- `label_source=llm_assisted_provisional`
- Gold/canonical dataset changed: no
- Dataset lock changed: no
- Thresholds/gates changed: no
- Human acceptance required before promotion.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_ledger(path: Path, *, report_path: Path, output_path: Path, audit_path: Path, counts: dict[str, Any]) -> None:
    _refuse_protected_output(path)
    entry = f"""

## {datetime.now(UTC).date().isoformat()} - Kimi NLI provisional relabel

Method status: `external_signal` / `llm_assisted_provisional`.

Ran staged heldout through `DiagnosisEngine` with Kimi `llm_entailment` and derived
whole-answer labels from per-claim NLI verdicts using the most-severe-claim rule.
No gold/canonical labels, locks, thresholds, gates, or engine/policy files were changed.

Relabeled staging output: `{output_path.relative_to(ROOT)}`

Audit worklist: `{audit_path.relative_to(ROOT)}`

Report: `{report_path.relative_to(ROOT)}`

{_format_counts(counts)}
Human acceptance required before any promotion.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-worklist", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--model", default="moonshot-v1-8k")
    parser.add_argument("--random-agrees", type=int, default=10)
    parser.add_argument("--audit-seed", type=int, default=20260619)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.disable(logging.CRITICAL)
    args = parse_args()
    rows = _load_rows(args.input)
    client = KimiClient(model=args.model)
    engine_config = {
        "llm_client": client,
        "claim_grounding_verifier_policy": "llm_entailment",
    }
    engine = DiagnosisEngine(
        config=engine_config,
        analyzers=[ClaimGroundingAnalyzer(engine_config)],
    )
    relabeled = relabel_rows(rows, engine, checkpoint_path=args.output)
    audit_rows = build_audit_worklist(
        relabeled,
        random_agrees=args.random_agrees,
        seed=args.audit_seed,
    )
    counts = _mismapping_counts(relabeled)

    _write_jsonl(args.audit_worklist, audit_rows)
    write_report(
        args.report,
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit_worklist,
        rows=relabeled,
        counts=counts,
        model=args.model,
        audit_seed=args.audit_seed,
    )
    if not args.no_ledger:
        append_ledger(
            args.ledger,
            report_path=args.report,
            output_path=args.output,
            audit_path=args.audit_worklist,
            counts=counts,
        )

    print(json.dumps({"rows": len(relabeled), "audit_rows": len(audit_rows), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
