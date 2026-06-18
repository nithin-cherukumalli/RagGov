"""Canonical RagGov scorer — single source of truth for accuracy measurement.

Replaces ad-hoc scoring scripts. Both the engine work and the sidekick measurement
harness import `score_file` / `build_run` from here so every number is produced the
same way.

Method status: this measures EXACT primary-failure accuracy against a dataset's
`expected_primary_failure`. It is uncalibrated and the induced probe is synthetic
(mutations of clean cases) — these numbers are diagnostic accuracy on the available
eval data, NOT a production-generalization guarantee. Config mode matters:
  - mode="default"  -> DiagnosisEngine() default config
  - mode="native"   -> DiagnosisEngine(config={"mode": "native"})
They differ (NCV behaves differently); always report the mode.

CLI:
    PYTHONPATH=/tmp/shim:src:. python scripts/raggov_score.py
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from raggov.engine import DiagnosisEngine
from raggov.models.chunk import RetrievedChunk
from raggov.models.run import RAGRun

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "evals" / "govrag_calib" / "govrag_calib_150.jsonl"
PROBE = ROOT / "evals" / "govrag_calib" / "staging" / "raw" / "induced_candidates.jsonl"

_PLACEHOLDER_IDS = {"gc-PENDING"}


def _engine(mode: str) -> DiagnosisEngine:
    if mode == "native":
        return DiagnosisEngine(config={"mode": "native"})
    if mode == "default":
        return DiagnosisEngine()
    raise ValueError(f"unknown mode: {mode!r} (use 'default' or 'native')")


def _citation_ids(case: dict[str, Any]) -> list[str]:
    raw = case.get("citations") or case.get("cited_doc_ids") or []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            doc = item.get("doc_id") or item.get("source_doc_id")
            if doc:
                out.append(str(doc))
        elif item is not None:
            out.append(str(item))
    return out


def build_run(case: dict[str, Any]) -> RAGRun:
    """Build a RAGRun from any of the calib/probe/heldout row schemas."""
    chunks: list[RetrievedChunk] = []
    for i, chunk in enumerate(case.get("retrieved_chunks", [])):
        doc_id = chunk.get("doc_id") or chunk.get("source_doc_id") or f"doc-{i}"
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk.get("chunk_id", f"chunk-{i}"),
                text=chunk.get("text", ""),
                source_doc_id=str(doc_id),
                score=chunk.get("score"),
                metadata=chunk.get("metadata") or {},
            )
        )
    return RAGRun(
        query=case.get("query", ""),
        retrieved_chunks=chunks,
        final_answer=case.get("answer") or case.get("final_answer") or "",
        cited_doc_ids=_citation_ids(case),
        metadata=case.get("metadata") or {},
    )


def _expected(case: dict[str, Any]) -> str | None:
    return case.get("expected_primary_failure") or case.get("primary_failure")


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_file(
    path: str | Path,
    mode: str = "default",
    splits: Iterable[str] | None = None,
    engine: DiagnosisEngine | None = None,
) -> dict[str, Any]:
    """Score a JSONL dataset. Returns overall + per-type accuracy and confidence.

    splits: if given, only rows whose `split` is in this set are scored (used for the
            canonical Calib train/dev/heldout subset). Placeholder ids are skipped.
    """
    path = Path(path)
    eng = engine or _engine(mode)
    split_set = set(splits) if splits is not None else None
    rows = _load_rows(path)
    n = correct = 0
    by_type: dict[str, list[int]] = {}
    confidences: list[float] = []
    is_calib = path.name == CALIB.name
    for case in rows:
        if is_calib and case.get("case_id") in _PLACEHOLDER_IDS:
            continue
        if split_set is not None and case.get("split") not in split_set:
            continue
        expected = _expected(case)
        if not expected:
            continue
        try:
            diagnosis = eng.diagnose(build_run(case))
            got = diagnosis.primary_failure.value
            conf = getattr(diagnosis, "confidence", None)
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
        except Exception as exc:  # measurement must never crash the whole run
            got = f"ERR:{type(exc).__name__}"
        ok = int(got == expected)
        n += 1
        correct += ok
        bucket = by_type.setdefault(expected, [0, 0])
        bucket[0] += 1
        bucket[1] += ok
    return {
        "dataset": path.name,
        "mode": mode,
        "splits": sorted(split_set) if split_set else None,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "confidence_mean": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "confidence_n": len(confidences),
        "per_type": {
            t: {"n": v[0], "correct": v[1], "accuracy": round(v[1] / v[0], 4)}
            for t, v in sorted(by_type.items(), key=lambda kv: -kv[1][0])
        },
    }


def _print(report: dict[str, Any]) -> None:
    print(
        f"{report['dataset']} [{report['mode']}] splits={report['splits']}: "
        f"{report['correct']}/{report['n']} = {report['accuracy']} "
        f"(confidence_mean={report['confidence_mean']})"
    )
    for t, v in report["per_type"].items():
        print(f"    {t:24} {v['correct']}/{v['n']} = {v['accuracy']}")


def main() -> None:
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Canonical RagGov scorer")
    parser.add_argument("--mode", default="default", choices=["default", "native"])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    reports = [
        score_file(CALIB, mode=args.mode, splits={"train", "dev", "heldout"}),
        score_file(PROBE, mode=args.mode),
    ]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            _print(report)


if __name__ == "__main__":
    main()
