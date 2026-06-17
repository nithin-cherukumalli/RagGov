#!/usr/bin/env python3
"""Convert seed-intake JSONL into live-format calibration *candidate* cases.

Tier-2 induction + labelled-source mapping (see AUTHORING_GUIDE.md):

  kind="clean"            -> emits a CLEAN case + induced failure variants
                            (INSUFFICIENT_CONTEXT, UNSUPPORTED_CLAIM, CITATION_MISMATCH),
                            each labelled *by construction*.
  kind="labelled_failure" -> maps source_label to a RagGov FailureType.

Output is written to staging/raw/induced_candidates.jsonl (gitignored) with
case_id="gc-PENDING" and split="unset". Nothing is added to the canonical
dataset here — review the candidates, then append the good ones with
scripts/add_calib_case.py (which assigns immutable ids and reminds you to
re-lock + log). Induced labels are by-construction; this script never invents a
gold label for a real (non-induced) case.

Usage:
    python scripts/induce_cases.py [seed_intake.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from add_calib_case import validate, _failure_types, _rows  # noqa: E402

DEFAULT_IN = ROOT / "evals" / "govrag_calib" / "staging" / "raw" / "starter_seed_intake.jsonl"
OUT = ROOT / "evals" / "govrag_calib" / "staging" / "raw" / "induced_candidates.jsonl"

LABEL_MAP = {
    "contradicted": ("CONTRADICTED_CLAIM", "GROUNDING", {"expected_answer_quality_issue": "ignores_context"}),
    "unsupported": ("UNSUPPORTED_CLAIM", "GROUNDING", {}),
    "citation": ("CITATION_MISMATCH", "CITATION", {}),
    "prompt_injection": ("PROMPT_INJECTION", "SECURITY", {"expected_security_issue": "prompt_injection"}),
    "poisoning": ("SUSPICIOUS_CHUNK", "SECURITY", {"expected_security_issue": "poisoning"}),
    "privacy": ("PRIVACY_VIOLATION", "SECURITY", {"expected_security_issue": "privacy"}),
}


def _chunks(passages: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(passages, 1):
        out.append({
            "chunk_id": f"chunk-{i}",
            "doc_id": p.get("doc_id") or f"doc-{i}",
            "text": p.get("text", ""),
            "rank": i,
            "score": p.get("score"),
            "metadata": {},
        })
    return out


def _base(seed: dict, **over) -> dict:
    """A live-format case skeleton with neutral expected_* defaults."""
    case = {
        "case_id": "gc-PENDING",
        "domain": seed.get("domain") or "wiki",
        "source_type": "synthetic_mutation",
        "query": seed.get("query") or "",
        "retrieved_chunks": _chunks(seed.get("passages") or []),
        "answer": seed.get("reference_answer") or "",
        "citations": [],
        "expected_primary_failure": "CLEAN",
        "expected_stage": "NONE",
        "expected_first_failing_node": None,
        "expected_root_cause": "",
        "expected_secondary_failures": [],
        "expected_claim_labels": [],
        "expected_citation_labels": [],
        "expected_retrieval_issue": "none",
        "expected_sufficiency_issue": "none",
        "expected_version_issue": "none",
        "expected_answer_quality_issue": "none",
        "expected_security_issue": "none",
        "expected_fix_category": "none",
        "expected_human_review_required": False,
        "label_source": "synthetic_mutation",
        "label_confidence": "high",
        "split": "unset",
        "notes": "",
    }
    case.update(over)
    return case


def from_clean(seed: dict) -> list[dict]:
    chunks = _chunks(seed.get("passages") or [])
    if not chunks:
        return []
    support = set(seed.get("supporting_doc_ids") or [])
    support_docs = [c["doc_id"] for c in chunks if c["doc_id"] in support] or [chunks[0]["doc_id"]]
    src = f"{seed.get('source_dataset')}:{seed.get('source_id')}"
    cases = []

    # 1) CLEAN
    cases.append(_base(seed, citations=support_docs,
                       expected_root_cause="Answer is supported by the retrieved evidence.",
                       notes=f"Induced from clean source {src}."))

    # 2) INSUFFICIENT_CONTEXT — drop the supporting chunk(s); keep distractors.
    remaining = [c for c in chunks if c["doc_id"] not in support]
    if remaining:
        cases.append(_base(
            seed, retrieved_chunks=remaining, citations=[],
            expected_primary_failure="INSUFFICIENT_CONTEXT", expected_stage="RETRIEVAL",
            expected_sufficiency_issue="insufficient",
            expected_fix_category="widen_retrieval",
            expected_root_cause="The chunk(s) containing the answer were removed; "
            "remaining context is insufficient.",
            notes=f"Induced INSUFFICIENT_CONTEXT from {src} (dropped {sorted(support)})."))

    # 3) UNSUPPORTED_CLAIM — append a claim with no support in any chunk.
    unsup_answer = (seed.get("reference_answer") or "").rstrip(". ") + \
        ". The source also notes this was formally reaffirmed at a later international summit."
    cases.append(_base(
        seed, answer=unsup_answer, citations=support_docs,
        expected_primary_failure="UNSUPPORTED_CLAIM", expected_stage="GROUNDING",
        expected_fix_category="add_citation_grounding",
        expected_root_cause="The appended 'reaffirmed at a later international summit' "
        "claim is not supported by any retrieved chunk.",
        notes=f"Induced UNSUPPORTED_CLAIM from {src} (appended unsupported sentence)."))

    # 4) CITATION_MISMATCH — cite a doc that is not in the retrieved set.
    cases.append(_base(
        seed, citations=["doc-not-retrieved-EXT"],
        expected_primary_failure="CITATION_MISMATCH", expected_stage="CITATION",
        expected_citation_labels=[{"doc_id": "doc-not-retrieved-EXT", "label": "not_retrieved"}],
        expected_fix_category="fix_citation",
        expected_root_cause="The answer cites doc-not-retrieved-EXT, which is absent "
        "from the retrieved context.",
        notes=f"Induced CITATION_MISMATCH from {src} (citation points outside retrieval)."))
    return cases


def from_labelled(seed: dict) -> list[dict]:
    label = (seed.get("source_label") or "").lower()
    if label not in LABEL_MAP:
        return []
    ftype, stage, extra = LABEL_MAP[label]
    src = f"{seed.get('source_dataset')}:{seed.get('source_id')}"
    # Injection seeds often carry no query/answer — synthesize a benign frame.
    query = seed.get("query") or "How do I complete this task?"
    answer = seed.get("reference_answer") or "Follow the documented steps in the product settings."
    confidence = "high" if label in ("contradicted", "unsupported", "prompt_injection") else "medium"
    case = _base(
        seed, query=query, answer=answer,
        citations=[c["doc_id"] for c in _chunks(seed.get("passages") or [])][:1],
        expected_primary_failure=ftype, expected_stage=stage,
        label_source="benchmark_migrated", label_confidence=confidence,
        expected_root_cause=f"Source-labelled {label} case migrated from {seed.get('source_dataset')}.",
        notes=f"Mapped from labelled source {src} (source_label={label}). "
        f"{'Citation mapping is heuristic — review before scoring.' if confidence == 'medium' else ''}".strip(),
    )
    case.update(extra)
    # Citation-mismatch labelled cases legitimately cite outside retrieval.
    if ftype == "CITATION_MISMATCH":
        case["citations"] = ["doc-cited-not-retrieved"]
    return [case]


def main() -> int:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    if not in_path.exists():
        print(f"induce_cases: input not found: {in_path}")
        print("Run scripts/pull_seed_intake.py first (on a machine with internet).")
        return 1

    seeds = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    ftypes = _failure_types()
    existing_ids = {r["case_id"] for r in _rows()}

    candidates: list[dict] = []
    for s in seeds:
        candidates += from_clean(s) if s.get("kind") == "clean" else from_labelled(s)

    # Validate every candidate (hard errors block; warns are fine).
    blocked = 0
    from collections import Counter
    by_type: Counter = Counter()
    with OUT.open("w") as fh:
        for c in candidates:
            errs = [e for e in validate(c, existing_ids, ftypes) if not e.startswith("WARN")]
            if errs:
                blocked += 1
                print(f"  blocked ({c['expected_primary_failure']}): {errs[0]}")
                continue
            fh.write(json.dumps(c) + "\n")
            by_type[c["expected_primary_failure"]] += 1

    print(f"induce_cases: {len(seeds)} seeds -> {sum(by_type.values())} valid candidates "
          f"({blocked} blocked) -> {OUT}")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:<22} {n}")
    print("Next: review candidates, then append good ones with scripts/add_calib_case.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
