from __future__ import annotations

import argparse
import json
import sys

from raggov.analyzers.grounding.claims import ExtractedClaim, LLMAtomicClaimExtractorV1
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate
from raggov.analyzers.grounding.verifiers import LLMClaimEntailmentVerifierV1
from raggov.connectors.groq_client import DEFAULT_GROQ_MODEL, build_groq_client_from_env


def _candidate(chunk_id: str, text: str, doc_id: str, score: float) -> EvidenceCandidate:
    return EvidenceCandidate(
        chunk_id=chunk_id,
        source_doc_id=doc_id,
        chunk_text=text,
        chunk_text_preview=text[:100],
        lexical_overlap_score=0.0,
        anchor_overlap_score=0.0,
        value_overlap_score=0.0,
        retrieval_score=score,
        rerank_score=None,
        candidate_reason="groq_smoke_test",
    )


def run_smoke(verbose: bool = False) -> tuple[dict[str, object], int]:
    client, reason = build_groq_client_from_env()
    model = getattr(client, "model_name", None) or DEFAULT_GROQ_MODEL
    report: dict[str, object] = {
        "connectivity": "skip",
        "extraction": "skip",
        "entailment": "skip",
        "model_used": model,
        "fallback_used": False,
        "rate_limited": False,
        "notes": "",
    }
    if client is None:
        report["notes"] = reason or "Groq client unavailable"
        return report, 2

    notes: list[str] = []

    try:
        response = client.chat('Return exactly this JSON: {"status":"ok","provider":"groq"}')
        payload = json.loads(response)
        report["connectivity"] = (
            "pass"
            if payload.get("status") == "ok" and payload.get("provider") == "groq"
            else "fail"
        )
        if verbose:
            notes.append("connectivity_json_ok" if report["connectivity"] == "pass" else "connectivity_json_bad")
    except Exception as exc:
        report["connectivity"] = "fail"
        notes.append(f"connectivity_error:{type(exc).__name__}")

    try:
        answer = (
            "The Andhra Pradesh order applies to government schools. "
            "It excludes private unaided schools."
        )
        claims = LLMAtomicClaimExtractorV1(client).extract_structured(answer)
        ExtractedClaim.model_validate(claims[0].model_dump())
        texts = [claim.claim_text.lower() for claim in claims]
        extraction_ok = (
            len(claims) >= 2
            and any("government school" in text for text in texts)
            and any("private unaided" in text for text in texts)
            and all(claim.atomicity_status in {"atomic", "unclear"} for claim in claims[:2])
        )
        report["extraction"] = "pass" if extraction_ok else "fail"
    except Exception as exc:
        report["extraction"] = "fail"
        notes.append(f"extraction_error:{type(exc).__name__}")

    try:
        verifier = LLMClaimEntailmentVerifierV1({"llm_client": client})
        result = verifier.verify(
            "The order applies to private unaided schools.",
            "What schools are covered?",
            [
                _candidate("chunk_1", "The order applies to all government schools in Andhra Pradesh.", "doc-1", 0.9),
                _candidate("chunk_2", "Private unaided schools are excluded from this order.", "doc-2", 0.88),
            ],
            metadata={
                "source_sentence": "The order applies to private unaided schools.",
                "claim_type": "policy_rule",
                "entities": ["private unaided schools"],
                "dates": [],
                "numbers": [],
                "atomicity_status": "atomic",
                "cited_doc_ids": [],
                "cited_chunk_ids": [],
            },
        )
        report["fallback_used"] = bool(result.fallback_used)
        report["entailment"] = (
            "pass"
            if result.support_label in {"contradicted", "insufficient_evidence"}
            and result.support_label != "supported"
            else "fail"
        )
        if result.fallback_used:
            notes.append("entailment_fallback_used")
    except Exception as exc:
        report["entailment"] = "fail"
        notes.append(f"entailment_error:{type(exc).__name__}")

    report["rate_limited"] = bool(getattr(client, "stats", None) and client.stats.rate_limited)
    report["notes"] = "; ".join(notes) if notes else "ok"
    exit_code = 0 if all(report[key] == "pass" for key in ("connectivity", "extraction", "entailment")) else 1
    return report, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Groq smoke test for GovRAG claim extraction and entailment.")
    parser.add_argument("--verbose", action="store_true", help="Include short diagnostic notes.")
    args = parser.parse_args()
    report, exit_code = run_smoke(verbose=args.verbose)
    print("Groq live smoke report")
    print(f"- connectivity: {report['connectivity']}")
    print(f"- extraction: {report['extraction']}")
    print(f"- entailment: {report['entailment']}")
    print(f"- model used: {report['model_used']}")
    print(f"- fallback_used: {str(report['fallback_used']).lower()}")
    print(f"- rate_limited: {str(report['rate_limited']).lower()}")
    print(f"- notes: {report['notes']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
