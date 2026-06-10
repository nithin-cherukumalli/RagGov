import json
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from raggov.engine import diagnose
from stresslab.cases.golden.rag_failures import GOLDEN_RAG_FAILURES
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
from raggov.models.run import RAGRun

target_cases = [
    "no_claim_clean",
    "retrieval_irrelevant_plausible_09",
    "retrieval_duplicate_chunks_11",
    "grounding_unsupported_17",
    "grounding_date_hallucination_20",
    "quality_incomplete_38",
    "quality_ambiguous_query_40",
    "quality_ignores_context_41",
    "retrieval_semantic_entropy_high_44",
    "grounding_complex_claim_split_45"
]

_EXTERNAL_CONFIG = {
    "mode": "external-enhanced",
    "enable_ncv": True,
    "enable_a2p": True,
    "use_llm": False,
    "enabled_external_providers": ["ragas", "deepeval"],
}

_NATIVE_CONFIG = {
    "mode": "native",
    "enable_ncv": True,
    "enable_a2p": True,
    "use_llm": False,
}

def load_fixture_run(fixture_path):
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = [RetrievedChunk.model_validate(c) for c in payload.get("retrieved_chunks", [])]
    entries = [
        CorpusEntry.model_validate(e)
        for e in payload.get("corpus_entries", payload.get("corpus_metadata", {}).get("entries", []))
    ]
    metadata = dict(payload.get("metadata", {}))
    if "corpus_metadata" in payload:
        metadata["corpus_metadata"] = payload["corpus_metadata"]
    if payload.get("parser_validation_profile") is not None:
        metadata["parser_validation_profile"] = payload["parser_validation_profile"]
    if "citations" in payload:
        metadata["citations"] = payload["citations"]
    return RAGRun(
        run_id=payload.get("run_id", payload.get("case_id", "test-case")),
        query=payload["query"],
        retrieved_chunks=chunks,
        final_answer=payload["final_answer"],
        cited_doc_ids=payload.get("cited_doc_ids", []),
        answer_confidence=payload.get("answer_confidence"),
        trace=payload.get("trace"),
        corpus_entries=entries,
        metadata=metadata,
    )

def evaluate_case(case_id):
    if case_id == "no_claim_clean":
        run = load_fixture_run(root / "tests" / "fixtures" / "govrag_evidence_30" / "no_claim_clean.json")
        config = _EXTERNAL_CONFIG
        expected_primary = "CLEAN"
        expected_stage = None
    else:
        golden_case = next(c for c in GOLDEN_RAG_FAILURES if c.case_id == case_id)
        run = RAGRun(
            run_id=golden_case.case_id,
            query=golden_case.query,
            retrieved_chunks=[RetrievedChunk(**c) for c in golden_case.retrieved_chunks],
            final_answer=golden_case.final_answer,
            cited_doc_ids=golden_case.cited_doc_ids or [],
        )
        config = _NATIVE_CONFIG
        expected_primary = getattr(golden_case.expected_primary_failure, "value", golden_case.expected_primary_failure) if golden_case.expected_primary_failure else "CLEAN"
        expected_stage = getattr(golden_case.expected_root_cause_stage, "value", golden_case.expected_root_cause_stage) if golden_case.expected_root_cause_stage else None

    diagnosis = diagnose(run, config=config)
    actual_primary = diagnosis.primary_failure.value if diagnosis.primary_failure else "CLEAN"
    actual_stage = diagnosis.root_cause_stage.value if diagnosis.root_cause_stage else None
    
    trace = diagnosis.diagnosis_decision_trace
    if not trace:
        trace = {}
        
    all_candidates = trace.get("alternatives_considered", []) + trace.get("suppressed_candidates", [])
    if trace.get("selected_primary_failure") and trace.get("selected_primary_failure") != "CLEAN":
        all_candidates.append({
            "failure_type": trace.get("selected_primary_failure"),
            "analyzer_name": trace.get("selected_analyzer"),
            "status": "fail", # approx
            "evidence_tier": trace.get("selected_tier"),
        })

    candidates = []
    for c in all_candidates:
        if c.get("failure_type"):
            candidates.append({
                "failure_type": c.get("failure_type"),
                "analyzer_name": c.get("analyzer_name"),
                "status": c.get("status"),
                "tier": c.get("evidence_tier"),
                "weight": c.get("weight", 1.0)
            })

    expected_candidate_emitted = any(c["failure_type"] == expected_primary for c in candidates)
    
    return {
        "case_id": case_id,
        "expected": {"primary": expected_primary, "stage": expected_stage},
        "actual": {"primary": actual_primary, "stage": actual_stage},
        "candidates": candidates,
        "selection_trace": {
            "selected_primary": trace.get("selected_primary_failure"),
            "selected_analyzer": trace.get("selected_analyzer"),
            "selection_reason": trace.get("selection_reason"),
            "suppressed_candidates": trace.get("suppressed_candidates", []),
            "warnings": trace.get("warnings", []),
        },
        "expected_candidate_emitted": expected_candidate_emitted,
        "issue_type": "policy_precedence" if expected_candidate_emitted else "analyzer_limitation"
    }

results = []
for case_id in target_cases:
    results.append(evaluate_case(case_id))

triage_json_path = root / "reports" / "decision_policy_cleanup_triage.json"
triage_md_path = root / "reports" / "decision_policy_cleanup_triage.md"

with open(triage_json_path, "w") as f:
    json.dump(results, f, indent=2)

md_content = "# Decision Policy Cleanup Triage\n\n"
for res in results:
    md_content += f"## {res['case_id']}\n"
    md_content += f"- **Expected:** {res['expected']['primary']} ({res['expected']['stage']})\n"
    md_content += f"- **Actual:** {res['actual']['primary']} ({res['actual']['stage']})\n"
    md_content += f"- **Expected Candidate Emitted:** {res['expected_candidate_emitted']}\n"
    md_content += f"- **Issue Type:** {res['issue_type']}\n"
    md_content += "### Candidates\n"
    for c in res['candidates']:
        md_content += f"  - `{c['failure_type']}` from `{c['analyzer_name']}` (status={c['status']}, tier={c['tier']})\n"
    md_content += "### Selection Trace\n"
    md_content += f"- **Selected Analyzer:** {res['selection_trace']['selected_analyzer']}\n"
    md_content += f"- **Reason:** {res['selection_trace']['selection_reason']}\n\n"

with open(triage_md_path, "w") as f:
    f.write(md_content)

print(f"Generated {triage_json_path} and {triage_md_path}")
