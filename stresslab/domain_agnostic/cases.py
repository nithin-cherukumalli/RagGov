from __future__ import annotations

from copy import deepcopy
from typing import Any


FAILURE_MODES = [
    ("retrieval_miss", "INSUFFICIENT_CONTEXT", "RETRIEVAL", "retriever", "COVERAGE_EXPANSION"),
    ("retrieval_noise", "RETRIEVAL_ANOMALY", "RETRIEVAL", "ranker", "NOISE_FILTER"),
    ("unsupported_claim", "UNSUPPORTED_CLAIM", "GROUNDING", "claim_verifier", "SOURCE_VERIFICATION"),
    ("contradicted_claim", "CONTRADICTED_CLAIM", "GROUNDING", "claim_verifier", "ANSWER_REWRITE"),
    ("citation_mismatch", "CITATION_MISMATCH", "GROUNDING", "citation_checker", "CITATION_REPAIR"),
    ("stale_deprecated_source", "STALE_RETRIEVAL", "RETRIEVAL", "source_validity", "FRESHNESS_FILTER"),
    ("insufficient_context", "INSUFFICIENT_CONTEXT", "SUFFICIENCY", "sufficiency", "ABSTENTION_THRESHOLD"),
    ("incomplete_answer", "INSUFFICIENT_CONTEXT", "GENERATION", "answer_completeness", "ANSWER_COMPLETION"),
    ("weak_grounding", "UNSUPPORTED_CLAIM", "GROUNDING", "grounding", "EVIDENCE_STRENGTHENING"),
]


DOMAINS = {
    "software_docs": {
        "query": "How do retries work in SDK version 2.4?",
        "answer": "Retries are enabled by default in SDK version 2.4.",
        "support": "SDK version 2.4 retries timeout errors twice by default.",
        "contradiction": "SDK version 2.4 disables retries by default.",
        "stale": "SDK version 1.9 retry API is deprecated by sdk-v2.4.",
    },
    "healthcare_guideline": {
        "query": "What dose applies if renal impairment is present?",
        "answer": "The dose is 5 mg daily for renal impairment.",
        "support": "For renal impairment, the guideline recommends 2.5 mg daily.",
        "contradiction": "For renal impairment, 5 mg daily is contraindicated.",
        "stale": "The 2020 dosing guideline is superseded by guideline-2025.",
    },
    "finance_insurance": {
        "query": "Which advisory fee applies after the 2026 effective date?",
        "answer": "The advisory fee is 1.2% after the 2026 effective date.",
        "support": "Effective 2026-01-01, the advisory fee is 1.2%.",
        "contradiction": "Effective 2026-01-01, the advisory fee is 0.8%.",
        "stale": "The 2024 advisory disclosure expired 2025.",
    },
    "product_manual_support": {
        "query": "Can every filter be washed?",
        "answer": "Every filter can be washed monthly.",
        "support": "Only the washable pre-filter can be rinsed monthly.",
        "contradiction": "The HEPA filter must not be washed.",
        "stale": "Manual version 1.0 is replaced by manual-v2.",
    },
    "scientific_paper": {
        "query": "How much did treatment improve outcomes compared with placebo?",
        "answer": "Treatment improved outcomes by 12% compared with placebo.",
        "support": "Treatment improved outcomes by 12% compared with placebo in the randomized trial.",
        "contradiction": "Treatment improved outcomes by 3% compared with placebo in the randomized trial.",
        "stale": "The preprint draft was withdrawn before peer review.",
    },
    "enterprise_kb": {
        "query": "When should support escalate priority incidents?",
        "answer": "Support should escalate priority incidents after 30 minutes.",
        "support": "Priority incidents must be escalated after 30 minutes without owner assignment.",
        "contradiction": "Priority incidents must be escalated after 10 minutes without owner assignment.",
        "stale": "The escalation KB article is deprecated by kb-2026-incident-flow.",
    },
}


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for domain, template in DOMAINS.items():
        for mode, primary, stage, node, fix in FAILURE_MODES:
            cases.append(_case(domain, template, mode, primary, stage, node, fix))
    return cases


def _case(
    domain: str,
    template: dict[str, str],
    mode: str,
    primary: str,
    stage: str,
    node: str,
    fix: str,
) -> dict[str, Any]:
    base = {
        "case_id": f"{domain}_{mode}",
        "domain": domain,
        "query": template["query"],
        "answer": template["answer"],
        "citations": ["c1"],
        "expected_primary_failure": primary,
        "expected_stage": stage,
        "expected_first_failing_node": node,
        "expected_fix_category": fix,
        "expected_no_government_logic_used": True,
    }

    chunks = [
        {
            "chunk_id": "c1",
            "source_doc_id": "doc-current",
            "text": template["support"],
            "score": 0.9,
            "metadata": {"status": "current"},
        }
    ]

    if mode == "retrieval_miss":
        chunks = []
        base["citations"] = []
    elif mode == "retrieval_noise":
        chunks.append(
            {
                "chunk_id": "c2",
                "source_doc_id": "doc-noise",
                "text": "Unrelated onboarding text with matching common terms but no answer evidence.",
                "score": 0.88,
                "metadata": {"status": "current", "noise": True},
            }
        )
    elif mode == "unsupported_claim":
        chunks[0]["text"] = template["support"]
        base["answer"] = template["answer"] + " It also includes an unsupported premium exception."
    elif mode == "contradicted_claim":
        chunks[0]["text"] = template["contradiction"]
    elif mode == "citation_mismatch":
        base["citations"] = ["missing-doc"]
    elif mode == "stale_deprecated_source":
        chunks[0]["text"] = template["stale"]
        chunks[0]["source_doc_id"] = "doc-stale"
        chunks[0]["metadata"] = {"status": "deprecated", "deprecated_by": "doc-current"}
    elif mode == "insufficient_context":
        chunks[0]["text"] = "The retrieved context covers the default case but omits the requested condition."
    elif mode == "incomplete_answer":
        base["answer"] = "The answer mentions the topic but omits the requested value, condition, or baseline."
    elif mode == "weak_grounding":
        chunks[0]["text"] = "The context is topically related but does not explicitly support the answer."

    base["retrieved_chunks"] = deepcopy(chunks)
    return base
