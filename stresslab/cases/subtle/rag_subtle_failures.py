"""Subtle RAG failure golden cases.

These cases focus on failures that are often masked by aggregate scores or 
plausible-sounding generations.
"""

from __future__ import annotations

from raggov.models.diagnosis import FailureStage, FailureType
from raggov.evaluators.base import ExternalEvaluatorProvider, ExternalSignalType
from stresslab.cases.models import RAGFailureGoldenCase

SUBTLE_RAG_FAILURES: list[RAGFailureGoldenCase] = [
    # 1. Correct answer, weak grounding
    RAGFailureGoldenCase(
        case_id="subtle_correct_unsupported_01",
        category="grounding",
        description="Factually correct answer but completely unsupported by retrieved context.",
        query="What is the capital of France?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The Louvre is a famous museum in Paris.", "source_doc_id": "doc1"}
        ],
        final_answer="The capital of France is Paris.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.UNSUPPORTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
        metadata={"factually_correct": True}
    ),
    
    # 2. Relevant retrieval, incomplete answer
    RAGFailureGoldenCase(
        case_id="subtle_incomplete_answer_02",
        category="answer_quality",
        description="Retrieval finds all requirements, but generator misses one.",
        query="List the requirements for a visa: age, income, and health.",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Visa requirements include: 1. Age over 18. 2. Income > $50k. 3. Health insurance.", "source_doc_id": "doc1"}
        ],
        final_answer="To get a visa you must be over 18 and have health insurance.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.INSUFFICIENT_CONTEXT.value, # Conflated with answer completeness in some taxonomies
        expected_root_cause_stage=FailureStage.GENERATION.value,
        expected_recommended_fix_category="generation",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 3. Irrelevant retrieval, plausible answer
    RAGFailureGoldenCase(
        case_id="subtle_plausible_hallucination_03",
        category="retrieval",
        description="Irrelevant chunks retrieved, but generator produces a plausible hallucination.",
        query="What is the interest rate for the 'Standard Plus' account?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The 'Basic' account has a 1% interest rate.", "source_doc_id": "doc1"},
            {"chunk_id": "c2", "text": "Our premium accounts offer higher rates.", "source_doc_id": "doc2"}
        ],
        final_answer="The interest rate for the Standard Plus account is 2.5%.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.SCOPE_VIOLATION.value,
        expected_root_cause_stage=FailureStage.RETRIEVAL.value,
        expected_recommended_fix_category="retrieval",
        expected_human_review_required=True,
        expected_should_have_answered=False,
    ),

    # 4. Related but non-supporting citation
    RAGFailureGoldenCase(
        case_id="subtle_related_non_supporting_04",
        category="citation",
        description="Citation points to a document about the right topic but doesn't contain the specific fact.",
        query="When was the company founded?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Our company has a long history of innovation in the tech sector.", "source_doc_id": "doc1"}
        ],
        final_answer="The company was founded in 1995 [doc1].",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.CITATION_MISMATCH.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 5. Partial context support
    RAGFailureGoldenCase(
        case_id="subtle_partial_support_05",
        category="grounding",
        description="One claim is supported, another is not.",
        query="Tell me about the founder and the headquarters.",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The company was founded by Jane Doe.", "source_doc_id": "doc1"}
        ],
        final_answer="The company was founded by Jane Doe and is headquartered in New York.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.UNSUPPORTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 6. Answer drift
    RAGFailureGoldenCase(
        case_id="subtle_answer_drift_06",
        category="answer_quality",
        description="Answer starts correct but ends with unsupported speculation.",
        query="Is the project on schedule?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The project is currently 80% complete.", "source_doc_id": "doc1"}
        ],
        final_answer="The project is 80% complete and we expect to finish ahead of time next week.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.UNSUPPORTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GENERATION.value,
        expected_recommended_fix_category="generation",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 7. Ambiguous query not detected
    RAGFailureGoldenCase(
        case_id="subtle_ambiguous_query_07",
        category="answer_quality",
        description="Ambiguous query answered confidently without clarification.",
        query="What is the policy for returns?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Returns for online orders are accepted within 30 days.", "source_doc_id": "doc1"},
            {"chunk_id": "c2", "text": "In-store purchases can be returned within 14 days.", "source_doc_id": "doc2"}
        ],
        final_answer="Our policy allows returns within 30 days.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.LOW_CONFIDENCE.value,
        expected_root_cause_stage=FailureStage.UNKNOWN.value,
        expected_recommended_fix_category="query",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 8. Good aggregate score hiding severe local failure
    RAGFailureGoldenCase(
        case_id="subtle_local_contradiction_08",
        category="grounding",
        description="Many claims are correct, but one is a direct contradiction.",
        query="Give me a summary of the report.",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Revenue increased by 10%. Expenses decreased by 5%. Net profit is $1M.", "source_doc_id": "doc1"}
        ],
        final_answer="Revenue grew by 10%, expenses went down by 5%, but net profit fell to $100k.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.CONTRADICTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 9. External evaluator disagreement ignored
    RAGFailureGoldenCase(
        case_id="subtle_external_disagreement_09",
        category="grounding",
        description="Native analyzer says PASS, but RAGAS says high hallucination.",
        query="What is the interest rate?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Interest rate is 5%.", "source_doc_id": "doc1"}
        ],
        final_answer="The interest rate is 5%.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.LOW_CONFIDENCE.value, # Blocked CLEAN due to high severity external signal
        expected_root_cause_stage=FailureStage.UNKNOWN.value,
        expected_recommended_fix_category="none",
        expected_human_review_required=True,
        expected_should_have_answered=True,
        expected_external_signals=[
            {
                "provider": ExternalEvaluatorProvider.ragas.value,
                "signal_type": ExternalSignalType.hallucination.value,
                "metric_name": "faithfulness",
                "value": 0.2,
                "label": "hallucination",
            }
        ]
    ),

    # 10. Keyword overlap false pass
    RAGFailureGoldenCase(
        case_id="subtle_keyword_overlap_10",
        category="grounding",
        description="High keyword overlap between chunk and answer, but semantic meaning is different.",
        query="Who is the manager?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The manager told John that the project is delayed.", "source_doc_id": "doc1"}
        ],
        final_answer="The manager is John.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.UNSUPPORTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),
    
    # 11. Silent failure: Table structure preserved but values swapped
    RAGFailureGoldenCase(
        case_id="subtle_table_value_swap_11",
        category="parser_chunking",
        description="Table structure looks okay, but values are swapped during extraction/chunking.",
        query="What was the revenue in Q1 and Q2?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "| Quarter | Revenue |\n| Q1 | $10M |\n| Q2 | $20M |", "source_doc_id": "doc1"}
        ],
        final_answer="Revenue was $20M in Q1 and $10M in Q2.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.CONTRADICTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 12. Correct answer, no citation
    RAGFailureGoldenCase(
        case_id="subtle_missing_citation_12",
        category="citation",
        description="Correct answer provided but no citation is present.",
        query="What is the tax rate?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The corporate tax rate is 21%.", "source_doc_id": "doc1"}
        ],
        final_answer="The tax rate is 21%.",
        cited_doc_ids=[],
        expected_primary_failure=FailureType.CITATION_MISMATCH.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 13. Near-miss retrieval
    RAGFailureGoldenCase(
        case_id="subtle_near_miss_retrieval_13",
        category="retrieval",
        description="Retrieved chunks are about the right entity but wrong attribute.",
        query="What is the CEO's salary?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "The CEO is responsible for strategic direction.", "source_doc_id": "doc1"},
            {"chunk_id": "c2", "text": "The CEO recently spoke at a conference.", "source_doc_id": "doc2"}
        ],
        final_answer="The CEO's salary is $1.2M.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.INSUFFICIENT_CONTEXT.value,
        expected_root_cause_stage=FailureStage.RETRIEVAL.value,
        expected_recommended_fix_category="retrieval",
        expected_human_review_required=True,
        expected_should_have_answered=False,
    ),

    # 14. Answer overrides specific context constraint
    RAGFailureGoldenCase(
        case_id="subtle_constraint_override_14",
        category="grounding",
        description="Context specifies a constraint (e.g., 'only for US citizens'), but answer ignores it.",
        query="Who can apply for the grant?",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Grants are available for research projects. Only US citizens are eligible.", "source_doc_id": "doc1"}
        ],
        final_answer="Anyone with a research project can apply for the grant.",
        cited_doc_ids=["doc1"],
        expected_primary_failure=FailureType.CONTRADICTED_CLAIM.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),

    # 15. Aggregated pass hiding multiple weak citations
    RAGFailureGoldenCase(
        case_id="subtle_many_weak_citations_15",
        category="citation",
        description="All claims are supported, but citations are imprecise (point to whole doc instead of specific chunk).",
        query="Summarize the three pillars of our strategy.",
        retrieved_chunks=[
            {"chunk_id": "c1", "text": "Pillar 1: Innovation.", "source_doc_id": "doc1"},
            {"chunk_id": "c2", "text": "Pillar 2: Customer Focus.", "source_doc_id": "doc1"},
            {"chunk_id": "c3", "text": "Pillar 3: Sustainability.", "source_doc_id": "doc1"}
        ],
        final_answer="Our strategy is based on innovation, customer focus, and sustainability.",
        cited_doc_ids=["doc1"], # Imprecise citation
        expected_primary_failure=FailureType.CITATION_MISMATCH.value,
        expected_root_cause_stage=FailureStage.GROUNDING.value,
        expected_recommended_fix_category="grounding",
        expected_human_review_required=True,
        expected_should_have_answered=True,
    ),
]
