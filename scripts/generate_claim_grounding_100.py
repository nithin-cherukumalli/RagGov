import json
from pathlib import Path
import random

def generate():
    cases = []
    case_idx = 1
    
    domains = ["software", "healthcare", "finance", "product_manuals", "science", "education"]
    gov_domain = "government"
    
    def get_domain(idx):
        if idx % 5 == 0:
            return gov_domain
        return domains[idx % len(domains)]
        
    def add_case(gold_label, domain, claim_text, chunk_text, **kwargs):
        nonlocal case_idx
        case = {
            "case_id": f"cgc-100-{case_idx:03d}",
            "domain": domain,
            "query": "What is the policy?",
            "answer": claim_text,
            "claim_text": claim_text,
            "retrieved_chunks": [
                {"chunk_id": "chunk_1", "text": chunk_text, "source_doc_id": "doc_1", "score": 0.9}
            ],
            "cited_doc_ids": ["doc_1"],
            "gold_label": gold_label,
            "claim_type": kwargs.get("claim_type", "general_factual"),
            "atomicity_status": "compound" if kwargs.get("is_compound") else "atomic",
            "gold_supporting_chunk_ids": ["chunk_1"] if gold_label == "entailed" else [],
            "gold_contradicting_chunk_ids": ["chunk_1"] if gold_label == "contradicted" else [],
            "critical_entities": kwargs.get("critical_entities", []),
            "critical_values": kwargs.get("critical_values", []),
            "critical_dates": kwargs.get("critical_dates", []),
            "is_compound": kwargs.get("is_compound", False),
            "expected_safety_gate": kwargs.get("expected_safety_gate", None)
        }
        cases.append(case)
        case_idx += 1

    # 1. 25 Supported Claims
    for i in range(25):
        domain = get_domain(i)
        claim = f"The {domain} policy allows full access."
        chunk = f"According to the guidelines, the {domain} policy allows full access to all users."
        add_case("entailed", domain, claim, chunk)

    # 2. 20 Insufficient
    for i in range(20):
        domain = get_domain(i)
        claim = f"The {domain} system requires a premium subscription."
        chunk = f"The {domain} system provides basic access for free."
        add_case("unsupported", domain, claim, chunk)

    # 3. 20 Contradicted
    for i in range(20):
        domain = get_domain(i)
        claim = f"The {domain} process is entirely manual."
        chunk = f"The {domain} process is fully automated and requires no manual intervention."
        add_case("contradicted", domain, claim, chunk)

    # 4. 10 Numeric Conflicts
    for i in range(10):
        domain = get_domain(i)
        claim = f"The limit is 500 users."
        chunk = f"The limit is established at 200 users for this tier."
        add_case("contradicted", domain, claim, chunk, critical_values=["500"], expected_safety_gate="missing_critical_fact_coverage")

    # 5. 10 Date/Time Conflicts
    for i in range(10):
        domain = get_domain(i)
        claim = f"The deadline is March 15, 2024."
        chunk = f"All submissions are due by April 1, 2024."
        add_case("contradicted", domain, claim, chunk, critical_dates=["March 15, 2024"], expected_safety_gate="missing_critical_fact_coverage")

    # 6. 10 Citation Cases
    for i in range(10):
        domain = get_domain(i)
        claim = f"The {domain} rule is strict."
        chunk = f"The {domain} rule is strict according to section 4."
        add_case("entailed", domain, claim, chunk)

    # 7. 5 Compound Claims
    for i in range(5):
        domain = get_domain(i)
        claim = f"The {domain} policy requires X and Y."
        chunk = f"The {domain} policy requires X but not Y."
        add_case("unsupported", domain, claim, chunk, is_compound=True, expected_safety_gate="compound_claim_not_fully_covered")

    # 8. 10 Paraphrase Support
    for i in range(10):
        domain = get_domain(i)
        claim = f"Users must authenticate via two factors."
        chunk = f"MFA (Multi-Factor Authentication) is a mandatory requirement for all individuals logging in."
        add_case("entailed", domain, claim, chunk)

    out_path = Path("evals/claim_grounding/claim_grounding_100.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
            
    print(f"Generated {len(cases)} cases to {out_path}")

if __name__ == "__main__":
    generate()
