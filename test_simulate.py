import json
from raggov.analyzers.grounding.verifiers import LocalNLIClaimVerifier
from raggov.analyzers.grounding.candidate_selection import EvidenceCandidate, EvidenceCandidateSelector
from raggov.models.chunk import RetrievedChunk

verifier = LocalNLIClaimVerifier({})
selector = EvidenceCandidateSelector()

with open("evals/govrag_calib/staging/raw/heldout_real_v1.jsonl") as f:
    count = 0
    for i, line in enumerate(f):
        case = json.loads(line)
        expected = case.get("expected_primary_failure")
        if expected != "CONTRADICTED_CLAIM":
            continue
        
        answer = case.get("answer", "")
        query = case.get("query", "")
        chunks = [
            RetrievedChunk(
                chunk_id=str(idx),
                text=c.get("text", ""),
                source_doc_id=c.get("doc_id", ""),
                score=c.get("score")
            )
            for idx, c in enumerate(case.get("context", []))
        ]
        
        from raggov.analyzers.grounding.claims import ClaimExtractor
        claims = ClaimExtractor().extract(answer)
        print(f"\n--- Case {i} (expected: {expected}) ---")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        
        for claim in claims:
            candidates = selector.select_candidates(claim, query, chunks)
            res = verifier.verify(claim, query, candidates)
            print(f"  Claim: {claim}")
            print(f"  Result label: {res.label} (support: {res.support_label})")
            print(f"  Rationale: {res.rationale}")
            
        count += 1
        if count >= 3:
            break
