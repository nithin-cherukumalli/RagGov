from raggov.engine import diagnose
from raggov.models.run import RAGRun
from raggov.models.chunk import RetrievedChunk
from raggov.models.corpus import CorpusEntry
import json
from pathlib import Path

payload = json.loads(Path("tests/fixtures/govrag_evidence_30/no_claim_clean.json").read_text())
chunks = [RetrievedChunk.model_validate(c) for c in payload.get("retrieved_chunks", [])]
entries = [CorpusEntry.model_validate(e) for e in payload.get("corpus_entries", [])]
run = RAGRun(
    run_id="test",
    query=payload["query"],
    retrieved_chunks=chunks,
    final_answer=payload["final_answer"],
    cited_doc_ids=[],
    corpus_entries=entries,
    metadata=payload.get("metadata", {})
)

config = {
    "mode": "external-enhanced",
    "enable_ncv": True,
    "enable_a2p": True,
    "use_llm": False,
    "enabled_external_providers": ["ragas", "deepeval"],
}
d = diagnose(run, config=config)
print("Primary:", d.primary_failure)
print("Trace:", json.dumps(d.diagnosis_decision_trace, indent=2))
