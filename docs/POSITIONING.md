# GovRAG Positioning

## Three Locked Statements

What GovRAG does that other tools do not:
GovRAG tells you which pipeline stage caused a bad RAG answer and what evidence supports that diagnosis, not just that the answer was bad.

Who needs it right now:
An engineer operating a document-QA or retrieval-backed product who keeps getting wrong-answer bug reports and needs stage-specific root-cause diagnosis.

Which single diagnosis saves hours:
`UNSUPPORTED_CLAIM` at `GROUNDING` with evidence showing the answer claim is not supported by the retrieved chunk set, likely because the governing clause was truncated or never retrieved.

## Positioning Boundary

GovRAG is not a RAGAS replacement and not a DeepEval replacement.
It is the diagnosis layer after evaluation reveals that something went wrong.

## Honesty Rule

Lead with native diagnosis strengths.
Do not imply production validation that the repo does not contain.

