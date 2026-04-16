"""Prompt construction for answer generation."""

from __future__ import annotations


def build_prompt(query: str, chunk_texts: list[str]) -> str:
    context_blocks = "\n\n".join(
        f"Context {index}:\n{chunk_text}"
        for index, chunk_text in enumerate(chunk_texts, start=1)
    )
    if not context_blocks:
        context_blocks = "No retrieved context was provided."

    return (
        "Answer the user's question using only the retrieved context.\n"
        "Every factual claim must cite the supporting context inline.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Context:\n{context_blocks}\n\n"
        "Answer:"
    )
