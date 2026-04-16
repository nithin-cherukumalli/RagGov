"""Fixed-window chunker for parsed ingest documents."""

from __future__ import annotations

from stresslab.ingest.models import ParsedDocument

from .base import BaseChunker, ChunkRecord


class FixedChunker(BaseChunker):
    """Split node text into fixed word windows with overlap."""

    def __init__(self, window_size: int = 200, overlap: int = 50) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")

        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for node in parsed_doc.nodes:
            words = node.text.split()
            if not words:
                continue

            for chunk_index, (start, end) in enumerate(self._window_ranges(len(words))):
                chunk_text = " ".join(words[start:end])
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{parsed_doc.doc_id}:{node.node_id}:{chunk_index}",
                        source_doc_id=parsed_doc.doc_id,
                        text=chunk_text,
                        page_start=node.page_start,
                        page_end=node.page_end,
                        section_path=list(node.section_path),
                        parent_node_id=node.parent_node_id,
                        chunk_strategy="fixed",
                    )
                )
        return chunks

    def _window_ranges(self, word_count: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        step = self.window_size - self.overlap
        start = 0

        while start < word_count:
            end = min(start + self.window_size, word_count)
            ranges.append((start, end))
            if end == word_count:
                break
            start += step

        return ranges
