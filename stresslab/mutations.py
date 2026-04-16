"""Failure injection mutations for stress testing."""

from __future__ import annotations

from stresslab.ingest.models import ParsedDocument, ParsedNode
from stresslab.chunking import ChunkRecord


def flatten_hierarchy(doc: ParsedDocument) -> ParsedDocument:
    """Flatten rule hierarchy by removing parent-child relationships.

    Injects: drop_parent_child_links
    Effect: All rules appear as siblings, loss of nested structure context.
    """
    if not doc.nodes:
        return doc

    # Create flat copies with section_path reduced to single level
    flattened_nodes = []
    for node in doc.nodes:
        flattened_node = ParsedNode(
            node_id=node.node_id,
            label=node.label,
            text=node.text,
            page_start=node.page_start,
            page_end=node.page_end,
            section_path=[node.label] if node.label else ["ROOT"],  # Flatten to single level
        )
        flattened_nodes.append(flattened_node)

    return ParsedDocument(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        title=doc.title,
        nodes=flattened_nodes,
    )


def collapse_tables(doc: ParsedDocument) -> ParsedDocument:
    """Collapse structured table content into plain text.

    Injects: flatten_statement_rows, collapse_statement_columns
    Effect: Loss of row/column associations, table structure destroyed.
    """
    if not doc.nodes:
        return doc

    # Convert all nodes to plain text, destroying any table markers
    collapsed_nodes = []
    for node in doc.nodes:
        # Remove table-like structure indicators
        text = node.text.replace("|", " ").replace("—", " ")

        collapsed_node = ParsedNode(
            node_id=node.node_id,
            label=node.label,
            text=text,
            page_start=node.page_start,
            page_end=node.page_end,
            section_path=node.section_path,
        )
        collapsed_nodes.append(collapsed_node)

    return ParsedDocument(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        title=doc.title,
        nodes=collapsed_nodes,
    )


def duplicate_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Duplicate near-identical chunks to collapse embeddings.

    Injects: duplicate_near_identical_chunks
    Effect: Multiple semantically identical chunks ranked together, loss of distinction.
    """
    if len(chunks) < 3:
        return chunks

    # Pick a chunk and duplicate it multiple times with slight variations
    duplicated = list(chunks)
    if chunks:
        base_chunk = chunks[0]
        for i in range(3):
            dup_chunk = ChunkRecord(
                chunk_id=f"{base_chunk.chunk_id}_dup_{i}",
                text=base_chunk.text,  # Exact duplicate
                source_doc_id=base_chunk.source_doc_id,
                page_start=base_chunk.page_start,
                page_end=base_chunk.page_end,
                section_path=base_chunk.section_path,
                parent_node_id=base_chunk.parent_node_id,
                chunk_strategy=base_chunk.chunk_strategy,
            )
            duplicated.append(dup_chunk)

    return duplicated


def constrain_top_k(chunks: list[ChunkRecord], exclude_indices: set[int]) -> list[ChunkRecord]:
    """Artificially exclude critical chunks from retrieval pool.

    Injects: top_k_excludes_exception_chunk
    Effect: Critical evidence missing from top-k results, incomplete answer.
    """
    # Keep all chunks but mark excluded ones as unretrievable
    # In real implementation, would remove from index
    return [chunk for i, chunk in enumerate(chunks) if i not in exclude_indices]


def oversegment(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Force excessive chunking by splitting each chunk.

    Injects: oversegmentation_ms15
    Effect: Evidence fragmented, context windows too small, incoherent retrieval.
    """
    oversegmented = []
    for chunk in chunks:
        # Split chunk into 2-3 parts
        text = chunk.text
        if len(text) > 100:
            mid = len(text) // 2

            # Part 1
            oversegmented.append(ChunkRecord(
                chunk_id=f"{chunk.chunk_id}_p1",
                text=text[:mid],
                source_doc_id=chunk.source_doc_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                parent_node_id=chunk.parent_node_id,
                chunk_strategy=chunk.chunk_strategy,
            ))

            # Part 2
            oversegmented.append(ChunkRecord(
                chunk_id=f"{chunk.chunk_id}_p2",
                text=text[mid:],
                source_doc_id=chunk.source_doc_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                parent_node_id=chunk.parent_node_id,
                chunk_strategy=chunk.chunk_strategy,
            ))
        else:
            oversegmented.append(chunk)

    return oversegmented


def undersegment(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Force insufficient chunking by merging all chunks.

    Injects: undersegmentation_ms20
    Effect: Context bloat, mixed topics, retrieval noise, hard to ground specific claims.
    """
    if not chunks:
        return chunks

    # Merge all chunks into 1 giant chunk
    all_text = " ".join(chunk.text for chunk in chunks)
    merged_chunk = ChunkRecord(
        chunk_id="merged_undersegmented",
        text=all_text,
        source_doc_id=chunks[0].source_doc_id,
        page_start=min(c.page_start for c in chunks),
        page_end=max(c.page_end for c in chunks),
        section_path=["ROOT"],
        parent_node_id=None,
        chunk_strategy="undersegmented",
    )

    return [merged_chunk]
