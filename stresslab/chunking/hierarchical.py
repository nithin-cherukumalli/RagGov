"""Hierarchy-aware chunker for parsed ingest documents."""

from __future__ import annotations

from collections import defaultdict

from stresslab.ingest.models import ParsedDocument, ParsedNode, ParsedTable

from .base import BaseChunker, ChunkRecord


class HierarchicalChunker(BaseChunker):
    """Chunk parsed nodes at clause boundaries while preserving descendants."""

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkRecord]:
        node_children = self._build_children_index(parsed_doc.nodes)
        descendants_by_node = self._build_descendant_index(parsed_doc.nodes, node_children)
        chunks = [
            self._build_node_chunk(parsed_doc, node, descendants_by_node)
            for node in parsed_doc.nodes
            if node.text.strip()
        ]
        chunks.extend(self._build_table_chunk(parsed_doc, table) for table in parsed_doc.tables)
        return chunks

    def _build_node_chunk(
        self,
        parsed_doc: ParsedDocument,
        node: ParsedNode,
        descendants_by_node: dict[str, list[ParsedNode]],
    ) -> ChunkRecord:
        lineage = [node, *descendants_by_node.get(node.node_id, [])]
        text = "\n\n".join(self._format_node_text(member) for member in lineage if member.text.strip())
        page_end = max(member.page_end for member in lineage)

        return ChunkRecord(
            chunk_id=f"{parsed_doc.doc_id}:{node.node_id}",
            source_doc_id=parsed_doc.doc_id,
            text=text,
            page_start=node.page_start,
            page_end=page_end,
            section_path=list(node.section_path),
            parent_node_id=node.parent_node_id,
            chunk_strategy="hierarchical",
        )

    def _build_table_chunk(self, parsed_doc: ParsedDocument, table: ParsedTable) -> ChunkRecord:
        lines: list[str] = []
        if table.title:
            lines.append(table.title)
        if table.headers:
            lines.append(" | ".join(cell for cell in table.headers if cell))
        for row in table.rows:
            row_text = " | ".join(cell for cell in row if cell)
            if row_text:
                lines.append(row_text)

        text = "\n".join(lines).strip() or table.table_id
        return ChunkRecord(
            chunk_id=f"{parsed_doc.doc_id}:{table.table_id}",
            source_doc_id=parsed_doc.doc_id,
            text=text,
            page_start=table.page,
            page_end=table.page,
            section_path=self._table_section_path(table),
            parent_node_id=None,
            chunk_strategy="hierarchical",
        )

    def _build_children_index(
        self,
        nodes: list[ParsedNode],
    ) -> dict[str | None, list[ParsedNode]]:
        children: dict[str | None, list[ParsedNode]] = defaultdict(list)
        for node in nodes:
            children[node.parent_node_id].append(node)
        return dict(children)

    def _build_descendant_index(
        self,
        nodes: list[ParsedNode],
        node_children: dict[str | None, list[ParsedNode]],
    ) -> dict[str, list[ParsedNode]]:
        descendants_by_node: dict[str, list[ParsedNode]] = {}

        for node in reversed(nodes):
            descendants: list[ParsedNode] = []
            for child in node_children.get(node.node_id, []):
                descendants.append(child)
                descendants.extend(descendants_by_node.get(child.node_id, []))
            descendants_by_node[node.node_id] = descendants

        return descendants_by_node

    def _format_node_text(self, node: ParsedNode) -> str:
        text = node.text.strip()
        if not text:
            return node.label
        return f"{node.label} {text}"

    def _table_section_path(self, table: ParsedTable) -> list[str]:
        if table.title and table.title.strip():
            return [table.title.strip(), table.table_id]
        return [table.table_id]
