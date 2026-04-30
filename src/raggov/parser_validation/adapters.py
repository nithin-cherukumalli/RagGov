from __future__ import annotations

from typing import Any

from raggov.parser_validation.models import (
    ChunkIR,
    ChunkingStrategyProfile,
    ChunkingStrategyType,
    ParsedDocumentIR,
    default_chunking_profile,
)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Read either obj.name or obj[name] without assuming object style."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _as_tuple(value: Any) -> tuple:
    """Normalize scalar/list/set/tuple/None values into a tuple."""
    if value is None:
        return ()

    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, set):
        return tuple(value)

    return (value,)


def _safe_int(value: Any) -> int | None:
    """Convert metadata page fields safely into int or None."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_from_chunk(raw_chunk: Any) -> dict[str, Any]:
    metadata = _get_attr_or_key(raw_chunk, "metadata", {}) or {}

    if not isinstance(metadata, dict):
        return {}

    return dict(metadata)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_tuple(value))


# ---------------------------------------------------------------------------
# Chunk adapter
# ---------------------------------------------------------------------------


def chunk_from_rag_chunk(raw_chunk: Any) -> ChunkIR:
    """
    Convert a project chunk or loader chunk into ChunkIR.

    Supported input styles:
    - existing GovRAG retrieved chunk objects
    - dict-based chunks
    - LangChain-like objects with page_content
    - future parser chunks carrying metadata fields

    This adapter is intentionally permissive but deterministic.
    """
    metadata = _metadata_from_chunk(raw_chunk)

    chunk_id = _first_present(
        _get_attr_or_key(raw_chunk, "chunk_id", None),
        _get_attr_or_key(raw_chunk, "id", None),
        metadata.get("chunk_id"),
        "unknown_chunk",
    )

    text = _first_present(
        _get_attr_or_key(raw_chunk, "text", None),
        _get_attr_or_key(raw_chunk, "content", None),
        _get_attr_or_key(raw_chunk, "page_content", None),
        "",
    )

    source_element_ids = _first_present(
        metadata.get("source_element_ids"),
        metadata.get("element_ids"),
        metadata.get("orig_element_ids"),
        metadata.get("orig_elements"),
    )

    source_table_ids = _first_present(
        metadata.get("source_table_ids"),
        metadata.get("table_ids"),
        metadata.get("table_id"),
    )

    page_start_raw = _first_present(
        metadata.get("page_start"),
        metadata.get("page_number"),
        metadata.get("page"),
    )

    page_end_raw = _first_present(
        metadata.get("page_end"),
        page_start_raw,
    )

    section_path = _first_present(
        metadata.get("section_path"),
        metadata.get("headers"),
        metadata.get("header_path"),
        metadata.get("toc_path"),
    )

    return ChunkIR(
        chunk_id=str(chunk_id),
        text=str(text),
        source_element_ids=_string_tuple(source_element_ids),
        source_table_ids=_string_tuple(source_table_ids),
        page_start=_safe_int(page_start_raw),
        page_end=_safe_int(page_end_raw),
        section_path=_string_tuple(section_path),
        metadata=metadata,
    )


def chunks_from_rag_run(run: Any) -> list[ChunkIR]:
    """Convert run.retrieved_chunks into ChunkIR objects."""
    raw_chunks = getattr(run, "retrieved_chunks", None)

    if raw_chunks is None:
        return []

    return [chunk_from_rag_chunk(chunk) for chunk in raw_chunks]


def parsed_doc_from_run_metadata(run: Any) -> ParsedDocumentIR | None:
    """
    Retrieve ParsedDocumentIR from a run if future ingestion stores it there.

    Supported locations:
    - run.parsed_document_ir
    - run.metadata["parsed_document_ir"]

    If unavailable, return None.
    """
    direct = getattr(run, "parsed_document_ir", None)
    if isinstance(direct, ParsedDocumentIR):
        return direct

    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, dict):
        candidate = metadata.get("parsed_document_ir")
        if isinstance(candidate, ParsedDocumentIR):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Chunking strategy resolution
# ---------------------------------------------------------------------------

_STRATEGY_ALIASES: dict[str, ChunkingStrategyType] = {
    # UNKNOWN
    "": ChunkingStrategyType.UNKNOWN,
    "unknown": ChunkingStrategyType.UNKNOWN,
    "none": ChunkingStrategyType.UNKNOWN,
    "null": ChunkingStrategyType.UNKNOWN,
    # FIXED_TOKEN
    "fixed": ChunkingStrategyType.FIXED_TOKEN,
    "fixed_token": ChunkingStrategyType.FIXED_TOKEN,
    "token": ChunkingStrategyType.FIXED_TOKEN,
    "token_chunker": ChunkingStrategyType.FIXED_TOKEN,
    "fixed_size": ChunkingStrategyType.FIXED_TOKEN,
    "fixed_size_token": ChunkingStrategyType.FIXED_TOKEN,
    # RECURSIVE_TEXT
    "recve": ChunkingStrategyType.RECURSIVE_TEXT,
    "recursive_text": ChunkingStrategyType.RECURSIVE_TEXT,
    "recursive_character": ChunkingStrategyType.RECURSIVE_TEXT,
    "recursive_character_text_splitter": ChunkingStrategyType.RECURSIVE_TEXT,
    # SENTENCE
    "sentence": ChunkingStrategyType.SENTENCE,
    "sentence_splitter": ChunkingStrategyType.SENTENCE,
    "sentence_chunker": ChunkingStrategyType.SENTENCE,
    # HIERARCHICAL
    "hierarchical": ChunkingStrategyType.HIERARCHICAL,
    "hierarchy": ChunkingStrategyType.HIERARCHICAL,
    "by_title": ChunkingStrategyType.HIERARCHICAL,
    "unstructured_by_title": ChunkingStrategyType.HIERARCHICAL,
    "title": ChunkingStrategyType.HIERARCHICAL,
    # MARKDOWN_HEADER
    "markdown_header": ChunkingStrategyType.MARKDOWN_HEADER,
    "markdown": ChunkingStrategyType.MARKDOWN_HEADER,
    "header": ChunkingStrategyType.MARKDOWN_HEADER,
    "headers": ChunkingStrategyType.MARKDOWN_HEADER,
    "markdown_headers": ChunkingStrategyType.MARKDOWN_HEADER,
    # SEMANTIC
    "semantic": ChunkingStrategyType.SEMANTIC,
    "semantic_chunker": ChunkingStrategyType.SEMANTIC,
    "semantic_splitter": ChunkingStrategyType.SEMANTIC,
    # TABLE_AWARE
    "table_aware": ChunkingStrategyType.TABLE_AWARE,
    "table": ChunkingStrategyType.TABLE_AWARE,
    "table_chunker": ChunkingStrategyType.TABLE_AWARE,
    "table_preserving": ChunkingStrategyType.TABLE_AWARE,
    # PARENT_CHILD
    "parent_child": ChunkingStrategyType.PARENT_CHILD,
    "parent": ChunkingStrategyType.PARENT_CHILD,
    "child": ChunkingStrategyType.PARENT_CHILD,
    "parent_document": ChunkingStrategyType.PARENT_CHILD,
    "parent_document_retriever": ChunkingStrategyType.PARENT_CHILD,
    # SUMMARY
    "summary": ChunkingStrategyType.SUMMARY,
    "summarized": ChunkingStrategyType.SUMMARY,
    "compressed": ChunkingStrategyType.SUMMARY,
    "compression": ChunkingStrategyType.SUMMARY,
    "contextual_compression": ChunkingStrategyType.SUMMARY,
    # LATE_CHUNKING
    "late_chunking": ChunkingStrategyType.LATE_CHUNKING,
    "late": ChunkingStrategyType.LATE_CHUNKING,
    "late_interaction": ChunkingStrategyType.LATE_CHUNKING,
}


def normalize_chunking_strategy(value: Any) -> ChunkingStrategyType:
    """
    Normalize a raw strategy name or value to a ChunkingStrategyType.

    Unknown strings return UNKNOWN.
    Run-level declarations override chunk-level declarations.
    """
    if value is None:
        return ChunkingStrategyType.UNKNOWN

    if isinstance(value, ChunkingStrategyType):
        return value

    normalized = str(value).lower().strip().replace(" ", "_").replace("-", "_")
    return _STRATEGY_ALIASES.get(normalized, ChunkingStrategyType.UNKNOWN)


def _resolve_chunking_strategy(run: Any) -> tuple[ChunkingStrategyType, str]:
    """
    Resolve chunking strategy from run or chunk metadata.

    Returns (resolved_type, source_label) where source_label is
    "run_metadata", "chunk_metadata", or "unknown".
    """
    # Run-level metadata takes precedence.
    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("chunking_strategy", "chunk_strategy", "splitter_type", "chunker_type"):
            value = metadata.get(key)
            if value is not None:
                return normalize_chunking_strategy(value), "run_metadata"

    # Fall back to the first non-null chunk-level metadata declaration.
    raw_chunks = getattr(run, "retrieved_chunks", None) or []
    for raw_chunk in raw_chunks:
        chunk_meta = getattr(raw_chunk, "metadata", None)
        if not isinstance(chunk_meta, dict):
            continue
        for key in ("chunk_strategy", "chunking_strategy", "splitter_type", "chunker_type"):
            value = chunk_meta.get(key)
            if value is not None:
                return normalize_chunking_strategy(value), "chunk_metadata"

    return ChunkingStrategyType.UNKNOWN, "unknown"


def chunking_profile_from_run_metadata(run: Any) -> ChunkingStrategyProfile:
    """Return the ChunkingStrategyProfile declared by run or chunk metadata."""
    strategy_type, _ = _resolve_chunking_strategy(run)
    return default_chunking_profile(strategy_type)


def chunking_strategy_source_from_run_metadata(run: Any) -> str:
    """
    Return where the chunking strategy was resolved from.

    Returns "run_metadata", "chunk_metadata", or "unknown".
    """
    _, source = _resolve_chunking_strategy(run)
    return source
