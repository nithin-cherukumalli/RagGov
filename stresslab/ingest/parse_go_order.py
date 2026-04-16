"""Conservative parser for Andhra Pradesh G.O. order PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

from .models import ParsedDocument, ParsedMetadata, ParsedNode, ParsedTable
from .pdf_extract import extract_pdf_text

_MULTISPACE_RE = re.compile(r"\s+")
_DEPARTMENT_RE = re.compile(r"DEPARTMENT\s*$")
_GO_LINE_RE = re.compile(r"^(?P<go>.+?)\s+Dated\s*:?\s*(?P<raw_date>.+)$", re.IGNORECASE)
_READ_LINE_RE = re.compile(r"^Read the following", re.IGNORECASE)
_ORDER_LINE_RE = re.compile(r"^ORDER:?$", re.IGNORECASE)
_NOTIFICATION_LINE_RE = re.compile(r"^NOTIFICATION\b", re.IGNORECASE)
_TABLE_TITLE_RE = re.compile(r"^(Statement|Annexure|ANNEXURE)\b", re.IGNORECASE)
_TOP_LEVEL_RE = re.compile(r"^(?P<label>\d+)\.\s*(?P<text>.*)$")
_PAREN_NUMERIC_RE = re.compile(r"^(?P<label>\(\d+\))\s*(?P<text>.*)$")
_PAREN_UPPER_RE = re.compile(r"^(?P<label>\([A-Z]+\))\s*(?P<text>.*)$")
_PAREN_LOWER_ROMAN_RE = re.compile(
    r"^(?P<label>\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\))\s*(?P<text>.*)$"
)
_PAREN_LOWER_ALPHA_RE = re.compile(r"^(?P<label>\([a-z]\))\s*(?P<text>.*)$")
_ALPHA_CLOSE_RE = re.compile(r"^(?P<label>[a-z]\))\s*(?P<text>.*)$")
_TRAILER_LINES = {"SECTION OFFICER"}


@dataclass(frozen=True, slots=True)
class _PageLine:
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class _Marker:
    label: str
    text: str
    level: int
    kind: str


def parse_go_order(path: Path) -> ParsedDocument:
    extracted = extract_pdf_text(path)
    lines = _flatten_lines(extracted.pages)

    title = _extract_title(lines) or path.stem
    department = _extract_department(lines)
    go_number, issued_date = _extract_go_metadata(lines)
    references = _extract_references(lines)
    distribution = _extract_distribution(lines)
    abstract = _extract_abstract(lines)
    tables = _extract_tables(lines)
    nodes = _extract_nodes(lines)

    metadata = ParsedMetadata(
        source_path=str(path),
        title=title,
        abstract=abstract,
        department=department,
        go_number=go_number,
        issued_date=issued_date,
        references=references,
        distribution=distribution,
    )
    return ParsedDocument(
        doc_id=path.stem,
        source_path=str(path),
        title=title,
        abstract=abstract,
        department=department,
        go_number=go_number,
        issued_date=issued_date,
        references=references,
        nodes=nodes,
        tables=tables,
        distribution=distribution,
        metadata=metadata,
    )


def _flatten_lines(pages: list[object]) -> list[_PageLine]:
    flattened: list[_PageLine] = []
    for page in pages:
        for raw_line in page.text.splitlines():
            text = _normalize_line(raw_line)
            if not text or text.isdigit():
                continue
            flattened.append(_PageLine(page=page.page_number, text=text))
    return flattened


def _normalize_line(line: str) -> str:
    return _MULTISPACE_RE.sub(" ", line).strip()


def _extract_title(lines: list[_PageLine]) -> str | None:
    first_page_lines = [line.text for line in lines if line.page == 1]
    collecting = False
    title_lines: list[str] = []
    for text in first_page_lines:
        if text.upper() == "ABSTRACT":
            collecting = True
            continue
        if not collecting:
            continue
        if _DEPARTMENT_RE.search(text) or _GO_LINE_RE.match(text):
            break
        title_lines.append(text)
    if title_lines:
        return " ".join(title_lines)
    return None


def _extract_department(lines: list[_PageLine]) -> str | None:
    for line in lines[:40]:
        if _DEPARTMENT_RE.search(line.text):
            return line.text
    return None


def _extract_go_metadata(lines: list[_PageLine]) -> tuple[str | None, date | None]:
    for line in lines[:60]:
        match = _GO_LINE_RE.match(line.text)
        if not match:
            continue
        go_number = match.group("go").strip()
        issued_date = _parse_date(match.group("raw_date"))
        return go_number, issued_date
    return None, None


def _parse_date(raw_value: str) -> date | None:
    parts = [int(part) for part in re.findall(r"\d+", raw_value)]
    if len(parts) < 3:
        return None
    day, month, year = parts[:3]
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_references(lines: list[_PageLine]) -> list[str]:
    references: list[str] = []
    collecting = False
    for line in lines[:80]:
        text = line.text
        if _READ_LINE_RE.match(text):
            collecting = True
            continue
        if not collecting:
            continue
        if _ORDER_LINE_RE.match(text) or _NOTIFICATION_LINE_RE.match(text) or set(text) == {"*"}:
            break
        references.append(text)
    return references


def _extract_abstract(lines: list[_PageLine]) -> str | None:
    body_started = False
    abstract_lines: list[str] = []
    for line in lines:
        text = line.text
        if not body_started:
            if _ORDER_LINE_RE.match(text) or _NOTIFICATION_LINE_RE.match(text):
                body_started = True
            continue
        marker = _match_marker(text, [])
        if marker:
            if abstract_lines:
                break
            continue
        if _TABLE_TITLE_RE.match(text):
            break
        if text in {"To", "Copy to :", "Copy to:"}:
            break
        abstract_lines.append(text)
        if len(abstract_lines) >= 4:
            break
    if not abstract_lines:
        return None
    return " ".join(abstract_lines)


def _extract_distribution(lines: list[_PageLine]) -> list[str]:
    distribution: list[str] = []
    collecting = False
    for line in lines:
        text = line.text
        if _is_distribution_start(text):
            collecting = True
            continue
        if not collecting:
            continue
        if _is_trailer_marker(text):
            collecting = False
            continue
        distribution.append(text)
    return distribution


def _extract_tables(lines: list[_PageLine]) -> list[ParsedTable]:
    tables: list[ParsedTable] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _TABLE_TITLE_RE.match(line.text):
            index += 1
            continue

        table_lines: list[str] = []
        index += 1
        while index < len(lines):
            candidate = lines[index].text
            if _TABLE_TITLE_RE.match(candidate):
                break
            if _is_table_break(candidate, table_lines):
                break
            table_lines.append(candidate)
            index += 1

        tables.append(
            ParsedTable(
                table_id=f"table-{len(tables) + 1}",
                page=line.page,
                title=line.text,
                rows=[_split_table_row(row) for row in table_lines if row],
            )
        )
    return tables


def _is_table_break(text: str, table_lines: list[str]) -> bool:
    if _is_distribution_start(text):
        return True
    if _is_trailer_marker(text):
        return True
    marker = _match_marker(text, [])
    if marker and marker.kind == "top" and table_lines:
        return True
    return False


def _split_table_row(text: str) -> list[str]:
    cells = [cell.strip() for cell in re.split(r"\s{2,}", text) if cell.strip()]
    if cells:
        return cells
    return [text]


def _extract_nodes(lines: list[_PageLine]) -> list[ParsedNode]:
    nodes: list[ParsedNode] = []
    body_started = False
    table_mode = False
    stack: list[tuple[int, ParsedNode, str]] = []
    current_node: ParsedNode | None = None
    current_kind: str | None = None

    for line in lines:
        text = line.text
        if not body_started:
            if _ORDER_LINE_RE.match(text) or _NOTIFICATION_LINE_RE.match(text):
                body_started = True
            continue

        if table_mode:
            if _TABLE_TITLE_RE.match(text):
                continue
            if _is_table_break(text, ["seen"]):
                table_mode = False
            else:
                continue

        if _TABLE_TITLE_RE.match(text):
            table_mode = True
            continue
        if _is_distribution_start(text) or _is_trailer_marker(text):
            current_node = None
            current_kind = None
            stack.clear()
            continue

        marker = _match_marker(text, stack)
        if marker:
            parent_node_id = _find_parent_id(stack, marker.level)
            section_path = [entry[1].label for entry in stack if entry[0] < marker.level]
            node = ParsedNode(
                node_id=f"node-{len(nodes) + 1}",
                label=marker.label,
                text=marker.text,
                page_start=line.page,
                page_end=line.page,
                parent_node_id=parent_node_id,
                section_path=section_path + [marker.label],
            )
            nodes.append(node)
            stack = [entry for entry in stack if entry[0] < marker.level]
            stack.append((marker.level, node, marker.kind))
            current_node = node
            current_kind = marker.kind
            continue

        if current_node is None:
            continue
        current_node.text = _append_text(current_node.text, text)
        current_node.page_end = line.page
        if current_kind == "top" and stack:
            stack[-1] = (stack[-1][0], current_node, stack[-1][2])

    return nodes


def _append_text(existing: str, text: str) -> str:
    if not existing:
        return text
    return f"{existing} {text}"


def _match_marker(
    text: str,
    stack: list[tuple[int, ParsedNode, str]],
) -> _Marker | None:
    match = _TOP_LEVEL_RE.match(text)
    if match:
        return _Marker(
            label=match.group("label"),
            text=match.group("text").strip(),
            level=1,
            kind="top",
        )

    match = _PAREN_UPPER_RE.match(text)
    if match:
        return _Marker(
            label=match.group("label"),
            text=match.group("text").strip(),
            level=2,
            kind="upper",
        )

    match = _PAREN_NUMERIC_RE.match(text)
    if match:
        level = 3 if stack and stack[-1][2] == "upper" else 2
        return _Marker(
            label=match.group("label"),
            text=match.group("text").strip(),
            level=level,
            kind="numeric",
        )

    match = _PAREN_LOWER_ROMAN_RE.match(text)
    if match:
        level = 4 if stack and stack[-1][2] == "alpha" else 3
        return _Marker(
            label=match.group("label"),
            text=match.group("text").strip(),
            level=level,
            kind="roman",
        )

    match = _PAREN_LOWER_ALPHA_RE.match(text)
    if match:
        level = 4 if stack and stack[-1][2] == "numeric" else 3
        return _Marker(
            label=match.group("label"),
            text=match.group("text").strip(),
            level=level,
            kind="alpha",
        )

    match = _ALPHA_CLOSE_RE.match(text)
    if match:
        label = f"({match.group('label')[0]})"
        level = 4 if stack and stack[-1][2] == "numeric" else 3
        return _Marker(
            label=label,
            text=match.group("text").strip(),
            level=level,
            kind="alpha",
        )

    return None


def _find_parent_id(
    stack: list[tuple[int, ParsedNode, str]],
    level: int,
) -> str | None:
    for entry_level, node, _kind in reversed(stack):
        if entry_level < level:
            return node.node_id
    return None


def _is_distribution_start(text: str) -> bool:
    return text == "To" or text.startswith("Copy to")


def _is_trailer_marker(text: str) -> bool:
    return text.startswith("//") or text.startswith("(BY ORDER") or text in _TRAILER_LINES
