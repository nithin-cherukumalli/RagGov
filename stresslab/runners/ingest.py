"""Runner for producing parsed ingest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stresslab.ingest import parse_go_order
from stresslab.reports import write_json_artifact


@dataclass(frozen=True)
class IngestRunResult:
    source_path: Path
    parsed_path: Path
    doc_id: str


def run_ingest(source_path: str | Path, output_dir: str | Path) -> IngestRunResult:
    source = Path(source_path)
    output = Path(output_dir)
    parsed_document = parse_go_order(source)
    parsed_path = output / f"{parsed_document.doc_id}.parsed.json"
    write_json_artifact(parsed_path, parsed_document)
    return IngestRunResult(
        source_path=source,
        parsed_path=parsed_path,
        doc_id=parsed_document.doc_id,
    )
