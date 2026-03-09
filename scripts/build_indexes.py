from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.schemas import PaperMetadata, SectionChunk
from app.indexing import PaperIndex, SectionIndex


logger = get_logger(__name__)


def load_parsed_data(parsed_dir: Path) -> tuple[list[PaperMetadata], list[SectionChunk]]:
    papers: list[PaperMetadata] = []
    chunks: list[SectionChunk] = []

    for json_path in sorted(parsed_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = PaperMetadata(**payload["metadata"])
        section_chunks = [SectionChunk(**item) for item in payload["sections"]]
        papers.append(metadata)
        chunks.extend(section_chunks)

    return papers, chunks


def build_indexes() -> None:
    settings = get_settings()
    parsed_dir = Path(settings.data_dir) / "parsed"
    papers, chunks = load_parsed_data(parsed_dir)

    paper_index = PaperIndex(settings.data_dir, settings.embedding_model)
    section_index = SectionIndex(settings.data_dir, settings.embedding_model)
    paper_index.build(papers)
    section_index.build(chunks)

    logger.info("Built paper index with %s papers", len(papers))
    logger.info("Built section index with %s chunks", len(chunks))


if __name__ == "__main__":
    build_indexes()
