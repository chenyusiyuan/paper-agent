from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

from app.core.config import Settings
from app.core.logger import get_logger
from app.core.schemas import PaperMetadata, SectionChunk
from app.indexing import PaperIndex, SectionIndex
from app.parsing import GrobidError, GrobidRunner, TeiParser


logger = get_logger(__name__)


class IngestService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runner = GrobidRunner(settings.grobid_url)
        self.parser = TeiParser()

    def ingest(self, pdf_dir: str) -> dict[str, int | list[dict[str, str]]]:
        pdf_root = Path(pdf_dir)
        parsed_dir = Path(self.settings.data_dir) / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = sorted(pdf_root.glob("*.pdf"))
        success_count = 0
        errors: list[dict[str, str]] = []

        for pdf_path in pdf_paths:
            paper_id = self._slugify(pdf_path.stem)
            logger.info("Parsing %s as %s", pdf_path, paper_id)
            try:
                tei_xml = self.runner.parse(str(pdf_path))
                metadata, chunks = self.parser.parse(tei_xml, paper_id)
                self._save_parsed_output(parsed_dir, metadata, chunks)
                success_count += 1
            except GrobidError as exc:
                logger.exception("Failed to process %s with GROBID", pdf_path)
                errors.append({"pdf": str(pdf_path), "error": str(exc)})
            except Exception as exc:
                logger.exception("Failed to parse TEI for %s", pdf_path)
                errors.append({"pdf": str(pdf_path), "error": str(exc)})

        papers, chunks = self._load_parsed_data(parsed_dir)
        self._build_indexes(papers, chunks)

        return {
            "total": len(pdf_paths),
            "success": success_count,
            "failed": len(errors),
            "errors": errors,
        }

    def _save_parsed_output(
        self,
        parsed_dir: Path,
        metadata: PaperMetadata,
        chunks: list[SectionChunk],
    ) -> None:
        output_path = parsed_dir / f"{metadata.paper_id}.json"
        payload = {
            "metadata": asdict(metadata),
            "sections": [asdict(chunk) for chunk in chunks],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved parsed output to %s", output_path)

    def _load_parsed_data(self, parsed_dir: Path) -> tuple[list[PaperMetadata], list[SectionChunk]]:
        papers: list[PaperMetadata] = []
        chunks: list[SectionChunk] = []

        for json_path in sorted(parsed_dir.glob("*.json")):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            papers.append(PaperMetadata(**payload["metadata"]))
            chunks.extend(SectionChunk(**item) for item in payload["sections"])

        return papers, chunks

    def _build_indexes(
        self,
        papers: list[PaperMetadata],
        chunks: list[SectionChunk],
    ) -> None:
        paper_index = PaperIndex(
            self.settings.data_dir,
            self.settings.embedding_model,
            embedding_batch_size=self.settings.embedding_batch_size,
            embedding_max_seq_length=self.settings.embedding_max_seq_length,
        )
        section_index = SectionIndex(
            self.settings.data_dir,
            self.settings.embedding_model,
            embedding_batch_size=self.settings.embedding_batch_size,
            embedding_max_seq_length=self.settings.embedding_max_seq_length,
        )
        paper_index.build(papers)
        section_index.build(chunks)
        logger.info("Built paper index with %s papers", len(papers))
        logger.info("Built section index with %s chunks", len(chunks))

    def _slugify(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
        return slug or "paper"
