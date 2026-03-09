from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger
from app.parsing import GrobidError, GrobidRunner, TeiParser


logger = get_logger(__name__)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "paper"


def ingest_papers(pdf_dir: Path) -> None:
    settings = get_settings()
    parsed_dir = Path(settings.data_dir) / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    runner = GrobidRunner(settings.grobid_url)
    parser = TeiParser()

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        paper_id = slugify(pdf_path.stem)
        logger.info("Parsing %s as %s", pdf_path, paper_id)
        try:
            tei_xml = runner.parse(str(pdf_path))
            metadata, chunks = parser.parse(tei_xml, paper_id)
        except GrobidError:
            logger.exception("Failed to process %s with GROBID", pdf_path)
            continue
        except Exception:
            logger.exception("Failed to parse TEI for %s", pdf_path)
            continue

        output_path = parsed_dir / f"{paper_id}.json"
        payload = {
            "metadata": asdict(metadata),
            "sections": [asdict(chunk) for chunk in chunks],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved parsed output to %s", output_path)


def main() -> None:
    settings = get_settings()
    default_pdf_dir = Path(settings.data_dir) / "raw_pdfs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir)
    args = parser.parse_args()

    ingest_papers(args.pdf_dir)


if __name__ == "__main__":
    main()
