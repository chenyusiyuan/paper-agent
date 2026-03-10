from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services import IngestService


logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    default_pdf_dir = Path(settings.data_dir) / "raw_pdfs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir)
    args = parser.parse_args()

    result = IngestService(settings).ingest(str(args.pdf_dir))
    logger.info("Ingest finished: %s", result)


if __name__ == "__main__":
    main()
