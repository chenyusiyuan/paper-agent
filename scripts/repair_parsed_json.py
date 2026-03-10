from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.core.logger import get_logger
from app.parsing.parsed_json_repair import repair_directory


logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", type=Path, default=Path(settings.data_dir) / "parsed")
    parser.add_argument("--pdf-dir", type=Path, default=Path(settings.data_dir) / "raw_pdfs")
    args = parser.parse_args()

    repaired_files = repair_directory(args.parsed_dir, args.pdf_dir)
    logger.info("Repaired %s parsed JSON files", len(repaired_files))


if __name__ == "__main__":
    main()
