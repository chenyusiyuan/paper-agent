from __future__ import annotations

from pathlib import Path

import requests

from app.core.logger import get_logger


logger = get_logger(__name__)


class GrobidError(Exception):
    pass


class GrobidRunner:
    def __init__(self, grobid_url: str) -> None:
        self.grobid_url = grobid_url.rstrip("/")

    def parse(self, pdf_path: str) -> str:
        path = Path(pdf_path)
        url = f"{self.grobid_url}/api/processFulltextDocument"
        try:
            pdf_bytes = path.read_bytes()
        except OSError as exc:
            logger.exception("Failed to read PDF: %s", path)
            raise GrobidError(f"Failed to read PDF: {path}") from exc

        files = {
            "input": (path.name, pdf_bytes, "application/pdf"),
        }

        try:
            response = requests.post(url, files=files, timeout=120)
        except requests.RequestException as exc:
            logger.exception("Failed to connect to GROBID at %s", url)
            raise GrobidError(f"Failed to connect to GROBID: {url}") from exc

        if response.status_code != 200:
            logger.error(
                "GROBID returned non-200 status code %s for %s",
                response.status_code,
                path,
            )
            raise GrobidError(
                f"GROBID returned status code {response.status_code}: {response.text}"
            )

        return response.text
