from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from app.parsing.grobid_runner import GrobidError, GrobidRunner


def test_parse_returns_tei_xml(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    response = Mock(status_code=200, text="<TEI />")

    with patch("app.parsing.grobid_runner.requests.post", return_value=response) as mock_post:
        runner = GrobidRunner("http://localhost:8070")
        result = runner.parse(str(pdf_path))

    assert result == "<TEI />"
    mock_post.assert_called_once()


def test_parse_raises_grobid_error_on_non_200(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    response = Mock(status_code=500, text="error")

    with patch("app.parsing.grobid_runner.requests.post", return_value=response):
        runner = GrobidRunner("http://localhost:8070")
        with pytest.raises(GrobidError, match="status code 500"):
            runner.parse(str(pdf_path))


def test_parse_raises_grobid_error_on_request_exception(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with patch(
        "app.parsing.grobid_runner.requests.post",
        side_effect=requests.RequestException("boom"),
    ):
        runner = GrobidRunner("http://localhost:8070")
        with pytest.raises(GrobidError, match="Failed to connect to GROBID"):
            runner.parse(str(pdf_path))
