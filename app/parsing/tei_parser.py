from __future__ import annotations

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree

from app.core.schemas import PaperMetadata, SectionChunk
from app.parsing.paper_normalizer import normalize_section_type


class TeiParser:
    def __init__(self) -> None:
        self.ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    def parse(self, tei_xml: str, paper_id: str) -> tuple[PaperMetadata, list[SectionChunk]]:
        root = etree.fromstring(tei_xml.encode("utf-8"))

        title = self._extract_text(root, ".//tei:titleStmt/tei:title")
        authors = self._extract_authors(root)
        year = self._extract_year(root)
        venue = self._extract_text(root, ".//tei:sourceDesc//tei:monogr/tei:title")
        abstract = self._extract_text(root, ".//tei:profileDesc/tei:abstract")
        keywords = self._extract_keywords(root)
        chunks = self._extract_sections(root, paper_id)

        metadata = PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue or None,
            abstract=abstract,
            keywords=keywords,
            section_titles=[chunk.section_title for chunk in chunks],
        )
        return metadata, chunks

    def _extract_text(self, root: object, path: str) -> str:
        node = root.find(path, self.ns)
        if node is None:
            return ""
        return " ".join(part.strip() for part in node.itertext() if part.strip()).strip()

    def _extract_authors(self, root: object) -> list[str]:
        author_nodes = root.findall(".//tei:sourceDesc//tei:biblStruct//tei:author", self.ns)
        authors: list[str] = []
        for author_node in author_nodes:
            forenames = [
                (value.text or "").strip()
                for value in author_node.findall(".//tei:forename", self.ns)
                if (value.text or "").strip()
            ]
            surnames = [
                (value.text or "").strip()
                for value in author_node.findall(".//tei:surname", self.ns)
                if (value.text or "").strip()
            ]
            full_name = " ".join(forenames + surnames).strip()
            if full_name:
                authors.append(full_name)
        return authors

    def _extract_year(self, root: object) -> int | None:
        date_nodes = root.findall(".//tei:sourceDesc//tei:biblStruct//tei:date", self.ns)
        for node in date_nodes:
            value = node.attrib.get("when", "")
            if len(value) >= 4 and value[:4].isdigit():
                return int(value[:4])

        for node in date_nodes:
            stripped = (node.text or "").strip()
            if len(stripped) >= 4 and stripped[:4].isdigit():
                return int(stripped[:4])
        return None

    def _extract_keywords(self, root: object) -> list[str]:
        term_nodes = root.findall(".//tei:profileDesc/tei:textClass/tei:keywords//tei:term", self.ns)
        if term_nodes:
            return [(node.text or "").strip() for node in term_nodes if (node.text or "").strip()]

        keyword_text = self._extract_text(root, ".//tei:profileDesc/tei:textClass/tei:keywords")
        if not keyword_text:
            return []
        return [value.strip() for value in keyword_text.split(",") if value.strip()]

    def _extract_sections(self, root: object, paper_id: str) -> list[SectionChunk]:
        body = root.find(".//tei:text/tei:body", self.ns)
        if body is None:
            return []
        body_divs = body.findall("./tei:div", self.ns)
        chunks: list[SectionChunk] = []
        for index, div in enumerate(body_divs):
            section_title = self._extract_div_head(div) or f"Section {index + 1}"
            paragraphs = [
                " ".join(part.strip() for part in node.itertext() if part.strip())
                for node in div.findall("./tei:p", self.ns)
            ]
            text = "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()
            if not text:
                continue

            chunks.append(
                SectionChunk(
                    chunk_id=f"{paper_id}_sec_{len(chunks)}",
                    paper_id=paper_id,
                    section_type=normalize_section_type(section_title),
                    section_title=section_title,
                    text=text,
                    page_start=0,
                    page_end=0,
                    order_in_paper=len(chunks),
                )
            )
        return chunks

    def _extract_div_head(self, div: object) -> str:
        head_node = div.find("./tei:head", self.ns)
        if head_node is None:
            return ""
        return " ".join(part.strip() for part in head_node.itertext() if part.strip()).strip()
