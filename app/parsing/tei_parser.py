from __future__ import annotations

import re

from lxml import etree

from app.core.schemas import PaperMetadata, SectionChunk
from app.parsing.paper_normalizer import normalize_section_type


class TeiParser:
    def __init__(self) -> None:
        self.ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        self._year_pattern = re.compile(r"\b(19|20)\d{2}\b")
        self._abstract_cut_markers = [
            "Equal contribution",
            "Corresponding Author",
            "Corresponding author",
        ]
        self._section_placeholder_pattern = re.compile(
            r"^(?:section|sec\.?|appendix)\s+[a-z]?\d+(?:\.\d+)*[a-z]?$",
            re.IGNORECASE,
        )
        self._section_sentence_verb_pattern = re.compile(
            r"\b(is|are|was|were|be|been|being|do|does|did|shows?|showed|demonstrates?|"
            r"improves?|contains?|announced|guarantee|come|comes|using|uses|learns?)\b",
            re.IGNORECASE,
        )

    def parse(self, tei_xml: str, paper_id: str) -> tuple[PaperMetadata, list[SectionChunk]]:
        root = etree.fromstring(tei_xml.encode("utf-8"))

        title = self._extract_text(root, ".//tei:titleStmt/tei:title")
        authors = self._extract_authors(root)
        year = self._extract_year(root)
        venue = self._extract_venue(root)
        abstract = self._clean_abstract(self._extract_text(root, ".//tei:profileDesc/tei:abstract"))
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
            section_titles=[chunk.section_title for chunk in chunks if chunk.level == 0],
        )
        return metadata, chunks

    def _extract_text(self, root: object, path: str) -> str:
        node = root.find(path, self.ns)
        if node is None:
            return ""
        return self._normalize_text(" ".join(part.strip() for part in node.itertext() if part.strip()))

    def _extract_authors(self, root: object) -> list[str]:
        author_nodes = root.findall(".//tei:sourceDesc//tei:biblStruct//tei:author", self.ns)
        authors: list[str] = []
        seen_authors: set[str] = set()
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
            full_name = self._clean_author_name(" ".join(forenames + surnames))
            if full_name and full_name not in seen_authors:
                seen_authors.add(full_name)
                authors.append(full_name)
        return authors

    def _extract_year(self, root: object) -> int | None:
        date_paths = [
            ".//tei:teiHeader//tei:sourceDesc//tei:imprint//tei:date",
            ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:date",
            ".//tei:teiHeader//tei:publicationStmt//tei:date",
        ]
        for path in date_paths:
            for node in root.findall(path, self.ns):
                year = self._extract_year_from_value(node.attrib.get("when", ""))
                if year is not None:
                    return year
                year = self._extract_year_from_value(self._normalize_text(" ".join(node.itertext())))
                if year is not None:
                    return year
        return None

    def _extract_venue(self, root: object) -> str | None:
        venue_paths = [
            ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:meeting/tei:title",
            ".//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:title",
            ".//tei:sourceDesc//tei:biblStruct/tei:series/tei:title",
            ".//tei:fileDesc/tei:publicationStmt/tei:publisher",
        ]
        for path in venue_paths:
            for node in root.findall(path, self.ns):
                venue = self._clean_venue_text(" ".join(part.strip() for part in node.itertext() if part.strip()))
                if venue:
                    return venue
        return None

    def _extract_keywords(self, root: object) -> list[str]:
        term_nodes = root.findall(".//tei:profileDesc/tei:textClass/tei:keywords//tei:term", self.ns)
        keywords: list[str] = []
        if term_nodes:
            keywords = [
                item
                for node in term_nodes
                for item in self._split_keywords((node.text or "").strip())
            ]
            return self._dedupe_preserve_order(keywords)

        keyword_text = self._extract_text(root, ".//tei:profileDesc/tei:textClass/tei:keywords")
        if not keyword_text:
            return []
        return self._dedupe_preserve_order(self._split_keywords(keyword_text))

    def _extract_sections(self, root: object, paper_id: str) -> list[SectionChunk]:
        body = root.find(".//tei:text/tei:body", self.ns)
        if body is None:
            return []
        chunks: list[SectionChunk] = []
        order_counter = 0
        for index, div in enumerate(body.findall("./tei:div", self.ns)):
            section_title = self._extract_div_head(div) or f"Section {index + 1}"
            coarse_chunk, fine_chunks, order_counter = self._parse_top_div(
                div,
                paper_id,
                order_counter,
                section_title,
            )
            if coarse_chunk is not None:
                chunks.append(coarse_chunk)
            chunks.extend(fine_chunks)
        return chunks

    def _parse_top_div(
        self,
        div: object,
        paper_id: str,
        order_counter: int,
        section_title: str,
    ) -> tuple[SectionChunk | None, list[SectionChunk], int]:
        section_type = normalize_section_type(section_title)
        all_text = self._collect_all_paragraphs(div)

        coarse_chunk: SectionChunk | None = None
        coarse_chunk_id: str | None = None
        if all_text:
            coarse_chunk = SectionChunk(
                chunk_id=f"{paper_id}_sec_{order_counter}",
                paper_id=paper_id,
                section_type=section_type,
                section_title=section_title,
                section_path=section_title,
                text=all_text,
                page_start=0,
                page_end=0,
                order_in_paper=order_counter,
                level=0,
                parent_chunk_id=None,
                granularity="coarse",
            )
            coarse_chunk_id = coarse_chunk.chunk_id
            order_counter += 1

        fine_chunks, order_counter = self._parse_subsections(
            div,
            paper_id,
            order_counter,
            parent_title=section_title,
            parent_path=section_title,
            parent_chunk_id=coarse_chunk_id,
            level=1,
        )

        return coarse_chunk, fine_chunks, order_counter

    def _parse_subsections(
        self,
        div: object,
        paper_id: str,
        order_counter: int,
        parent_title: str,
        parent_path: str,
        parent_chunk_id: str | None,
        level: int,
    ) -> tuple[list[SectionChunk], int]:
        chunks: list[SectionChunk] = []
        for sub_div in div.findall("./tei:div", self.ns):
            sub_title = self._extract_div_head(sub_div)
            chunk_title = sub_title or parent_title
            section_path = self._build_section_path(parent_path, sub_title)
            sub_text = self._collect_direct_paragraphs(sub_div)

            chunk_id: str | None = None
            if sub_text:
                chunk_id = f"{paper_id}_sec_{order_counter}"
                chunks.append(
                    SectionChunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        section_type=normalize_section_type(chunk_title),
                        section_title=chunk_title,
                        section_path=section_path,
                        text=sub_text,
                        page_start=0,
                        page_end=0,
                        order_in_paper=order_counter,
                        level=level,
                        parent_chunk_id=parent_chunk_id,
                        granularity="fine",
                    )
                )
                order_counter += 1

            child_chunks, order_counter = self._parse_subsections(
                sub_div,
                paper_id,
                order_counter,
                parent_title=chunk_title,
                parent_path=section_path,
                parent_chunk_id=chunk_id or parent_chunk_id,
                level=level + 1,
            )
            chunks.extend(child_chunks)

        return chunks, order_counter

    def _collect_all_paragraphs(self, div: object) -> str:
        paragraphs = [
            " ".join(part.strip() for part in node.itertext() if part.strip())
            for node in div.findall(".//tei:p", self.ns)
        ]
        return "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()

    def _collect_direct_paragraphs(self, div: object) -> str:
        paragraphs = [
            " ".join(part.strip() for part in node.itertext() if part.strip())
            for node in div.findall("./tei:p", self.ns)
        ]
        return "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()

    def _build_section_path(self, parent_path: str, section_title: str) -> str:
        if parent_path and section_title:
            return f"{parent_path} > {section_title}"
        return section_title or parent_path

    def _extract_div_head(self, div: object) -> str:
        head_node = div.find("./tei:head", self.ns)
        if head_node is None:
            return ""
        title = self._normalize_text(" ".join(part.strip() for part in head_node.itertext() if part.strip()))
        return self._clean_section_title(title)

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _clean_author_name(self, name: str) -> str:
        cleaned = self._normalize_text(name)
        cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;*†")
        return cleaned

    def _extract_year_from_value(self, value: str) -> int | None:
        match = self._year_pattern.search(value)
        if match is None:
            return None
        return int(match.group(0))

    def _clean_venue_text(self, text: str) -> str:
        cleaned = self._normalize_text(text).strip(" ,;:")
        if not cleaned or "http" in cleaned.lower():
            return ""
        return cleaned

    def _clean_abstract(self, text: str) -> str:
        cleaned = self._normalize_text(text)
        if not cleaned:
            return ""
        cut_positions = [cleaned.find(marker) for marker in self._abstract_cut_markers if marker in cleaned]
        if cut_positions:
            cleaned = cleaned[:min(cut_positions)].rstrip(" ,;:-*†")
        cleaned = re.sub(r"\s+\d+\s+\*\s*$", "", cleaned).strip()
        return cleaned

    def _split_keywords(self, keyword_text: str) -> list[str]:
        normalized = self._normalize_text(keyword_text)
        if not normalized:
            return []
        return [
            item.strip(" ,;:•·")
            for item in re.split(r"[;,•·|]", normalized)
            if item.strip(" ,;:•·")
        ]

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _clean_section_title(self, title: str) -> str:
        cleaned = self._normalize_text(title).lstrip("#").strip(" ,;:.-")
        if not cleaned:
            return ""

        cleaned = self._strip_leading_section_label(cleaned)
        cleaned = cleaned.strip(" ,;:.-?")
        if not cleaned:
            return ""
        if self._section_placeholder_pattern.fullmatch(cleaned):
            return ""
        if cleaned.endswith(("&", "/", "-")):
            return ""
        if len(cleaned) > 120 or "http" in cleaned.lower():
            return ""
        if self._looks_like_bad_section_title(cleaned):
            canonical_title = self._canonical_section_title(cleaned)
            return canonical_title or ""
        return cleaned

    def _strip_leading_section_label(self, title: str) -> str:
        canonical_title = self._canonical_prefixed_title(title)
        if canonical_title is not None:
            return canonical_title

        patterns = [
            r"^(?:section|sec\.?)\s+\d+(?:\.\d+)*\s*[:.)-]?\s*",
            r"^(?:appendix\s+)?(?:\d+|[ivxlcdm]+)(?:\.\d+)*\s+",
            r"^(?:appendix\s+)?[a-z](?:\.\d+)*\s+",
        ]
        cleaned = title
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _canonical_prefixed_title(self, title: str) -> str | None:
        match = re.match(
            r"^(abstract|introduction|background|related work|method(?:ology)?|approach|"
            r"experiment(?:s)?|evaluation|result(?:s)?(?: and discussion)?|conclusion|summary)"
            r"\s+\d+(?:\.\d+)*\b",
            title,
            re.IGNORECASE,
        )
        if match is None:
            return None
        return self._canonical_section_title(match.group(1))

    def _looks_like_bad_section_title(self, title: str) -> bool:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
        if len(words) < 5:
            return False
        trailing_stop_words = {"a", "an", "and", "by", "for", "from", "in", "not", "of", "on", "or", "the", "to", "with"}
        lowercase_words = sum(1 for word in words[1:] if word[:1].islower())
        lowercase_ratio = lowercase_words / max(len(words) - 1, 1)
        if words[-1].lower() in trailing_stop_words:
            return True
        return lowercase_ratio > 0.45 and self._section_sentence_verb_pattern.search(title) is not None

    def _canonical_section_title(self, title: str) -> str:
        section_type = normalize_section_type(title)
        canonical_titles = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "method": "Method",
            "experiment": "Experiments",
            "related_work": "Related Work",
            "conclusion": "Conclusion",
        }
        return canonical_titles.get(section_type, "")
