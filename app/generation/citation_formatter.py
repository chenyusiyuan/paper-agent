from __future__ import annotations

import re
from xml.sax.saxutils import escape

from app.core.constants import SECTION_TYPES
from app.core.schemas import Evidence, PaperMetadata


def format_evidences_xml(evidences: list[Evidence], metadata: PaperMetadata) -> str:
    authors = "".join(f"<author>{escape(author)}</author>" for author in metadata.authors)
    keywords = "".join(f"<keyword>{escape(keyword)}</keyword>" for keyword in metadata.keywords)
    evidence_nodes = []
    for index, evidence in enumerate(evidences, start=1):
        evidence_nodes.append(
            (
                f'<evidence id="{index}" paper_id="{escape(evidence.paper_id)}" '
                f'section_type="{escape(evidence.section_type)}" '
                f'section_title="{escape(evidence.section_title)}">{escape(evidence.text)}</evidence>'
            )
        )

    year = "" if metadata.year is None else str(metadata.year)
    venue = "" if metadata.venue is None else metadata.venue
    return (
        "<paper>"
        "<metadata>"
        f"<paper_id>{escape(metadata.paper_id)}</paper_id>"
        f"<title>{escape(metadata.title)}</title>"
        f"<authors>{authors}</authors>"
        f"<year>{escape(year)}</year>"
        f"<venue>{escape(venue)}</venue>"
        f"<abstract>{escape(metadata.abstract)}</abstract>"
        f"<keywords>{keywords}</keywords>"
        "</metadata>"
        f"{''.join(evidence_nodes)}"
        "</paper>"
    )


def add_citation_marks(answer: str, evidences: list[Evidence]) -> str:
    if not answer or not evidences:
        return answer

    evidence_by_type: dict[str, Evidence] = {}
    for evidence in evidences:
        key = evidence.section_type.lower()
        if key not in evidence_by_type:
            evidence_by_type[key] = evidence

    patterns = [
        r"related[ _]work" if section_type == "related_work" else re.escape(section_type)
        for section_type in SECTION_TYPES
    ]
    marker_pattern = re.compile(
        r"\[(" + "|".join(patterns) + r")\]",
        re.IGNORECASE,
    )

    def replace_marker(match: re.Match[str]) -> str:
        original_text = match.group(1)
        evidence = evidence_by_type.get(original_text.lower().replace(" ", "_"))
        if evidence is None:
            return match.group(0)
        if evidence.page is not None:
            return f"[{original_text}·p.{evidence.page}]"
        return f"[{original_text}·{evidence.section_title}]"

    return marker_pattern.sub(replace_marker, answer)
