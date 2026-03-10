from __future__ import annotations

from xml.sax.saxutils import escape

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
    return answer
