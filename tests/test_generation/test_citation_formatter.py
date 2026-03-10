from app.core.schemas import Evidence
from app.generation.citation_formatter import add_citation_marks


def test_add_citation_marks_appends_page_number() -> None:
    answer = "The method is effective [method]."
    evidences = [
        Evidence(
            text="method text",
            paper_id="paper-1",
            section_type="method",
            section_title="Method",
            page=5,
        )
    ]

    result = add_citation_marks(answer, evidences)

    assert result == "The method is effective [method·p.5]."


def test_add_citation_marks_appends_section_title_when_page_none() -> None:
    answer = "The paper overview is here [abstract]."
    evidences = [
        Evidence(
            text="abstract text",
            paper_id="paper-1",
            section_type="abstract",
            section_title="Abstract",
            page=None,
        )
    ]

    result = add_citation_marks(answer, evidences)

    assert result == "The paper overview is here [abstract·Abstract]."


def test_add_citation_marks_leaves_unknown_marker_unchanged() -> None:
    answer = "This stays the same [unknown_section]."

    result = add_citation_marks(answer, [])

    assert result == "This stays the same [unknown_section]."


def test_add_citation_marks_handles_empty_answer() -> None:
    evidences = [
        Evidence(
            text="method text",
            paper_id="paper-1",
            section_type="method",
            section_title="Method",
            page=5,
        )
    ]

    result = add_citation_marks("", evidences)

    assert result == ""
