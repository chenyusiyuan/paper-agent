import pytest

from app.parsing.paper_normalizer import normalize_section_type


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Introduction", "introduction"),
        ("1. Introduction", "introduction"),
        ("INTRODUCTION", "introduction"),
        ("Method", "method"),
        ("Methodology", "method"),
        ("Our Approach", "method"),
        ("Proposed Method", "method"),
        ("Experiments", "experiment"),
        ("Experimental Evaluation", "experiment"),
        ("Results and Analysis", "experiment"),
        ("Related Work", "related_work"),
        ("Background", "related_work"),
        ("Conclusion", "conclusion"),
        ("Summary", "conclusion"),
        ("Abstract", "abstract"),
        ("Appendix", "other"),
    ],
)
def test_normalize_section_type(title: str, expected: str) -> None:
    assert normalize_section_type(title) == expected
