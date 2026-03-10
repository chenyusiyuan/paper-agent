from __future__ import annotations

import re

from app.core.constants import SECTION_TYPE_MAP


def normalize_section_type(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.strip().lower())
    keyword_order = [
        "abstract",
        "introduction",
        "method",
        "approach",
        "proposed",
        "experiment",
        "evaluation",
        "result",
        "ablation",
        "related work",
        "background",
        "conclusion",
        "summary",
    ]

    for keyword in keyword_order:
        if keyword in normalized:
            return SECTION_TYPE_MAP[keyword]
    return "other"
