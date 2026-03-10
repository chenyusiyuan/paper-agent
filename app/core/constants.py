SECTION_TYPE_MAP: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "method": "method",
    "methodology": "method",
    "approach": "method",
    "proposed": "method",
    "experiment": "experiment",
    "evaluation": "experiment",
    "result": "experiment",
    "ablation": "ablation",
    "related work": "related_work",
    "background": "related_work",
    "conclusion": "conclusion",
    "summary": "conclusion",
}

DEFAULT_TOP_K = 5

SECTION_TYPES = [
    "abstract",
    "introduction",
    "method",
    "experiment",
    "ablation",
    "related_work",
    "conclusion",
    "other",
]
