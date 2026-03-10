from __future__ import annotations

from pathlib import Path

from app.parsing.parsed_json_repair import repair_payload


def test_repair_payload_applies_metadata_and_section_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: ["retrieval", "generation"],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-020",
            "title": "RQ-RAG: LEARNING TO REFINE QUERY FOR RETRIEVAL AUGMENTED GENERATION",
            "authors": ["Chi-Min Chan", "Chunpu Xu"],
            "year": None,
            "venue": None,
            "abstract": "Example abstract. 1 * Lead contributors.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-020_sec_0",
                "paper_id": "paper-020",
                "section_type": "introduction",
                "section_title": "Introduction",
                "section_path": "Introduction",
                "text": "Intro text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 0,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-020_sec_1",
                "paper_id": "paper-020",
                "section_type": "method",
                "section_title": "RQ-RAG: Learning to Refine Query for Retrieval Augmented Generation",
                "section_path": "RQ-RAG",
                "text": "Method text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 1,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-020_sec_2",
                "paper_id": "paper-020",
                "section_type": "related_work",
                "section_title": "Related Works",
                "section_path": "Related Works",
                "text": "Related work text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 12,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
        ],
    }

    repaired = repair_payload(payload, "paper-020", Path("data/raw_pdfs/paper_020.pdf"))

    assert repaired["metadata"]["year"] == 2024
    assert repaired["metadata"]["venue"] == "arXiv"
    assert repaired["metadata"]["title"] == "RQ-RAG: LEARNING TO REFINE QUERIES FOR RETRIEVAL AUGMENTED GENERATION"
    assert repaired["metadata"]["abstract"] == "Example abstract."
    assert repaired["metadata"]["keywords"] == ["retrieval", "generation"]
    assert repaired["sections"][1]["section_title"] == "RQ-RAG: Learning to Refine Query for Retrieval Augmented Generation"
    assert repaired["sections"][2]["section_title"] == "Related Works"
    assert repaired["metadata"]["section_titles"] == [
        "Introduction",
        "RQ-RAG: Learning to Refine Query for Retrieval Augmented Generation",
        "Related Works",
    ]


def test_repair_payload_applies_author_override_and_lowercase_title_cleanup(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: [],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-021",
            "title": "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads",
            "authors": [
                "Guangxuan Xiao",
                "Jiaming Tang",
                "Tsinghua University",
                "Sjtu",
            ],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-021_sec_0",
                "paper_id": "paper-021",
                "section_type": "conclusion",
                "section_title": "conclusion",
                "section_path": "conclusion",
                "text": "Conclusion text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 0,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            }
        ],
    }

    repaired = repair_payload(payload, "paper-021", Path("data/raw_pdfs/paper_021.pdf"))

    assert repaired["metadata"]["year"] == 2024
    assert repaired["metadata"]["venue"] == "arXiv"
    assert repaired["metadata"]["authors"] == [
        "Guangxuan Xiao",
        "Jiaming Tang",
        "Jingwei Zuo",
        "Junxian Guo",
        "Shang Yang",
        "Haotian Tang",
        "Yao Fu",
        "Song Han",
    ]
    assert repaired["sections"][0]["section_title"] == "Conclusion"
