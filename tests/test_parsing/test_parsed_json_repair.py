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


def test_repair_payload_merges_and_reindexes_known_bad_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: [],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-015",
            "title": "Adaptive Retrieval-Augmented Generation for Conversational Systems",
            "authors": ["Xi Wang", "Procheta Sen", "Ruizhe Li", "Emine Yilmaz"],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-015_sec_0",
                "paper_id": "paper-015",
                "section_type": "introduction",
                "section_title": "Introduction",
                "section_path": "Introduction",
                "text": "Recently... lack of up-to-date Can you find me some interesting things to do?",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 0,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-015_sec_1",
                "paper_id": "paper-015",
                "section_type": "other",
                "section_title": "Knowledge",
                "section_path": "Knowledge",
                "text": "Figure example block.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 1,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-015_sec_2",
                "paper_id": "paper-015",
                "section_type": "other",
                "section_title": "Not Use Knowledge",
                "section_path": "Not Use Knowledge",
                "text": "Figure 1 : Example conversation.\nknowledge and the rest of the introduction.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 2,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-015_sec_3",
                "paper_id": "paper-015",
                "section_type": "related_work",
                "section_title": "Related Work",
                "section_path": "Related Work",
                "text": "Related work text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 3,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
        ],
    }

    repaired = repair_payload(payload, "paper-015", Path("data/raw_pdfs/paper_015.pdf"))

    assert len(repaired["sections"]) == 2
    assert repaired["sections"][0]["chunk_id"] == "paper-015_sec_0"
    assert repaired["sections"][1]["chunk_id"] == "paper-015_sec_1"
    assert repaired["sections"][0]["text"] == "Recently... lack of up-to-date knowledge and the rest of the introduction."
    assert repaired["metadata"]["section_titles"] == ["Introduction", "Related Work"]


def test_repair_payload_strips_review_banner_and_fixes_known_authors(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: [],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-024",
            "title": "Under review as a conference paper at ICLR 2025 CORRECTIVE RETRIEVAL AUGMENTED GENERATION",
            "authors": [],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-024_sec_0",
                "paper_id": "paper-024",
                "section_type": "introduction",
                "section_title": "INTRODUCTION",
                "section_path": "INTRODUCTION",
                "text": "Text Under review as a conference paper at ICLR 2025 more text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 0,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            }
        ],
    }

    repaired = repair_payload(payload, "paper-024", Path("data/raw_pdfs/paper_024.pdf"))

    assert repaired["metadata"]["title"] == "CORRECTIVE RETRIEVAL AUGMENTED GENERATION"
    assert "Under review as a conference paper at ICLR 2025" not in repaired["sections"][0]["text"]

    payload = {
        "metadata": {
            "paper_id": "paper-010",
            "title": "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach",
            "authors": ["Zhuowan Li", "Google DeepMind"],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [],
    }

    repaired = repair_payload(payload, "paper-010", Path("data/raw_pdfs/paper_010.pdf"))

    assert repaired["metadata"]["authors"] == [
        "Zhuowan Li",
        "Cheng Li",
        "Mingyang Zhang",
        "Qiaozhu Mei",
        "Michael Bendersky",
    ]


def test_repair_payload_removes_inline_caption_noise(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: [],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-010",
            "title": "Retrieval Augmented Generation or Long-Context LLMs?",
            "authors": [],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-010_sec_0",
                "paper_id": "paper-010",
                "section_type": "analysis",
                "section_title": "Motivation",
                "section_path": "Motivation",
                "text": (
                    "As demonstrated in Sec. 3, RAG lags behind long-context LLMs in terms of performance. "
                    "Table 1 : Results of Gemini-1.5-Pro, GPT-3.5-Turbo, and GPT-4O using the Contriever "
                    "retriever. LC consistently outperforms RAG, while SELF-ROUTE achieves performance "
                    "comparable to LC using much less tokens. most queries, RAG scores and LC scores are "
                    "highly similar."
                ),
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 5,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            }
        ],
    }

    repaired = repair_payload(payload, "paper-010", Path("data/raw_pdfs/paper_010.pdf"))

    assert "Table 1 :" not in repaired["sections"][0]["text"]
    assert "Most queries" in repaired["sections"][0]["text"]


def test_repair_payload_drops_pure_table_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.parsing.parsed_json_repair._extract_keywords_from_pdf",
        lambda _: [],
    )

    payload = {
        "metadata": {
            "paper_id": "paper-011",
            "title": "Reasoning in Retrieval-Augmented Generation",
            "authors": [],
            "year": None,
            "venue": None,
            "abstract": "Example abstract.",
            "keywords": [],
            "section_titles": [],
        },
        "sections": [
            {
                "chunk_id": "paper-011_sec_0",
                "paper_id": "paper-011",
                "section_type": "results",
                "section_title": "Main results",
                "section_path": "Main results",
                "text": "Results text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 0,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-011_sec_1",
                "paper_id": "paper-011",
                "section_type": "other",
                "section_title": "NQ TriviaQA StrategyQA",
                "section_path": "NQ TriviaQA StrategyQA",
                "text": "Ret Sub Ret Sub Ret Sub Llama2 13B 3.4 5.3 2.9 4.0 3.6 6.8 Table 4 : Evaluation.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 16,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
            {
                "chunk_id": "paper-011_sec_2",
                "paper_id": "paper-011",
                "section_type": "analysis",
                "section_title": "Efficiency Analysis",
                "section_path": "Efficiency Analysis",
                "text": "Efficiency text.",
                "page_start": 0,
                "page_end": 0,
                "order_in_paper": 17,
                "level": 0,
                "parent_chunk_id": None,
                "granularity": "coarse",
            },
        ],
    }

    repaired = repair_payload(payload, "paper-011", Path("data/raw_pdfs/paper_011.pdf"))

    assert [section["section_title"] for section in repaired["sections"]] == [
        "Main results",
        "Efficiency Analysis",
    ]
