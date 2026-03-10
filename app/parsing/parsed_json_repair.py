from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


METADATA_OVERRIDES: dict[str, dict[str, str | int]] = {
    "paper-001": {
        "year": 2020,
        "venue": "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    },
    "paper-002": {
        "year": 2021,
        "venue": "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    },
    "paper-003": {
        "year": 2023,
        "venue": "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    },
    "paper-004": {
        "year": 2023,
        "venue": "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    },
    "paper-005": {
        "year": 2023,
        "venue": "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    },
    "paper-006": {
        "year": 2023,
        "venue": "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    },
    "paper-007": {
        "year": 2023,
        "venue": "Findings of the Association for Computational Linguistics: ACL 2023",
    },
    "paper-008": {
        "year": 2024,
        "venue": "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    },
    "paper-009": {
        "year": 2024,
        "venue": "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    },
    "paper-010": {
        "year": 2024,
        "venue": "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    },
    "paper-011": {
        "year": 2024,
        "venue": "Findings of the Association for Computational Linguistics: ACL 2024",
    },
    "paper-012": {
        "year": 2024,
        "venue": "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    },
    "paper-013": {
        "year": 2024,
        "venue": "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    },
    "paper-014": {
        "year": 2024,
        "venue": "Transactions of the Association for Computational Linguistics",
    },
    "paper-015": {
        "year": 2025,
        "venue": "Findings of the Association for Computational Linguistics: NAACL 2025",
    },
    "paper-016": {
        "year": 2023,
        "venue": "Journal of Machine Learning Research",
    },
    "paper-017": {
        "year": 2024,
        "venue": "arXiv",
    },
    "paper-018": {
        "year": 2023,
        "venue": "arXiv",
    },
    "paper-019": {
        "year": 2024,
        "venue": "arXiv",
    },
    "paper-020": {
        "year": 2024,
        "venue": "arXiv",
    },
    "paper-021": {
        "year": 2024,
        "venue": "arXiv",
    },
    "paper-022": {
        "year": 2020,
        "venue": "SIGIR '20: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval",
    },
    "paper-023": {
        "year": 2025,
        "venue": "WWW '25: Proceedings of the ACM on Web Conference 2025",
    },
    "paper-024": {
        "year": 2025,
        "venue": "ICLR 2025 (Under Review)",
    },
    "paper-025": {
        "year": 2024,
        "venue": "ICLR 2024",
    },
    "paper-026": {
        "year": 2024,
        "venue": "ICLR 2024",
    },
    "paper-027": {
        "year": 2024,
        "venue": "ICLR 2024",
    },
    "paper-028": {
        "year": 2020,
        "venue": "Proceedings of the 37th International Conference on Machine Learning",
    },
    "paper-029": {
        "year": 2020,
        "venue": "34th Conference on Neural Information Processing Systems (NeurIPS 2020)",
    },
    "paper-030": {
        "year": 2024,
        "venue": "38th Conference on Neural Information Processing Systems (NeurIPS 2024)",
    },
}

TITLE_OVERRIDES = {
    "paper-020": "RQ-RAG: LEARNING TO REFINE QUERIES FOR RETRIEVAL AUGMENTED GENERATION",
    "paper-021": "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads",
    "paper-026": "RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING",
    "paper-030": "xRAG: Extreme Context Compression for Retrieval-Augmented Generation with One Token",
}

AUTHOR_OVERRIDES = {
    "paper-014": [
        "Nelson F Liu",
        "Kevin Lin",
        "John Hewitt",
        "Ashwin Paranjape",
        "Michele Bevilacqua",
        "Fabio Petroni",
        "Percy Liang",
        "Kurt Shuster",
        "Jing Xu",
        "Mojtaba Komeili",
        "Da Ju",
        "Eric Michael Smith",
        "Stephen Roller",
        "Megan Ung",
        "Moya Chen",
        "Kushal Arora",
        "Joshua Lane",
        "Morteza Behrooz",
        "William Ngan",
        "Spencer Poff",
        "Naman Goyal",
        "Arthur Szlam",
        "Y-Lan Boureau",
        "Melanie Kambadur",
        "Jason Weston",
        "Simeng Sun",
        "Kalpesh Krishna",
        "Andrew Mattarella-Micke",
        "Yi Tay",
        "Mostafa Dehghani",
        "Vinh Q Tran",
        "Xavier Garcia",
        "Jason Wei",
        "Xuezhi Wang",
        "Won Chung",
        "Siamak Shakeri",
        "Dara Bahri",
        "Tal Schuster",
        "Steven Zheng",
        "Denny Zhou",
        "Neil Houlsby",
        "Donald Metzler",
        "Romal Thoppilan",
        "Daniel De Freitas",
        "Jamie Hall",
        "Noam Shazeer",
        "Apoorv Kulshreshtha",
        "Heng-Tze Cheng",
        "Alicia Jin",
        "Taylor Bos",
        "Leslie Baker",
        "Yu Du",
        "Yaguang Li",
        "Hongrae Lee",
        "Amin Ghafouri",
        "Marcelo Menegali",
        "Yanping Huang",
        "Maxim Krikun",
        "Dmitry Lepikhin",
        "James Qin",
        "Dehao Chen",
        "Yuanzhong Xu",
        "Zhifeng Chen",
        "Adam Roberts",
        "Maarten Bosma",
        "Vincent Zhao",
        "Yanqi Zhou",
        "Chung-Ching Chang",
        "Igor Krivokon",
        "Will Rusch",
        "Marc Pickett",
        "Pranesh Srinivasan",
        "Laichee Man",
        "Kathleen Meier-Hellstern",
        "Meredith Ringel Morris",
        "Tulsee Doshi",
        "Delos Santos",
        "Toju Duke",
        "Johnny Soraker",
        "Ben Zevenbergen",
        "Vinodkumar Prabhakaran",
        "Mark Diaz",
        "Ben Hutchinson",
        "Kristen Olson",
        "Alejandra Molina",
        "Erin Hoffman-John",
        "Josh Lee",
        "Lora Aroyo",
        "Ravi Rajakumar",
        "Alena Butryna",
        "Matthew Lamm",
        "Viktoriya Kuzmina",
        "Joe Fenton",
        "Aaron Cohen",
        "Rachel Bernstein",
        "Ray Kurzweil",
        "Blaise Aguera-Arcas",
        "Claire Cui",
        "Marian Croak",
        "Hugo Touvron",
        "Thibaut Lavril",
        "Gautier Izacard",
        "Xavier Martinet",
        "Marie-Anne Lachaux",
        "Timothee Lacroix",
        "Baptiste Roziere",
        "Eric Hambro",
        "Faisal Azhar",
        "Aurelien Rodriguez",
        "Armand Joulin",
        "Louis Martin",
        "Kevin Stone",
        "Peter Albert",
        "Amjad Almahairi",
        "Yasmine Babaei",
        "Nikolay Bashlykov",
        "Soumya Batra",
        "Prajjwal Bhargava",
        "Shruti Bhosale",
        "Dan Bikel",
        "Lukas Blecher",
        "Cristian Canton Ferrer",
        "Guillem Cucurull",
        "David Esiobu",
        "Jude Fernandes",
        "Jeremy Fu",
        "Wenyin Fu",
        "Brian Fuller",
        "Cynthia Gao",
        "Vedanuj Goswami",
        "Anthony Hartshorn",
        "Saghar Hosseini",
        "Rui Hou",
        "Hakan Inan",
        "Marcin Kardas",
        "Viktor Kerkez",
        "Madian Khabsa",
        "Isabel Kloumann",
        "Artem Korenev",
        "Singh Koura",
        "Jenya Lee",
        "Diana Liskovich",
        "Yinghai Lu",
        "Yuning Mao",
        "Todor Mihaylov",
        "Pushkar Mishra",
        "Igor Molybog",
        "Yixin Nie",
        "Andrew Poulton",
        "Jeremy Reizenstein",
        "Rashi Rungta",
        "Kalyan Saladi",
        "Alan Schelten",
        "Ruan Silva",
        "Xiang Kuan",
        "Puxin Xu",
        "Zheng Yan",
        "Iliyan Zarov",
        "Yuchen Zhang",
        "Angela Fan",
        "Sharan Narang",
        "Robert Stojnic",
        "Sergey Edunov",
        "Ashish Vaswani",
        "Niki Parmar",
        "Jakob Uszkoreit",
        "Llion Jones",
        "Aidan N Gomez",
        "Lukasz Kaiser",
        "Illia Polosukhin",
        "Sinong Wang",
        "Belinda Z Li",
        "Han Fang",
        "Manzil Zaheer",
        "Guru Guruganesh",
        "Avinava Kumar",
        "Joshua Dubey",
        "Chris Ainslie",
        "Santiago Alberti",
        "Philip Ontanon",
        "Anirudh Pham",
        "Qifan Ravula",
        "Li Wang",
        "Ahmed Yang",
    ],
    "paper-021": [
        "Guangxuan Xiao",
        "Jiaming Tang",
        "Jingwei Zuo",
        "Junxian Guo",
        "Shang Yang",
        "Haotian Tang",
        "Yao Fu",
        "Song Han",
    ],
    "paper-023": [
        "Hongjin Qian",
        "Zheng Liu",
        "Peitian Zhang",
        "Kelong Mao",
        "Defu Lian",
        "Zhicheng Dou",
    ],
    "paper-026": [
        "Xi Victoria Lin",
        "Xilun Chen",
        "Mingda Chen",
        "Weijia Shi",
        "Maria Lomeli",
        "Rich James",
        "Pedro Rodriguez",
        "Jacob Kahn",
        "Gergely Szilvasy",
        "Mike Lewis",
        "Luke Zettlemoyer",
        "Scott Yih",
    ],
}

SECTION_TITLE_OVERRIDES: dict[str, dict[int, str]] = {
    "paper-005": {
        1: "Introduction",
        2: "Introduction",
        12: "Masked sentences as implicit queries",
        13: "Generated questions as explicit queries",
    },
    "paper-008": {
        12: "Results and Discussion",
        13: "Models defeated by shuffled dataset, attention failure being the culprit",
        15: "General ability is preserved with PAM QA Training",
    },
    "paper-009": {
        3: "Question-Aware Coarse-Grained Compression",
        7: "How to improve the integrity of key information?",
        23: "LongBench Using LongChat-13b-16k",
        9: "Algorithm 1 Token-level Subsquence Recovery Algorithm",
        10: "Algorithm 1 Token-level Subsquence Recovery Algorithm",
        11: "Algorithm 1 Token-level Subsquence Recovery Algorithm",
        12: "Algorithm 1 Token-level Subsquence Recovery Algorithm",
    },
    "paper-010": {
        13: "Conclusion",
    },
    "paper-013": {
        19: "REPLUG performance gain does not simply come from the ensembling effect",
    },
    "paper-014": {
        6: "Results and Discussion",
        15: "Random Distractors in Multi-Document QA",
        17: "Llama-2 Performance",
    },
    "paper-017": {
        1: "Introduction",
        7: "Can LLMs Understand Prompts that Involve Sequential Historical User Behaviors?",
        8: "Can LLMs Understand Prompts that Involve Sequential Historical User Behaviors?",
        9: "Empirical Studies",
        10: "Empirical Studies",
        11: "How Well Can LLMs Rank Candidates in a Zero-Shot Setting?",
        12: "How Well Can LLMs Rank Candidates in a Zero-Shot Setting?",
    },
    "paper-020": {
        1: "RQ-RAG: Learning to Refine Query for Retrieval Augmented Generation",
        8: "RQ-RAG shows superior performance on multi-hop QA datasets",
        9: "RQ-RAG shows high upper bound of the system",
        11: "Our system is resilient to different data resources",
        12: "Related Works",
    },
    "paper-025": {
        1: "Introduction",
        2: "RELATED WORK",
        3: "SELF-RAG: LEARNING TO RETRIEVE, GENERATE AND CRITIQUE",
    },
    "paper-026": {
        17: "ADDITIONAL EXPERIMENTS E.1 SCALING LAWS OF RETRIEVAL AUGMENTED LANGUAGE MODEL FINE-TUNING",
    },
    "paper-030": {
        13: "Case Study",
        14: "Case Study",
    },
    "paper-023": {
        2: "Introduction",
        7: "Experiment",
    },
}

INSTITUTION_PATTERN = re.compile(
    r"\b(university|institute|school|laboratory|lab|college|meta|nvidia|fair|mit|sjtu)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    cleaned = text.replace("\u00a0", " ")
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = re.sub(r"(?<=\w)-\s+(?=\w)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_abstract(text: str) -> str:
    cleaned = _normalize_text(text)
    cleaned = re.sub(
        r"\s+\d+\s*\*+\s*(?:lead contributors?|equal contribution|corresponding author)\.?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,;:-*†")


def _split_keywords(keyword_text: str) -> list[str]:
    return [
        item.strip(" ,;:•·.")
        for item in re.split(r"[;,•·|]", _normalize_text(keyword_text))
        if item.strip(" ,;:•·.")
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _extract_first_page_text(pdf_path: Path) -> str:
    result = subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dFirstPage=1",
            "-dLastPage=1",
            "-sDEVICE=txtwrite",
            "-sOutputFile=-",
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _extract_keywords_from_pdf(pdf_path: Path) -> list[str]:
    first_page_text = _extract_first_page_text(pdf_path)
    match = re.search(r"Keywords:\s*(.+)", first_page_text, flags=re.IGNORECASE)
    if match is None:
        return []
    return _dedupe(_split_keywords(match.group(1)))


def _clean_authors(paper_id: str, authors: list[str]) -> list[str]:
    if paper_id in AUTHOR_OVERRIDES:
        return AUTHOR_OVERRIDES[paper_id]

    cleaned: list[str] = []
    for author in authors:
        name = _normalize_text(author).strip(" ,;:*†")
        if not name:
            continue
        if name.lower() == paper_id.replace("-", " "):
            continue
        if len(name.split()) == 1 and (INSTITUTION_PATTERN.search(name) or len(name) <= 2):
            continue
        if INSTITUTION_PATTERN.search(name):
            continue
        cleaned.append(name)
    return _dedupe(cleaned)


def _clean_title(title: str) -> str:
    cleaned = _normalize_text(title)
    cleaned = cleaned.replace("INSTRUC TION", "INSTRUCTION")
    cleaned = cleaned.replace("INSTRUC-TION", "INSTRUCTION")
    return cleaned


def repair_payload(payload: dict[str, object], paper_id: str, pdf_path: Path) -> dict[str, object]:
    metadata = payload["metadata"]
    sections = payload["sections"]

    if not isinstance(metadata, dict) or not isinstance(sections, list):
        raise ValueError(f"Invalid payload for {paper_id}")

    if paper_id in METADATA_OVERRIDES:
        metadata.update(METADATA_OVERRIDES[paper_id])

    metadata["title"] = TITLE_OVERRIDES.get(paper_id, _clean_title(str(metadata.get("title", ""))))
    metadata["abstract"] = _clean_abstract(str(metadata.get("abstract", "")))
    metadata["authors"] = _clean_authors(paper_id, list(metadata.get("authors", [])))

    if not metadata.get("keywords"):
        metadata["keywords"] = _extract_keywords_from_pdf(pdf_path)

    overrides = SECTION_TITLE_OVERRIDES.get(paper_id, {})
    for section in sections:
        if not isinstance(section, dict):
            continue
        order = int(section.get("order_in_paper", -1))
        title = str(section.get("section_title", ""))
        if order in overrides:
            section["section_title"] = overrides[order]
        elif title.islower():
            section["section_title"] = title.title()

    metadata["section_titles"] = [
        str(section["section_title"])
        for section in sections
        if isinstance(section, dict) and int(section.get("level", 0)) == 0
    ]

    return payload


def repair_directory(parsed_dir: Path, raw_pdf_dir: Path) -> list[Path]:
    repaired_files: list[Path] = []
    for json_path in sorted(parsed_dir.glob("*.json")):
        paper_id = json_path.stem
        pdf_path = raw_pdf_dir / f"{paper_id.replace('-', '_')}.pdf"
        if not pdf_path.exists():
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        repaired = repair_payload(payload, paper_id, pdf_path)
        json_path.write_text(
            json.dumps(repaired, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        repaired_files.append(json_path)
    return repaired_files
