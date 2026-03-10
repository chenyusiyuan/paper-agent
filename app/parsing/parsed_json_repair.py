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
    "paper-024": "CORRECTIVE RETRIEVAL AUGMENTED GENERATION",
    "paper-026": "RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING",
    "paper-030": "xRAG: Extreme Context Compression for Retrieval-Augmented Generation with One Token",
}

AUTHOR_OVERRIDES = {
    "paper-005": [
        "Zhengbao Jiang",
        "Frank F Xu",
        "Luyu Gao",
        "Zhiqing Sun",
        "Qian Liu",
        "Jane Dwivedi-Yu",
        "Yiming Yang",
        "Jamie Callan",
        "Graham Neubig",
    ],
    "paper-010": [
        "Zhuowan Li",
        "Cheng Li",
        "Mingyang Zhang",
        "Qiaozhu Mei",
        "Michael Bendersky",
    ],
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
    "paper-017": [
        "Yupeng Hou",
        "Junjie Zhang",
        "Zihan Lin",
        "Hongyu Lu",
        "Ruobing Xie",
        "Julian McAuley",
        "Wayne Xin Zhao",
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


def _reindex_sections(paper_id: str, sections: list[dict[str, object]]) -> list[dict[str, object]]:
    for index, section in enumerate(sections):
        section["chunk_id"] = f"{paper_id}_sec_{index}"
        section["paper_id"] = paper_id
        section["order_in_paper"] = index
    return sections


def _clean_inline_block(text: str, pattern: str, replacement: str = "") -> str:
    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def _apply_known_payload_repairs(
    metadata: dict[str, object],
    sections: list[dict[str, object]],
    paper_id: str,
) -> list[dict[str, object]]:
    by_order = {
        int(section.get("order_in_paper", -1)): section
        for section in sections
        if isinstance(section, dict)
    }

    if paper_id == "paper-005" and 0 in by_order and 2 in by_order:
        intro = str(by_order[0].get("text", ""))
        intro_prefix = intro.split(
            "For example, to generate a summary about a particular topic, the initial retrieval based on the topic name",
            1,
        )[0]
        intro_continuation = str(by_order[2].get("text", ""))
        if "details." in intro_continuation:
            intro_continuation = intro_continuation.split("details.", 1)[1].strip()
        by_order[0]["text"] = (
            intro_prefix
            + "For example, to generate a summary about a particular topic, the initial retrieval based on the topic name "
            + "(e.g., Joe Biden) may not cover all aspects and details.\n"
            + intro_continuation
        ).strip()
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in {1, 2}]

    if paper_id == "paper-012":
        if 0 in by_order:
            text = str(by_order[0].get("text", ""))
            text = re.sub(
                r"Figure 1 : QA performance .*? base LLM\.\s*",
                "",
                text,
                flags=re.DOTALL,
            )
            text = re.sub(
                r",\s*furthermore,\s*In response to a query,.*?classifier\.\s*often",
                ", furthermore, often",
                text,
                flags=re.DOTALL,
            )
            by_order[0]["text"] = text.strip()
        if 6 in by_order:
            datasets_text = str(by_order[6].get("text", ""))
            datasets_text = re.sub(
                r"Table 1 : Averaged results .*? easy comparisons\.\s*",
                "",
                datasets_text,
                flags=re.DOTALL,
            )
            if 7 in by_order:
                continuation = re.sub(
                    r"^FLAN-T5-XXL \(11B\) GPT-3\.5 \(Turbo\)\s*",
                    "",
                    str(by_order[7].get("text", "")),
                )
                datasets_text = f"{datasets_text.rstrip()}\n{continuation.strip()}".strip()
            by_order[6]["text"] = datasets_text
        if 10 in by_order and 11 in by_order:
            extra = re.sub(
                r"^Figure 3 : .*?\(Right\)\.\s*",
                "",
                str(by_order[11].get("text", "")),
                flags=re.DOTALL,
            )
            by_order[10]["text"] = f"{str(by_order[10].get('text', '')).rstrip()}{extra}"
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in {7, 11}]

    if paper_id == "paper-015" and 0 in by_order and 2 in by_order:
        intro = str(by_order[0].get("text", ""))
        intro_prefix = intro.split("Can you find me some interesting things to do?", 1)[0]
        intro_continuation = str(by_order[2].get("text", ""))
        intro_continuation = re.sub(r"^Figure 1 : .*?\.\s*", "", intro_continuation, flags=re.DOTALL)
        by_order[0]["text"] = f"{intro_prefix}{intro_continuation}".strip()
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in {1, 2}]

    if paper_id == "paper-016":
        if 45 in by_order:
            by_order[45]["text"] = _clean_inline_block(
                str(by_order[45].get("text", "")),
                r"More details on the hyperparameters and .*?development set results are reported in Appendix A\.3\.\s*",
                "More details on the hyperparameters and development set results are reported in Appendix A.3. ",
            ).strip()
        if 47 in by_order:
            text = str(by_order[47].get("text", ""))
            text = text.replace(
                "As [0, 1) [1, 2) [2, 4) [4, 8) [8, 16) [16 shown in the left panel of Figure 3 ,",
                "As shown in the left panel of Figure 3,",
            )
            text = text.replace(
                " 6. Note: Depending on the question, it may not be important or useful to retrieve the exact text of the answer in MMLU, and as such, a hits@k value of 30% does not imply that retrieval fails to surface useful information in 70% of cases",
                "",
            )
            by_order[47]["text"] = text.strip()
        if 48 in by_order:
            text = str(by_order[48].get("text", ""))
            text = _clean_inline_block(
                text,
                r"for example, Question: Theo Walcott plays for ___ Answer: Arsenal F\.C\. \( 2017 Using this data set,",
                (
                    "for example, Question: Theo Walcott plays for ___ Answer: Arsenal F.C. (2017), "
                    "Everton F.C. (2020), and form a small training set of 248 training, 112 development "
                    "and 806 test questions.\nUsing this data set,"
                ),
            )
            text = _clean_inline_block(
                text,
                r"Table 12 : Impact of index data temporality on Natural Questions\..*?leads to the best result\.\s*",
                "",
            )
            by_order[48]["text"] = text.strip()
        if 51 in by_order:
            by_order[51]["text"] = _clean_inline_block(
                str(by_order[51].get("text", "")),
                r"Table 13 : MMLU scores with de-biasing\.\s*$",
                "",
            ).strip()
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in {3, 4}]

    if paper_id == "paper-017":
        metadata["abstract"] = str(metadata.get("abstract", "")).replace(
            "Recently, large language models (LLMs) (e.g., have",
            "Recently, large language models (LLMs) (e.g., GPT-4) have",
            1,
        )
        if 0 in by_order and 2 in by_order:
            appendix_finding = str(by_order[2].get("text", ""))
            appendix_finding = appendix_finding.replace("Ranking w/ LLMs (e.g. ChatGPT)", "")
            appendix_finding = appendix_finding.replace("Parsing outputs", "")
            appendix_finding = appendix_finding.replace("🏅 🥈 🥉", "")
            appendix_finding = appendix_finding.strip()
            by_order[0]["text"] = f"{str(by_order[0].get('text', '')).rstrip()}\n{appendix_finding}"
        for order in (3, 5):
            if order not in by_order:
                continue
            text = str(by_order[order].get("text", ""))
            text = re.sub(r"^Historical User Behaviors\?\s*", "", text)
            text = text.replace(
                "Historical User Behaviors? In LLM-based methods, historical interactions are naturally arranged in an ordered sequence.",
                "In LLM-based methods, historical interactions are naturally arranged in an ordered sequence.",
                1,
            )
            text = _clean_inline_block(
                text,
                r"Table 2 : Performance comparison on randomly retrieved candidates\..*?$",
                "",
            )
            by_order[order]["text"] = text.strip()
        for order in (6, 8, 10):
            if order in by_order:
                by_order[order]["text"] = re.sub(
                    r"^N@1 N@5 N@10 N@20 N@1 N@5 N@10 N@20\s*",
                    "",
                    str(by_order[order].get("text", "")),
                ).strip()
        if 7 in by_order:
            by_order[7]["text"] = str(by_order[7].get("text", "")).replace(
                "usually will not 0 5 10 15 20 Ground-Truth Item Pos. affect",
                "usually will not affect",
            )
        if 9 in by_order:
            by_order[9]["text"] = str(by_order[9].get("text", "")).replace(
                "We follow the same setting in Section items are randomly retrieved.",
                "We follow the same setting in Section 3.1 where items are randomly retrieved.",
            )
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in {1, 2}]

    if paper_id == "paper-024":
        banner = "Under review as a conference paper at ICLR 2025"
        for section in sections:
            text = str(section.get("text", ""))
            section["text"] = text.replace(banner, "").strip()
        if 3 in by_order:
            text = str(by_order[3].get("text", ""))
            if "After optimizing the retrieval results, an arbitrary generative model can be adopted." in text:
                text = text.split(
                    "After optimizing the retrieval results, an arbitrary generative model can be adopted.",
                    1,
                )[0] + "After optimizing the retrieval results, an arbitrary generative model can be adopted."
            by_order[3]["text"] = text.strip()
        if 16 in by_order:
            text = str(by_order[16].get("text", ""))
            text = _clean_inline_block(text, r"PopQA \(Mallen et al\., 2023\).*", "")
            by_order[16]["text"] = text.strip()

    if paper_id == "paper-003" and 13 in by_order:
        by_order[13]["text"] = _clean_inline_block(
            str(by_order[13].get("text", "")),
            r"Table 6 : .*",
            "",
        ).strip()

    if paper_id == "paper-010" and 5 in by_order:
        by_order[5]["text"] = _clean_inline_block(
            str(by_order[5].get("text", "")),
            r"\s*Table 1 : Results of Gemini-1\.5-Pro, GPT-3\.5-Turbo, and GPT-4O using the Contriever retriever\..*?much less tokens\.\s*most queries,",
            " Most queries,",
        ).strip()

    if paper_id == "paper-011":
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) != 16]

    if paper_id == "paper-013" and 14 in by_order:
        by_order[14]["text"] = _clean_inline_block(
            str(by_order[14].get("text", "")),
            r"Table 1 : Both REPLUG and REPLUG LSR consistently enhanced the performance of different language models\..*?original language model\.\s*",
            "",
        ).strip()

    if paper_id == "paper-014":
        if 9 in by_order:
            text = str(by_order[9].get("text", ""))
            text = _clean_inline_block(text, r"Figure 9 : .*?slightly decreases\.\s*", "")
            text = _clean_inline_block(text, r"Figure 10 : .*?performance trends\.\s*", "")
            by_order[9]["text"] = text.strip()
        if 15 in by_order:
            text = str(by_order[15].get("text", ""))
            text = re.sub(r"^Multi-Document QA\s*", "", text)
            text = _clean_inline_block(
                text,
                r"Figure 13 : .*?rather than retrieved distractors\.\s*",
                "",
            )
            text = _clean_inline_block(
                text,
                r"Figure 14 presents the results of this experiment\..*?Figure 14 : .*?the prompt\.\s*",
                "Figure 14 presents the results of this experiment. ",
            )
            by_order[15]["text"] = text.strip()
        if 17 in by_order:
            text = str(by_order[17].get("text", ""))
            text = _clean_inline_block(
                text,
                r"with and without additional Figure 15 : .*?input context\.\s*Figure 16 : .*?models\.\s*supervised fine-tuning",
                "with and without additional supervised fine-tuning",
            )
            by_order[17]["text"] = text.strip()

    if paper_id == "paper-019":
        if 2 in by_order:
            by_order[2]["text"] = _clean_inline_block(
                str(by_order[2].get("text", "")),
                r"Figure 2 : Overview of our RAFT method\..*?At test time, all methods follow the standard RAG setting, provided with a top-k retrieved documents in the context\.\s*",
                "",
            ).strip()
        if 15 in by_order:
            by_order[15]["text"] = _clean_inline_block(
                str(by_order[15].get("text", "")),
                r"Figure 2 : RAG-Token document posterior .*?The posterior for document 1 is high when generating \"A Farewell to Arms\" and for document 2 when generating \"The Sun Also Rises\"\.\s*",
                "",
            ).strip()
        if 7 in by_order:
            by_order[7]["text"] = _clean_inline_block(
                str(by_order[7].get("text", "")),
                r"Question: The Oberoi family .*?Table 1 : RAFT improves RAG performance for all specialized domains: .*?$",
                "",
            ).strip()

    if paper_id == "paper-021" and 6 in by_order:
        by_order[6]["text"] = _clean_inline_block(
            str(by_order[6].get("text", "")),
            r"DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads .*?Figure 6 : .*?GQA model\.\s*",
            "",
        ).strip()

    if paper_id == "paper-022":
        if 11 in by_order:
            text = str(by_order[11].get("text", ""))
            text = _clean_inline_block(
                text,
                r"We Method MRR@10 .*?Table 2 : End-to-end retrieval results on MS MARCO\..*?document collection\.\s*compare against KNRM",
                "We compare against KNRM",
            )
            by_order[11]["text"] = text.strip()
        if 14 in by_order:
            by_order[14]["text"] = _clean_inline_block(
                str(by_order[14].get("text", "")),
                r"Figure 6 Table 4 : Space Footprint vs MRR@10 \(Dev\) on MS MARCO\.\s*",
                "",
            ).strip()

    if paper_id == "paper-025":
        if 0 in by_order and 1 in by_order:
            continuation = str(by_order[1].get("text", ""))
            continuation = _clean_inline_block(
                continuation,
                r"^Step 1: Retrieve K documents .*?Some states including Texas and Utah, are named after\s*",
                "",
            )
            by_order[0]["text"] = (
                str(by_order[0].get("text", "")).rstrip()
                + " left), which "
                + continuation.lstrip()
            ).strip()
            sections = [section for section in sections if int(section.get("order_in_paper", -1)) != 1]
        if 2 in by_order:
            by_order[2]["text"] = _clean_inline_block(
                str(by_order[2].get("text", "")),
                r"^x, y \{5, 4, 3, 2, 1\} y is a useful response to x\.\s*Table 1 : .*?x, y, d indicate input, output, and a relevant passage, respectively\.\s*",
                "",
            ).strip()
        for order, table_num in ((32, 8), (34, 9)):
            if order in by_order:
                by_order[order]["text"] = _clean_inline_block(
                    str(by_order[order].get("text", "")),
                    rf"Table {table_num} : .*",
                    "",
                ).strip()

    if paper_id == "paper-026" and 3 in by_order:
        by_order[3]["text"] = _clean_inline_block(
            str(by_order[3].get("text", "")),
            r"For each example \(x i , y i \) ∈ Table 1 : .*?\. D L , we retrieve",
            "For each example (x_i, y_i) in D_L, we retrieve",
        ).strip()

    if paper_id == "paper-027":
        if 4 in by_order:
            text = str(by_order[4].get("text", ""))
            text = _clean_inline_block(
                text,
                r"Figure 4 : Querying Process: .*?higher-layer summaries\.\s*our results demonstrate",
                "our results demonstrate",
            )
            text = _clean_inline_block(
                text,
                r"Table 1 : NarrativeQA Performance .*?Likewise, in the QuALITY dataset as shown in Table 4 ,",
                "Likewise, in the QuALITY dataset as shown in Table 4,",
            )
            text = _clean_inline_block(
                text,
                r"Table 3 : Controlled comparison of F-1 scores on the QASPER dataset,.*?$",
                "",
            )
            by_order[4]["text"] = text.strip()
        if 6 in by_order:
            text = str(by_order[6].get("text", ""))
            text = _clean_inline_block(
                text,
                r"Table 6 : Performance comparison .*?Kočiskỳ et al\., 2018\) 6\.2 5\.7 0\.3 3\.7 BM25 \+ BERT .*?Recursively Summarizing Books \(Wu et al\., 2021\) 21\s*",
                "",
            )
            by_order[6]["text"] = text.strip()
        if 14 in by_order:
            by_order[14]["text"] = _clean_inline_block(
                str(by_order[14].get("text", "")),
                r"Table 9 : Ablation study results comparing RAPTOR with a recency-based tree approach\s*",
                "",
            ).strip()

    if paper_id == "paper-029":
        if 0 in by_order:
            by_order[0]["text"] = _clean_inline_block(
                str(by_order[0].get("text", "")),
                r"Figure 1 : Overview of our approach\..*?",
                "",
            ).strip()
        if 15 in by_order:
            by_order[15]["text"] = _clean_inline_block(
                str(by_order[15].get("text", "")),
                r"Figure 2 : RAG-Token document posterior .*?The posterior for document 1 is high when generating \"A Farewell to Arms\" and for document 2 when generating \"The Sun Also Rises\"\.\s*",
                "",
            ).strip()

    if paper_id == "paper-030" and 25 in by_order:
        extra_orders = [26, 27, 28, 29, 30, 31]
        extra_chunks: list[str] = []
        for order in extra_orders:
            if order not in by_order:
                continue
            text = str(by_order[order].get("text", ""))
            text = _clean_inline_block(text, r"^Figure \d+ : .*?\n?", "")
            text = text.replace("RAG w/o Retrieval xRAG", "")
            text = text.replace("Background: [X]", "")
            extra_chunks.append(text.strip())
        merged = str(by_order[25].get("text", "")).rstrip()
        if extra_chunks:
            merged = f"{merged}\n" + "\n".join(chunk for chunk in extra_chunks if chunk)
        by_order[25]["text"] = merged.strip()
        sections = [section for section in sections if int(section.get("order_in_paper", -1)) not in set(extra_orders)]

    return _reindex_sections(
        paper_id,
        [section for section in sections if isinstance(section, dict)],
    )


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

    sections = _apply_known_payload_repairs(metadata, sections, paper_id)
    if paper_id == "paper-017":
        repaired_titles = {
            0: "Introduction",
            1: "Empirical Studies",
            2: "Empirical Studies",
            3: "How Well Can LLMs Rank Candidates in a Zero-Shot Setting?",
            4: "How Well Can LLMs Rank Candidates in a Zero-Shot Setting?",
            5: "Related Work",
            6: "Conclusion",
            7: "Limitations",
        }
        for section in sections:
            order = int(section.get("order_in_paper", -1))
            if order in repaired_titles:
                section["section_title"] = repaired_titles[order]
    payload["sections"] = sections

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
