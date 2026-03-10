from __future__ import annotations

from xml.sax.saxutils import escape

from openai import OpenAI

from app.core.logger import get_logger
from app.core.schemas import AgentState, Evidence, IntentType, PaperMetadata
from app.generation.citation_formatter import add_citation_marks, format_evidences_xml
from app.generation.prompts import (
    CANDIDATE_SUMMARY_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    PAPER_READING_PROMPT,
    SECTION_QA_PROMPT,
)


logger = get_logger(__name__)


class AnswerBuilder:
    def __init__(self, llm_api_key: str, llm_base_url: str, llm_model: str) -> None:
        self.client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self.llm_model = llm_model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def classify_intent(self, query: str, state: AgentState) -> IntentType:
        user_prompt = (
            f"Query: {query}\n"
            f"Current Paper ID: {state.current_paper_id}\n"
            f"Last Intent: {state.last_intent.value if state.last_intent else 'None'}\n"
            "Output:"
        )
        raw_output = self._call_llm(INTENT_CLASSIFICATION_PROMPT, user_prompt).strip().lower()
        if raw_output in {intent.value for intent in IntentType}:
            return IntentType(raw_output)

        for intent in IntentType:
            if intent.value in raw_output:
                return intent

        logger.warning("Unexpected intent classification output: %s", raw_output)
        return IntentType.PAPER_SEARCH

    def generate_paper_reading(
        self,
        query: str,
        evidences: list[Evidence],
        metadata: PaperMetadata,
    ) -> str:
        prompt = PAPER_READING_PROMPT.format(
            query=query,
            title=metadata.title,
            authors=", ".join(metadata.authors) or "未知",
            year=metadata.year if metadata.year is not None else "未知",
            venue=metadata.venue or "未知",
            evidences_xml=format_evidences_xml(evidences, metadata),
        )
        answer = self._call_llm(
            "你是严谨的论文阅读助手，只能根据给定证据回答。",
            prompt,
        )
        return add_citation_marks(answer, evidences)

    def generate_section_qa(
        self,
        query: str,
        evidences: list[Evidence],
        metadata: PaperMetadata,
    ) -> str:
        prompt = SECTION_QA_PROMPT.format(
            query=query,
            title=metadata.title,
            authors=", ".join(metadata.authors) or "未知",
            year=metadata.year if metadata.year is not None else "未知",
            venue=metadata.venue or "未知",
            evidences_xml=format_evidences_xml(evidences, metadata),
        )
        answer = self._call_llm(
            "你是严谨的论文问答助手，只能根据给定证据回答。",
            prompt,
        )
        return add_citation_marks(answer, evidences)

    def generate_candidates_summary(
        self,
        candidates: list[PaperMetadata],
        query: str,
    ) -> str:
        candidates_xml = self._format_candidates_xml(candidates)
        return self._call_llm(
            "你是论文推荐助手，请根据候选论文列表生成简明总结。",
            CANDIDATE_SUMMARY_PROMPT.format(query=query, candidates_xml=candidates_xml),
        )

    def _format_candidates_xml(self, candidates: list[PaperMetadata]) -> str:
        if not candidates:
            return "<papers />"

        paper_nodes: list[str] = []
        for paper in candidates:
            authors = "".join(f"<author>{escape(author)}</author>" for author in paper.authors)
            keywords = "".join(f"<keyword>{escape(keyword)}</keyword>" for keyword in paper.keywords)
            year = "" if paper.year is None else str(paper.year)
            venue = paper.venue or ""
            paper_nodes.append(
                "<paper>"
                f"<paper_id>{escape(paper.paper_id)}</paper_id>"
                f"<title>{escape(paper.title)}</title>"
                f"<authors>{authors}</authors>"
                f"<year>{escape(year)}</year>"
                f"<venue>{escape(venue)}</venue>"
                f"<abstract>{escape(paper.abstract)}</abstract>"
                f"<keywords>{keywords}</keywords>"
                "</paper>"
            )
        return f"<papers>{''.join(paper_nodes)}</papers>"
