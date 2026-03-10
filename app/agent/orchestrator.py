from __future__ import annotations

from dataclasses import replace

from app.agent.intent_router import IntentRouter
from app.agent.state_manager import StateManager
from app.agent.tools import AgentTools
from app.core.schemas import AgentResponse, AgentState, Evidence, IntentType, SectionChunk
from app.generation.answer_builder import AnswerBuilder


class Orchestrator:
    def __init__(
        self,
        intent_router: IntentRouter,
        state_manager: StateManager,
        tools: AgentTools,
        answer_builder: AnswerBuilder,
    ) -> None:
        self.intent_router = intent_router
        self.state_manager = state_manager
        self.tools = tools
        self.answer_builder = answer_builder

    def run(self, query: str, state: AgentState) -> AgentResponse:
        intent = self.intent_router.classify(query, state)
        updated_state = self.state_manager.transition(state, intent, query)

        if intent == IntentType.PAPER_SEARCH:
            candidates = self.tools.search_papers(query)
            updated_state = replace(
                updated_state,
                candidate_papers=[candidate.paper_id for candidate in candidates],
            )
            if len(candidates) == 1:
                updated_state = self.state_manager.lock_paper(
                    updated_state,
                    candidates[0].paper_id,
                )
            answer = self.answer_builder.generate_candidates_summary(candidates, query)
            return AgentResponse(
                answer=answer,
                evidences=[],
                intent=intent,
                updated_state=updated_state,
                candidate_papers=candidates,
            )

        paper_id = updated_state.current_paper_id
        if paper_id is None:
            raise RuntimeError(
                f"Expected a locked paper_id for intent={intent}, but current_paper_id is None. "
                "IntentRouter should have converted this intent to PAPER_SEARCH."
            )

        metadata = self.tools.get_paper_metadata(paper_id)
        if metadata is None:
            raise ValueError(f"Paper metadata not found for paper_id={paper_id}")

        if intent == IntentType.PAPER_READING:
            sections = self.tools.retrieve_sections(
                paper_id,
                query,
                target_sections=["abstract", "conclusion"],
            )
            sections.extend(
                self.tools.retrieve_sections(
                    paper_id,
                    query,
                    target_sections=["method", "experiment"],
                )
            )
            unique_sections = self._deduplicate_sections(sections)
            evidences = self._build_evidences(unique_sections)
            answer = self.answer_builder.generate_paper_reading(query, evidences, metadata)
            updated_state = self.state_manager.to_answering(updated_state)
            return AgentResponse(
                answer=answer,
                evidences=evidences,
                intent=intent,
                updated_state=updated_state,
                candidate_papers=None,
            )

        sections = self.tools.retrieve_sections(paper_id, query, top_k=3)
        evidences = self._build_evidences(sections)
        answer = self.answer_builder.generate_section_qa(query, evidences, metadata)
        updated_state = self.state_manager.to_answering(updated_state)
        updated_state = replace(
            updated_state,
            current_focus_section=sections[0].section_type if sections else None,
        )
        return AgentResponse(
            answer=answer,
            evidences=evidences,
            intent=intent,
            updated_state=updated_state,
            candidate_papers=None,
        )

    def _build_evidences(self, sections: list[SectionChunk]) -> list[Evidence]:
        return [
            Evidence(
                text=section.text,
                paper_id=section.paper_id,
                section_type=section.section_type,
                section_title=section.section_title,
                page=section.page_start,
            )
            for section in sections
        ]

    def _deduplicate_sections(self, sections: list[SectionChunk]) -> list[SectionChunk]:
        deduplicated: list[SectionChunk] = []
        seen_chunk_ids: set[str] = set()
        for section in sections:
            if section.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(section.chunk_id)
            deduplicated.append(section)
        return deduplicated
